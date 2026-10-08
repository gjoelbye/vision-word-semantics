"""Class-token features of every image in a manifest, at ten depths or at the final block.

A backbone without a class token (SigLIP2) gives the mean over its patches
instead. Each cache is a float16 ``.cls.npy`` memory map with a ``.meta.json`` sidecar
that records the sha of the manifest it was built against, the layers kept and
how far the run got. ``src.analysis.directions`` refuses a cache whose sha is not the
current manifest's.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ..config import EMBEDDINGS, local_path
from ..models import free, has_class_token, hidden_size, load, select_layer_indices
from ..stimuli.render import fingerprint

N_LAYERS_SWEEP = 10

#: Batches between two writes of the resume cursor.
_CHECKPOINT_EVERY = 200

#: Threads decoding the next batch of PNGs while the current one runs.
_DECODE_WORKERS = 8


def _forward(model, proc, images: list[Image.Image], layers: list[int],
             device: str, class_token: bool, *, final_only: bool) -> np.ndarray:
    """``(images, layers, width)`` summary vectors.

    The summary is the class token, or the mean over all patches for a backbone
    that has none (SigLIP2).
    """
    px = proc(images=images, return_tensors="pt")["pixel_values"].to(
        device, dtype=next(model.parameters()).dtype)
    out = model(pixel_values=px, output_hidden_states=not final_only)
    hs = (out.last_hidden_state,) if final_only else out.hidden_states
    selected = [0] if final_only else layers
    if class_token:
        cls = torch.stack([hs[i][:, 0] for i in selected], 1)
    else:
        cls = torch.stack([hs[i].mean(1) for i in selected], 1)
    return cls.float().cpu().numpy()


def cache_paths(key: str, output_dir: Path = EMBEDDINGS) -> dict[str, Path]:
    """The ``.cls.npy`` features and the ``.meta.json`` sidecar of one model's cache."""
    return {"cls": output_dir / f"{key}.cls.npy",
            "meta": output_dir / f"{key}.meta.json"}


def _read_meta(key: str, output_dir: Path = EMBEDDINGS) -> dict | None:
    f = cache_paths(key, output_dir)["meta"]
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except json.JSONDecodeError:
        return None


def embed_manifest(key: str, manifest: pd.DataFrame, *, batch_size: int = 128,
                   output_dir: Path = EMBEDDINGS, layer_mode: str = "sweep",
                   manifest_sha: str | None = None,
                   refuse_stale: bool = False) -> Path:
    """Embed every manifest row with model ``key`` and cache the class tokens.

    ``layer_mode`` is ``"sweep"`` (ten depths) or ``"final"`` (last block
    only). A cache whose sidecar matches the manifest sha and layer mode is
    reused or resumed at its recorded cursor; any other cache is rebuilt, or
    raises when ``refuse_stale`` is set.
    """
    if layer_mode not in {"sweep", "final"}:
        raise ValueError("layer_mode must be 'sweep' or 'final'")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(key, output_dir)
    sha = fingerprint(manifest) if manifest_sha is None else manifest_sha
    meta = _read_meta(key, output_dir)
    config_matches = bool(
        meta
        and meta.get("sha") == sha
        and meta.get("layer_mode", "sweep") == layer_mode)

    if config_matches and meta.get("done") and paths["cls"].exists():
        print(f"{key}: cached")
        return paths["cls"]

    resume_at = 0
    if config_matches and paths["cls"].exists():
        resume_at = int(meta.get("cursor", 0))
        if resume_at:
            print(f"{key}: resuming at {resume_at:,} of {len(manifest):,}")
    elif meta and not config_matches and refuse_stale:
        raise RuntimeError(
            f"{key}: the cache was built for another manifest or layer mode; "
            f"delete {paths['meta']} to rebuild it")
    elif meta and not config_matches:
        print(f"{key}: cache configuration changed, re-embedding")

    t0 = time.time()
    device = "cuda"
    model, proc, n_layers, _ = load(key, device)
    final_only = layer_mode == "final"
    layers = ([n_layers] if final_only
              else select_layer_indices(n_layers, N_LAYERS_SWEEP))
    class_token = has_class_token(model)
    all_paths = [local_path(p) for p in manifest.image_path]
    d_hidden = hidden_size(model)

    mode = "r+" if resume_at and paths["cls"].exists() else "w+"
    cls_all = np.lib.format.open_memmap(
        paths["cls"], mode=mode, dtype=np.float16,
        shape=(len(all_paths), len(layers), d_hidden))

    def save_meta(cursor: int, done: bool) -> None:
        paths["meta"].write_text(json.dumps(
            dict(sha=sha, cursor=int(cursor), done=bool(done),
                 layers=[int(x) for x in layers], n_layers=int(n_layers),
                 dim=int(d_hidden), class_token=bool(class_token),
                 layer_mode=layer_mode)))

    # Futures are read in submission order, so rows keep the manifest order.
    decoder = ThreadPoolExecutor(max_workers=_DECODE_WORKERS)

    def _open(path: Path) -> Image.Image:
        with Image.open(path) as im:
            return im.convert("RGB")

    def submit(lo: int, hi: int) -> list:
        return [decoder.submit(_open, q) for q in all_paths[lo:hi]]

    i = resume_at
    bs = batch_size
    since_checkpoint = 0
    pending = submit(i, i + bs) if i < len(all_paths) else []
    pbar = tqdm(total=len(all_paths), initial=i, desc=f"{key} ({n_layers}L)",
                unit="img")
    with torch.no_grad():
        while i < len(all_paths):
            imgs = [f.result() for f in pending]
            try:
                c = _forward(model, proc, imgs, layers, device, class_token,
                             final_only=final_only)
            except torch.cuda.OutOfMemoryError:
                if bs == 1:
                    raise
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                tqdm.write(f"[OOM] batch -> {bs}")
                pending = submit(i, i + bs)
                continue
            nxt = i + len(imgs)
            pending = submit(nxt, nxt + bs) if nxt < len(all_paths) else []
            cls_all[i:i + len(imgs)] = c.astype(np.float16)
            i += len(imgs)
            pbar.update(len(imgs))
            since_checkpoint += 1
            if since_checkpoint >= _CHECKPOINT_EVERY:
                cls_all.flush()
                save_meta(i, done=False)
                since_checkpoint = 0
    pbar.close()
    decoder.shutdown(wait=False)
    cls_all.flush()
    save_meta(len(all_paths), done=True)
    free(model)

    gb = paths["cls"].stat().st_size / 1e9
    print(f"{key}: {len(all_paths)}x{len(layers)}x{d_hidden} in "
          f"{time.time() - t0:.0f}s -> {gb:.1f} GB")
    return paths["cls"]
