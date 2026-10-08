"""Select the 96 background photographs from the ImageNet-1k validation split.

About 9,000 validation photographs are sampled and center-cropped to 256
pixels. A photograph is kept when the frame is informative (luminance spread,
entropy and color, no dominant flat tone, little pure white or black) and its
center is neither blank nor too busy. The kept photographs are ranked by a CLIP
score against prompts for good and for poor photographs, and the best of each
class, up to 200, are checked for text by OCR. OCR reads the full-size
photograph upscaled to at least 1,200 pixels on the long edge, and three
high-pass versions of it, each also rotated 30 degrees either way. The first 96
with no text are written as ``backgrounds/bg_000.png`` to ``bg_095.png`` with
``backgrounds/backgrounds.parquet``. The candidate pools live in a temporary
directory.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.config import BACKGROUNDS, IMAGE_SIZE, SEED, ensure_dirs, manifest_path
from src.stimuli.imagenet import IMAGENET, image_key, square_resize
from src.stimuli.typography import n_slots
from src.text_detector import LINK_THRESHOLD, LOW_TEXT, TEXT_THRESHOLD

CLIP_REPO = "openai/clip-vit-base-patch16"

#: Validation photographs sampled, spread evenly over the shards.
N_SCAN = 9000

#: Candidates, at most one per class, passed to the OCR screen.
SHORTLIST = 200

#: A candidate is rejected on any OCR confidence at or above this.
OCR_THRESHOLD = 0.01

AESTHETIC_POSITIVE = [
    "a beautiful professional photograph",
    "a scenic landscape with soft natural light",
    "a vivid colourful nature photograph",
    "a well composed photograph with pleasing colours",
]
AESTHETIC_NEGATIVE = [
    "a blurry low quality snapshot",
    "a cluttered messy indoor photograph",
    "a plain flat empty background",
    "a document, screenshot, or product packaging",
]


def describe(img: Image.Image) -> dict:
    """Clutter, tone and color statistics of one square photograph."""
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    grey = rgb.mean(2)
    lap = (-4 * grey
           + np.roll(grey, 1, 0) + np.roll(grey, -1, 0)
           + np.roll(grey, 1, 1) + np.roll(grey, -1, 1))[1:-1, 1:-1]
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colourfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                          + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    hist = np.histogram(grey, bins=32, range=(0, 1))[0].astype(np.float64)
    p = hist / hist.sum()
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    c = grey[grey.shape[0] // 4: -grey.shape[0] // 4,
             grey.shape[1] // 4: -grey.shape[1] // 4]
    clap = (-4 * c + np.roll(c, 1, 0) + np.roll(c, -1, 0)
            + np.roll(c, 1, 1) + np.roll(c, -1, 1))[1:-1, 1:-1]
    return dict(clutter=float(lap.var()), centre_clutter=float(clap.var()),
                lum_std=float(grey.std()), colourfulness=colourfulness,
                entropy=entropy, flat_share=float(p.max()),
                near_white=float((grey > 0.92).mean()),
                near_black=float((grey < 0.08).mean()))


def load_candidates(n_scan: int, pool: Path) -> pd.DataFrame:
    """Save sampled validation photographs into ``pool``, square and full size; return their statistics."""
    import pyarrow.parquet as pq

    shards = sorted((IMAGENET / "data").glob("validation-*.parquet"))
    if not shards:
        raise SystemExit(f"no validation shards under {IMAGENET / 'data'}")
    rows, raw = [], pool / "square"
    full = pool / "full"
    raw.mkdir(exist_ok=True)
    full.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    per_shard = int(np.ceil(n_scan / len(shards)))
    for sh in tqdm(shards, desc="scanning shards"):
        t = pq.read_table(sh, columns=["image", "label"])
        idx = rng.choice(t.num_rows, min(per_shard, t.num_rows), replace=False)
        imgs, labels = t.column("image").to_pylist(), t.column("label").to_pylist()
        for i in idx:
            try:
                im = Image.open(io.BytesIO(imgs[i]["bytes"])).convert("RGB")
            except Exception:
                continue
            sq = square_resize(im, IMAGE_SIZE)
            key = image_key(imgs[i]["bytes"])
            path = raw / f"{key}.png"
            if not path.exists():
                sq.save(path, "PNG")
            # Watermarks are legible only at the original resolution.
            fpath = full / f"{key}.png"
            if not fpath.exists():
                im.save(fpath, "PNG")
            rows.append(dict(key=key, path=str(path), full_path=str(fpath),
                             label=int(labels[i]), **describe(sq)))
    return pd.DataFrame(rows).drop_duplicates("key").reset_index(drop=True)


#: High-pass radii as a fraction of the long edge, one per watermark scale.
HP_RADII = (0.008, 0.02, 0.04)


def _watermark_variants(path: str, upscale: int = 1200) -> dict[str, "np.ndarray"]:
    """The photograph upscaled, and its high-pass versions at the ``HP_RADII`` scales.

    Subtracting a heavy blur removes the scene's tones and leaves a faint
    overlay, such as a stock watermark, on flat gray.
    """
    from PIL import ImageFilter, ImageOps

    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = max(1.0, upscale / max(w, h))
    big = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    grey = big.convert("L")
    out = {"plain": np.asarray(big)}
    for frac in HP_RADII:
        r = max(2, int(round(frac * max(grey.size))))
        a = (np.asarray(grey, float)
             - np.asarray(grey.filter(ImageFilter.GaussianBlur(r)), float) + 128)
        out[f"hp{frac}"] = np.asarray(ImageOps.autocontrast(
            Image.fromarray(np.clip(a, 0, 255).astype("uint8")), cutoff=1))
    return out


def ocr_confidence(paths: list[str], gpu: bool) -> list[float]:
    """Strongest OCR confidence for a string of three or more characters with a
    letter, over every variant and rotation; 1.0 if OCR fails."""
    import easyocr

    reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    out = []
    for p in tqdm(paths, desc="ocr", unit="img"):
        best = 0.0
        try:
            for name, arr in _watermark_variants(p).items():
                rots = [None] if name == "plain" else [None, [30, -30]]
                for rot in rots:
                    for _, txt, c in reader.readtext(
                            arr, text_threshold=TEXT_THRESHOLD, low_text=LOW_TEXT,
                            link_threshold=LINK_THRESHOLD, rotation_info=rot):
                        t = str(txt).strip()
                        # Rules out single-glyph hits on foliage and rock texture.
                        if len(t) >= 3 and any(ch.isalpha() for ch in t):
                            best = max(best, float(c))
        except Exception:
            best = 1.0
        out.append(best)
    return out


def aesthetic_score(paths: list[str], gpu: bool) -> list[float]:
    """Mean CLIP similarity to the positive prompts minus the mean to the negative ones."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
    model = CLIPModel.from_pretrained(CLIP_REPO).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_REPO)

    def _as_tensor(x):
        # Some transformers versions return a model output rather than a tensor.
        return x if torch.is_tensor(x) else (
            x.pooler_output if hasattr(x, "pooler_output") else x.last_hidden_state)

    with torch.no_grad():
        t = proc(text=AESTHETIC_POSITIVE + AESTHETIC_NEGATIVE,
                 return_tensors="pt", padding=True).to(device)
        tf = _as_tensor(model.get_text_features(**t))
        tf = tf / tf.norm(dim=-1, keepdim=True)
        npos = len(AESTHETIC_POSITIVE)
        out = []
        for i in tqdm(range(0, len(paths), 64), desc="aesthetic"):
            batch = [Image.open(p).convert("RGB") for p in paths[i:i + 64]]
            v = proc(images=batch, return_tensors="pt").to(device)
            f = _as_tensor(model.get_image_features(**v))
            f = f / f.norm(dim=-1, keepdim=True)
            s = (f @ tf.T).float().cpu().numpy()
            out += (s[:, :npos].mean(1) - s[:, npos:].mean(1)).tolist()
    del model
    torch.cuda.empty_cache()
    return out


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    import torch

    ensure_dirs()
    gpu = torch.cuda.is_available()
    n = n_slots()

    # The candidates take about 4 GB, so the pool sits beside the output rather than in /tmp.
    with tempfile.TemporaryDirectory(dir=BACKGROUNDS, prefix="_pool_") as pool:
        df = load_candidates(N_SCAN, Path(pool))
        print(f"scanned {len(df)} ImageNet validation images "
              f"across {df.label.nunique()} classes -> selecting {n}")

        keep = df[(df.lum_std > 0.13) & (df.entropy > 4.0) & (df.flat_share < 0.20)
                  & (df.near_white < 0.15) & (df.near_black < 0.15)
                  & (df.colourfulness > 0.10)].copy()
        print(f"informative: {len(keep)}/{len(df)} survive")

        lo, hi = keep.centre_clutter.quantile([0.10, 0.70])
        keep = keep[(keep.centre_clutter >= lo) & (keep.centre_clutter <= hi)]
        print(f"legible centre (clutter in [{lo:.4f}, {hi:.4f}]): {len(keep)}")

        keep = keep.copy()
        keep["aesthetic"] = aesthetic_score(keep.path.tolist(), gpu)
        keep = keep.sort_values("aesthetic", ascending=False)
        short = keep.groupby("label", sort=False).head(1).head(SHORTLIST).copy()
        print(f"shortlist: {len(short)} images, at most one per class, "
              f"aesthetic {short.aesthetic.min():+.3f} to {short.aesthetic.max():+.3f}")

        short["ocr_conf"] = ocr_confidence(short.full_path.tolist(), gpu)
        clean = short[short.ocr_conf < OCR_THRESHOLD].reset_index(drop=True)
        print(f"text-free: {len(clean)}/{len(short)} carry no legible text")
        if len(clean) < n:
            raise SystemExit(f"only {len(clean)} clean candidates for {n} slots")

        chosen = clean.head(n).reset_index(drop=True)
        manifest = []
        for i, r in enumerate(chosen.itertuples()):
            out = BACKGROUNDS / f"bg_{i:03d}.png"
            Image.open(r.path).save(out, "PNG")
            manifest.append(dict(background_index=i, path=manifest_path(out),
                                 key=r.key, imagenet_label=int(r.label),
                                 clutter=round(r.clutter, 4),
                                 centre_clutter=round(r.centre_clutter, 5),
                                 entropy=round(r.entropy, 3),
                                 colourfulness=round(r.colourfulness, 4),
                                 aesthetic=round(r.aesthetic, 4),
                                 ocr_conf=round(r.ocr_conf, 3),
                                 sha256_16=hashlib.sha256(out.read_bytes()).hexdigest()[:16]))
    m = pd.DataFrame(manifest)
    m.to_parquet(BACKGROUNDS / "backgrounds.parquet")
    print(f"\nwrote {len(m)} backgrounds to backgrounds/")
    print(f"  {m.imagenet_label.nunique()} distinct classes | "
          f"aesthetic {m.aesthetic.min():+.3f} to {m.aesthetic.max():+.3f} | "
          f"max OCR confidence {m.ocr_conf.max():.3f}")


if __name__ == "__main__":
    main()
