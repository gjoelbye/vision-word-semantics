"""Compute the frozen caption targets the SPEECH-COCO finetunes are trained against.

The written captions are embedded with Qwen3-Embedding-0.6B, pooled at the last
token of a 77-token left-padded context. The spoken captions are embedded with
w2v-BERT 2.0, averaged over the unpadded encoder frames of each recording. Each
cache is written under `data/captions/speech-coco/<tower>/` as float16
`vectors.npy`, `keys.txt` (sample keys) and `meta.json`. A finished cache
is skipped, and the speech pass resumes after the last finished parquet shard.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from src.config import ensure_dirs
from src.features.speech import SAMPLE_RATE
from src.features.text import Qwen3Tower, tokenizer
from src.finetune import data
from src.models import ENCODERS

TEXT_TOWER = "qwen3-0p6b-frozen"
SPEECH_TOWER = "w2v-bert-2.0-frozen"

# Token context of the written captions.
CAPTION_CONTEXT = 77
TEXT_BATCH = 256

# Recordings per forward pass.
SPEECH_BATCH = 32


def written(device: str) -> None:
    """Embed every written caption in the shards with Qwen3-0.6B, in shard order, and write the cache."""
    import webdataset as wds

    repo, dim, _ = ENCODERS[TEXT_TOWER]
    out = data.caption_dir(TEXT_TOWER)
    out.mkdir(parents=True, exist_ok=True)
    tower = Qwen3Tower(repo).to(device).eval()
    tokenize = tokenizer(TEXT_TOWER, CAPTION_CONTEXT)

    shards = data.shards()
    dataset = (wds.WebDataset(shards, shardshuffle=False,
                              handler=wds.warn_and_continue, empty_check=False)
               .decode(handler=wds.warn_and_continue)
               .to_tuple("__key__", "txt", handler=wds.warn_and_continue))

    keys: list[str] = []
    blocks: list[np.ndarray] = []
    pending_keys: list[str] = []
    pending_text: list[str] = []

    def flush() -> None:
        if not pending_text:
            return
        batch = {k: v.to(device) for k, v in tokenize(pending_text).items()}
        with torch.no_grad():
            vectors = tower(batch).float().cpu().numpy().astype(np.float16)
        blocks.append(vectors)
        keys.extend(pending_keys)
        pending_keys.clear()
        pending_text.clear()

    for key, text in tqdm(dataset, desc=TEXT_TOWER, unit="cap"):
        text = (text or "").strip()
        if not text:
            continue
        pending_keys.append(key)
        pending_text.append(text)
        if len(pending_text) >= TEXT_BATCH:
            flush()
    flush()

    vectors = np.concatenate(blocks) if blocks else np.zeros((0, dim), np.float16)
    np.save(out / "vectors.npy", vectors)
    (out / "keys.txt").write_text("\n".join(keys))
    (out / "meta.json").write_text(json.dumps({
        "tower": TEXT_TOWER, "repo": repo, "dim": int(dim),
        "captions": int(len(keys)), "context": CAPTION_CONTEXT,
        "corpus": "speech-coco", "dtype": "float16",
        "shard_fingerprint": data.fingerprint(), "n_shards": len(shards),
        "recorded": dt.date.today().isoformat(),
    }, indent=2) + "\n")
    print(f"{len(keys):,} written captions -> captions/speech-coco/{TEXT_TOWER}")


def audio_rows(source: Path):
    """`(caption_id, waveform)` for each decodable recording of one parquet shard, at 16 kHz in float32."""
    from scipy.io import wavfile

    frame = pq.read_table(source, columns=["id", "audio"]).to_pandas()
    for row in frame.itertuples(index=False):
        try:
            rate, wave = wavfile.read(io.BytesIO(row.audio["bytes"]))
        except Exception:
            continue
        if rate != SAMPLE_RATE:
            from scipy.signal import resample_poly
            wave = resample_poly(wave, SAMPLE_RATE // 50, rate // 50)
        yield int(row.id), np.asarray(wave, dtype=np.float32) / 32768.0


def spoken(device: str) -> None:
    """Embed every recording with w2v-BERT 2.0, as the mean over unpadded frames, and write the cache."""
    from transformers import AutoFeatureExtractor, AutoModel

    repo, dim, _ = ENCODERS[SPEECH_TOWER]
    out = data.caption_dir(SPEECH_TOWER)
    out.mkdir(parents=True, exist_ok=True)
    features = AutoFeatureExtractor.from_pretrained(repo)
    tower = AutoModel.from_pretrained(repo, dtype=torch.float32).to(device).eval()

    sources = data.source_shards()
    total = len(pd.read_parquet(data.SPEECH_COCO_MANIFEST, columns=["key"]))

    # Resume at the first parquet shard not yet finished.
    cursor_path = out / "cursor.json"
    keys: list[str] = []
    first_shard = 0
    if cursor_path.exists() and (out / "vectors.npy").exists():
        state = json.loads(cursor_path.read_text())
        existing = np.load(out / "vectors.npy", mmap_mode="r")
        if existing.shape == (total, dim):
            first_shard = int(state["shards_done"])
            keys = [k for k in (out / "keys.txt").read_text().split("\n") if k]
            print(f"resuming at shard {first_shard} of {len(sources)}")
        del existing
    vectors = np.lib.format.open_memmap(
        out / "vectors.npy", mode="r+" if first_shard else "w+",
        dtype=np.float16, shape=(total, dim))

    def flush(waves: list[np.ndarray], ids: list[int]) -> None:
        batch = features(waves, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                         padding=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            hidden = tower(**batch).last_hidden_state.float()
        mask = batch.get("attention_mask")
        if mask is not None and mask.shape[1] != hidden.shape[1]:
            # The extractor stacks two mel frames per encoder frame; fold the mask the same way.
            mask = mask[:, :hidden.shape[1] * 2:2][:, :hidden.shape[1]]
        if mask is None:
            pooled = hidden.mean(1)
        else:
            weight = mask.to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * weight).sum(1) / weight.sum(1).clamp(min=1)
        pooled = pooled.cpu().numpy().astype(np.float16)
        start = len(keys)
        room = min(len(ids), vectors.shape[0] - start)
        vectors[start:start + room] = pooled[:room]
        keys.extend(data.key_of(i) for i in ids[:room])

    # Padding makes attention cost grow with the longest recording in a batch,
    # so a batch is capped by the work of SPEECH_BATCH recordings of 3.5 s.
    budget = SPEECH_BATCH * (3.5 * SAMPLE_RATE) ** 2

    def full(pending: list[np.ndarray], candidate: np.ndarray) -> bool:
        if not pending:
            return False
        if len(pending) >= SPEECH_BATCH:
            return True
        longest = max(max(len(w) for w in pending), len(candidate))
        return (len(pending) + 1) * longest ** 2 > budget

    waves: list[np.ndarray] = []
    ids: list[int] = []
    progress = tqdm(total=total, initial=len(keys), desc="audio", unit="utt")
    for index, source in enumerate(sources):
        if index < first_shard:
            continue
        for caption_id, wave in audio_rows(source):
            if full(waves, wave):
                flush(waves, ids)
                progress.update(len(ids))
                waves, ids = [], []
            waves.append(wave)
            ids.append(caption_id)
        if waves:
            flush(waves, ids)
            progress.update(len(ids))
            waves, ids = [], []
        # Written after the vectors, so the cursor never claims unwritten rows.
        vectors.flush()
        (out / "keys.txt").write_text("\n".join(keys) + "\n")
        cursor_path.write_text(json.dumps(
            {"shards_done": index + 1, "rows": len(keys)}) + "\n")
    progress.close()
    vectors.flush()

    if len(keys) < total:
        # Rows that failed to decode leave the tail unwritten; keep only what was written.
        del vectors
        trimmed = np.array(np.load(out / "vectors.npy", mmap_mode="r")[:len(keys)])
        np.save(out / "vectors.npy", trimmed)
    (out / "keys.txt").write_text("\n".join(keys) + "\n")
    (out / "meta.json").write_text(json.dumps({
        "tower": SPEECH_TOWER, "repo": repo, "dim": dim,
        "corpus": "speech-coco", "captions": len(keys),
        "pooling": "mean over unpadded encoder frames",
        "sample_rate": SAMPLE_RATE, "dtype": "float16",
        "shard_fingerprint": data.fingerprint(), "n_shards": len(sources),
        "recorded": dt.date.today().isoformat(),
    }, indent=2) + "\n")
    cursor_path.unlink(missing_ok=True)
    print(f"{len(keys):,} spoken captions -> captions/speech-coco/{SPEECH_TOWER}")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    ensure_dirs()
    device = "cuda"
    for tower, build in ((TEXT_TOWER, written), (SPEECH_TOWER, spoken)):
        if (data.caption_dir(tower) / "meta.json").exists():
            print(f"{tower}: cached")
            continue
        build(device)


if __name__ == "__main__":
    main()
