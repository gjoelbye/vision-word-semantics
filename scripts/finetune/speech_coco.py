"""Rewrite SPEECH-COCO as webdataset shards and exclude the photographs with text.

``--shards`` reads the parquet export under ``spokencoco_hf/data/`` and writes
one train and one validation tar per parquet file under ``spokencoco-wds-288/``,
images resized to a short side of 288, two percent of the photographs held out
for validation. It also writes ``corpus/speech-coco_manifest.parquet`` and
``corpus/speech-coco_provenance.json``. ``--scan`` runs the CRAFT detector once
per photograph and writes ``corpus/speech-coco_text_scan.jsonl`` and
``corpus/speech-coco_excluded_keys.txt``, which lists every sample of a
photograph with any detected text. With no flag both stages run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import tarfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from src.config import CORPUS, DATA_ROOT, ensure_dirs
from src.finetune.corpus import PREFIX, SHARDS

#: The downloaded parquet export, recorded relative to the data root.
SOURCE = DATA_ROOT / "spokencoco_hf" / "data"

#: Short side of the written images; training crops 224 from it.
SHORT_SIDE = 288

#: Share of photographs held out for validation, chosen by photograph so that
#: the five captions of one photograph fall on one side.
VAL_PERCENT = 2

WORKERS = 12


def val_side(image_id: int) -> bool:
    """Whether a photograph is held out, by a crc32 hash of its COCO id."""
    return zlib.crc32(str(image_id).encode()) % 100 < VAL_PERCENT


def key_of(caption_id: int) -> str:
    """The sample name: the caption id as nine digits."""
    return f"{caption_id:09d}"


def source_shards() -> list[Path]:
    found = sorted(SOURCE.glob("*.parquet"))
    if not found:
        raise SystemExit(f"no parquet shards under {SOURCE}")
    return found


def resize(image: Image.Image) -> Image.Image:
    """Short side to ``SHORT_SIDE``, aspect kept, bicubic."""
    w, h = image.size
    scale = SHORT_SIDE / min(w, h)
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                        Image.BICUBIC)


def write_one(source: Path) -> dict:
    """Rewrite one parquet shard as a train tar and a val tar; return its manifest rows."""
    stem = source.stem.replace("train-", "")
    writers = {
        False: tarfile.open(SHARDS / f"spokencoco-train-{stem}.tar", "w"),
        True: tarfile.open(SHARDS / f"spokencoco-val-{stem}.tar", "w"),
    }
    rows = []
    table = pq.read_table(source, columns=["id", "image_id", "image", "text",
                                           "speaker", "gender", "speed",
                                           "duration"])
    frame = table.to_pandas()
    del table
    for row in frame.itertuples(index=False):
        key = key_of(int(row.id))
        held = val_side(int(row.image_id))
        try:
            image = Image.open(io.BytesIO(row.image["bytes"])).convert("RGB")
        except Exception:
            continue
        buffer = io.BytesIO()
        resize(image).save(buffer, format="JPEG", quality=95)
        payload = buffer.getvalue()
        caption = str(row.text).strip()
        meta = json.dumps({"key": key, "image_id": int(row.image_id),
                           "speaker": str(row.speaker),
                           "gender": str(row.gender),
                           "speed": float(row.speed),
                           "duration": float(row.duration)}).encode()
        for suffix, blob in ((".jpg", payload), (".txt", caption.encode()),
                             (".json", meta)):
            info = tarfile.TarInfo(key + suffix)
            info.size = len(blob)
            writers[held].addfile(info, io.BytesIO(blob))
        rows.append({"key": key, "caption_id": int(row.id),
                     "image_id": int(row.image_id), "text": caption,
                     "speaker": str(row.speaker), "gender": str(row.gender),
                     "speed": float(row.speed), "duration": float(row.duration),
                     "split": "val" if held else "train",
                     "source_shard": source.name})
    for writer in writers.values():
        writer.close()
    return {"shard": source.name, "rows": rows}


def build_shards() -> None:
    SHARDS.mkdir(parents=True, exist_ok=True)
    sources = source_shards()
    print(f"rewriting {len(sources)} parquet shards into {SHARDS.name}/")
    collected = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for result in tqdm(pool.map(write_one, sources), total=len(sources),
                           desc="shards", unit="shard"):
            collected.append(result)

    order = {s.name: i for i, s in enumerate(sources)}
    collected.sort(key=lambda r: order[r["shard"]])
    frame = pd.DataFrame([row for r in collected for row in r["rows"]])
    if frame.key.duplicated().any():
        raise SystemExit("caption ids are not unique across shards")
    frame.to_parquet(CORPUS / f"{PREFIX}manifest.parquet", index=False)

    counts = frame.split.value_counts().to_dict()
    record = {
        "corpus": "speech-coco",
        "source": str(SOURCE.relative_to(DATA_ROOT)),
        "shards": len(sources),
        "rows": int(len(frame)),
        "images": int(frame.image_id.nunique()),
        "train_rows": int(counts.get("train", 0)),
        "val_rows": int(counts.get("val", 0)),
        "val_images": int(frame[frame.split == "val"].image_id.nunique()),
        "short_side": SHORT_SIDE,
        "val_percent": VAL_PERCENT,
        "scan_rows": int(len(frame)),
        "recorded": dt.date.today().isoformat(),
    }
    (CORPUS / f"{PREFIX}provenance.json").write_text(
        json.dumps(record, indent=2) + "\n")
    print(f"{len(frame):,} samples over {record['images']:,} photographs")
    print(f"  train {record['train_rows']:,}   val {record['val_rows']:,}")


def build_scan() -> None:
    """Scan each photograph once, then exclude every sample that shows it."""
    import torch
    import webdataset as wds

    from src.textscan import is_excluded, reader, text_density

    path = CORPUS / f"{PREFIX}manifest.parquet"
    if not path.exists():
        raise SystemExit(f"no manifest at {path}; run with --shards")
    rows = pd.read_parquet(path)
    first = rows.drop_duplicates("image_id")
    wanted = dict(zip(first.key, first.image_id))
    detector = reader(torch.cuda.is_available())

    shards = sorted(str(p) for p in SHARDS.glob("*.tar"))
    # Select by key before decoding, so a photograph is decoded once.
    dataset = (wds.WebDataset(shards, shardshuffle=False, empty_check=False,
                              handler=wds.warn_and_continue)
               .select(lambda sample: sample.get("__key__") in wanted)
               .decode("pil", handler=wds.warn_and_continue)
               .to_tuple("__key__", "jpg", handler=wds.warn_and_continue))
    out = CORPUS / f"{PREFIX}text_scan.jsonl"
    seen = 0
    with open(out, "w") as fh:
        for key, image in tqdm(dataset, total=len(wanted), desc="scan",
                               unit="img"):
            record = {"key": key, "image_id": int(wanted[key])}
            record |= text_density(detector, image)
            fh.write(json.dumps(record) + "\n")
            seen += 1
            if seen >= len(wanted):
                break
    print(f"{seen:,} photographs scanned")

    dropped_images = set()
    with open(out) as fh:
        for line in fh:
            record = json.loads(line)
            if is_excluded(record):
                dropped_images.add(int(record["image_id"]))
    excluded = rows[rows.image_id.isin(dropped_images)].key.tolist()
    (CORPUS / f"{PREFIX}excluded_keys.txt").write_text(
        "\n".join(sorted(excluded)) + "\n")
    print(f"{len(dropped_images):,} photographs carry text, "
          f"{len(excluded):,} of {len(rows):,} samples excluded")

    record_path = CORPUS / f"{PREFIX}provenance.json"
    record = json.loads(record_path.read_text())
    record |= {"scanned_images": seen,
               "excluded_images": len(dropped_images),
               "n_excluded": len(excluded),
               "scan_recorded": dt.date.today().isoformat()}
    record_path.write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", action="store_true",
                    help="rewrite the parquet export as webdataset shards")
    ap.add_argument("--scan", action="store_true",
                    help="detect text in the photographs and write the exclude list")
    a = ap.parse_args()
    ensure_dirs()
    both = not (a.shards or a.scan)
    if a.shards or both:
        build_shards()
    if a.scan or both:
        build_scan()


if __name__ == "__main__":
    main()
