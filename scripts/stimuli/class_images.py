"""Draw the class names on the 96 slots, write the class photographs, and scan them for text.

The names go into the ``renders/`` tree at the text centers recorded for the
vocabulary, and ``concepts/manifest.parquet`` lists them. The photographs are
every ImageNet-1k validation image except the 96 backgrounds, center-cropped
to 256 pixels under ``concepts/images/``. ``concepts/images.parquet`` lists
them with the CRAFT box count and text area of each. Existing images are kept,
and the text scan runs again on every photograph.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src import concepts
from src.audit import summarise
from src.config import IMAGE_SIZE, LEXICON, ensure_dirs, local_path, manifest_path
from src.render import CENTRES_PATH, build_manifest, fingerprint
from src.slots import n_slots
from src.textscan import reader, text_density

WORKERS = 12


def render() -> None:
    """Draw the class names on every slot at the vocabulary's text centers and write their manifest."""
    if not CENTRES_PATH.exists():
        raise SystemExit(f"{CENTRES_PATH} does not exist. Render the vocabulary "
                         f"first with scripts/stimuli/render.py.")
    audit = pd.read_parquet(LEXICON / "concepts_audit.parquet")
    print(summarise(audit))
    if ((audit.status == "FAIL") & audit.blocking).any():
        raise SystemExit("blocking gates are failing; fix lexicon/concepts.tsv")

    words = sorted(concepts.concept_words().word)
    print(f"{len(words):,} words x {n_slots()} slots x 2 conditions "
          f"+ {n_slots()} clean = {len(words) * n_slots() * 2 + n_slots():,} images")
    man = build_manifest(words, workers=WORKERS)
    concepts.RENDER_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    man.to_parquet(concepts.RENDER_MANIFEST)
    print(f"wrote concepts/manifest.parquet: {len(man):,} rows")
    print(f"manifest fingerprint {fingerprint(man)[:16]}")


def images() -> pd.DataFrame:
    """Write every validation photograph except the backgrounds; return their table in (class, key) order."""
    skip = concepts.background_keys()
    concepts.IMAGES.mkdir(parents=True, exist_ok=True)
    rows, n_skipped, n_failed = [], 0, 0

    def write_one(key: str, wnid: str, payload: bytes) -> str | None:
        out = concepts.image_path(wnid, key)
        if out.exists():
            return manifest_path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            concepts.square_resize(concepts.decode(payload), IMAGE_SIZE).save(out)
        except Exception:                                     # noqa: BLE001
            return None
        return manifest_path(out)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for shard in tqdm(concepts.validation_shards(), desc="shards", unit="shard"):
            batch = []
            for key, wnid, payload in concepts.read_validation(shard):
                if key in skip:
                    n_skipped += 1
                    continue
                batch.append((key, wnid, payload))
            for (key, wnid, _), path in zip(
                    batch, pool.map(lambda t: write_one(*t), batch)):
                if path is None:
                    n_failed += 1
                    continue
                rows.append(dict(wnid=wnid, key=key, image_path=path))

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("no validation images were written")
    # Row i of an embedding cache is the embedding of row i here, so the order is fixed.
    frame = frame.sort_values(["wnid", "key"], kind="mergesort").reset_index(drop=True)
    per_class = frame.groupby("wnid").size()
    print(f"\n{len(frame):,} photographs over {frame.wnid.nunique()} classes")
    print(f"  {n_skipped} skipped as backgrounds, {n_failed} failed to decode")
    print(f"  per class: min {per_class.min()}, median {int(per_class.median())}, "
          f"max {per_class.max()}")
    return frame


def scan(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with the CRAFT text scan of each photograph appended.

    An unreadable photograph gets -1 boxes and no text area.
    """
    detector = reader(torch.cuda.is_available())
    found = []
    for path in tqdm(frame.image_path, desc="scan", unit="img"):
        try:
            with Image.open(local_path(path)) as im:
                found.append(text_density(detector, im.convert("RGB")))
        except Exception:                                     # noqa: BLE001
            found.append(dict(n_boxes=-1, text_area_ratio=float("nan")))
    scanned = pd.DataFrame(found, index=frame.index)
    merged = frame.join(scanned)
    clean = merged.n_boxes == 0
    per_class = merged.assign(clean=clean).groupby("wnid").clean.sum()
    print(f"\n{int(clean.sum()):,} of {len(merged):,} photographs carry no "
          f"detected text ({clean.mean():.1%})")
    print(f"  classes with fewer than {concepts.MIN_IMAGES} text-free: "
          f"{int((per_class < concepts.MIN_IMAGES).sum())}")
    return merged


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    ensure_dirs()
    render()
    frame = scan(images())
    frame.to_parquet(concepts.IMAGE_MANIFEST)
    print("wrote concepts/images.parquet")
    print(f"image manifest fingerprint {fingerprint(frame)[:16]}")


if __name__ == "__main__":
    main()
