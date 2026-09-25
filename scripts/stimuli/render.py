"""Render every word of the vocabulary on the 96 slots, with its scrambled control.

Writes the images under ``renders/``, the canonical ``lexicon/manifest.parquet``
and, on the first run, ``lexicon/centres.parquet``. Existing images are kept.
Stops if a blocking gate of ``lexicon/audit.parquet`` fails.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.audit import summarise
from src.config import LEXICON, ensure_dirs
from src.render import build_manifest, fingerprint
from src.slots import n_slots

WORKERS = 12


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    ensure_dirs()

    audit = pd.read_parquet(LEXICON / "audit.parquet")
    print(summarise(audit))
    if ((audit.status == "FAIL") & audit.blocking).any():
        raise SystemExit("blocking gates are failing; fix the lexicon first")

    words = sorted(pd.read_parquet(LEXICON / "words.parquet").word.unique())
    print(f"{len(words):,} words x {n_slots()} slots x 2 conditions "
          f"+ {n_slots()} clean = {len(words) * n_slots() * 2 + n_slots():,} images")

    man = build_manifest(words, workers=WORKERS)
    man.to_parquet(LEXICON / "manifest.parquet")
    print(f"wrote lexicon/manifest.parquet: {len(man):,} rows")
    print(f"manifest fingerprint {fingerprint(man)[:16]}")


if __name__ == "__main__":
    main()
