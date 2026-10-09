"""Word-by-word cosine similarity matrices for the 14 image models and 4 encoders.

An image model contributes its word directions at the final layer, centered
over words, and an encoder its text or speech vectors, centered the same way.
Writes `data/similarity/<model>.npy` and the index
`data/similarity/index.parquet`, which is rewritten as each matrix is saved. The
index records the word fingerprint of every matrix and the manifest
fingerprint of every image model's matrix.
"""
from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import pandas as pd

from src.analysis.directions import (HEADLINE, manifest, rownorm,
                                     word_directions, word_order)
from src.config import SIMILARITY, SIMILARITY_INDEX, ensure_dirs, line_fingerprint
from src.features import ESTIMATOR, encoder_directions, speech, text
from src.models import DEFAULT_SWEEP
from src.stimuli.render import fingerprint

# Every matrix, in index order: the image models, the text encoders, the
# speech encoder.
MODELS = DEFAULT_SWEEP + text.TEXT_MODELS + speech.SPEECH_MODELS


def kind_of(key: str) -> str:
    if key in text.TEXT_MODELS:
        return "text"
    if key in speech.SPEECH_MODELS:
        return "speech"
    return "vision"


def directions(key: str, words: list[str]) -> np.ndarray:
    """The per-word directions of one model or encoder, centered over words."""
    kind = kind_of(key)
    if kind == "vision":
        return word_directions(key, -1, center=True)
    return encoder_directions(key, words)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    ensure_dirs()

    words = list(word_order())
    word_sha = line_fingerprint(words)
    manifest_sha = fingerprint(manifest())
    print(f"{len(words):,} words | word set {word_sha[:12]} | "
          f"manifest {manifest_sha[:12]}")

    # The index lists only matrices written by this run.
    SIMILARITY_INDEX.unlink(missing_ok=True)
    rows = []
    for key in MODELS:
        kind = kind_of(key)
        started = time.time()
        direction = directions(key, words)
        unit = rownorm(np.asarray(direction, np.float32))
        similarity = (unit @ unit.T).astype(np.float32)
        np.save(SIMILARITY / f"{key}.npy", similarity)
        del direction, unit, similarity
        gc.collect()
        rows.append({
            "model": key,
            "kind": kind,
            "estimator": HEADLINE if kind == "vision" else ESTIMATOR[kind],
            "word_sha": word_sha,
            "manifest_sha": manifest_sha if kind == "vision" else pd.NA,
        })
        pd.DataFrame(rows).to_parquet(SIMILARITY_INDEX)
        print(f"{key}: {time.time() - started:.0f}s")

    print(f"{len(rows)} matrices indexed in similarity/index.parquet")


if __name__ == "__main__":
    main()
