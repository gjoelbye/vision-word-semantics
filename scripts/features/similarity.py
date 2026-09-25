"""Word-by-word cosine similarity matrices for the 14 image models and 4 encoders.

An image model's word directions come from the headline estimator at the final
layer, and an encoder's from its text or speech vectors centred over words.
Writes ``results/similarity/<model>.npy`` and the index
``results/similarity.parquet``, which is rewritten as each matrix is saved and
records the word and manifest fingerprints of every matrix.
"""
from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import pandas as pd

from src import wordspeech, wordtext
from src.config import RESULTS, ensure_dirs
from src.models import DEFAULT_SWEEP
from src.render import fingerprint
from src.tau import HEADLINE, estimator, manifest, rownorm, word_order

#: Every matrix, in index order: the image models, the text encoders, the
#: speech encoder.
MODELS = DEFAULT_SWEEP + wordtext.TEXT_MODELS + wordspeech.SPEECH_MODELS

OUTPUT = RESULTS / "similarity"
INDEX = RESULTS / "similarity.parquet"


def kind_of(key: str) -> str:
    if key in wordtext.TEXT_MODELS:
        return "text"
    if key in wordspeech.SPEECH_MODELS:
        return "speech"
    return "vision"


def directions(key: str, words: list[str]) -> np.ndarray:
    """The per-word directions of one model or encoder, centred over words."""
    kind = kind_of(key)
    if kind == "vision":
        return estimator(HEADLINE)(key, -1)
    loader = wordspeech if kind == "speech" else wordtext
    vectors = loader.load(key, words)
    return vectors - vectors.mean(0, keepdims=True)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    ensure_dirs()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    words = list(word_order())
    word_sha = wordtext.fingerprint(words)
    manifest_sha = fingerprint(manifest())
    print(f"{len(words):,} words | word set {word_sha[:12]} | "
          f"manifest {manifest_sha[:12]}")

    # The index lists only matrices written by this run.
    INDEX.unlink(missing_ok=True)
    rows = []
    for key in MODELS:
        kind = kind_of(key)
        started = time.time()
        direction = directions(key, words)
        unit = rownorm(np.asarray(direction, np.float32))
        similarity = (unit @ unit.T).astype(np.float32)
        np.save(OUTPUT / f"{key}.npy", similarity)
        del direction, unit, similarity
        gc.collect()
        rows.append({
            "model": key,
            "kind": kind,
            "estimator": (HEADLINE if kind == "vision"
                          else "carrier_mean_96" if kind == "text"
                          else "rendition_mean_96"),
            "word_sha": word_sha,
            "manifest_sha": manifest_sha if kind == "vision" else pd.NA,
        })
        pd.DataFrame(rows).to_parquet(INDEX)
        print(f"{key}: {time.time() - started:.0f}s")

    print(f"{len(rows)} matrices indexed in results/similarity.parquet")


if __name__ == "__main__":
    main()
