"""Speak every word and embed it with the frozen w2v-BERT 2.0 encoder.

Each word is spoken in the 96 renditions of `src.features.speech`, and the pooled
vectors are averaged and written to `data/word_speech/<encoder>/vectors.npy` and
`meta.json`. A cache that matches the current words and carrier sentences is
skipped. Needs `phonemizer` and the `espeak-ng` binary.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time

import numpy as np
import torch

from src.analysis.directions import word_order
from src.config import ensure_dirs
from src.features import speech, text
from src.models import ENCODERS


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    words = list(word_order())
    word_sha = text.fingerprint(words)
    cells = speech.grid()
    print(f"{len(words):,} words x {len(cells)} renditions "
          f"({len(speech.SPEAKERS)} voices, {len(speech.RATES)} rates, "
          f"{len(speech.CARRIERS)} carriers) | device {device}")

    for key in speech.SPEECH_MODELS:
        repo, dim, _ = ENCODERS[key]
        out = speech.cache_dir(key)
        meta_file = out / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            if (meta.get("word_sha") == word_sha
                    and list(meta.get("carriers", []))
                    == list(speech.CARRIERS)):
                print(f"{key}: already embedded for these words and carriers")
                continue
            print(f"{key}: cache is for other words or carriers, rebuilding")

        started = time.time()
        renditions = speech.build(words, device=device)
        if renditions.shape[2] != speech.BINS * dim:
            raise SystemExit(
                f"{key} returned {renditions.shape[2]} dimensions, expected "
                f"{speech.BINS} bins x {dim}")
        vectors = renditions.mean(1)
        print(f"{key}: {vectors.shape} in {time.time() - started:.0f}s")

        # Metadata last, so an interrupted write leaves a cache that load refuses.
        meta_file.unlink(missing_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "vectors.npy", vectors)
        meta_file.write_text(json.dumps({
            "tower": key, "repo": repo, "dim": int(dim),
            "bins": speech.BINS, "width": int(renditions.shape[2]),
            "words": len(words), "word_sha": word_sha,
            "synthesiser": speech.TTS_REPO,
            "sample_rate": speech.SAMPLE_RATE,
            "carriers": list(speech.CARRIERS),
            "speakers": list(speech.SPEAKERS),
            "rates": list(speech.RATES),
            "renditions": len(cells),
            "estimator": "rendition_mean_96",
            "device": device, "dtype": "float32",
            "input": "the word inside each of 96 spoken renditions, averaged",
            "recorded": dt.date.today().isoformat(),
        }, indent=2) + "\n")
        print(f"  wrote {key}/vectors.npy")


if __name__ == "__main__":
    main()
