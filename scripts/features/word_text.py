"""Embed every word as text with the three frozen Qwen3 embedding models.

Each word is embedded inside the 96 carriers of ``src.wordtext`` and the
vectors are averaged. Writes ``word_text/<encoder>/vectors.npy`` and
``meta.json``; an encoder whose cache matches the current words and carriers
is skipped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time

import numpy as np

from src import wordtext
from src.config import ensure_dirs
from src.models import ENCODERS
from src.tau import word_order


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    ensure_dirs()

    words = list(word_order())
    word_sha = wordtext.fingerprint(words)
    print(f"{len(words):,} words x {len(wordtext.CARRIERS)} carriers "
          f"({len(wordtext.TEMPLATE_FRAMES)} frames x {wordtext.VARIANTS} "
          f"variants) | fingerprint {word_sha[:12]}")

    for key in wordtext.TEXT_MODELS:
        repo, dim, _ = ENCODERS[key]
        out = wordtext.cache_dir(key)
        meta_file = out / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            if (meta.get("word_sha") == word_sha
                    and list(meta.get("carriers", [])) == list(wordtext.CARRIERS)):
                print(f"{key}: already embedded for these words and carriers")
                continue
            print(f"{key}: cache is for other words or carriers, rebuilding")

        started = time.time()
        per_carrier, device = wordtext.embed(key, words)
        if per_carrier.shape[2] != dim:
            raise SystemExit(
                f"{key} returned {per_carrier.shape[2]} dimensions, the "
                f"registry says {dim}")
        vectors = per_carrier.mean(1)

        spread = float(np.mean([per_carrier[i].std(0).mean()
                                for i in range(len(words))]))
        if spread < 1e-6:
            raise SystemExit(
                f"{key}: the {len(wordtext.CARRIERS)} carrier vectors are "
                f"identical")
        within, between = wordtext.crossing_gap(per_carrier)
        print(f"{key}: {vectors.shape} on {device} in {time.time() - started:.0f}s"
              f" | spread across carriers {spread:.4f}"
              f" | crossing within {within:+.3f} between {between:+.3f}")
        if within <= between:
            raise SystemExit(
                f"{key}: the variants of a frame are no more alike than "
                f"variants of different frames")

        # Metadata last, so an interrupted write leaves a cache that load refuses.
        meta_file.unlink(missing_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "vectors.npy", vectors)
        meta_file.write_text(json.dumps({
            "tower": key, "repo": repo, "dim": int(dim),
            "words": len(words), "word_sha": word_sha,
            "carriers": list(wordtext.CARRIERS),
            "frames": len(wordtext.TEMPLATE_FRAMES),
            "variants": wordtext.VARIANTS,
            "estimator": "carrier_mean_96",
            "context": wordtext.CONTEXT, "device": device,
            "dtype": "float32",
            "input": "the word inside each of 96 shared carriers, averaged",
            "recorded": dt.date.today().isoformat(),
        }, indent=2) + "\n")
        print(f"  wrote {key}/vectors.npy")


if __name__ == "__main__":
    main()
