"""Cache the class-token features of every rendered image.

By default the fourteen sweep models embed the vocabulary manifest,
``lexicon/manifest.parquet``, at ten depths into ``embeddings/``.
``--finetuned`` embeds it with the twenty SPEECH-COCO finetunes instead.
``--manifest concepts`` embeds the class-name renders and the class photographs
with the sweep models, into ``embeddings/concepts/renders/`` and
``embeddings/concepts/images/``. ``--manifest two_clue`` embeds the two-clue
renders with the sweep models at the final layer only, into
``composition/embeddings/``. A finished cache is skipped and an unfinished one
resumes.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src import composition_stimuli as two_clue
from src import concepts
from src.config import EMBEDDINGS, LEXICON, ensure_dirs
from src.embed import embed_manifest
from src.models import DEFAULT_SWEEP, FINETUNES

#: Images per forward pass; halved automatically on an out-of-memory error.
BATCH_SIZE = 128

#: The script that writes each manifest.
PRODUCED_BY = {"main": "scripts/stimuli/render.py",
               "concepts": "scripts/stimuli/class_images.py",
               "two_clue": "scripts/stimuli/two_clue_stimuli.py"}


def two_clue_manifest() -> tuple[pd.DataFrame, str]:
    """The two-clue manifest and its fingerprint, refused unless fully rendered."""
    manifest = pd.read_parquet(two_clue.MANIFEST_PATH).reset_index(drop=True)
    manifest_sha = two_clue.manifest_fingerprint(manifest)
    meta = json.loads(two_clue.MANIFEST_META_PATH.read_text())
    if (meta.get("manifest_sha") != manifest_sha
            or meta.get("render_complete") is not True):
        raise SystemExit(f"{two_clue.MANIFEST_PATH} is stale or not fully "
                         f"rendered; run {PRODUCED_BY['two_clue']}")
    return manifest, manifest_sha


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="main",
                    choices=("main", "concepts", "two_clue"))
    ap.add_argument("--finetuned", action="store_true",
                    help="embed the vocabulary with the SPEECH-COCO finetunes")
    a = ap.parse_args()
    if a.finetuned and a.manifest != "main":
        raise SystemExit("--finetuned embeds the vocabulary manifest only")
    ensure_dirs()

    if a.manifest == "main":
        jobs = [("vocabulary", LEXICON / "manifest.parquet", EMBEDDINGS)]
    elif a.manifest == "concepts":
        jobs = [("class names", concepts.RENDER_MANIFEST,
                 EMBEDDINGS / "concepts" / "renders"),
                ("class photographs", concepts.IMAGE_MANIFEST,
                 EMBEDDINGS / "concepts" / "images")]
    else:
        jobs = [("two-clue items", two_clue.MANIFEST_PATH,
                 two_clue.EMBEDDINGS_ROOT)]
    models = FINETUNES if a.finetuned else DEFAULT_SWEEP

    missing = [str(p) for _, p, _ in jobs if not p.exists()]
    if missing:
        raise SystemExit(f"no manifest at {', '.join(missing)}. Render the set "
                         f"first: {PRODUCED_BY[a.manifest]}")

    for label, path, out in jobs:
        if a.manifest == "two_clue":
            # Final layer only, and a cache built for another manifest is refused.
            man, manifest_sha = two_clue_manifest()
            options = dict(layer_mode="final", manifest_sha=manifest_sha,
                           refuse_stale=True)
        else:
            man, options = pd.read_parquet(path).reset_index(drop=True), {}
        print(f"\n{label}: {len(man):,} images | {len(models)} models")
        for key in models:
            embed_manifest(key, man, batch_size=BATCH_SIZE, output_dir=out,
                           **options)


if __name__ == "__main__":
    main()
