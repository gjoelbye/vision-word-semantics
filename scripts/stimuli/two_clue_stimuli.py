"""Validate, lay out and render the two-clue items.

Reads ``lexicon/two_clue_pilot.tsv`` and ``lexicon/two_clue_confirmatory.tsv``,
records the layout and manifest under ``composition/``, and renders the six
conditions of every item on the 96 slots into ``composition/renders/``.
Rendering resumes where it stopped. ``scripts/features/embed.py --manifest
two_clue`` embeds the renders.
"""
from __future__ import annotations

import argparse
import json

from src.composition_stimuli import (
    CONDITIONS,
    MANIFEST_META_PATH,
    build_manifest,
    load_curated_items,
    load_or_create_layout,
    manifest_fingerprint,
    render_dataset,
)
from src.slots import n_slots


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    items, report, dataset_sha = load_curated_items()
    errors = int(report.status.eq("error").sum())
    n_images = len(items) * n_slots() * len(CONDITIONS)
    print(f"curated items: {len(items):,}")
    print(f"deterministic checks: {len(report) - errors}/{len(report)} passing")
    print(f"dataset fingerprint: {dataset_sha}")
    print(f"images: {len(items):,} x {n_slots()} x {len(CONDITIONS)} = "
          f"{n_images:,}")

    layout, layout_sha = load_or_create_layout(items, dataset_sha)
    expected = build_manifest(items, layout, dataset_sha, layout_sha)
    if len(expected) != n_images:
        raise RuntimeError(f"expected {n_images} manifest rows, got {len(expected)}")
    manifest = render_dataset(items, layout, dataset_sha)
    manifest_sha = manifest_fingerprint(manifest)
    meta = json.loads(MANIFEST_META_PATH.read_text())
    if meta.get("manifest_sha") != manifest_sha:
        raise RuntimeError("written manifest fingerprint does not match metadata")
    print(f"\ncomplete: {len(manifest):,} images")
    print(f"manifest fingerprint: {manifest_sha}")


if __name__ == "__main__":
    main()
