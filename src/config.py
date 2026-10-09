"""Paths and constants shared by every stage.

Everything the pipeline downloads, and every cache it generates, lives under
`DATA_ROOT`, which is `data/` in the repository root unless the
`VIT_SEMANTIC_DATA` environment variable names another directory. The committed
inputs (`lexicon/`) and the committed outputs the paper reads (`results/`,
`figures/`, `tables/`) sit in the repository root. Manifests record image paths
as they were on the rendering machine, and `data_relative` and `local_path`
map them onto this data root.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]

# The committed word lists. `vocabulary.tsv` holds the eight lexical lists,
# one row per word. `class_names.tsv` is the class-name curation record. The
# two two-clue tables are hashed cell by cell into the two-clue fingerprints.
VOCABULARY_TSV = REPO_ROOT / "lexicon" / "vocabulary.tsv"
CLASS_NAMES_TSV = REPO_ROOT / "lexicon" / "class_names.tsv"
TWO_CLUE_PILOT_TSV = REPO_ROOT / "lexicon" / "two_clue_pilot.tsv"
TWO_CLUE_CONFIRMATORY_TSV = REPO_ROOT / "lexicon" / "two_clue_confirmatory.tsv"

DATA_ROOT = Path(os.environ.get("VIT_SEMANTIC_DATA", REPO_ROOT / "data"))

BACKGROUNDS = DATA_ROOT / "backgrounds"
# Tables generated from the committed word lists, not the lists themselves.
LEXICON = DATA_ROOT / "lexicon"
RENDERS = DATA_ROOT / "renders"
EMBEDDINGS = DATA_ROOT / "embeddings"

# Word-by-word cosine matrices, one per model and encoder, and their index.
SIMILARITY = DATA_ROOT / "similarity"

# The bold face cut from the installed variable sans for the figures.
FONTS = DATA_ROOT / "fonts"

RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"
TABLES = REPO_ROOT / "tables"

# Finetuned backbone weights, one directory per finetuned model, each holding
# `best.slim.pt` and a `provenance.json` with the recipe and the step.
CHECKPOINTS = DATA_ROOT / "checkpoints"

# Records of the SPEECH-COCO training corpus: its manifest, text scan, exclude
# list and provenance.
CORPUS = DATA_ROOT / "corpus"

# Word vectors from the frozen text encoders, one cache per encoder.
WORD_TEXT = DATA_ROOT / "word_text"

# Word vectors from the frozen speech encoder, averaged over spoken renditions.
WORD_SPEECH = DATA_ROOT / "word_speech"

# Caption target vectors from the frozen towers, one cache per tower.
CAPTIONS = DATA_ROOT / "captions"

# The class-name experiment: the manifest of its rendered words (the images
# live under `RENDERS`) and the ImageNet validation photographs.
CONCEPTS = DATA_ROOT / "concepts"

# The two-clue experiment: its layout, manifest, renders and embedding caches.
COMPOSITION = DATA_ROOT / "composition"

_ALL_DIRS = (BACKGROUNDS, LEXICON, RENDERS, EMBEDDINGS, SIMILARITY, RESULTS,
             FIGURES, TABLES, CHECKPOINTS, CORPUS, CAPTIONS, CONCEPTS,
             COMPOSITION, WORD_TEXT, WORD_SPEECH)


# The directory names that sit directly under a data root, where
# `data_relative` splits a recorded path.
_DATA_SUBDIRS = frozenset({
    "backgrounds", "lexicon", "renders", "embeddings", "checkpoints", "corpus",
    "captions", "composition", "concepts", "word_text", "word_speech",
})


def data_relative(recorded: str | Path) -> PurePosixPath:
    """The part of a recorded image path below its data root, such as `renders/zynga/094.png`.

    Recorded paths may carry the absolute prefix of the machine that rendered
    them, and that prefix is part of the manifest fingerprint, so it is split
    off rather than rewritten. The split is at the leftmost data subdirectory,
    so a word directory named like a subdirectory is kept.
    """
    parts = PurePosixPath(str(recorded)).parts
    for i, part in enumerate(parts):
        if part in _DATA_SUBDIRS:
            return PurePosixPath(*parts[i:])
    raise ValueError(f"recorded path is below no known data directory: {recorded}")


def local_path(recorded: str | Path) -> Path:
    """Where a recorded manifest `image_path` is on this machine."""
    return DATA_ROOT / data_relative(recorded)


def manifest_path(local: str | Path) -> str:
    """The path to record for an image written under the data root, relative to it."""
    pure = Path(local).resolve()
    root = DATA_ROOT.resolve()
    if not pure.is_relative_to(root):
        raise ValueError(f"not under the data root: {local}")
    return str(PurePosixPath(pure.relative_to(root)))


def line_fingerprint(items) -> str:
    """sha256 of the items in order, each as its string and a newline.

    The manifest fingerprint hashes the recorded `image_path` strings this
    way, and the word fingerprint of the text and speech caches hashes the word
    list.
    """
    h = hashlib.sha256()
    for item in items:
        h.update(str(item).encode())
        h.update(b"\n")
    return h.hexdigest()


def ensure_dirs() -> None:
    """Create every output directory. Safe to call repeatedly."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# Side of the square backgrounds and renders, in pixels.
IMAGE_SIZE = 256

# Fraction of the shortest edge that survives every backbone's preprocessing.
# Text is always placed inside this window. 0.875 is the DINOv2 crop of 224
# from 256; the other processors resize the whole canvas.
CROP_SAFE_FRAC = 0.875

# Longest word rendered. Longer words at the largest size band would not fit
# the crop-safe window.
MAX_WORD_LENGTH = 10

# Global seed. Stimulus construction, the untrained ViT-B and the finetuning
# runs derive their generators from it.
SEED = 20260821
