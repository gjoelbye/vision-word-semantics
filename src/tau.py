"""Word directions: the mean over slots of a word's embedding minus its scrambled control's.

For a word ``w``, the direction is the mean over slots ``s`` of
``f(background_s + w) - f(background_s + scramble_s(w))``. The estimator the
results use, ``residual_centred``, also subtracts the mean over words. Caches
are checked against the manifest they were built from before they are read.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EMBEDDINGS, LEXICON
from .stimuli.imagenet import IMAGE_MANIFEST, RENDER_MANIFEST
from .stimuli.render import CLEAN, fingerprint
from .stimuli.typography import split_halves


@dataclass(frozen=True)
class Dataset:
    """One rendered image set: the manifest that fixes its row order, and its cache directory.

    ``produced_by`` and ``rebuild`` name the scripts that write the manifest and
    the caches.
    """

    name: str
    manifest_path: Path
    embeddings: Path
    produced_by: str = "scripts/stimuli/render.py"
    rebuild: str = "scripts/features/embed.py"


#: The 1,561-word vocabulary.
MAIN = Dataset("main", LEXICON / "manifest.parquet", EMBEDDINGS)

#: The 200 class names, rendered on the same slots.
CONCEPT_WORDS = Dataset("concept_words", RENDER_MANIFEST,
                        EMBEDDINGS / "concepts" / "renders",
                        produced_by="scripts/stimuli/class_images.py",
                        rebuild="scripts/features/embed.py --manifest concepts")

#: The ImageNet validation photographs of the classes.
CONCEPT_IMAGES = Dataset("concept_images", IMAGE_MANIFEST,
                         EMBEDDINGS / "concepts" / "images",
                         produced_by="scripts/stimuli/class_images.py",
                         rebuild="scripts/features/embed.py --manifest concepts")


class MissingCondition(FileNotFoundError):
    """Raised when a rendered condition the caller asked for does not exist."""


class DegenerateTau(ValueError):
    """Raised when every direction at a layer is exactly zero.

    At hidden-state index 0 the class token has not seen the image, so a word
    image and its control give the same vector.
    """


@lru_cache(maxsize=None)
def manifest(ds: Dataset = MAIN) -> pd.DataFrame:
    p = ds.manifest_path
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run {ds.produced_by}")
    return pd.read_parquet(p).reset_index(drop=True)


@lru_cache(maxsize=8)
def load_cache(key: str, *,
               ds: Dataset = MAIN) -> tuple[np.ndarray, np.ndarray, int]:
    """Class-token ``(features, layer_indices, n_layers)`` of one model on one image set.

    Features come back as a read-only memory map. The cache is refused unless it
    was built, to the end, from the current manifest of ``ds``.
    """
    import json

    meta_f = ds.embeddings / f"{key}.meta.json"
    arr_f = ds.embeddings / f"{key}.cls.npy"
    run = ds.rebuild
    if not meta_f.exists():
        raise FileNotFoundError(
            f"no {ds.name} cache for {key}; run {run}")
    meta = json.loads(meta_f.read_text())
    man = manifest(ds)
    got = fingerprint(man)
    if meta.get("sha") != got:
        raise RuntimeError(
            f"{key} was built against a different manifest row order "
            f"({str(meta.get('sha'))[:12]} != {got[:12]}). "
            f"Re-run {run}")
    if not meta.get("done"):
        raise RuntimeError(
            f"{key} is only embedded to row {meta.get('cursor')} of {len(man)}; "
            f"re-run {run} to finish it")
    x = np.load(arr_f, mmap_mode="r")
    assert len(x) == len(man), f"{key} has {len(x)} rows, manifest has {len(man)}"
    return x, np.array(meta["layers"]), int(meta["n_layers"])


@lru_cache(maxsize=None)
def word_order(ds: Dataset = MAIN) -> tuple[str, ...]:
    man = manifest(ds)
    return tuple(sorted(set(man.loc[man.word != CLEAN, "word"])))


def residual_tau(key: str, layer: int = -1, slots: tuple[int, ...] | None = None,
                 *, center: bool = False, ds: Dataset = MAIN) -> np.ndarray:
    """Word directions: the mean over slots of f(background + word) - f(background + scrambled word).

    Rows follow word_order(), and the scrambled word keeps the word's ink and
    spatial frequency but loses its glyph structure. ``slots`` restricts the mean,
    ``center`` subtracts the mean over words, and DegenerateTau is raised where
    every direction is zero.
    """
    x, _, _ = load_cache(key, ds=ds)
    man = manifest(ds)
    if "condition" not in man.columns:
        raise MissingCondition(
            f"the {ds.name} manifest has no 'condition' column, so it holds no "
            f"scrambled control. Re-render it with {ds.produced_by}.")
    word_mask = ((man.word != CLEAN) & man.condition.eq("word")).to_numpy()
    control_mask = ((man.word != CLEAN)
                    & man.condition.eq("scramble")).to_numpy()
    if not control_mask.any():
        raise MissingCondition(
            f"the {ds.name} manifest has no scramble rows. Re-render it with "
            f"{ds.produced_by}.")
    word_keys = man.loc[word_mask, ["word", "slot"]].reset_index(drop=True)
    control_keys = man.loc[control_mask, ["word", "slot"]].reset_index(drop=True)
    if not word_keys.equals(control_keys):
        raise MissingCondition("word and scramble rows are not paired by word and slot")

    # The clean background cancels in the difference and is not read.
    selected = x[:, layer, :]
    word_rows = np.flatnonzero(word_mask)
    control_rows = np.flatnonzero(control_mask)
    slot = word_keys.slot.to_numpy()
    words = word_order(ds)
    index = {word: i for i, word in enumerate(words)}
    rows = np.fromiter((index[word] for word in word_keys.word), int,
                       count=len(word_keys))
    keep = (np.ones(len(word_rows), bool) if slots is None
            else np.isin(slot, slots))
    out = np.zeros((len(words), selected.shape[1]), np.float32)
    count = np.zeros(len(words))
    positions = np.flatnonzero(keep)
    for start in range(0, len(positions), 4096):
        pos = positions[start:start + 4096]
        differences = (np.asarray(selected[word_rows[pos]], np.float32)
                       - np.asarray(selected[control_rows[pos]], np.float32))
        np.add.at(out, rows[pos], differences)
        np.add.at(count, rows[pos], 1)
    if not count.all():
        raise ValueError("a word received no paired word and scramble images")
    out /= count[:, None]
    if not np.abs(out).any():
        raise DegenerateTau(
            f"{key} layer {layer}: every residual direction is exactly zero")
    if center:
        out = out - out.mean(0, keepdims=True)
    return out


def shared_mean_share(T: np.ndarray) -> float:
    """Squared norm of the mean row of ``T`` over the mean squared row norm.

    Near 1, every row points the same way and a cosine between two rows mostly
    reflects that shared direction.
    """
    T = np.asarray(T, np.float32)
    return float(np.linalg.norm(T.mean(0)) ** 2 / (T ** 2).sum(1).mean())


def estimator(name: str):
    """The estimator called ``name``: ``residual`` (uncentered) or ``residual_centred``."""
    table = {
        "residual": lambda k, l=-1, s=None, *, ds=MAIN:
            residual_tau(k, l, s, center=False, ds=ds),
        "residual_centred": lambda k, l=-1, s=None, *, ds=MAIN:
            residual_tau(k, l, s, center=True, ds=ds),
    }
    if name not in table:
        raise ValueError(f"unknown estimator {name!r}; choose from {sorted(table)}")
    return table[name]


ESTIMATORS = ("residual", "residual_centred")
HEADLINE = "residual_centred"


def rownorm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def split_half_cosine(key: str, appearances, layer: int = -1) -> float:
    """Mean over words of the cosine between the two matched-appearance halves.

    Each half's word directions come from the ``HEADLINE`` estimator on that
    half's slots, restricted to ``appearances`` (``None`` keeps every appearance
    tuple).
    """
    a, b = split_halves(appearances)
    ta = rownorm(estimator(HEADLINE)(key, layer, tuple(a)))
    tb = rownorm(estimator(HEADLINE)(key, layer, tuple(b)))
    return float((ta * tb).sum(1).mean())
