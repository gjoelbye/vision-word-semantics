"""Word directions and visual directions, as Section 3.3 of the paper defines them.

`f` is a model's class token at one depth. For a word `w` rendered on the 96
slots `s` (a slot is one background with its typography setting, see
`src.stimuli.typography`), the uncentered word direction is the mean over slots
of `f(x_s + w) - f(x_s + scrambled w)`. The control keeps the background,
position, color and backing of the word image and replaces the letters with a
phase-scrambled patch, so the difference is the response to intact lettering.
The word direction `tau_w` subtracts the mean of the uncentered directions over
the vocabulary, and the result tables record it as `residual_centred`. The
vocabulary is the 1,561 words, names and pseudowords for `MAIN` and the 200
class names for `CLASS_NAMES`, so a name in both has two word directions.

For a class `c`, the uncentered visual direction is the mean of `f(x_i)` over
its retained validation photographs. The visual direction `nu_c` subtracts the
mean of those class means over every retained class, which is all 1,000 when
every photograph is kept.

Every analysis script reads the caches through `load_cache`, which refuses a
cache not built, to the end, from the current manifest.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import EMBEDDINGS, LEXICON
from ..features.image import cache_paths
from ..stimuli.imagenet import IMAGE_MANIFEST, MIN_IMAGES, RENDER_MANIFEST
from ..stimuli.render import CLEAN, fingerprint
from ..stimuli.typography import split_halves


@dataclass(frozen=True)
class Dataset:
    """One rendered image set: the manifest that fixes its row order, and its cache directory.

    `produced_by` and `rebuild` name the scripts that write the manifest and
    the caches.
    """

    name: str
    manifest_path: Path
    embeddings: Path
    produced_by: str = "scripts/stimuli/render.py"
    rebuild: str = "scripts/features/embed.py"


# The lexical vocabulary: 1,481 words and names and 80 pseudowords.
MAIN = Dataset("main", LEXICON / "manifest.parquet", EMBEDDINGS)

# The 200 class names, rendered on the same slots.
CLASS_NAMES = Dataset("class_names", RENDER_MANIFEST,
                      EMBEDDINGS / "concepts" / "renders",
                      produced_by="scripts/stimuli/class_images.py",
                      rebuild="scripts/features/embed.py --manifest concepts")

# The ImageNet validation photographs of the classes.
CLASS_IMAGES = Dataset("class_images", IMAGE_MANIFEST,
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
    """Class-token `(features, layer_indices, n_layers)` of one model on one image set.

    Features come back as a read-only memory map. The cache is refused unless it
    was built, to the end, from the current manifest of `ds`.
    """
    import json

    paths = cache_paths(key, ds.embeddings)
    meta_f, arr_f = paths["meta"], paths["cls"]
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


def word_directions(key: str, layer: int = -1,
                    slots: tuple[int, ...] | None = None, *, center: bool,
                    ds: Dataset = MAIN) -> np.ndarray:
    """Word directions: the mean over slots of f(background + word) - f(background + scrambled word).

    Rows follow word_order(), and the scrambled word keeps the word's ink and
    spatial frequency but loses its glyph structure. `slots` restricts the mean,
    `center` subtracts the mean over words, and DegenerateTau is raised where
    every direction is zero. `center` has no default because the results use
    both: the centered directions everywhere, the uncentered ones to measure
    what centering removes.
    """
    x, _, _ = load_cache(key, ds=ds)
    man = manifest(ds)
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
            f"{key} layer {layer}: every word direction is exactly zero")
    if center:
        out = out - out.mean(0, keepdims=True)
    return out


def shared_mean_share(T: np.ndarray) -> float:
    """Squared norm of the mean row of `T` over the mean squared row norm.

    Near 1, every row points the same way and a cosine between two rows mostly
    reflects that shared direction.
    """
    T = np.asarray(T, np.float32)
    return float(np.linalg.norm(T.mean(0)) ** 2 / (T ** 2).sum(1).mean())


# The name the result tables and the similarity index record for the centered
# word directions.
HEADLINE = "residual_centred"


def rownorm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def split_half_cosine(key: str, appearances, layer: int = -1) -> float:
    """Mean over words of the cosine between the two matched-appearance halves.

    Each half's centered word directions come from that half's slots,
    restricted to `appearances` (`None` keeps every appearance tuple).
    """
    a, b = split_halves(appearances)
    ta = rownorm(word_directions(key, layer, tuple(a), center=True))
    tb = rownorm(word_directions(key, layer, tuple(b), center=True))
    return float((ta * tb).sum(1).mean())


# Photograph subsets a visual direction can be built from: every photograph, or
# those in which the text detector found no writing.
SUBSETS = ("all", "text_free")


def _rows(subset: str) -> pd.DataFrame:
    man = manifest(CLASS_IMAGES)
    if subset == "all":
        return man
    if subset != "text_free":
        raise ValueError(f"unknown subset {subset!r}; choose from {SUBSETS}")
    if "n_boxes" not in man.columns:
        raise FileNotFoundError(
            "the image manifest carries no text scan; run "
            "scripts/stimuli/class_images.py")
    return man[man.n_boxes == 0]


@lru_cache(maxsize=8)
def retained(subset: str = "all") -> tuple[str, ...]:
    """The classes with at least `MIN_IMAGES` photographs in `subset`, sorted by class id."""
    counts = _rows(subset).groupby("wnid").size()
    return tuple(sorted(counts.index[counts >= MIN_IMAGES]))


def class_mean(key: str, layer: int = -1, subset: str = "all") -> np.ndarray:
    """`(n_classes, d)` visual directions, rows aligned to `retained(subset)`.

    Each class is averaged over its photographs in `subset`, and the mean over
    the retained classes is subtracted.
    """
    x, _, _ = load_cache(key, ds=CLASS_IMAGES)
    man = manifest(CLASS_IMAGES)
    rows = _rows(subset)
    classes = retained(subset)
    index = {c: i for i, c in enumerate(classes)}
    keep = rows.wnid.isin(index)
    position = np.flatnonzero(man.index.isin(rows.index[keep]))
    target = np.fromiter((index[c] for c in man.wnid.iloc[position]), int,
                         count=len(position))

    out = np.zeros((len(classes), x.shape[2]), np.float32)
    count = np.zeros(len(classes))
    # Sum the memory-mapped rows of the one layer in blocks of 4096.
    for start in range(0, len(position), 4096):
        chunk = position[start:start + 4096]
        block = np.asarray(x[chunk, layer, :], np.float32)
        np.add.at(out, target[start:start + 4096], block)
        np.add.at(count, target[start:start + 4096], 1)
    if not count.all():
        empty = [classes[i] for i in np.flatnonzero(count == 0)][:5]
        raise ValueError(f"a retained class received no photograph: {empty}")
    out /= count[:, None]
    if not np.abs(out).any():
        raise DegenerateTau(
            f"{key} layer {layer}: every class mean is exactly zero. At "
            f"hidden-state index 0 the class token is still its learned "
            f"initial value.")
    out = out - out.mean(0, keepdims=True)
    return out
