"""Class directions from photographs, and the readouts that compare them with word directions.

For a class ``c`` with photographs ``I_c``, the class direction is the mean of
``f(x_i)`` over ``I_c`` minus the mean of those class means over every retained
class. Caches are read through :func:`src.tau.load_cache`, so each is checked
against its own manifest.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .neighbors import K, N_PERM, SUBSAMPLE_FRAC
from .stats import (binomial_se, permutation_p, subsample_indices,
                    subsample_summary)
from .stimuli.imagenet import MIN_IMAGES
from .tau import CONCEPT_IMAGES, DegenerateTau, load_cache, manifest

#: Photograph subsets a class direction can be built from: every photograph, or
#: those in which the text detector found no writing.
SUBSETS = ("all", "text_free")


def _rows(subset: str) -> pd.DataFrame:
    man = manifest(CONCEPT_IMAGES)
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
    """The classes with at least :data:`src.stimuli.imagenet.MIN_IMAGES` photographs in ``subset``, sorted by class id."""
    counts = _rows(subset).groupby("wnid").size()
    return tuple(sorted(counts.index[counts >= MIN_IMAGES]))


def class_mean(key: str, layer: int = -1, subset: str = "all") -> np.ndarray:
    """``(n_classes, d)`` class directions, rows aligned to ``retained(subset)``.

    Each class is averaged over its photographs in ``subset``, and the mean over
    the retained classes is subtracted.
    """
    x, _, _ = load_cache(key, ds=CONCEPT_IMAGES)
    man = manifest(CONCEPT_IMAGES)
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


def class_coherence(key: str, layer: int = -1,
                    subset: str = "all") -> np.ndarray:
    """Mean cosine between two photographs of a class, one number per retained class.

    Photographs are centered on the mean over all photographs and scaled to unit
    length. For unit ``v_i`` the mean off-diagonal cosine is
    ``(||sum v_i||^2 - n) / (n (n - 1))``.
    """
    x, _, _ = load_cache(key, ds=CONCEPT_IMAGES)
    man = manifest(CONCEPT_IMAGES)
    rows = _rows(subset)
    classes = retained(subset)
    index = {c: i for i, c in enumerate(classes)}
    position = np.flatnonzero(man.index.isin(rows.index)
                              & man.wnid.isin(index).to_numpy())
    target = np.fromiter((index[c] for c in man.wnid.iloc[position]), int,
                         count=len(position))

    total = np.zeros(x.shape[2], np.float64)
    for start in range(0, len(position), 8192):
        chunk = position[start:start + 8192]
        total += np.asarray(x[chunk, layer, :], np.float32).sum(0)
    mean = (total / len(position)).astype(np.float32)

    summed = np.zeros((len(classes), x.shape[2]), np.float64)
    count = np.zeros(len(classes))
    for start in range(0, len(position), 8192):
        chunk = position[start:start + 8192]
        block = np.asarray(x[chunk, layer, :], np.float32) - mean
        block /= np.linalg.norm(block, axis=1, keepdims=True) + 1e-9
        np.add.at(summed, target[start:start + 8192], block)
        np.add.at(count, target[start:start + 8192], 1)
    energy = (summed ** 2).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (energy - count) / (count * (count - 1))
    return out.astype(np.float32)


def nearest_class(N: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each class direction, the index of the nearest other one and its cosine."""
    sim = N @ N.T
    np.fill_diagonal(sim, -np.inf)
    who = sim.argmax(1)
    return who, sim[np.arange(len(N)), who]


def cross_rank(query: np.ndarray, candidates: np.ndarray,
               target: np.ndarray) -> np.ndarray:
    """Rank of each query's own target among the candidates, by cosine; 0 is nearest.

    Both arrays are unit rows. Ties count for the target, as in
    :func:`src.neighbors.partner_rank`, and no candidate is excluded.
    """
    sim = query @ candidates.T
    own = sim[np.arange(len(target)), target]
    return (sim > own[:, None]).sum(1)


def word_class_rows(words: pd.Series, wnids: pd.Series,
                    subset: str = "all") -> tuple[np.ndarray, np.ndarray]:
    """Positions of the words whose class is retained, and their rows in ``retained(subset)``."""
    classes = retained(subset)
    index = {c: i for i, c in enumerate(classes)}
    keep = np.array([w in index for w in wnids], bool)
    return np.flatnonzero(keep), np.array(
        [index[w] for w in np.asarray(wnids)[keep]], int)


def concept_alignment(T: np.ndarray, N: np.ndarray, rows: np.ndarray,
                      target: np.ndarray, seed: int = 0) -> dict:
    """Mean cosine between each word direction and its class direction.

    ``T`` and ``N`` are unit rows; ``rows`` picks the words and ``target`` their
    classes. The null permutes which class goes with which word, the p-value is
    ``(count + 1) / (n + 1)``, and the standard error comes from subsamples of
    words without replacement (see :func:`src.stats.subsample_summary`).
    """
    per_word = (T[rows] * N[target]).sum(1)
    observed = float(per_word.mean())
    rng = np.random.default_rng(seed)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = (T[rows] * N[target[rng.permutation(len(target))]]).sum(1).mean()
    n = len(per_word)
    m = max(2, int(SUBSAMPLE_FRAC * n))
    draws = np.array([per_word[idx].mean()
                      for idx in subsample_indices(n, SUBSAMPLE_FRAC)])
    _, _, se = subsample_summary(observed, draws, n, m)
    return dict(n_words=len(rows), alignment=observed, se=se,
                null_mean=float(null.mean()),
                p=permutation_p(observed, null), n_permutations=N_PERM)


def concept_retrieval(T: np.ndarray, N: np.ndarray, rows: np.ndarray,
                      target: np.ndarray) -> list[dict]:
    """Word-to-class and class-to-word retrieval, one row per direction.

    Each row carries the queries whose target ranks first (``hits``) and among
    the K nearest (``hits10``), the binomial standard error of ``rate10``, the
    chance level ``K / n_candidates`` and the 1-based median rank of the target.
    """
    out = []
    for name, query, candidates, want, n_cand in (
            ("word_to_class", T[rows], N, target, len(N)),
            ("class_to_word", N[target], T[rows], np.arange(len(rows)), len(rows))):
        rank = cross_rank(query, candidates, want)
        hits, hits10 = int((rank == 0).sum()), int((rank < K).sum())
        n = len(rank)
        out.append(dict(direction=name, n=n, n_candidates=n_cand,
                        hits=hits, hits10=hits10, rate10=hits10 / n,
                        rate10_se=binomial_se(hits10, n),
                        chance10=K / n_cand,
                        median_rank=float(np.median(rank)) + 1))
    return out
