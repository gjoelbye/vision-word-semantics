"""Readouts that compare word directions with the visual directions of the classes they name.

The visual directions are built in ``directions``, and each cache is checked
against its own manifest there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..stats import (N_PERM, SUBSAMPLE_FRAC, binomial_se, permutation_p,
                     subsample_indices, subsample_summary)
from .directions import CLASS_IMAGES, load_cache, manifest, retained
from .lexical import K


def class_coherence(key: str, layer: int = -1) -> np.ndarray:
    """Mean cosine between two photographs of a class, one number per retained class.

    Every photograph counts, text or not. Photographs are centered on the mean
    over all photographs and scaled to unit length. For unit ``v_i`` the mean
    off-diagonal cosine is ``(||sum v_i||^2 - n) / (n (n - 1))``.
    """
    x, _, _ = load_cache(key, ds=CLASS_IMAGES)
    man = manifest(CLASS_IMAGES)
    classes = retained()
    index = {c: i for i, c in enumerate(classes)}
    position = np.flatnonzero(man.wnid.isin(index).to_numpy())
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
    ``lexical.partner_rank``, and no candidate is excluded.
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


def class_alignment(T: np.ndarray, N: np.ndarray, rows: np.ndarray,
                    target: np.ndarray, seed: int = 0) -> dict:
    """Mean cosine between each word direction and its class direction.

    ``T`` and ``N`` are unit rows; ``rows`` picks the words and ``target`` their
    classes. The null permutes which class goes with which word, the p-value is
    ``(count + 1) / (n + 1)``, and the standard error comes from subsamples of
    words without replacement (see ``stats.subsample_summary``).
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


def class_retrieval(T: np.ndarray, N: np.ndarray, rows: np.ndarray,
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
