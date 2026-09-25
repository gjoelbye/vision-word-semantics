"""Standard errors, subsampling intervals, permutation p-values and p-value corrections.

Words are never resampled with replacement, because a duplicated word is its
own nearest neighbour and inflates purity. Subsamples are drawn without
replacement, and a share over pairs or items takes the binomial standard error.
"""
from __future__ import annotations

import numpy as np


def binomial_se(k: int, n: int) -> float:
    """Standard error of the proportion ``k / n``, ``sqrt(p (1 - p) / n)``."""
    if n == 0:
        return float("nan")
    p = k / n
    return float(np.sqrt(p * (1 - p) / n))


def subsample_summary(point: float, draws: np.ndarray, n: int,
                      m: int) -> tuple[float, float, float]:
    """95% interval and standard error from subsamples of ``m`` of ``n`` words.

    Both rescale the spread of the draws by the finite-population factor
    ``sqrt(m (n - 1) / (n (n - m)))``: the interval rescales their percentiles
    about their mean and centres them on ``point``, and the standard error
    rescales their standard deviation. The draws keep their own dtype.
    """
    draws = np.asarray(draws)
    scale = np.sqrt(m * (n - 1) / (n * (n - m)))
    lo, hi = point + scale * np.percentile(draws - draws.mean(), [2.5, 97.5])
    return float(lo), float(hi), float(scale * draws.std(ddof=1))


def subsample_indices(n: int, frac: float = 0.8, draws: int = 1000,
                      seed: int = 0):
    """Row indices for ``draws`` subsamples of ``max(2, int(frac * n))`` rows, without replacement."""
    rng = np.random.default_rng(seed)
    m = max(2, int(frac * n))
    for _ in range(draws):
        yield rng.choice(n, m, replace=False)


def permutation_p(observed: float, null: np.ndarray,
                  two_sided: bool = False) -> float:
    """Permutation p-value ``(count + 1) / (n + 1)``, where count is the null draws at or above ``observed``."""
    null = np.asarray(null, float)
    hits = (np.abs(null) >= abs(observed)) if two_sided else (null >= observed)
    return float((int(hits.sum()) + 1) / (len(null) + 1))


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the order given.

    The r-th smallest of n values is multiplied by ``n - r`` (r from 0), the
    running maximum over the ascending order keeps the sequence monotone, and
    the result is capped at 1. Needs finite one-dimensional input.
    """
    p_values = np.asarray(pvalues, dtype=float)
    if p_values.ndim != 1 or not np.isfinite(p_values).all():
        raise ValueError("Holm correction needs finite one-dimensional p-values")
    order = np.argsort(p_values, kind="mergesort")
    ranked = p_values[order]
    adjusted_ranked = np.maximum.accumulate(
        ranked * np.arange(len(ranked), 0, -1))
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, in the order given, monotone and capped at one."""
    order = np.argsort(p)
    scaled = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out
