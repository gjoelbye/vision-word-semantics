"""Statistics shared by the analysis code: standard errors, intervals, tests and corrections.

Words are subsampled without replacement, because a duplicated word is its own
nearest neighbor and inflates purity. A share over pairs or items takes the
binomial standard error. The two-clue tests treat each clue grid as one unit.
"""
from __future__ import annotations

import functools
import itertools

import numpy as np

# Permutation draws of the purity, category-accuracy and word-class nulls.
N_PERM = 2000
# Subsamples behind the purity interval, and the share of words each keeps.
N_SUBSAMPLE = 1000
SUBSAMPLE_FRAC = 0.8


def binomial_se(k: int, n: int) -> float:
    """Standard error of the proportion `k / n`, `sqrt(p (1 - p) / n)`."""
    if n == 0:
        return float("nan")
    p = k / n
    return float(np.sqrt(p * (1 - p) / n))


def subsample_summary(point: float, draws: np.ndarray, n: int,
                      m: int) -> tuple[float, float, float]:
    """95% interval and standard error from subsamples of `m` of `n` words.

    Both rescale the spread of the draws by the finite-population factor
    `sqrt(m (n - 1) / (n (n - m)))`: the interval rescales their percentiles
    about their mean and centers them on `point`, and the standard error
    rescales their standard deviation. The draws keep their own dtype.
    """
    draws = np.asarray(draws)
    scale = np.sqrt(m * (n - 1) / (n * (n - m)))
    lo, hi = point + scale * np.percentile(draws - draws.mean(), [2.5, 97.5])
    return float(lo), float(hi), float(scale * draws.std(ddof=1))


def subsample_indices(n: int, frac: float = SUBSAMPLE_FRAC,
                      draws: int = N_SUBSAMPLE, seed: int = 0):
    """Row indices for `draws` subsamples of `max(2, int(frac * n))` rows, without replacement."""
    rng = np.random.default_rng(seed)
    m = max(2, int(frac * n))
    for _ in range(draws):
        yield rng.choice(n, m, replace=False)


def permutation_p(observed: float, null: np.ndarray) -> float:
    """Permutation p-value `(count + 1) / (n + 1)`, where count is the null draws at or above `observed`."""
    null = np.asarray(null, float)
    return float((int((null >= observed).sum()) + 1) / (len(null) + 1))


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the order given.

    The r-th smallest of n values is multiplied by `n - r` (r from 0), the
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


def sign_test(differences: np.ndarray) -> dict:
    """Items the synonym wins, and the two-sided exact sign test.

    A tie counts for neither side and leaves the denominator.
    """
    from scipy.stats import binomtest

    wins = int((differences > 0).sum())
    losses = int((differences < 0).sum())
    p = (float(binomtest(wins, wins + losses, 0.5).pvalue)
         if wins + losses else float("nan"))
    return dict(n=int(len(differences)), wins=wins, sign_p=p)


def vocabulary_spread(cosine: np.ndarray) -> tuple[float, float]:
    """The mean and standard deviation of the off-diagonal cells of a cosine matrix.

    Cosines are expressed as z-scores against these two numbers, so they can be
    compared across models.
    """
    upper = np.triu_indices(len(cosine), 1)
    off = cosine[upper]
    return float(off.mean()), float(off.std())


@functools.lru_cache(maxsize=8)
def _sign_patterns(n: int) -> np.ndarray:
    """Every sign vector of length `n`."""
    return np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)))


def exact_sign_flip_p(values: np.ndarray, alternative: str = "greater") -> float:
    """Exact sign-flip p-value over every sign pattern of the grid means.

    The null is symmetric about zero. `alternative` is `greater` or
    `two-sided`.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("sign-flip values must be a finite nonempty vector")
    if len(values) > 20:
        raise ValueError("exact enumeration is limited to 20 units")
    observed = float(values.mean())
    null = _sign_patterns(len(values)) @ values / len(values)
    tolerance = np.finfo(float).eps * max(1.0, abs(observed)) * 8
    if alternative == "greater":
        return float(np.mean(null >= observed - tolerance))
    if alternative == "two-sided":
        return float(np.mean(np.abs(null) >= abs(observed) - tolerance))
    raise ValueError("alternative must be greater or two-sided")


def bootstrap_mean_ci(values: np.ndarray, draws: np.ndarray,
                      confidence: float = 0.95) -> tuple[float, float]:
    """Percentile interval of the mean, one resample per row of `draws`."""
    values = np.asarray(values, dtype=float)
    draws = np.asarray(draws, dtype=int)
    if draws.ndim != 2 or draws.shape[1] != len(values):
        raise ValueError("bootstrap draws do not match the family vector")
    means = values[draws].mean(axis=1)
    tail = (1 - confidence) / 2
    return tuple(float(x) for x in np.quantile(means, [tail, 1 - tail]))
