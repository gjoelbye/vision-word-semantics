"""Nearest-neighbor readouts on word directions: top-10 accuracy and neighbor purity.

Both count the ``K`` nearest words. A partner's rank counts the words strictly
nearer than it, so a tie counts for the partner. Intervals subsample words
without replacement, because a resampled duplicate is its own nearest neighbor.
"""
from __future__ import annotations

import numpy as np

from .stats import binomial_se, subsample_summary
from .tau import rownorm

#: Nearest words counted by both readouts: top-10 accuracy and neighbor purity.
K = 10
N_PERM = 2000
N_SUBSAMPLE = 1000
SUBSAMPLE_FRAC = 0.8


def similarity(T: np.ndarray) -> np.ndarray:
    """Word-by-word cosine similarity with the diagonal set to ``-inf``.

    A word would otherwise be its own nearest neighbor and fill a place in
    every query's top k.
    """
    Tn = rownorm(np.asarray(T, np.float32))
    S = Tn @ Tn.T
    np.fill_diagonal(S, -np.inf)
    return S


# --- partner retrieval -------------------------------------------------------

def partner_rank(S: np.ndarray, i: int, j: int) -> int:
    """How many words are strictly nearer to ``i`` than ``j`` is; 0 means ``j`` is nearest."""
    row = S[i]
    return int((row > row[j]).sum())


def partner_ranks(S: np.ndarray, index: dict[str, int],
                  pairs: list[tuple[str, str]], reverse: bool = False
                  ) -> np.ndarray:
    """Partner rank for every pair in one direction; a pair with a word outside ``index`` is skipped."""
    out = []
    for a, b in pairs:
        if reverse:
            a, b = b, a
        if a in index and b in index:
            out.append(partner_rank(S, index[a], index[b]))
    return np.asarray(out, int)


def retrieval(S: np.ndarray, index: dict[str, int],
              pairs: list[tuple[str, str]], relation: str) -> list[dict]:
    """One row per direction of a relation, each with its own denominator.

    ``hits`` counts partners ranked first and ``hits10`` partners among the K
    nearest (zero-based rank below K). ``rate10_se`` is the binomial standard
    error of ``rate10``, and ``chance10`` is ``K / (N - 1)``.
    """
    chance10 = K / (len(index) - 1)
    rows = []
    for name, reverse in (("forward", False), ("backward", True)):
        ranks = partner_ranks(S, index, pairs, reverse=reverse)
        hits10 = (ranks < K)
        rows.append(dict(
            relation=relation, direction=name, n=int(len(ranks)),
            hits=int((ranks == 0).sum()), hits10=int(hits10.sum()),
            rate10=float(hits10.mean()) if len(ranks) else np.nan,
            rate10_se=binomial_se(int(hits10.sum()), len(ranks)),
            chance10=chance10))
    return rows


# --- neighbor purity ---------------------------------------------------------

def knn_purity(S: np.ndarray, labels: np.ndarray, k: int = K,
               restrict: np.ndarray | None = None) -> float:
    """Share of each word's ``k`` nearest neighbors that carry its own label.

    ``restrict`` limits each word's neighbors to words sharing its value.
    """
    nn = neighbour_indices(S, k, restrict)
    return float((labels[nn] == labels[:, None]).mean())


def neighbour_indices(S: np.ndarray, k: int = K,
                      restrict: np.ndarray | None = None) -> np.ndarray:
    """``(n, k)`` row indices of each word's ``k`` nearest neighbors.

    The diagonal is excluded, and ``restrict`` excludes words that do not share
    its value. A word with fewer than ``k`` admissible neighbors raises
    ValueError.
    """
    work = np.array(S, dtype=np.float32, copy=True)
    np.fill_diagonal(work, -np.inf)
    if restrict is not None:
        work[restrict[:, None] != restrict[None, :]] = -np.inf
    admissible = np.isfinite(work).sum(1)
    if (admissible < k).any():
        raise ValueError(
            f"{int((admissible < k).sum())} words have fewer than k={k} "
            f"admissible neighbours (fewest {int(admissible.min())})")
    return np.argsort(-work, axis=1)[:, :k]


def chance_purity(labels: np.ndarray) -> float:
    """``sum_c n_c(n_c - 1) / (n(n - 1))``: the purity of randomly assigned labels."""
    labels = np.asarray(labels)
    n = len(labels)
    _, counts = np.unique(labels, return_counts=True)
    return float((counts * (counts - 1)).sum() / (n * (n - 1)))


def restricted_chance_purity(labels: np.ndarray,
                             restrict: np.ndarray) -> float:
    """Chance purity when every neighbor must share ``restrict`` with its query.

    For each query, the share of the other words in its restriction group that
    carry its label, averaged over queries.
    """
    labels = np.asarray(labels)
    restrict = np.asarray(restrict)
    if len(labels) != len(restrict):
        raise ValueError("labels and restrict must have the same length")
    probabilities = []
    for label, group in zip(labels, restrict):
        in_group = restrict == group
        denominator = int(in_group.sum()) - 1
        if denominator <= 0:
            raise ValueError("a restriction group has no candidate neighbours")
        numerator = int((in_group & (labels == label)).sum()) - 1
        probabilities.append(numerator / denominator)
    return float(np.mean(probabilities))


def shuffle_groups(n: int, *keys) -> np.ndarray:
    """Cell index per word for the intersection of the given groupings; ``None`` keys are ignored."""
    present = [np.asarray(key) for key in keys if key is not None]
    if not present:
        return np.zeros(n, dtype=int)
    pairs = list(zip(*(key.tolist() for key in present)))
    order = {cell: i for i, cell in enumerate(sorted(set(pairs)))}
    return np.array([order[cell] for cell in pairs])


def purity_permutation(S: np.ndarray, labels: np.ndarray, k: int = K,
                       n_perm: int = N_PERM, seed: int = 0,
                       restrict: np.ndarray | None = None,
                       strata: np.ndarray | None = None) -> dict:
    """Observed purity against a null that shuffles the labels with the geometry fixed.

    Labels are shuffled within the cells of ``restrict`` crossed with
    ``strata``; ``restrict`` also limits the neighbors, ``strata`` does not.
    The p-value is ``(count + 1) / (n_perm + 1)``.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    nn = neighbour_indices(S, k, restrict)

    def score(y: np.ndarray) -> float:
        return float((y[nn] == y[:, None]).mean())

    observed = score(labels)
    cells = shuffle_groups(len(labels), restrict, strata)
    members = [cells == value for value in np.unique(cells)]

    def shuffle() -> np.ndarray:
        if len(members) == 1:
            return rng.permutation(labels)
        out = labels.copy()
        for m in members:
            out[m] = rng.permutation(labels[m])
        return out

    null = np.array([score(shuffle()) for _ in range(n_perm)])
    p = (int((null >= observed).sum()) + 1) / (n_perm + 1)
    chance = (restricted_chance_purity(labels, restrict)
              if restrict is not None else chance_purity(labels))
    return dict(purity=observed, p=float(p), null_mean=float(null.mean()),
                null_sd=float(null.std()), chance=chance, k=k, n_perm=n_perm)


def purity_subsample(S: np.ndarray, labels: np.ndarray, k: int = K,
                     n: int = N_SUBSAMPLE, frac: float = SUBSAMPLE_FRAC,
                     seed: int = 0,
                     restrict: np.ndarray | None = None) -> dict:
    """Purity with a 95% interval and standard error from subsamples of words.

    Each draw keeps ``frac`` of the words without replacement and recomputes
    purity on the reduced similarity matrix. The spread is rescaled by the
    finite-population factor and the interval is centered on the point estimate
    (see :func:`src.stats.subsample_summary`).
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    N = len(labels)
    m = int(frac * N)
    if restrict is not None:
        restrict = np.asarray(restrict)
        if len(restrict) != N:
            raise ValueError("labels and restrict must have the same length")
    point = knn_purity(S, labels, k, restrict)
    draws = np.array([knn_purity(
        S[np.ix_(i, i)], labels[i], k,
        None if restrict is None else restrict[i])
                      for i in (rng.choice(N, m, replace=False)
                                for _ in range(n))])
    lo, hi, se = subsample_summary(point, draws, N, m)
    return dict(purity=float(point), ci_lo=lo, ci_hi=hi, se=se, n=N, m=m)
