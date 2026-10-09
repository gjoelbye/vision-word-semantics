"""The lexical measures of Section 4 and Appendix C, computed on word-by-word cosines.

`scripts/analysis/table1_models.py` runs them for every model and encoder, and
`figure1_neighbours.py` and `figure6_finetune.py` reuse them. Partner retrieval
and neighbor purity count the `K` nearest words. A partner's rank counts the
words strictly nearer than it, so a tie counts for the partner. Intervals
subsample words without replacement, because a resampled duplicate is its own
nearest neighbor.

The comparisons within an item use the three triplet lists, here called blocks:
each item holds an anchor, its synonym (`kept`) and a competitor (`other`: the
one-edit word, the antonym or the base), and is scored by which of the two is
nearer to the anchor. Nearest-centroid accuracy assigns held-out nouns and
brands to the nearest centroid on a fixed half split of every
category-by-spelling-cluster cell. Both depend on the vocabulary alone, so
vision models and encoders are scored on the same items and the same held-out
words.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..stats import (N_PERM, N_SUBSAMPLE, SUBSAMPLE_FRAC, binomial_se,
                     permutation_p, subsample_indices, subsample_size,
                     subsample_summary)
from .directions import rownorm

# Nearest words counted by both measures: top-10 accuracy and neighbor purity.
K = 10


def similarity(T: np.ndarray) -> np.ndarray:
    """Word-by-word cosine similarity with the diagonal set to `-inf`.

    A word would otherwise be its own nearest neighbor and fill a place in
    every query's top k.
    """
    Tn = rownorm(np.asarray(T, np.float32))
    S = Tn @ Tn.T
    np.fill_diagonal(S, -np.inf)
    return S


# Partner retrieval.

def partner_rank(S: np.ndarray, i: int, j: int) -> int:
    """How many words are strictly nearer to `i` than `j` is; 0 means `j` is nearest."""
    row = S[i]
    return int((row > row[j]).sum())


def partner_ranks(S: np.ndarray, index: dict[str, int],
                  pairs: list[tuple[str, str]], reverse: bool = False
                  ) -> np.ndarray:
    """Partner rank for every pair in one direction; a pair with a word outside `index` is skipped."""
    out = []
    for a, b in pairs:
        if reverse:
            a, b = b, a
        if a in index and b in index:
            out.append(partner_rank(S, index[a], index[b]))
    return np.asarray(out, int)


def retrieval(S: np.ndarray, index: dict[str, int],
              pairs: list[tuple[str, str]], relation: str) -> list[dict]:
    """One row per query direction of a relation, each with its own denominator.

    `hits` counts partners ranked first and `hits10` partners among the K
    nearest (zero-based rank below K). `rate10_se` is the binomial standard
    error of `rate10`, and `chance10` is `K / (N - 1)`.
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


# Neighbor purity.

def knn_purity(S: np.ndarray, labels: np.ndarray, k: int = K,
               restrict: np.ndarray | None = None) -> float:
    """Share of each word's `k` nearest neighbors that carry its own label.

    `restrict` limits each word's neighbors to words sharing its value.
    """
    nn = neighbour_indices(S, k, restrict)
    return float((labels[nn] == labels[:, None]).mean())


def neighbour_indices(S: np.ndarray, k: int = K,
                      restrict: np.ndarray | None = None) -> np.ndarray:
    """`(n, k)` row indices of each word's `k` nearest neighbors.

    The diagonal is excluded, and `restrict` excludes words that do not share
    its value. A word with fewer than `k` admissible neighbors raises
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
            f"admissible neighbors (fewest {int(admissible.min())})")
    return np.argsort(-work, axis=1)[:, :k]


def chance_purity(labels: np.ndarray) -> float:
    """`sum_c n_c(n_c - 1) / (n(n - 1))`: the purity of randomly assigned labels."""
    labels = np.asarray(labels)
    n = len(labels)
    _, counts = np.unique(labels, return_counts=True)
    return float((counts * (counts - 1)).sum() / (n * (n - 1)))


def restricted_chance_purity(labels: np.ndarray,
                             restrict: np.ndarray) -> float:
    """Chance purity when every neighbor must share `restrict` with its query.

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
            raise ValueError("a restriction group has no candidate neighbors")
        numerator = int((in_group & (labels == label)).sum()) - 1
        probabilities.append(numerator / denominator)
    return float(np.mean(probabilities))


def purity_permutation(S: np.ndarray, labels: np.ndarray, k: int = K,
                       n_perm: int = N_PERM, seed: int = 0,
                       strata: np.ndarray | None = None) -> dict:
    """Observed purity against a null that shuffles the labels with the geometry fixed.

    Labels are shuffled within each value of `strata`, or over all words
    without it. The p-value is `(count + 1) / (n_perm + 1)`.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    nn = neighbour_indices(S, k)

    def score(y: np.ndarray) -> float:
        return float((y[nn] == y[:, None]).mean())

    observed = score(labels)
    cells = (np.zeros(len(labels), dtype=int) if strata is None
             else np.asarray(strata))
    members = [cells == value for value in np.unique(cells)]

    def shuffle() -> np.ndarray:
        if len(members) == 1:
            return rng.permutation(labels)
        out = labels.copy()
        for m in members:
            out[m] = rng.permutation(labels[m])
        return out

    null = np.array([score(shuffle()) for _ in range(n_perm)])
    return dict(purity=observed, p=permutation_p(observed, null), null_mean=float(null.mean()),
                null_sd=float(null.std()), chance=chance_purity(labels), k=k,
                n_perm=n_perm)


def purity_subsample(S: np.ndarray, labels: np.ndarray, k: int = K,
                     n: int = N_SUBSAMPLE, frac: float = SUBSAMPLE_FRAC,
                     seed: int = 0) -> dict:
    """Purity with a 95% interval and standard error from subsamples of words.

    Each draw keeps `frac` of the words without replacement and recomputes
    purity on the reduced similarity matrix. The spread is rescaled by the
    finite-population factor and the interval is centered on the point estimate
    (see `stats.subsample_summary`).
    """
    labels = np.asarray(labels)
    N = len(labels)
    m = subsample_size(N, frac)
    point = knn_purity(S, labels, k)
    draws = np.array([knn_purity(S[np.ix_(i, i)], labels[i], k)
                      for i in subsample_indices(N, frac, n, seed)])
    lo, hi, se = subsample_summary(point, draws, N, m)
    return dict(purity=float(point), ci_lo=lo, ci_hi=hi, se=se, n=N, m=m)


# Comparisons within an item.

BLOCKS = (
    # name, the relation giving the synonym, the relation giving the
    # competitor, the two column names, and the number of items.
    ("synonym_vs_twin", "synonym", "spelling_twin", "synonym", "twin", 150),
    ("synonym_vs_antonym", "antonym_synonym", "antonym", "synonym", "antonym", 50),
    ("synonym_vs_base", "negation", "negation_base", "synonym", "base", 35),
)


def build_items(pairs: pd.DataFrame, words: pd.DataFrame) -> dict:
    """The three blocks as frames with `anchor`, `kept` (the synonym) and `other` (the competitor).

    `class` holds the anchor's picturability from the word table. Built from
    `pairs.parquet` by joining two relations on their shared anchor. Raises
    SystemExit unless every block has its number of items and block 1 holds 75
    concrete and 75 abstract anchors.
    """
    picturability = dict(zip(words.word, words.picturability))
    out = {}
    for name, kept_rel, other_rel, kept, other, expected in BLOCKS:
        left = (pairs.loc[pairs.relation == kept_rel, ["a", "b"]]
                .rename(columns={"b": kept}))
        right = (pairs.loc[pairs.relation == other_rel, ["a", "b"]]
                 .rename(columns={"b": other}))
        frame = left.merge(right, on="a").rename(columns={"a": "anchor"})
        if len(frame) != expected:
            raise SystemExit(
                f"block {name} joined to {len(frame)} items, not {expected}; "
                f"the word set changed and the readout's n with it")
        frame["kept"], frame["other"] = frame[kept], frame[other]
        frame["class"] = frame.anchor.map(picturability)
        out[name] = frame
    block1 = out["synonym_vs_twin"]
    counts = block1["class"].value_counts().to_dict()
    if counts.get("concrete") != 75 or counts.get("abstract") != 75:
        raise SystemExit(
            f"block 1 holds {counts}, not 75 concrete and 75 abstract; the "
            f"class contrast is no longer balanced and its Fisher test would "
            f"be reading an imbalance instead of an effect")
    return out


# Nearest-centroid accuracy.

# The noun and brand lists, by the `role` their words take in `words.parquet`.
DATASETS = {"common_noun": "category", "brand": "brand"}

SPLIT_SEED = 0


def balanced_split(frame: pd.DataFrame, seed: int) -> np.ndarray:
    """A `train` or `test` label per row, halving every category-by-spelling-cluster cell."""
    rng = np.random.default_rng(seed)
    split = np.full(len(frame), "", dtype=object)
    for label in sorted(frame.label.unique()):
        for ortho in sorted(frame.ortho.unique()):
            rows = frame.index[
                frame.label.eq(label) & frame.ortho.eq(ortho)
            ].to_numpy()
            if len(rows) < 2 or len(rows) % 2:
                raise ValueError(
                    f"{label!r} × ortho {ortho!r} has {len(rows)} words; "
                    "the fixed half split requires a positive even count")
            rows = rng.permutation(rows)
            half = len(rows) // 2
            split[rows[:half]] = "train"
            split[rows[half:]] = "test"
    return split


def length_bands(n_chars: pd.Series) -> np.ndarray:
    """Word-length tercile (0, 1 or 2) per word.

    Lengths are tied integers, so the bands can be unequal in size. Raises
    SystemExit unless three bands survive.
    """
    band = pd.qcut(n_chars, 3, labels=False, duplicates="drop")
    band = band.astype(int).to_numpy()
    if len(np.unique(band)) != 3:
        raise SystemExit(
            f"word length gave {len(np.unique(band))} bands, not three; the "
            f"stratified null needs three and the vocabulary has changed")
    return band


def dataset_frame(words: pd.DataFrame, dataset: str, role: str) -> pd.DataFrame:
    """The words of one list sorted by word, with label, spelling cluster, length band and split."""
    frame = (words.loc[words.role.eq(role),
                       ["word", "category", "ortho", "n_chars"]]
             .rename(columns={"category": "label"})
             .sort_values("word").reset_index(drop=True))
    frame.insert(0, "dataset", dataset)
    frame["ortho"] = frame.ortho.astype(int)
    frame["n_chars"] = frame.n_chars.astype(int)
    frame["length_band"] = length_bands(frame.n_chars)
    frame["split"] = balanced_split(frame, SPLIT_SEED)
    return frame


def nearest_centroid_prediction(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    label_order: np.ndarray,
) -> np.ndarray:
    """The label of the nearest training centroid, by cosine, per held-out word."""
    centroids = np.stack([X_train[y_train == label].mean(0)
                          for label in label_order])
    return label_order[np.argmax(X_test @ rownorm(centroids).T, axis=1)]


def nearest_centroid_accuracy(
    X: np.ndarray, frame: pd.DataFrame, label_order: np.ndarray,
) -> tuple[float, int, int, np.ndarray]:
    """Held-out accuracy, the number correct, the number held out, and the predictions.

    `X` holds one unit-length row per word of `frame`, in its row order.
    """
    train = frame.split.eq("train").to_numpy()
    test = frame.split.eq("test").to_numpy()
    prediction = nearest_centroid_prediction(
        X[train], frame.loc[train, "label"].to_numpy(), X[test], label_order)
    correct = int((prediction == frame.loc[test, "label"].to_numpy()).sum())
    held = int(test.sum())
    return correct / held, correct, held, prediction
