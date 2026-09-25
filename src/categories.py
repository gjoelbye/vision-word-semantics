"""Nearest-centroid accuracy on a fixed half split of the noun and brand lists.

Every category-by-spelling-cluster cell is halved with a fixed seed, centroids
are averaged over one half, and each word of the other half is assigned to the
nearest centroid by cosine. The split depends on the lexicon alone, so vision
models and encoders are scored on the same held-out words.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tau import rownorm

#: The two grids, by the ``role`` each takes in ``words.parquet``.
DATASETS = {"common_noun": "category", "brand": "brand"}

#: The seed of the half split.
SPLIT_SEED = 0


def balanced_split(frame: pd.DataFrame, seed: int) -> np.ndarray:
    """Return train/test labels, halving every label×orthography cell."""
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
    if (split == "").any():
        raise AssertionError("a word was not assigned to either split")
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


def dataset_frame(words: pd.DataFrame, dataset: str, role: str,
                  split_seed: int) -> pd.DataFrame:
    """The words of one list sorted by word, with label, spelling cluster, length band and split."""
    frame = (words.loc[words.role.eq(role),
                       ["word", "category", "ortho", "n_chars"]]
             .rename(columns={"category": "label"})
             .sort_values("word").reset_index(drop=True))
    frame.insert(0, "dataset", dataset)
    frame["ortho"] = frame.ortho.astype(int)
    frame["n_chars"] = frame.n_chars.astype(int)
    frame["length_band"] = length_bands(frame.n_chars)
    frame["split"] = balanced_split(frame, split_seed)
    return frame


def centred_centroids(
    X: np.ndarray, labels: np.ndarray, label_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Label centroids and directions from their unweighted grand mean."""
    centroids = np.stack([X[labels == label].mean(0)
                          for label in label_order])
    directions = centroids - centroids.mean(0, keepdims=True)
    return centroids, directions


def nearest_centroid_prediction(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    label_order: np.ndarray,
) -> np.ndarray:
    """The label of the nearest training centroid, by cosine, per held-out word."""
    centroids, _ = centred_centroids(X_train, y_train, label_order)
    return label_order[np.argmax(X_test @ rownorm(centroids).T, axis=1)]


def nearest_centroid_accuracy(
    X: np.ndarray, frame: pd.DataFrame, label_order: np.ndarray,
) -> tuple[float, int, int]:
    """Held-out accuracy, the number correct, and the number held out.

    ``X`` holds one unit-length row per word of ``frame``, in its row order.
    """
    train = frame.split.eq("train").to_numpy()
    test = frame.split.eq("test").to_numpy()
    prediction = nearest_centroid_prediction(
        X[train], frame.loc[train, "label"].to_numpy(), X[test], label_order)
    correct = int((prediction == frame.loc[test, "label"].to_numpy()).sum())
    held = int(test.sum())
    return correct / held, correct, held
