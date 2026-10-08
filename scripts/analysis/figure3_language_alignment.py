"""Mutual 10-NN alignment and linear CKA between 12 image models and 6 encoders.

The encoders are CLIP-B, SigLIP2-B, the three Qwen3 text encoders and w2v-BERT
2.0, and each cell is tested against 1,000 relabelings of the encoder's words
among words of the same length, with Benjamini-Hochberg adjustment within each
measure. Reads the matrices of ``scripts/features/similarity.py`` and writes
``results/language_alignment.parquet``.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.analysis.directions import manifest, word_order
from src.config import LEXICON, RESULTS, SEED, SIMILARITY, ensure_dirs
from src.features import speech, text
from src.models import DEFAULT_SWEEP
from src.stats import benjamini_hochberg
from src.stimuli.render import fingerprint

INDEX = SIMILARITY / "index.parquet"
OUT = RESULTS / "language_alignment.parquet"

#: Neighbors per word in mutual k-NN alignment.
K_NEAREST = 10

#: Relabelings in the length-stratified null. The smallest attainable p is
#: 1 / (N_PERM + 1).
N_PERM = 1000

#: The two measures, in the order ``measure_cells`` returns them.
MEASURE_NAMES = ("mutual_knn", "cka")

LANGUAGE_SUPERVISED = ("clip-b", "siglip2-b")
COLUMNS = [m for m in DEFAULT_SWEEP if m not in LANGUAGE_SUPERVISED]
ROWS = [*LANGUAGE_SUPERVISED, *text.TEXT_MODELS, *speech.SPEECH_MODELS]


def nearest(matrix: np.ndarray, k: int = K_NEAREST) -> np.ndarray:
    """Each string's ``k`` nearest other strings, as ``(n, k)`` indices."""
    ranked = np.array(matrix, np.float32)
    np.fill_diagonal(ranked, -np.inf)
    return np.argpartition(-ranked, k, axis=1)[:, :k]


def adjacency(neighbors: np.ndarray) -> np.ndarray:
    """``A[i, j]`` is true where ``j`` is among the nearest of ``i``."""
    out = np.zeros((len(neighbors), len(neighbors)), bool)
    out[np.arange(len(neighbors))[:, None], neighbors] = True
    return out


def centred(matrix: np.ndarray) -> np.ndarray:
    """``H S H`` scaled to unit Frobenius norm, in single precision."""
    out = np.array(matrix, np.float64)
    out -= out.mean(0, keepdims=True)
    out -= out.mean(1, keepdims=True)
    out /= np.linalg.norm(out)
    return out.astype(np.float32)


def cka_chance(column_kernel: np.ndarray, row_kernel: np.ndarray) -> float:
    """CKA expected when one encoder's strings are relabeled uniformly."""
    return float(np.trace(column_kernel, dtype=np.float64)
                 * np.trace(row_kernel, dtype=np.float64)
                 / (len(row_kernel) - 1))


def measure_cells(column_adjacency, column_kernels, row_neighbors, row_kernel
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Mutual k-NN alignment and linear CKA of one row against every column.

    Mutual k-NN alignment is a gather from the column adjacency at the row's
    neighbor indices. CKA is the inner product of unit-norm centered matrices.
    """
    n, k = row_neighbors.shape
    rows = np.arange(n)[:, None]
    mutual = column_adjacency[:, rows, row_neighbors].sum((1, 2)) / (n * k)
    cka = column_kernels.reshape(len(column_kernels), -1) @ row_kernel.ravel()
    return mutual, cka.astype(np.float64)


def relabel(generator, strata: np.ndarray) -> np.ndarray:
    """A permutation of the strings within strata."""
    out = np.empty(len(strata), int)
    for value in np.unique(strata):
        members = np.flatnonzero(strata == value)
        out[members] = generator.permutation(members)
    return out


def measure(strata: np.ndarray) -> pd.DataFrame:
    """Every cell's value, chance, length-null mean and p, one row per measure.

    The generator is reseeded for each row encoder, so every row draws the same
    relabelings.
    """
    column_adjacency, column_kernels = [], []
    for model in COLUMNS:
        matrix = np.load(SIMILARITY / f"{model}.npy")
        column_adjacency.append(adjacency(nearest(matrix)))
        column_kernels.append(centred(matrix))
        print(f"  read {model}", flush=True)
    column_adjacency = np.stack(column_adjacency)
    column_kernels = np.stack(column_kernels)

    mutual_chance = K_NEAREST / (len(strata) - 1)
    records = []
    for reference in ROWS:
        matrix = np.load(SIMILARITY / f"{reference}.npy")
        neighbors, kernel = nearest(matrix), centred(matrix)
        observed = measure_cells(column_adjacency, column_kernels, neighbors,
                                 kernel)
        chance = {"mutual_knn": np.full(len(COLUMNS), mutual_chance),
                  "cka": np.array([cka_chance(c, kernel)
                                   for c in column_kernels])}
        generator = np.random.default_rng(SEED)
        draws = [[], []]
        for _ in range(N_PERM):
            order = relabel(generator, strata)
            inverse = np.argsort(order)
            for store, value in zip(draws, measure_cells(
                    column_adjacency, column_kernels, inverse[neighbors[order]],
                    kernel[np.ix_(order, order)])):
                store.append(value)
        for name, value, null in zip(MEASURE_NAMES, observed, draws):
            null = np.stack(null)
            for column, model in enumerate(COLUMNS):
                records.append({
                    "reference": reference, "model": model, "measure": name,
                    "value": float(value[column]),
                    "chance": float(chance[name][column]),
                    "length_null_mean": float(null[:, column].mean()),
                    "p": float((1 + (null[:, column] >= value[column]).sum())
                               / (N_PERM + 1)),
                })
        print(f"  measured {reference}", flush=True)
    frame = pd.DataFrame(records)
    frame["p_bh"] = frame.groupby("measure").p.transform(
        lambda p: benjamini_hochberg(p.to_numpy()))
    return frame


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    ensure_dirs()
    # A fixed BLAS thread count keeps the floating-point order of the products.
    threadpool_limits(8)

    index = pd.read_parquet(INDEX).set_index("model")
    words = (pd.DataFrame({"word": word_order()})
               .merge(pd.read_parquet(LEXICON / "words.parquet"),
                      on="word", how="left"))
    word_sha = text.fingerprint(words.word)
    manifest_sha = fingerprint(manifest())
    for model in COLUMNS + ROWS:
        if model not in index.index or index.word_sha[model] != word_sha:
            raise RuntimeError(f"{model} was computed for a different string set; "
                               f"re-run scripts/features/similarity.py")
        if index.kind[model] == "vision" and index.manifest_sha[model] != manifest_sha:
            raise RuntimeError(f"{model} was computed from a different manifest; "
                               f"re-run scripts/features/similarity.py")
    print(f"{len(COLUMNS)} columns x {len(ROWS)} rows over {len(words):,} strings")

    frame = measure(words.n_chars.to_numpy())
    frame = frame.assign(n_perm=N_PERM, word_sha=word_sha,
                         manifest_sha=manifest_sha)
    frame.to_parquet(OUT)
    print(f"wrote results/{OUT.name}: {len(frame)} cells")


if __name__ == "__main__":
    main()
