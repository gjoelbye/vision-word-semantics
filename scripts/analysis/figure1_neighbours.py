"""Nearest words, held-out country-to-capital offsets and the Figure 1b projection.

Each of the 14 models is read at the final layer. For every word, the four
nearest words among the other 1,560 are recorded. For the 60 country-capital
pairs in five folds, the number of capitals ranked first is counted with the
country as the query, with and without the mean country-to-capital offset of
the other folds, and the mean cosine between each held-out offset and that mean
is tested against a null that re-pairs countries and capitals. For DINOv3-H,
the six displayed pairs are projected onto the mean offset of the other 54
pairs and the leading component of their midpoints. Writes
``results/{neighbours,relation_offsets,figure1b_projection}.parquet``.
"""
from __future__ import annotations

import argparse
import gc

import numpy as np
import pandas as pd

from src import neighbors as NB
from src.config import LEXICON, RESULTS, ensure_dirs
from src.models import DEFAULT_SWEEP
from src.render import fingerprint
from src.tau import HEADLINE, estimator, load_cache, manifest, rownorm, word_order

#: Nearest words kept per query.
NEIGHBOURS = 4
RELATION = "country_capital"
N_FOLDS = 5
FOLD_SEED = 0
PERMUTATION_SEED = 2
N_PERM = 999
#: The model and the six pairs of Figure 1b.
PROJECTED_MODEL = "dinov3-h"
DISPLAY = ("colombia", "finland", "nigeria", "poland", "turkey", "vietnam")


def neighbour_rows(model: str, S: np.ndarray, frame: pd.DataFrame, pool: str,
                   meaning_pairs: set[tuple[str, str]]) -> pd.DataFrame:
    """The nearest words of every word of ``frame``, in the row order of ``S``."""
    nn = NB.neighbour_indices(S, k=NEIGHBOURS)
    query_i = np.repeat(np.arange(len(frame)), NEIGHBOURS)
    neighbour_i = nn.reshape(-1)
    query = frame.iloc[query_i].reset_index(drop=True)
    neighbour = frame.iloc[neighbour_i].reset_index(drop=True)
    return pd.DataFrame({
        "model": model,
        "pool": pool,
        "query": query.word.to_numpy(),
        "rank": np.tile(np.arange(1, NEIGHBOURS + 1), len(frame)),
        "neighbour": neighbour.word.to_numpy(),
        "cosine": S[query_i, neighbour_i],
        "semantic_partner": [pair in meaning_pairs for pair in
                             zip(query.word, neighbour.word)],
        "same_category": (query.category.notna().to_numpy()
                          & neighbour.category.notna().to_numpy()
                          & query.category.eq(neighbour.category).to_numpy()),
    })


def balanced_folds(n: int, n_folds: int, seed: int) -> np.ndarray:
    """Fold per pair: a seeded permutation of the rows, assigned round robin."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.empty(n, dtype=int)
    folds[order] = np.arange(n) % n_folds
    return folds


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """A uniform permutation with no fixed point, by rejection."""
    identity = np.arange(n)
    while True:
        candidate = rng.permutation(n)
        if np.all(candidate != identity):
            return candidate


def ranks(scores: np.ndarray, targets: np.ndarray,
          excluded: np.ndarray) -> np.ndarray:
    """One-based competition rank of each row's target, with one item per row excluded."""
    scores = np.asarray(scores).copy()
    scores[np.arange(len(scores)), excluded] = -np.inf
    target_scores = scores[np.arange(len(scores)), targets]
    return 1 + np.sum(scores > target_scores[:, None], axis=1)


def fold_offsets(A: np.ndarray, B: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Mean offset ``B - A`` over the pairs outside each fold."""
    offsets = B - A
    return np.stack([offsets[folds != fold].mean(0) for fold in range(N_FOLDS)])


def alignment_mean(A: np.ndarray, B: np.ndarray, folds: np.ndarray) -> float:
    """Mean cosine between each held-out offset and the mean offset of the other folds."""
    means = fold_offsets(A, B, folds)
    directions = rownorm(means)[folds]
    return float(np.sum(rownorm(B - A) * directions, axis=1).mean())


def offset_row(model: str, T: np.ndarray, source_rows: np.ndarray,
               target_rows: np.ndarray) -> dict:
    """Capitals ranked first among all words, and held-out offset alignment with its null.

    A query is the country, or the country plus the mean offset of the other
    folds; the country itself is excluded from the candidates. The null
    re-pairs countries with capitals by a derangement and refits every fold.
    """
    n = len(source_rows)
    folds = balanced_folds(n, N_FOLDS, FOLD_SEED)
    A, B = T[source_rows], T[target_rows]
    means = fold_offsets(A, B, folds)
    with_offset = ranks(rownorm(A + means[folds]) @ T.T, target_rows,
                        source_rows)
    without_offset = ranks(A @ T.T, target_rows, source_rows)
    observed = alignment_mean(A, B, folds)
    rng = np.random.default_rng(PERMUTATION_SEED)
    null = np.array([alignment_mean(A, B[derangement(n, rng)], folds)
                     for _ in range(N_PERM)])
    return dict(model=model, relation=RELATION, n_pairs=n,
                first_without_offset=int((without_offset == 1).sum()),
                first_with_offset=int((with_offset == 1).sum()),
                alignment_mean=observed,
                alignment_mean_null_mean=float(null.mean()),
                alignment_mean_null_p=float(
                    (np.sum(null >= observed) + 1) / (len(null) + 1)),
                n_folds=N_FOLDS, n_perm=N_PERM)


def unit(vector: np.ndarray) -> np.ndarray:
    """``vector`` in float64, scaled to unit length."""
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def leading_component(matrix: np.ndarray) -> np.ndarray:
    """Leading right-singular vector of the centered rows, signed so the largest projection is positive."""
    centred = np.asarray(matrix, dtype=float)
    centred = centred - centred.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    axis = vh[0].copy()
    projections = centred @ axis
    if projections[int(np.argmax(np.abs(projections)))] < 0:
        axis *= -1
    return axis


def midpoint_axis(axis1: np.ndarray, midpoints: np.ndarray) -> np.ndarray:
    """Leading component of the midpoints after the ``axis1`` direction is removed."""
    centred = midpoints - midpoints.mean(axis=0, keepdims=True)
    residual = centred - np.outer(centred @ axis1, axis1)
    axis2 = leading_component(residual)
    axis2 -= axis1 * float(axis1 @ axis2)
    return unit(axis2)


def mean_pairwise_cosine(differences: np.ndarray) -> float:
    """Mean cosine over every two rows of ``differences``."""
    directions = rownorm(np.asarray(differences, dtype=np.float32))
    gram = directions @ directions.T
    return float(gram[np.triu_indices(len(gram), 1)].mean())


def projection_rows(model: str, pairs: pd.DataFrame, a: np.ndarray,
                    b: np.ndarray) -> list[dict]:
    """The twelve endpoints of the displayed pairs, and one row of offset cosines.

    Axis 1 is the mean offset of the undisplayed pairs; axis 2 is the leading
    component of the displayed midpoints orthogonal to it. Coordinates are
    relative to the mean of the displayed endpoints.
    """
    selected = pairs.a.isin(DISPLAY).to_numpy()
    differences = b - a
    midpoints = (a + b) / 2
    axis1 = unit(differences[~selected].mean(axis=0))
    basis = np.column_stack([axis1, midpoint_axis(axis1, midpoints[selected])])
    centre = np.concatenate([a[selected], b[selected]], axis=0).mean(axis=0)
    rows = [dict(record_type="diagnostic", model=model,
                 selected_pairwise_cosine=mean_pairwise_cosine(
                     differences[selected]),
                 all_pairwise_cosine=mean_pairwise_cosine(differences),
                 n_display=int(selected.sum()),
                 n_undisplayed=int((~selected).sum()))]
    for index in np.flatnonzero(selected):
        for endpoint, word, vector in (
                ("source", pairs.a.iloc[index], a[index]),
                ("target", pairs.b.iloc[index], b[index])):
            xy = (vector - centre) @ basis
            rows.append(dict(record_type="coordinate", model=model,
                             pair_index=int(index),
                             source=pairs.a.iloc[index],
                             target=pairs.b.iloc[index], endpoint=endpoint,
                             word=word, x=float(xy[0]), y=float(xy[1])))
    return rows


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    ensure_dirs()
    words = pd.read_parquet(LEXICON / "words.parquet")
    pair_table = pd.read_parquet(LEXICON / "pairs.parquet")
    vocab = list(word_order())
    index = {w: i for i, w in enumerate(vocab)}
    vocab_frame = pd.DataFrame({"word": vocab}).merge(words, on="word",
                                                      how="left")
    meaning = pair_table[pair_table.kind == "meaning"]
    meaning_pairs = (set(zip(meaning.a, meaning.b))
                     | set(zip(meaning.b, meaning.a)))
    pairs = (pair_table[pair_table.relation.eq(RELATION)]
             .sort_values("a").reset_index(drop=True))
    source_rows = np.array([index[w] for w in pairs.a])
    target_rows = np.array([index[w] for w in pairs.b])

    neighbours, offsets, projection = [], [], []
    for model in DEFAULT_SWEEP:
        _, layers, n_layers = load_cache(model)
        if int(layers[-1]) != n_layers:
            raise SystemExit(f"{model}: the last cached layer is not the final one")
        T = rownorm(estimator(HEADLINE)(model, -1))
        neighbours.append(neighbour_rows(model, NB.similarity(T), vocab_frame,
                                         "vocabulary", meaning_pairs))
        offsets.append(offset_row(model, T, source_rows, target_rows))
        if model == PROJECTED_MODEL:
            projection += projection_rows(model, pairs, T[source_rows],
                                          T[target_rows])
        del T
        load_cache.cache_clear()
        gc.collect()
        print(f"  {model}")

    sha = fingerprint(manifest())
    for name, frame in (("neighbours", pd.concat(neighbours, ignore_index=True)),
                        ("relation_offsets", pd.DataFrame(offsets)),
                        ("figure1b_projection", pd.DataFrame(projection))):
        frame["manifest_sha"] = sha
        frame.to_parquet(RESULTS / f"{name}.parquet", index=False)
        print(f"wrote results/{name}.parquet ({len(frame)} rows)")


if __name__ == "__main__":
    main()
