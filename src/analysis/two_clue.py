"""Difference vectors, margins and tests for the two-clue items.

The unit of inference is the clue grid of eight items. Embedding differences
are averaged over the 96 slots before any cosine is taken.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..stimuli.two_clue_render import CONDITIONS

#: The difference vectors that are scored against the targets.
READOUTS = ("joint", "feature", "base", "interaction")

BLOCK_FIELDS = (
    "equation_id", "slot", "appearance", "replicate", "background_index",
    "font", "size", "color", "placement", "box", "feature_position",
    "target_position", "upper_x", "upper_y", "lower_x", "lower_y",
)


def validate_dense_blocks(manifest: pd.DataFrame, *, n_equations: int,
                          n_slots: int) -> None:
    """Raise unless the rows reshape to ``(equation, slot, condition)`` blocks.

    Within a block the conditions follow ``CONDITIONS`` and every field of
    ``BLOCK_FIELDS`` is constant, and each equation holds every slot once.
    """
    required = set((*BLOCK_FIELDS, "condition"))
    if missing := required - set(manifest):
        raise ValueError(f"manifest lacks dense-layout fields {sorted(missing)}")
    n_conditions = len(CONDITIONS)
    expected_rows = n_equations * n_slots * n_conditions
    if len(manifest) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(manifest)}")
    conditions = np.asarray(manifest.condition, dtype=object).reshape(
        -1, n_conditions)
    expected = np.asarray(CONDITIONS, dtype=object)
    if not np.all(conditions == expected):
        raise ValueError("conditions are not canonical within every dense block")
    for field in BLOCK_FIELDS:
        values = np.asarray(manifest[field], dtype=object).reshape(
            -1, n_conditions)
        if not np.all(values == values[:, :1]):
            raise ValueError(f"{field} changes within an equation-slot block")
    equation_slots = manifest.iloc[::n_conditions].groupby(
        "equation_id", sort=False).slot.agg(["count", "nunique", "min", "max"])
    valid = (
        len(equation_slots) == n_equations
        and equation_slots["count"].eq(n_slots).all()
        and equation_slots["nunique"].eq(n_slots).all()
        and equation_slots["min"].eq(0).all()
        and equation_slots["max"].eq(n_slots - 1).all()
    )
    if not valid:
        raise ValueError("each equation must contain every canonical slot once")


def row_unit(values: np.ndarray) -> np.ndarray:
    """Unit-normalize rows, leaving a row of zero length at zero.

    A zero row scores a cosine of zero against every target and so a margin of
    zero. A non-finite value raises.
    """
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all():
        raise ValueError("cannot score a non-finite two-clue vector")
    return np.divide(values, norms, out=np.zeros_like(values),
                     where=norms > 1e-12)


def condition_vectors(block: np.ndarray) -> dict[str, np.ndarray]:
    """Reduce ``(slot, condition, dimension)`` embeddings to five directions.

    Within a slot and in float32, ``joint``, ``feature`` and ``base`` subtract
    the image with both clues scrambled, and ``target`` subtracts the scrambled
    target. Each is then averaged over slots, and ``interaction`` is
    ``joint - feature - base``.
    """
    block = np.asarray(block, dtype=np.float32)
    if block.ndim != 3 or block.shape[1] != len(CONDITIONS):
        raise ValueError(
            f"expected (slot, {len(CONDITIONS)}, dimension), got "
            f"{block.shape}")
    both, feature_only, base_only, scrambled, target, target_scrambled = (
        block[:, index] for index in range(len(CONDITIONS)))
    feature = (feature_only - scrambled).mean(axis=0)
    base = (base_only - scrambled).mean(axis=0)
    joint = (both - scrambled).mean(axis=0)
    target_direction = (target - target_scrambled).mean(axis=0)
    return {
        "joint": joint,
        "feature": feature,
        "base": base,
        "target": target_direction,
        "interaction": joint - feature - base,
    }


def target_metrics(items: pd.DataFrame,
                   vectors: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Margins of every readout against its item's target, and the axis gains.

    From the cosine with the target, ``*_margin`` subtracts the mean cosine with
    the other seven targets of the grid, ``*_feature_axis_margin`` the cosine
    with the one target sharing the base, and ``*_base_axis_margin`` the mean
    cosine with the three sharing the feature.
    """
    missing = set((*READOUTS, "target")) - set(vectors)
    if missing:
        raise ValueError(f"missing two-clue vectors: {sorted(missing)}")
    n = len(items)
    for name in (*READOUTS, "target"):
        if np.asarray(vectors[name]).shape[0] != n:
            raise ValueError(f"{name} has the wrong number of equations")

    result = items.reset_index(drop=True).copy()
    for readout in READOUTS:
        result[f"{readout}_margin"] = np.nan
        result[f"{readout}_feature_axis_margin"] = np.nan
        result[f"{readout}_base_axis_margin"] = np.nan

    groups = result.groupby("family_id", sort=False).indices
    for _, raw_indices in groups.items():
        indices = np.asarray(raw_indices, dtype=int)
        family = result.iloc[indices]
        if len(indices) != 8:
            raise ValueError("target scoring requires eight equations per family")
        targets = row_unit(np.asarray(vectors["target"])[indices])
        for readout in READOUTS:
            queries = row_unit(np.asarray(vectors[readout])[indices])
            scores = queries @ targets.T
            own = np.diag(scores)
            margins = own - (scores.sum(axis=1) - own) / 7
            feature_margins = np.empty(8, dtype=np.float32)
            base_margins = np.empty(8, dtype=np.float32)
            for local, row in enumerate(family.itertuples(index=False)):
                same_base_other_feature = (
                    family.base.eq(row.base) & ~family.feature.eq(row.feature)
                ).to_numpy()
                same_feature_other_base = (
                    family.feature.eq(row.feature) & ~family.base.eq(row.base)
                ).to_numpy()
                if same_base_other_feature.sum() != 1:
                    raise ValueError("family is not a complete two-feature grid")
                if same_feature_other_base.sum() != 3:
                    raise ValueError("family is not a complete four-base grid")
                feature_margins[local] = (
                    own[local] - scores[local, same_base_other_feature][0])
                base_margins[local] = (
                    own[local] - scores[local, same_feature_other_base].mean())
            result.loc[indices, f"{readout}_margin"] = margins
            result.loc[indices, f"{readout}_feature_axis_margin"] = feature_margins
            result.loc[indices, f"{readout}_base_axis_margin"] = base_margins

    result["feature_axis_gain_over_base"] = (
        result.joint_feature_axis_margin - result.base_feature_axis_margin)
    result["base_axis_gain_over_feature"] = (
        result.joint_base_axis_margin - result.feature_base_axis_margin)
    return result
