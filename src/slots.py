"""The slot table and the matched-appearance split.

A slot is one background paired with one typography setting. Every word is drawn
on all of them, so background and typography cancel in any comparison between
words. Each typography setting is used on ``REPLICATES`` different backgrounds,
and :func:`split_halves` puts half of the replicates on each side, so the two
halves carry the same typography and differ only in the photographs.
"""
from __future__ import annotations

import functools
import random

import pandas as pd

from .config import SEED
from .typography import FACTORS, build_appearance_design

# Backgrounds each appearance tuple is drawn on. Four gives 96 slots from 24
# tuples and two replicates per half of the matched-appearance split. The
# background count is n_slots().
REPLICATES = 4


@functools.lru_cache(maxsize=None)
def slot_table(n_backgrounds: int | None = None) -> pd.DataFrame:
    """One row per slot: ``slot, appearance, replicate, background_index`` plus
    every appearance factor. Deterministic.

    Backgrounds are assigned so that no two slots share one, and so that the
    background a given appearance receives in replicate 0 is unrelated to the
    one it receives in replicate 1.
    """
    appearance = build_appearance_design()
    n_slots = len(appearance) * REPLICATES
    if n_backgrounds is None:
        n_backgrounds = n_slots
    if n_backgrounds < n_slots:
        raise ValueError(
            f"{n_slots} slots need at least that many backgrounds, got {n_backgrounds}")
    order = list(range(n_backgrounds))
    random.Random(SEED).shuffle(order)
    rows = []
    for rep in range(REPLICATES):
        for _, a in appearance.iterrows():
            s = len(rows)
            rows.append(dict(slot=s, appearance=int(a.appearance), replicate=rep,
                             background_index=order[s],
                             **{f: a[f] for f in FACTORS},
                             monospace=bool(a.monospace)))
    return pd.DataFrame(rows)


def n_slots() -> int:
    return len(slot_table())


def slots_for(appearances) -> tuple[int, ...]:
    """Every slot carrying one of these appearance tuples, all replicates."""
    t = slot_table()
    return tuple(sorted(t.loc[t.appearance.isin(list(appearances)), "slot"]))


def split_halves(appearances=None) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Matched-appearance halves: replicates {0, 1} against {2, 3}.

    Each half carries every appearance tuple twice, and the halves differ only
    in which photographs they were drawn on. ``appearances`` restricts both
    halves to those appearance tuples; ``None`` keeps all of them. Slots are
    returned in ascending order.
    """
    t = slot_table()
    if appearances is not None:
        t = t[t.appearance.isin(list(appearances))]
    half = REPLICATES // 2
    lo, hi = t.replicate < half, t.replicate >= half
    # Sort on (appearance, replicate), which is a unique key within a half.
    key = ["appearance", "replicate"]
    ta = t.loc[lo].sort_values(key, kind="mergesort")[list(FACTORS)].to_numpy()
    tb = t.loc[hi].sort_values(key, kind="mergesort")[list(FACTORS)].to_numpy()
    assert ta.shape == tb.shape and (ta == tb).all(), \
        "the two halves do not carry identical appearance tuples"
    return (tuple(sorted(t.loc[lo, "slot"].tolist())),
            tuple(sorted(t.loc[hi, "slot"].tolist())))
