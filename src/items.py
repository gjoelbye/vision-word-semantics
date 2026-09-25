"""The three blocks of comparisons within an item, built from the word lists.

Each item holds an anchor and two competitors, and the readout asks which
competitor is nearer to the anchor. The blocks depend on the lexicon alone, so
vision models and encoders are scored on the same items.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BLOCKS = (
    # name, the relation giving the kept competitor, the relation giving the
    # competitor it is measured against, and the two column names.
    ("synonym_vs_twin", "synonym", "spelling_twin", "synonym", "twin", 150),
    ("synonym_vs_antonym", "antonym_synonym", "antonym", "synonym", "antonym", 50),
    ("synonym_vs_base", "negation", "negation_base", "synonym", "base", 35),
)


def build_items(pairs: pd.DataFrame, words: pd.DataFrame) -> dict:
    """The three blocks as frames with ``anchor``, ``kept``, ``other`` and the anchor's class.

    Built from ``pairs.parquet`` by joining two relations on their shared
    anchor; raises SystemExit if a block's size or the 75/75 class split of
    block 1 changes.
    """
    picturability = dict(zip(words.word, words.picturability))
    chars = dict(zip(words.word, words.n_chars))
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
        frame["anchor_chars"] = frame.anchor.map(chars)
        frame["kept_chars"] = frame.kept.map(chars)
        out[name] = frame
    block1 = out["synonym_vs_twin"]
    counts = block1["class"].value_counts().to_dict()
    if counts.get("concrete") != 75 or counts.get("abstract") != 75:
        raise SystemExit(
            f"block 1 holds {counts}, not 75 concrete and 75 abstract; the "
            f"class contrast is no longer balanced and its Fisher test would "
            f"be reading an imbalance instead of an effect")
    return out


def sign_test(differences: np.ndarray) -> dict:
    """Items won by the kept competitor, and the two-sided exact sign test.

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
