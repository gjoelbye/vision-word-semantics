"""Deterministic checks applied before rendering and reporting.

Nothing renders until the blocking checks pass. Named-entity rates are
reported rather than gated because the geography pairs contain proper nouns by
design.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .typography import crop_safe_window

#: Largest allowed excess of letter overlap within a meaning relation over
#: length-matched random word pairs.
GATE_MAX_LETTER_EXCESS = 0.05

#: Largest allowed Cramer's V between category and spelling cluster; the
#: crossing makes it zero by construction.
GATE_MAX_CRAMERS_V = 0.01


def gate(name: str, detail: str, value, threshold, ok: bool,
         blocking: bool = True) -> dict:
    """One row of an audit report. A failed check is FAIL when blocking, else warn."""
    return dict(gate=name, detail=detail, value=value, threshold=threshold,
                status="pass" if ok else ("FAIL" if blocking else "warn"),
                blocking=blocking)


def audit_words(words: pd.DataFrame) -> list[dict]:
    """Rendering safety: every word must fit the crop-safe window everywhere."""
    _, _, side = crop_safe_window()
    over = words.loc[~words.fits, "word"].tolist() if "fits" in words else []
    return [gate("render/fits",
                 f"widest word {words.max_px.max():.0f}px against a {side}px "
                 f"window" + (f", e.g. {', '.join(over[:5])}" if over else ""),
                 int(len(over)), 0, len(over) == 0)]


def audit_offensive(words: pd.DataFrame) -> list[dict]:
    """Reject blocked strings before committing or rendering the vocabulary."""
    from .lexicon import is_blocked
    from nltk.corpus import wordnet as wn

    bad = [w for w in words.word if is_blocked(w, wn.synsets(w))]
    return [gate("words/offensive",
                 "blocked strings in the rendered set"
                 + (f": {', '.join(bad[:6])}" if bad else ""),
                 len(bad), 0, not bad)]


def audit_pairs_orthography(pairs: pd.DataFrame,
                            vocab: list[str]) -> list[dict]:
    """Gates that no meaning pair is a pair of word forms and no meaning relation overlaps in letters."""
    from .lexicon import orthographic_relatives
    from .neighbors import letter_overlap

    out = []
    relatives = orthographic_relatives(pairs)
    out.append(gate(
        "pairs/orthography",
        "meaning pairs whose two words are forms of each other"
        + (f", e.g. {relatives.a.iloc[0]}/{relatives.b.iloc[0]}"
           if len(relatives) else ""),
        int(len(relatives)), 0, len(relatives) == 0))

    for relation, group in pairs[pairs.kind == "meaning"].groupby("relation"):
        ov = letter_overlap(list(zip(group.a, group.b)), vocab)
        out.append(gate(
            f"pairs/letters/{relation}",
            f"letter overlap within the relation ({ov['within']:.3f}) against "
            f"two random words ({ov['random']:.3f})",
            round(ov["excess"], 4), GATE_MAX_LETTER_EXCESS,
            ov["excess"] <= GATE_MAX_LETTER_EXCESS))
    return out


def audit_pairs_proper_nouns(pairs: pd.DataFrame) -> list[dict]:
    """Share of each relation's words that WordNet knows only as a named entity; reported, never blocking."""
    from .lexicon import is_named_entity

    rows = []
    for relation, group in pairs.groupby("relation"):
        words = pd.concat([group.a, group.b])
        share = float(np.mean([is_named_entity(w) for w in words]))
        rows.append(gate(f"pairs/proper-nouns/{relation}",
                         "share of the relation's words WordNet knows only as "
                         "a named entity",
                         round(share, 3), "reported, not gated",
                         True, blocking=False))
    return rows


def report(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    # thresholds are a mix of numbers and strings; keep the column one type so
    # the table round-trips through parquet
    df["threshold"] = df["threshold"].astype(str)
    order = {"FAIL": 0, "warn": 1, "pass": 2}
    return df.sort_values("status", key=lambda s: s.map(order)).reset_index(drop=True)


def summarise(df: pd.DataFrame) -> str:
    failing = int(((df.status == "FAIL") & df.blocking).sum())
    warn = int((df.status == "warn").sum())
    verdict = ("READY TO RENDER" if failing == 0
               else f"BLOCKED: {failing} gate(s) failing")
    return (f"{verdict}   ({int((df.status == 'pass').sum())} pass, "
            f"{warn} warn, {failing} fail)")
