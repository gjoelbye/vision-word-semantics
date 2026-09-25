"""The committed word set: block reader, pair table, form metrics and checks.

Form metrics come at three levels: gross form (rendered width and ink area),
character identity (edit distance) and character order (anagram membership).
Rendered geometry is measured from the fonts and sizes the slot table uses.
Width is reported in ems so it is comparable across size bands, and in pixels
for the fit check.
"""
from __future__ import annotations

import csv
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import WORD_LISTS
from .typography import (FONT_FAMILIES, MONOSPACE_FAMILIES, crop_safe_window,
                         font_px, ink_mask, load_font)

REFERENCE_PX = 100          # metrics are measured here and scaled linearly

# Letters whose glyphs rise above or fall below the x-height. Word shape, the
# outline a reader sees before resolving individual letters, is largely this.
ASCENDERS = set("bdfhklt")
DESCENDERS = set("gjpqy")


@lru_cache(maxsize=None)
def _char_metrics(family: str) -> tuple[dict[str, float], dict[str, float]]:
    """Per-character advance width and ink area at ``REFERENCE_PX``."""
    font = load_font(family, REFERENCE_PX)
    adv = {c: float(font.getlength(c)) for c in "abcdefghijklmnopqrstuvwxyz"}
    area = {c: float(ink_mask(family, _band_for_reference(), c).sum())
            for c in "abcdefghijklmnopqrstuvwxyz"}
    return adv, area


def _band_for_reference() -> str:
    # ink_mask takes a size band; measure at the largest and rescale.
    from .typography import SIZE_BAND_NAMES
    return SIZE_BAND_NAMES[-1]


@lru_cache(maxsize=None)
def _ink_scale(family: str) -> float:
    """Convert the reference band's ink area to ems squared."""
    return 1.0 / (font_px(_band_for_reference()) ** 2)


def rendered_width_em(word: str, family: str) -> float:
    """Advance width of the whole string, in ems. Linear in font size."""
    adv, _ = _char_metrics(family)
    return sum(adv[c] for c in word) / REFERENCE_PX


def rendered_ink_em2(word: str, family: str) -> float:
    """Total inked area, in ems squared. Independent of size band."""
    _, area = _char_metrics(family)
    return sum(area[c] for c in word) * _ink_scale(family)


def max_rendered_px(word: str, size_band: str) -> float:
    """Widest this word gets, in pixels, over every font at this size band."""
    return max(rendered_width_em(word, f) * font_px(size_band)
               for f in FONT_FAMILIES)


def fits_everywhere(word: str) -> bool:
    """Does the word stay inside the crop-safe window in every slot?"""
    from .typography import SIZE_BAND_NAMES
    _, _, side = crop_safe_window()
    return all(max_rendered_px(word, b) <= side for b in SIZE_BAND_NAMES)


def add_form_metrics(df: pd.DataFrame, word_col: str = "word") -> pd.DataFrame:
    """Attach every form metric to a word table. Returns a new frame."""
    out = df.copy()
    words = out[word_col].tolist()
    prop_fonts = [f for f in FONT_FAMILIES if f not in MONOSPACE_FAMILIES]

    # Width averaged over the slots that can express width variation at all.
    out["width_em"] = [float(np.mean([rendered_width_em(w, f) for f in prop_fonts]))
                       for w in words]
    out["ink_em2"] = [float(np.mean([rendered_ink_em2(w, f) for f in prop_fonts]))
                      for w in words]
    out["glyph_width_em"] = out.width_em / out[word_col].str.len()
    out["max_px"] = [max_rendered_px(w, "large") for w in words]
    _, _, side = crop_safe_window()
    out["fits"] = out.max_px <= side

    out["n_ascenders"] = [sum(c in ASCENDERS for c in w) for w in words]
    out["n_descenders"] = [sum(c in DESCENDERS for c in w) for w in words]
    out["n_distinct_chars"] = [len(set(w)) for w in words]
    out["first_char"] = [w[0] for w in words]
    out["last_char"] = [w[-1] for w in words]
    out["anagram_key"] = [anagram_key(w) for w in words]
    return out


def anagram_key(word: str) -> str:
    """Words sharing this key use exactly the same characters."""
    return "".join(sorted(word))


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance. Small strings, so the simple table is fine."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def read_block(name: str) -> list[dict]:
    """Rows of one block file in ``lexicon/``, as dicts keyed by the header.

    A row with more fields than the header or an empty cell stops the run with
    the file and line it is on.
    """
    out = []
    with open(WORD_LISTS / name) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for n, r in enumerate(reader, start=2):
            if None in r:                      # a stray extra tab
                raise SystemExit(f"{name} line {n}: more fields than the header")
            row = {k: (v or "").strip() for k, v in r.items()}
            blank = [k for k, v in row.items() if not v]
            if blank:
                raise SystemExit(f"{name} line {n}: empty cell in {blank}")
            out.append(row)
    return out


#: The pair table: ``(block, column a, column b, relation, kind)``. Meaning
#: pairs carry the claim; spelling pairs are the control.
PAIRS = [
    ("block1.txt", "anchor", "synonym",       "synonym",         "meaning"),
    ("block1.txt", "anchor", "spelling_twin", "spelling_twin",   "spelling"),
    ("block2.txt", "anchor", "synonym",       "antonym_synonym", "meaning"),
    ("block2.txt", "anchor", "antonym",       "antonym",         "meaning"),
    ("block3.txt", "negation", "synonym",     "negation",        "meaning"),
    ("block3.txt", "negation", "base",        "negation_base",   "spelling"),
    ("block4.txt", "singular", "plural",      "singular_plural", "spelling"),
    ("block5.txt", "country", "capital",      "country_capital", "meaning"),
]


# --- the crossed grids ---------------------------------------------------------

def crossing_report(words: pd.DataFrame) -> dict:
    table = pd.crosstab(words.category, words.ortho).to_numpy(float)
    n = table.sum()
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / n
    chi2 = float(((table - expected) ** 2
                  / np.where(expected == 0, 1, expected)).sum())
    v = (float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
         if min(table.shape) > 1 else 0.0)
    return dict(
        cells_equal=bool(len(np.unique(table)) == 1),
        smallest_cell=int(table.min()),
        largest_cell=int(table.max()),
        n_categories=int(table.shape[0]),
        n_clusters=int(table.shape[1]),
        category_counts_equal=bool(len(set(table.sum(1))) == 1),
        cluster_counts_equal=bool(len(set(table.sum(0))) == 1),
        cramers_v=round(v, 6),
    )


def crossing_passes(report: dict, max_cramers_v: float = 0.01) -> bool:
    return bool(report["cells_equal"]
                and report["category_counts_equal"]
                and report["cluster_counts_equal"]
                and report["cramers_v"] <= max_cramers_v)


# --- vocabulary checks ---------------------------------------------------------

BLOCKED_USAGE_DOMAINS = frozenset({
    "vulgarism.n.01", "obscenity.n.02", "ethnic_slur.n.01", "slur.n.01",
    "profanity.n.01", "swearword.n.01", "expletive.n.01",
})

BLOCKED_WORDS = frozenset({
    "coon", "fuck", "fucked", "fucking", "twat", "arse", "arsed", "snatch",
    "cunt", "wank", "wanker", "prick", "dick", "cock", "tits", "titties",
    "nigger", "spic", "chink", "kike", "wop", "gook", "paki", "dago",
    "faggot", "fag", "dyke", "tranny", "retard", "retarded", "spastic",
    "whore", "slut", "bitch", "bastard", "bollocks", "shit", "shitty",
    "piss", "pissed", "crap", "penis", "vagina", "boner", "horny",
})

DERIVATIONAL_ENDINGS = ("ress", "esse", "ess", "ine", "ina", "trix",
                         "ette", "enne")


def is_blocked(word: str, synsets) -> bool:
    if word in BLOCKED_WORDS:
        return True
    return any(domain.name() in BLOCKED_USAGE_DOMAINS
               for synset in synsets for domain in synset.usage_domains())


def is_named_entity(word: str, synsets=None) -> bool:
    from nltk.corpus import wordnet as wn

    synsets = wn.synsets(word) if synsets is None else synsets
    if not synsets:
        return False
    return all(all(lemma.name()[0].isupper() for lemma in synset.lemmas()
                   if lemma.name().lower() == word) for synset in synsets)


def is_orthographic_relative(a: str, b: str) -> bool:
    """Whether two strings are inflectional or derivational forms."""
    from nltk.corpus import wordnet as wn

    if a == b:
        return True
    lo, hi = sorted((a, b), key=len)
    if lo in hi:
        return True
    prefix = next((i for i, (x, y) in enumerate(zip(lo, hi)) if x != y),
                  min(len(lo), len(hi)))
    if prefix >= 4 and edit_distance(a, b) / max(len(a), len(b)) <= 0.5:
        return True
    for suffix in DERIVATIONAL_ENDINGS:
        if hi.endswith(suffix) and len(hi) - len(suffix) >= 3:
            stem = hi[:-len(suffix)]
            if edit_distance(stem, lo) / max(len(stem), len(lo)) <= 0.5:
                return True

    def derivatives(word: str) -> set[str]:
        return {relative.name().lower()
                for synset in wn.synsets(word)
                for lemma in synset.lemmas()
                if lemma.name().lower() == word
                for relative in lemma.derivationally_related_forms()}

    return b in derivatives(a) or a in derivatives(b)


def orthographic_relatives(pairs: pd.DataFrame) -> pd.DataFrame:
    meaning = pairs[pairs.kind == "meaning"]
    bad = [row for row in meaning.itertuples()
           if is_orthographic_relative(row.a, row.b)]
    return pd.DataFrame(
        [dict(relation=row.relation, a=row.a, b=row.b) for row in bad],
        columns=["relation", "a", "b"])
