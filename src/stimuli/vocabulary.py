"""The committed word set: the vocabulary reader, the lists and pairs, form measures and checks.

A word's rendered width is measured from the fonts and sizes the slot table
uses, in ems so it is comparable across size bands and in pixels for the fit
check. Two words are compared by edit distance and by letter overlap.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ..config import VOCABULARY_TSV
from .typography import (FONT_FAMILIES, SIZE_BAND_NAMES, crop_safe_window,
                         font_px, load_font)

REFERENCE_PX = 100          # font size at which advance widths are measured


@lru_cache(maxsize=None)
def _advance_widths(family: str) -> dict[str, float]:
    """Per-character advance width at ``REFERENCE_PX``."""
    font = load_font(family, REFERENCE_PX)
    return {c: float(font.getlength(c)) for c in "abcdefghijklmnopqrstuvwxyz"}


def rendered_width_em(word: str, family: str) -> float:
    """Advance width of the whole string, in ems. Linear in font size."""
    adv = _advance_widths(family)
    return sum(adv[c] for c in word) / REFERENCE_PX


def max_rendered_px(word: str, size_band: str) -> float:
    """Largest advance width of ``word`` in pixels, over every font at ``size_band``."""
    return max(rendered_width_em(word, f) * font_px(size_band)
               for f in FONT_FAMILIES)


def fits_everywhere(word: str) -> bool:
    """Whether the word is no wider than the crop-safe window in every font at every size band."""
    _, _, side = crop_safe_window()
    return all(max_rendered_px(word, b) <= side for b in SIZE_BAND_NAMES)


def add_fit(words: pd.DataFrame) -> pd.DataFrame:
    """A copy of a word table with ``max_px``, the widest rendering at the
    largest size band, and ``fits``, whether that is inside the crop-safe window."""
    out = words.copy()
    out["max_px"] = [max_rendered_px(w, "large") for w in out.word]
    _, _, side = crop_safe_window()
    out["fits"] = out.max_px <= side
    return out


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance, from the full dynamic-programming table."""
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


#: The eight lists of ``vocabulary.tsv`` in the paper's order, with brands
#: before pseudowords: ``(slug, title, roles, example, purpose)``. ``roles``
#: are the item roles of one item, in table order. ``example`` is the first
#: word of the item Appendix A quotes, and ``purpose`` is what the list tests.
LISTS = (
    ("synonym_one_edit", "Synonym and one-edit comparison",
     ("anchor", "synonym", "spelling_twin"), "torch", "meaning and spelling"),
    ("synonym_antonym", "Synonym and antonym",
     ("anchor", "synonym", "antonym"), "brave", "synonymy and opposition"),
    ("synonym_negation", "Synonym and apparent negation/base",
     ("negation", "synonym", "base"), "disappear", "meaning and similar form"),
    ("singular_plural", "Singular and plural",
     ("singular", "plural"), "road", "form with related meaning"),
    ("country_capital", "Country and capital",
     ("country", "capital"), "germany", "factual association"),
    ("common_nouns", "Common nouns", ("category",), None, "Category membership"),
    ("brands", "Brands", ("brand",), None, "Industry membership"),
    ("pseudowords", "Pseudowords", ("pseudoword",), None,
     "Retrieval candidates and centering"),
)
ROLES = {slug: roles for slug, _, roles, _, _ in LISTS}

#: The pair table: ``(list, role a, role b, relation, kind)``. Meaning pairs
#: carry the claim; spelling pairs are the control.
PAIRS = (
    ("synonym_one_edit", "anchor", "synonym",       "synonym",         "meaning"),
    ("synonym_one_edit", "anchor", "spelling_twin", "spelling_twin",   "spelling"),
    ("synonym_antonym",  "anchor", "synonym",       "antonym_synonym", "meaning"),
    ("synonym_antonym",  "anchor", "antonym",       "antonym",         "meaning"),
    ("synonym_negation", "negation", "synonym",     "negation",        "meaning"),
    ("synonym_negation", "negation", "base",        "negation_base",   "spelling"),
    ("singular_plural",  "singular", "plural",      "singular_plural", "spelling"),
    ("country_capital",  "country", "capital",      "country_capital", "meaning"),
)

COLUMNS = ("list", "item", "item_role", "word", "category", "spelling_cluster",
           "concreteness")


def read_vocabulary(path=VOCABULARY_TSV) -> pd.DataFrame:
    """``lexicon/vocabulary.tsv``, one row per word in file order.

    Blank cells become missing values. ``spelling_cluster`` becomes the integer
    column ``ortho`` and ``concreteness`` becomes ``picturability``, the names
    the derived tables use. A file that breaks the schema raises ``ValueError``
    naming the rows, counted from the header as line 1.
    """
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if tuple(table.columns) != COLUMNS:
        raise ValueError(f"{path.name}: columns {list(table.columns)}, "
                         f"expected {list(COLUMNS)}")

    def lines(mask):
        return (table.index[mask] + 2).tolist()[:10]

    blank = table[["list", "item", "item_role", "word"]].eq("").any(axis=1)
    if blank.any():
        raise ValueError(f"{path.name} lines {lines(blank)}: empty list, item, "
                         f"item_role or word")
    repeated = table.word.duplicated(keep=False)
    if repeated.any():
        raise ValueError(f"{path.name} lines {lines(repeated)}: a word appears twice")
    malformed = ~table.word.str.fullmatch(r"[a-z]{3,10}")
    if malformed.any():
        raise ValueError(f"{path.name} lines {lines(malformed)}: a word is not "
                         f"three to ten lowercase letters")
    roles = table.groupby(["list", "item"], sort=False).item_role.agg(sorted)
    wrong = [key for key, got in roles.items()
             if got != sorted(ROLES.get(key[0], ()))]
    if wrong:
        raise ValueError(f"{path.name}: items without exactly their list's roles: "
                         f"{wrong[:10]}")

    table = table.mask(table.eq(""))
    table["spelling_cluster"] = table.spelling_cluster.astype("Int64")
    return table.rename(columns={"spelling_cluster": "ortho",
                                 "concreteness": "picturability"})


def item_table(vocabulary: pd.DataFrame, slug: str) -> pd.DataFrame:
    """One row per item of a list in file order, one column per role."""
    rows = vocabulary[vocabulary.list == slug]
    return (rows.pivot(index="item", columns="item_role", values="word")
            .reindex(index=rows.item.unique(), columns=list(ROLES[slug])))


def pair_table(vocabulary: pd.DataFrame) -> pd.DataFrame:
    """The pairs of ``PAIRS``, relation by relation and in item order within one.

    The order is part of the result: the letter-overlap audit draws its random
    words pair by pair.
    """
    rows = []
    for slug, a, b, relation, kind in PAIRS:
        items = item_table(vocabulary, slug)
        rows += [dict(item=f"pair/{relation}/{x}", relation=relation, kind=kind,
                      a=x, b=y) for x, y in zip(items[a], items[b])]
    return pd.DataFrame(rows)


def list_summary(vocabulary: pd.DataFrame) -> pd.DataFrame:
    """Items, distinct words and pairs per list, with an example: Table 3.

    Pairs are counted before the query direction is reversed. A dash marks a
    cell the paper leaves empty.
    """
    pairs = pair_table(vocabulary)
    pair_list = pairs.relation.map({relation: slug
                                    for slug, _, _, relation, _ in PAIRS})
    rows = []
    for slug, title, _, anchor, purpose in LISTS:
        words = vocabulary[vocabulary.list == slug]
        listed = pairs[pair_list == slug]
        example = listed[listed.a == anchor]
        shown = ", ".join(f"{a}-{b}" for a, b in zip(example.a, example.b))
        rows.append({"List": title, "Items": words.item.nunique(),
                     "Words": words.word.nunique(), "Pairs": len(listed),
                     "Example and purpose": f"{shown}: {purpose}" if anchor else purpose})
    counts = pd.DataFrame(rows)
    total = {"List": "Total", "Items": 0, "Words": counts.Words.sum(),
             "Pairs": counts.Pairs.sum(), "Example and purpose": ""}
    summary = pd.concat([counts, pd.DataFrame([total])]).set_index("List")
    for column in ("Items", "Words", "Pairs"):
        summary[column] = summary[column].map(lambda n: f"{n:,}" if n else "-")
    return summary


def _markdown_table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def readme_markdown(vocabulary: pd.DataFrame, class_names: pd.DataFrame) -> str:
    """The text of ``lexicon/README.md``, a readable view of the committed lists.

    ``class_names`` is the kept rows of ``class_names.tsv``.
    """
    heading = {"spelling_twin": "One-edit word"}
    titles = {slug: title for slug, title, *_ in LISTS}
    summary = list_summary(vocabulary)
    pseudo = vocabulary.list == "pseudowords"
    out = [
        "_Generated from `vocabulary.tsv` and `class_names.tsv` by "
        "`python scripts/stimuli/vocabulary.py --readme`. Edit those files, "
        "not this page._",
        "",
        "# Word lists",
        "",
        f"`vocabulary.tsv` is the lexical vocabulary: {(~pseudo).sum():,} words "
        f"and names and {pseudo.sum()} pseudowords in eight lists, one row per "
        f"word. `class_names.tsv` is the curation record of the ImageNet-1k "
        f"class names, one row per candidate with its decision; "
        f"{len(class_names)} are kept. `two_clue_pilot.tsv` and "
        f"`two_clue_confirmatory.tsv` hold the two-clue items.",
        "",
        "No fingerprint covers the roles, pairs or categories in "
        "`vocabulary.tsv` and `class_names.tsv`, so a cache built before an "
        "edit to either file still loads after it. After an edit, rerun every "
        "stage from `scripts/run_stimuli.sh` on.",
        "",
        "## Reading the vocabulary",
        "",
        "```python",
        "from src.stimuli.vocabulary import read_vocabulary",
        "",
        "vocabulary = read_vocabulary()",
        "# or, without the checks and renames:",
        "import pandas as pd",
        "vocabulary = pd.read_csv(\"lexicon/vocabulary.tsv\", sep=\"\\t\", dtype=str,",
        "                         keep_default_na=False)",
        "```",
        "",
        _markdown_table(("Column", "Content"), [
            ("`list`", "The list the word belongs to, named by the slugs below"),
            ("`item`", "The first word of the word's item. An item of a relation "
                       "list is one row of its table below; in the other lists "
                       "an item is one word."),
            ("`item_role`", "The word's part in its item. `spelling_twin` is the "
                            "paper's one-edit word. A noun's role is `category`, "
                            "a brand's `brand`, a pseudoword's `pseudoword`."),
            ("`word`", "The rendered string: lowercase, three to ten letters, "
                       "and unique in the file"),
            ("`category`", "The category of a noun or the industry of a brand"),
            ("`spelling_cluster`", "The spelling cluster of a noun or brand, 0 "
                                   "to 3. The derived tables call it `ortho`."),
            ("`concreteness`", "Whether the item of list 1 is concrete or "
                               "abstract. The derived tables call it "
                               "`picturability`."),
        ]),
        "",
        "The derived tables and the results name each pair relation as follows.",
        "",
        _markdown_table(("Relation", "List", "Pair"), [
            (f"`{relation}`", titles[slug],
             f"{heading.get(a, a).lower()} and {heading.get(b, b).lower()}")
            for slug, a, b, relation, _ in PAIRS]),
        "",
        "## The eight lists",
        "",
        "Words counts distinct words, and Pairs counts pairs before the query "
        "direction is reversed.",
        "",
        _markdown_table(("List", *summary.columns),
                        [(name, *row) for name, *row in summary.itertuples()]),
    ]
    for number, (slug, title, roles, _, _) in enumerate(LISTS, start=1):
        rows = vocabulary[vocabulary.list == slug]
        out += ["", f"### ({number}) {title}", "",
                f"`{slug}`: {rows.item.nunique()} items, {len(rows)} words.", ""]
        if slug == "pseudowords":
            out.append(", ".join(rows.word))
        elif len(roles) == 1:
            label = "Category" if slug == "common_nouns" else "Industry"
            clusters = sorted(rows.ortho.unique())
            grid = rows.groupby(["category", "ortho"], sort=False).word.agg(", ".join)
            out.append(_markdown_table(
                (label, *(f"Cluster {c}" for c in clusters)),
                [(category, *(grid[category, c] for c in clusters))
                 for category in rows.category.unique()]))
        else:
            items = item_table(vocabulary, slug)
            header = [heading.get(role, role.capitalize()) for role in roles]
            if rows.picturability.notna().any():
                items["concreteness"] = rows.groupby("item").picturability.first()
                header.append("Concreteness")
            out.append(_markdown_table(header, items.itertuples(index=False)))
    by_category = class_names.groupby("category").word.agg(", ".join)
    out += ["", "## Class names", "",
            f"The {len(class_names)} kept class names of `class_names.tsv`, by "
            f"category.", "",
            _markdown_table(("Category", "Class names"), by_category.items()),
            "", "## Two-clue items", "",
            "`two_clue_pilot.tsv` holds the pilot grids used during development "
            "and `two_clue_confirmatory.tsv` the confirmatory grids. Their cells "
            "and column names are hashed into the two-clue dataset "
            "fingerprint, which seeds the layout and the scrambles, so an edit "
            "to either file, a column name included, means rendering the two-clue "
            "stimuli again. In them the `feature` column holds the broad clue and "
            "`base` the narrow clue."]
    return "\n".join(out) + "\n"


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
    """Whether two strings are inflectional or derivational forms of one word.

    Two strings are related when one contains the other, when they share a
    prefix of at least four letters and their edit distance is at most half the
    longer length, when the longer less a derivational ending is that close to
    the shorter, or when WordNet lists them as derivationally related forms.
    """
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


# --- letter overlap ----------------------------------------------------------

def letter_overlap(pairs: list[tuple[str, str]], vocab: list[str],
                   n_random: int = 2000, seed: int = 0) -> dict:
    """Mean letter-set Jaccard within the pairs, against length-matched random word pairs.

    Each random pair draws one word of the same length as each word of a real
    pair, because letter-set overlap rises with word length.
    """
    def jaccard(a: str, b: str) -> float:
        A, B = set(a), set(b)
        return len(A & B) / len(A | B)

    by_length: dict[int, list[str]] = {}
    for w in vocab:
        by_length.setdefault(len(w), []).append(w)

    within = [jaccard(a, b) for a, b in pairs]
    rng = np.random.default_rng(seed)
    between = []
    draws = max(1, n_random // max(1, len(pairs)))
    for a, b in pairs:
        for _ in range(draws):
            x = rng.choice(by_length.get(len(a)) or vocab)
            y = rng.choice(by_length.get(len(b)) or vocab)
            if x != y:
                between.append(jaccard(x, y))
    return dict(n_pairs=len(within),
                within=float(np.mean(within)) if within else float("nan"),
                random=float(np.mean(between)) if between else float("nan"),
                excess=float(np.mean(within) - np.mean(between))
                if within and between else float("nan"))


# --- the audit ---------------------------------------------------------------
# Deterministic checks on the word set, run before it is rendered. A failed
# blocking check stops rendering.

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
    """One row counting the words wider than the crop-safe window at the largest size band."""
    _, _, side = crop_safe_window()
    over = words.loc[~words.fits, "word"].tolist() if "fits" in words else []
    return [gate("render/fits",
                 f"widest word {words.max_px.max():.0f}px against a {side}px "
                 f"window" + (f", e.g. {', '.join(over[:5])}" if over else ""),
                 int(len(over)), 0, len(over) == 0)]


def audit_offensive(words: pd.DataFrame) -> list[dict]:
    """One row counting the words on the blocklist or in an offensive WordNet usage domain."""
    from nltk.corpus import wordnet as wn

    bad = [w for w in words.word if is_blocked(w, wn.synsets(w))]
    return [gate("words/offensive",
                 "blocked strings in the rendered set"
                 + (f": {', '.join(bad[:6])}" if bad else ""),
                 len(bad), 0, not bad)]


def audit_pairs_orthography(pairs: pd.DataFrame,
                            vocab: list[str]) -> list[dict]:
    """Rows checking that no meaning pair joins two forms of one word, and that no
    meaning relation exceeds length-matched random pairs in letter overlap by
    more than ``GATE_MAX_LETTER_EXCESS``."""
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


def report(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    # Thresholds mix numbers and strings, and a parquet column holds one type.
    # Each is written as given, so an integer threshold stays an integer.
    df["threshold"] = [str(row["threshold"]) for row in rows]
    order = {"FAIL": 0, "warn": 1, "pass": 2}
    return df.sort_values("status", key=lambda s: s.map(order)).reset_index(drop=True)


def summarise(df: pd.DataFrame) -> str:
    failing = int(((df.status == "FAIL") & df.blocking).sum())
    warn = int((df.status == "warn").sum())
    verdict = ("READY TO RENDER" if failing == 0
               else f"BLOCKED: {failing} gate(s) failing")
    return (f"{verdict}   ({int((df.status == 'pass').sum())} pass, "
            f"{warn} warn, {failing} fail)")
