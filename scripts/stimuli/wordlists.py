"""Read the word set from ``lexicon/vocabulary.tsv`` and write the word, pair and audit tables.

Writes ``lexicon/words.parquet`` (one row per word with its role, frequency and
form metrics), ``lexicon/pairs.parquet`` (the pairs of ``src.lexicon.PAIRS``)
and ``lexicon/audit.parquet`` (the checks on the word set). Stops if a word
does not fit the crop-safe window or a blocking gate fails. With ``--readme``
it only writes the committed ``lexicon/README.md`` from the committed lists.
"""
from __future__ import annotations

import argparse

import pandas as pd
from wordfreq import zipf_frequency

from src import audit
from src.concepts import concept_words
from src.config import LEXICON, VOCABULARY_TSV, ensure_dirs
from src.lexicon import (LISTS, PAIRS, add_form_metrics, crossing_passes,
                         crossing_report, pair_table, read_vocabulary,
                         readme_markdown)

README = VOCABULARY_TSV.parent / "README.md"

# The relation and relation kind each word carries in ``words.parquet``. A
# word takes its list's relation, except the two roles that complete a
# spelling pair.
LIST_RELATION = {"synonym_one_edit": "synonym", "synonym_antonym": "antonym",
                 "synonym_negation": "negation", "singular_plural": "singular_plural",
                 "country_capital": "country_capital", "common_nouns": "category",
                 "brands": "brand", "pseudowords": "pseudoword"}
ROLE_RELATION = {"spelling_twin": "spelling_twin", "base": "negation_base"}
RELATION_KIND = {relation: kind for _, _, _, relation, kind in PAIRS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", action="store_true",
                        help="write lexicon/README.md and stop")
    args = parser.parse_args()
    vocabulary = read_vocabulary()
    readme = readme_markdown(vocabulary, concept_words())
    if args.readme:
        README.write_text(readme)
        print("wrote lexicon/README.md")
        return
    if not README.exists() or README.read_text() != readme:
        print("lexicon/README.md is out of date; run with --readme to refresh it")
    ensure_dirs()
    for slug, *_ in LISTS:
        print(f"  {slug:<17} {vocabulary.item[vocabulary.list == slug].nunique():4d} items")

    # --- one row per rendered word ------------------------------------------
    # ``role`` is ``pair`` for the five relation lists; ``item_role`` is the
    # part the word plays inside its own item.
    relation = vocabulary.item_role.map(ROLE_RELATION).fillna(
        vocabulary.list.map(LIST_RELATION))
    words = pd.DataFrame(dict(
        word=vocabulary.word,
        role=vocabulary.item_role.where(
            ~vocabulary.list.isin([slug for slug, *_ in PAIRS]), "pair"),
        relation=relation,
        relation_kind=relation.map(RELATION_KIND).fillna(relation),
        item_role=vocabulary.item_role,
        category=vocabulary.category,
        ortho=vocabulary.ortho,
        picturability=vocabulary.picturability))
    # Zipf floor: 3.0 for pair words, 2.2 for nouns and brands, none for pseudowords.
    words["zipf_floor"] = words.role.map(
        {"category": 2.2, "brand": 2.2}).fillna(3.0).where(
            words.role != "pseudoword")
    words["n_chars"] = words.word.str.len()
    # Pseudowords get Zipf 0 from wordfreq; recorded, never gated on.
    words["zipf"] = [zipf_frequency(w, "en") for w in words.word]
    words = add_form_metrics(words).sort_values("word").reset_index(drop=True)
    if not words.fits.all():
        raise SystemExit(f"does not fit the crop-safe window: "
                         f"{words.loc[~words.fits, 'word'].tolist()}")

    pairs = pair_table(vocabulary)

    # --- deterministic gates ------------------------------------------------
    cats = words[words.role == "category"]
    brands = words[words.role == "brand"]
    rows = (audit.audit_words(words)
            + audit.audit_offensive(words)
            + audit.audit_pairs_orthography(pairs, words.word.tolist())
            + audit.audit_pairs_proper_nouns(pairs))
    v = crossing_report(cats)
    rows.append(dict(gate="categories/crossing",
                     detail="Cramer's V between category and spelling cluster",
                     value=round(v["cramers_v"], 4), threshold=audit.GATE_MAX_CRAMERS_V,
                     status="pass" if crossing_passes(
                         v, audit.GATE_MAX_CRAMERS_V) else "FAIL",
                     blocking=True))
    v = crossing_report(brands)
    rows.append(dict(gate="brands/crossing",
                     detail="Cramer's V between brand industry and spelling cluster",
                     value=round(v["cramers_v"], 4), threshold=audit.GATE_MAX_CRAMERS_V,
                     status="pass" if crossing_passes(
                         v, audit.GATE_MAX_CRAMERS_V) else "FAIL",
                     blocking=True))
    report = audit.report(rows)

    blocking = [r for r in rows if r["status"] == "FAIL" and r.get("blocking", True)]

    words.to_parquet(LEXICON / "words.parquet")
    pairs.to_parquet(LEXICON / "pairs.parquet")
    report.to_parquet(LEXICON / "audit.parquet")

    print(f"\n{len(words):,} words: "
          f"{(words.role == 'pair').sum()} in pairs, "
          f"{(words.role == 'category').sum()} category, "
          f"{(words.role == 'brand').sum()} brands, "
          f"{(words.role == 'pseudoword').sum()} pseudowords")
    print(f"{len(pairs):,} pairs across {pairs.relation.nunique()} relations")
    print(pairs.groupby(["kind", "relation"]).size().to_string())
    print(f"\npartner-retrieval chance is 1/{len(words) - 1} = "
          f"{1 / (len(words) - 1):.5f}")
    print()
    print(report[["gate", "value", "threshold", "status"]].to_string(index=False))
    print()
    print(audit.summarise(report))
    if blocking:
        raise SystemExit(f"{len(blocking)} blocking gate(s) failing: "
                         f"{[r['gate'] for r in blocking]}")


if __name__ == "__main__":
    main()
