"""Read the word set from `lexicon/vocabulary.tsv` and write the word, pair and audit tables.

Writes `data/lexicon/words.parquet` (one row per word with its role, frequency
and widest rendering), `data/lexicon/pairs.parquet` (the pairs of
`src.stimuli.vocabulary.PAIRS`) and `data/lexicon/audit.parquet` (the checks
on the word set). Stops if a word does not fit the crop-safe window or a
blocking gate fails. With `--readme` it only writes the committed
`lexicon/README.md` from the committed lists.
"""
from __future__ import annotations

import argparse

import pandas as pd
from wordfreq import zipf_frequency

from src.config import LEXICON, VOCABULARY_TSV, ensure_dirs
from src.stimuli import vocabulary
from src.stimuli.imagenet import kept_class_names

README = VOCABULARY_TSV.parent / "README.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", action="store_true",
                        help="write lexicon/README.md and stop")
    args = parser.parse_args()
    vocab = vocabulary.read_vocabulary()
    readme = vocabulary.readme_markdown(vocab, kept_class_names())
    if args.readme:
        README.write_text(readme)
        print("wrote lexicon/README.md")
        return
    if not README.exists() or README.read_text() != readme:
        print("lexicon/README.md is out of date; run with --readme to refresh it")
    ensure_dirs()
    for slug, *_ in vocabulary.LISTS:
        print(f"  {slug:<17} {vocab.item[vocab.list == slug].nunique():4d} items")

    # One row per rendered word.
    # `role` is `pair` for the five relation lists; `item_role` is the
    # part the word plays inside its own item.
    words = pd.DataFrame(dict(
        word=vocab.word,
        role=vocab.item_role.where(
            ~vocab.list.isin([slug for slug, *_ in vocabulary.PAIRS]), "pair"),
        item_role=vocab.item_role,
        category=vocab.category,
        ortho=vocab.ortho,
        picturability=vocab.picturability))
    words["n_chars"] = words.word.str.len()
    # Pseudowords get Zipf 0 from wordfreq; recorded, never gated on.
    words["zipf"] = [zipf_frequency(w, "en") for w in words.word]
    words = vocabulary.add_fit(words).sort_values("word").reset_index(drop=True)
    if not words.fits.all():
        raise SystemExit(f"does not fit the crop-safe window: "
                         f"{words.loc[~words.fits, 'word'].tolist()}")

    pairs = vocabulary.pair_table(vocab)

    # Deterministic gates.
    cats = words[words.role == "category"]
    brands = words[words.role == "brand"]
    rows = (vocabulary.audit_words(words)
            + vocabulary.audit_offensive(words)
            + vocabulary.audit_pairs_orthography(pairs, words.word.tolist()))
    for name, group, detail in [
            ("categories/crossing", cats,
             "Cramer's V between category and spelling cluster"),
            ("brands/crossing", brands,
             "Cramer's V between brand industry and spelling cluster")]:
        v, ok = vocabulary.crossing(group)
        rows.append(vocabulary.gate(name, detail, round(v, 4),
                                    vocabulary.GATE_MAX_CRAMERS_V, ok))
    report = vocabulary.report(rows)

    blocking = [r for r in rows if r["status"] == "FAIL" and r["blocking"]]

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
    print(vocabulary.summarise(report))
    if blocking:
        raise SystemExit(f"{len(blocking)} blocking gate(s) failing: "
                         f"{[r['gate'] for r in blocking]}")


if __name__ == "__main__":
    main()
