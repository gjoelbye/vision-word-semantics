"""Read the word set from ``lexicon/block*.txt`` and write the word, pair and audit tables.

Writes ``lexicon/words.parquet`` (one row per word with its role, frequency and
form metrics), ``lexicon/pairs.parquet`` (the pairs of ``src.lexicon.PAIRS``)
and ``lexicon/audit.parquet`` (the checks on the word set). Stops if a word
appears twice, a word does not fit the crop-safe window, or a blocking gate
fails.
"""
from __future__ import annotations

import argparse

import pandas as pd
from wordfreq import zipf_frequency

from src import audit
from src.config import LEXICON, ensure_dirs
from src.lexicon import (PAIRS, add_form_metrics, crossing_passes,
                         crossing_report, read_block)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    ensure_dirs()
    blocks = {n: read_block(n) for n in
              [f"block{i}.txt" for i in range(1, 9)]}
    for n, rows in blocks.items():
        print(f"  {n}  {len(rows):4d} items")

    # --- one row per rendered word ------------------------------------------
    rows = []

    def add(word, role, relation, kind, item, category=None, ortho=None, cls=None):
        # ``item_role`` is the part the word plays inside its own item.
        rows.append(dict(word=word, role=role, relation=relation,
                         relation_kind=kind, item_role=item, category=category,
                         ortho=ortho, picturability=cls))
    for r in blocks["block1.txt"]:
        add(r["anchor"], "pair", "synonym", "meaning", "anchor", cls=r["class"])
        add(r["synonym"], "pair", "synonym", "meaning", "synonym", cls=r["class"])
        add(r["spelling_twin"], "pair", "spelling_twin", "spelling",
            "spelling_twin", cls=r["class"])
    for r in blocks["block2.txt"]:
        add(r["anchor"], "pair", "antonym", "meaning", "anchor")
        add(r["synonym"], "pair", "antonym", "meaning", "synonym")
        add(r["antonym"], "pair", "antonym", "meaning", "antonym")
    for r in blocks["block3.txt"]:
        add(r["negation"], "pair", "negation", "meaning", "negation")
        add(r["synonym"], "pair", "negation", "meaning", "synonym")
        add(r["base"], "pair", "negation_base", "spelling", "base")
    for r in blocks["block4.txt"]:
        add(r["singular"], "pair", "singular_plural", "spelling", "singular")
        add(r["plural"], "pair", "singular_plural", "spelling", "plural")
    for r in blocks["block5.txt"]:
        add(r["country"], "pair", "country_capital", "meaning", "country")
        add(r["capital"], "pair", "country_capital", "meaning", "capital")
    for r in blocks["block6.txt"]:
        add(r["word"], "category", "category", "category", "category",
            category=r["category"], ortho=int(r["ortho"]))
    for r in blocks["block7.txt"]:
        add(r["pseudoword"], "pseudoword", "pseudoword", "pseudoword", "pseudoword")
    # Brands are a separate role from the nouns: their categories are industries.
    for r in blocks["block8.txt"]:
        add(r["word"], "brand", "brand", "brand", "brand",
            category=r["category"], ortho=int(r["ortho"]))

    words = pd.DataFrame(rows)
    dupes = words.word[words.word.duplicated()].tolist()
    if dupes:
        raise SystemExit(f"a word appears twice across the blocks: {dupes[:10]}")
    # Nullable integer: ortho, the spelling cluster, exists only for nouns and brands.
    words["ortho"] = words.ortho.astype("Int64")
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

    # --- the pair table -----------------------------------------------------
    pairs = []
    for f, ca, cb, relation, kind in PAIRS:
        for r in blocks[f]:
            if r.get(ca) and r.get(cb):
                pairs.append(dict(item=f"pair/{relation}/{r[ca]}",
                                  relation=relation, kind=kind,
                                  a=r[ca], b=r[cb]))
    pairs = pd.DataFrame(pairs)

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
