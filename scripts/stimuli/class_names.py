"""Check the class names of ``lexicon/class_names.tsv`` and write the class-name tables.

Every rule that can be recomputed from the table is checked again: form,
uniqueness, the class each word names, frequency, blocked words and fit at the
recorded text centers. The recorded decisions are counted, not checked. Writes
``lexicon/concepts.parquet`` and ``lexicon/concepts_audit.parquet``, and stops
if a blocking gate fails. Run it after ``scripts/stimuli/render.py``, which
records the text centers the names must fit.
"""
from __future__ import annotations

import argparse
import re

import pandas as pd
from wordfreq import zipf_frequency

from src.config import CLASS_NAMES_TSV, LEXICON, MAX_WORD_LENGTH, ensure_dirs
from src.stimuli import imagenet, vocabulary
from src.stimuli.render import CENTRES_PATH, slot_centres
from src.stimuli.typography import crop_safe_window, slot_table
from src.stimuli.vocabulary import is_blocked

#: Zipf floor, the same as for the pair words of the vocabulary.
ZIPF_FLOOR = 3.0

WORD_PATTERN = re.compile(rf"^[a-z]{{3,{MAX_WORD_LENGTH}}}$")


def gates(kept: pd.DataFrame) -> list[dict]:
    """One audit row for each rule that can be recomputed from the table."""
    from nltk.corpus import wordnet as wn

    rows = []
    g = vocabulary.gate

    bad_form = [w for w in kept.word if not WORD_PATTERN.fullmatch(str(w))]
    rows.append(g("concepts/form",
                  f"one lowercase word of 3 to {MAX_WORD_LENGTH} letters"
                  + (f", e.g. {', '.join(bad_form[:5])}" if bad_form else ""),
                  len(bad_form), 0, not bad_form))

    dup_word = kept.word[kept.word.duplicated()].tolist()
    rows.append(g("concepts/one_row_per_word",
                  "a word appears once"
                  + (f", e.g. {', '.join(dup_word[:5])}" if dup_word else ""),
                  len(dup_word), 0, not dup_word))

    dup_class = kept.wnid[kept.wnid.duplicated()].tolist()
    rows.append(g("concepts/one_word_per_class",
                  "a class is named by one word"
                  + (f", e.g. {', '.join(dup_class[:5])}" if dup_class else ""),
                  len(dup_class), 0, not dup_class))

    known = set(imagenet.wnids())
    unknown = [w for w in kept.wnid if w not in known]
    rows.append(g("concepts/class_exists",
                  "every class id is one of the 1,000"
                  + (f", e.g. {', '.join(unknown[:5])}" if unknown else ""),
                  len(unknown), 0, not unknown))

    mispaired = [f"{w}/{c}" for w, c in zip(kept.word, kept.wnid)
                 if c not in imagenet.classes_named(w)]
    rows.append(g("concepts/word_names_its_class",
                  "every kept word is a name of the class it is paired with"
                  + (f": {', '.join(mispaired[:5])}" if mispaired else ""),
                  len(mispaired), 0, not mispaired))

    ambiguous = [w for w in kept.word if len(imagenet.classes_named(w)) > 1]
    rows.append(g("concepts/names_one_class",
                  "no kept word names two classes once case and the export's "
                  "disambiguating digits are normalised"
                  + (f": {', '.join(ambiguous[:5])}" if ambiguous else ""),
                  len(ambiguous), 0, not ambiguous))

    zipf = {w: zipf_frequency(w, "en") for w in kept.word}
    rare = sorted((w for w, z in zipf.items() if z < ZIPF_FLOOR),
                  key=lambda w: zipf[w])
    rows.append(g("concepts/frequency",
                  f"every word at Zipf {ZIPF_FLOOR} or above, lowest "
                  f"{min(zipf.values()):.2f}"
                  + (f", e.g. {', '.join(rare[:5])}" if rare else ""),
                  len(rare), 0, not rare))

    blocked = [w for w in kept.word if is_blocked(w, wn.synsets(w))]
    rows.append(g("concepts/offensive",
                  "no blocked string or offensive usage domain"
                  + (f": {', '.join(blocked[:6])}" if blocked else ""),
                  len(blocked), 0, not blocked))

    # The names are drawn at the text centers recorded for the vocabulary.
    _, _, side = crop_safe_window()
    widest = int(kept.max_px.max()) if "max_px" in kept else 0
    fits = True
    detail = f"widest word {widest}px against a {side}px window"
    if CENTRES_PATH.exists():
        try:
            slot_centres(tuple(sorted(kept.word)))
            detail += ", and every word fits at every recorded centre"
        except ValueError as exc:
            fits = False
            detail = str(exc).replace("\n", " ")[:200]
    else:
        detail += f", but {CENTRES_PATH.name} does not exist yet"
    rows.append(g("concepts/fits", detail, widest, side, fits))

    return rows


def composition(kept: pd.DataFrame, table: pd.DataFrame) -> list[dict]:
    """Non-blocking audit rows for the counts that describe the set."""
    rows = []
    g = vocabulary.gate
    n = len(kept)

    def count(name, detail, value):
        rows.append(g(f"count/{name}", detail, int(value), "-", True,
                      blocking=False))

    count("candidates", "rows in the committed file", len(table))
    count("kept", "words in the set", n)
    if "sense_rank" in kept:
        first = int((kept.sense_rank == 0).sum())
        count("kept_first_sense",
              "kept where the class is the word's first WordNet sense", first)
        count("kept_adjudicated",
              "kept where it is not, so rule 4 decided it", n - first)
    dropped = table[table.decision != "keep"]
    for reason, group in dropped.groupby(dropped.reason.str.slice(0, 46)):
        count(f"dropped/{reason[:38]}", f"dropped: {reason}", len(group))
    if "in_vocab" in kept:
        count("in_main_vocabulary",
              "kept words also in the 1,561-word vocabulary",
              int(kept.in_vocab.astype(bool).sum()))
    if "category" in kept:
        for cat, group in kept.groupby("category"):
            count(f"category/{cat}", f"kept words whose class is a {cat}",
                  len(group))
    return rows


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    ensure_dirs()

    table = pd.read_csv(CLASS_NAMES_TSV, sep="\t")
    kept = imagenet.kept_class_names()
    print(f"{CLASS_NAMES_TSV.name}: {len(table)} candidates, "
          f"{len(kept)} kept")

    rows = gates(kept) + composition(kept, table)
    report = vocabulary.report(rows)
    verdict = vocabulary.summarise(report)
    shown = report[report.gate.str.startswith("concepts/")]
    print(shown[["gate", "detail", "value", "threshold", "status"]]
          .to_string(index=False))
    print(f"\n{verdict}")

    counts = report[report.gate.str.startswith("count/")]
    print("\ncomposition")
    print(counts[["gate", "value"]].to_string(index=False))

    if ((report.status == "FAIL") & report.blocking).any():
        raise SystemExit("blocking gates are failing; fix lexicon/class_names.tsv")

    words = kept.copy()
    words["zipf"] = [zipf_frequency(w, "en") for w in words.word]
    words["class_name"] = words.wnid.map(imagenet.class_names())
    words["n_slots"] = len(slot_table())
    words.to_parquet(LEXICON / "concepts.parquet")
    report.to_parquet(LEXICON / "concepts_audit.parquet")
    print(f"\nwrote lexicon/concepts.parquet ({len(words)} words)")
    print("wrote lexicon/concepts_audit.parquet")


if __name__ == "__main__":
    main()
