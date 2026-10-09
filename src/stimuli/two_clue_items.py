"""Reading and validating the two-clue item tables.

A clue grid crosses two broad clues (``feature``) with four narrow clues
(``base``), with one target per cell. The pilot and confirmatory tables are
recorded inputs. The checks cover the shape of every clue grid and the
measurable properties of its words.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from ..config import SEED


COLUMNS = (
    "family_id",
    "feature_axis",
    "feature",
    "base_axis",
    "base",
    "target",
    "target_sense",
    "source_uri",
    "source_note",
)
SPLITS = ("pilot", "confirmatory")
EXPECTED_FAMILIES = {"pilot": 6, "confirmatory": 12}
FEATURES_PER_FAMILY = 2
BASES_PER_FAMILY = 4
ITEMS_PER_FAMILY = FEATURES_PER_FAMILY * BASES_PER_FAMILY
ZIPF_FLOOR = 3.0
MAX_FEATURE_TARGET_LENGTH_GAP = 1.0
MAX_LETTER_EXCESS = 0.05

_WORD = re.compile(r"^[a-z]{3,10}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_SOURCE = re.compile(r"^(?:wordnet:[A-Za-z0-9_.-]+|https?://\S+)$")


class TwoClueValidationError(ValueError):
    """Raised when an item table fails its checks."""


def read_table(path: str | Path, split: str) -> pd.DataFrame:
    """Read one canonical TSV without interpreting empty strings as NaN."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {SPLITS}")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    got = tuple(frame.columns)
    if got != COLUMNS:
        raise TwoClueValidationError(
            f"{path} columns are {got}; expected exactly {COLUMNS}")
    frame = frame.copy()
    for column in COLUMNS:
        frame[column] = frame[column].str.strip()
    frame.insert(0, "split", split)
    return frame


def _check(rows: list[dict], *, split: str, check: str, value,
           requirement: str, ok: bool, detail: str = "") -> None:
    rows.append({
        "split": split,
        "check": check,
        "value": value,
        "requirement": requirement,
        "status": "ok" if ok else "error",
        "detail": detail,
    })


def _family_payload(group: pd.DataFrame) -> str:
    columns = ["split", "family_id", "feature_axis", "feature",
               "base_axis", "base", "target", "target_sense",
               "source_uri", "source_note"]
    return group.sort_values(["feature", "base"])[columns].to_csv(
        index=False, lineterminator="\n")


def add_question_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``question_id``, a hash of the row and of its whole clue grid.

    A changed target, source or grid member changes every ID in that grid.
    """
    out = frame.copy()
    family_digest = {}
    for key, group in out.groupby(["split", "family_id"], sort=True):
        family_digest[key] = hashlib.sha256(
            _family_payload(group).encode()).hexdigest()
    ids = []
    for row in out.itertuples(index=False):
        payload = (f"{row.split}|{row.family_id}|{row.feature}|{row.base}|"
                   f"{row.target}|{family_digest[(row.split, row.family_id)]}")
        ids.append(hashlib.sha256(payload.encode()).hexdigest()[:20])
    out["question_id"] = ids
    return out


def _validate_split(frame: pd.DataFrame, split: str,
                    expected_families: int) -> list[dict]:
    rows: list[dict] = []
    expected_rows = expected_families * ITEMS_PER_FAMILY
    _check(rows, split=split, check="row_count", value=len(frame),
           requirement=str(expected_rows), ok=len(frame) == expected_rows)
    _check(rows, split=split, check="family_count",
           value=frame.family_id.nunique(), requirement=str(expected_families),
           ok=frame.family_id.nunique() == expected_families)

    blanks = {column: int(frame[column].eq("").sum()) for column in COLUMNS}
    n_blank = sum(blanks.values())
    _check(rows, split=split, check="nonempty_fields", value=n_blank,
           requirement="0 blank cells", ok=n_blank == 0,
           detail=str({k: v for k, v in blanks.items() if v}))

    identifiers = pd.concat(
        [frame.family_id, frame.feature_axis, frame.base_axis], ignore_index=True)
    bad_ids = sorted(set(x for x in identifiers if not _IDENTIFIER.fullmatch(x)))
    _check(rows, split=split, check="identifier_format", value=len(bad_ids),
           requirement="lowercase snake_case", ok=not bad_ids,
           detail=", ".join(bad_ids))

    words = pd.concat([frame.feature, frame.base, frame.target],
                      ignore_index=True)
    bad_words = sorted(set(x for x in words if not _WORD.fullmatch(x)))
    _check(rows, split=split, check="word_format", value=len(bad_words),
           requirement="lowercase a-z, 3-10 characters", ok=not bad_words,
           detail=", ".join(bad_words))

    bad_sources = sorted(set(x for x in frame.source_uri
                             if not _SOURCE.fullmatch(x)))
    _check(rows, split=split, check="source_format", value=len(bad_sources),
           requirement="wordnet:<synset> or an HTTP(S) URL",
           ok=not bad_sources, detail=", ".join(bad_sources))

    duplicate_targets = sorted(frame.loc[
        frame.target.duplicated(keep=False), "target"].unique())
    _check(rows, split=split, check="unique_targets",
           value=len(duplicate_targets), requirement="0 duplicates",
           ok=not duplicate_targets, detail=", ".join(duplicate_targets))

    cue_words = set(frame.feature) | set(frame.base)
    target_cues = sorted(set(frame.target) & cue_words)
    _check(rows, split=split, check="target_cue_disjointness",
           value=len(target_cues), requirement="0 shared words",
           ok=not target_cues, detail=", ".join(target_cues))

    base_families = frame.groupby("base").family_id.nunique()
    reused_bases = sorted(base_families[base_families > 1].index)
    _check(rows, split=split, check="base_family_uniqueness",
           value=len(reused_bases), requirement="each base belongs to one family",
           ok=not reused_bases, detail=", ".join(reused_bases))

    family_errors = []
    length_gaps = []
    for family_id, group in frame.groupby("family_id", sort=True):
        features = sorted(group.feature.unique())
        bases = sorted(group.base.unique())
        expected = {(f, b) for f in features for b in bases}
        observed = set(zip(group.feature, group.base))
        axes_ok = (group.feature_axis.nunique() == 1
                   and group.base_axis.nunique() == 1)
        ok = (len(group) == ITEMS_PER_FAMILY
              and len(features) == FEATURES_PER_FAMILY
              and len(bases) == BASES_PER_FAMILY
              and observed == expected
              and axes_ok)
        if not ok:
            family_errors.append(family_id)
        if len(features) == FEATURES_PER_FAMILY:
            means = group.assign(n=group.target.str.len()).groupby("feature").n.mean()
            length_gaps.append((family_id, float(abs(means.iloc[0] - means.iloc[1]))))
    _check(rows, split=split, check="complete_2x4_families",
           value=len(family_errors), requirement="0 malformed families",
           ok=not family_errors, detail=", ".join(family_errors))

    large_gaps = [(f, gap) for f, gap in length_gaps
                  if gap > MAX_FEATURE_TARGET_LENGTH_GAP]
    _check(rows, split=split, check="target_length_balance",
           value=len(large_gaps),
           requirement=("within-family feature-level mean target length gap "
                        f"<= {MAX_FEATURE_TARGET_LENGTH_GAP:.1f}"),
           ok=not large_gaps,
           detail=", ".join(f"{f}:{gap:.2f}" for f, gap in large_gaps))

    exact_or_concat = []
    for row in frame.itertuples():
        if (row.target in {row.feature, row.base}
                or row.target in {row.feature + row.base,
                                  row.base + row.feature}):
            exact_or_concat.append(
                f"{row.feature}+{row.base}->{row.target}")
    _check(rows, split=split, check="no_exact_or_concat_targets",
           value=len(exact_or_concat), requirement="0 shortcuts",
           ok=not exact_or_concat, detail=", ".join(exact_or_concat))

    if len(frame):
        rows.extend(_lexical_checks(frame, split))
    return rows


def _lexical_checks(frame: pd.DataFrame, split: str) -> list[dict]:
    from nltk.corpus import wordnet as wn
    from wordfreq import zipf_frequency

    from .vocabulary import (fits_everywhere, is_blocked, is_named_entity,
                             is_orthographic_relative, letter_overlap)

    rows: list[dict] = []
    words = sorted(set(frame.feature) | set(frame.base) | set(frame.target))
    frequencies = {word: float(zipf_frequency(word, "en")) for word in words}
    rare = [(word, value) for word, value in frequencies.items()
            if value < ZIPF_FLOOR]
    _check(rows, split=split, check="zipf_floor", value=len(rare),
           requirement=f"every word >= {ZIPF_FLOOR:.1f}", ok=not rare,
           detail=", ".join(f"{w}:{v:.2f}" for w, v in rare))

    too_wide = [word for word in words if not fits_everywhere(word)]
    _check(rows, split=split, check="render_fit", value=len(too_wide),
           requirement="every word fits every typography slot",
           ok=not too_wide, detail=", ".join(too_wide))

    blocked = [word for word in words if is_blocked(word, wn.synsets(word))]
    _check(rows, split=split, check="safe_words", value=len(blocked),
           requirement="0 blocked words", ok=not blocked,
           detail=", ".join(blocked))

    named = [word for word in words if is_named_entity(word)]
    _check(rows, split=split, check="ordinary_words", value=len(named),
           requirement="0 proper-name-only words", ok=not named,
           detail=", ".join(named))

    bad_relatives = []
    for row in frame.itertuples():
        for cue in (row.feature, row.base):
            if is_orthographic_relative(cue, row.target):
                bad_relatives.append(f"{cue}/{row.target}")
    bad_relatives = sorted(set(bad_relatives))
    _check(rows, split=split, check="cue_target_orthography",
           value=len(bad_relatives), requirement="0 orthographic relatives",
           ok=not bad_relatives, detail=", ".join(bad_relatives))

    source_errors = []
    for row in frame.itertuples():
        if not row.source_uri.startswith("wordnet:"):
            continue
        name = row.source_uri.removeprefix("wordnet:")
        try:
            synset = wn.synset(name)
        except (LookupError, ValueError):
            source_errors.append(f"{row.target}:{name}")
            continue
        lemmas = {lemma.name().lower().replace("_", "-")
                  for lemma in synset.lemmas()}
        if row.target not in lemmas:
            source_errors.append(f"{row.target}:{name}")
    _check(rows, split=split, check="wordnet_sources",
           value=len(source_errors),
           requirement="target is a lemma of every cited WordNet synset",
           ok=not source_errors, detail=", ".join(source_errors))

    vocab = words
    for cue_role in ("feature", "base"):
        pairs = list(zip(frame[cue_role], frame.target))
        overlap = letter_overlap(pairs, vocab, seed=SEED)
        _check(rows, split=split,
               check=f"{cue_role}_target_letter_excess",
               value=round(overlap["excess"], 4),
               requirement=f"<= {MAX_LETTER_EXCESS:.2f}",
               ok=overlap["excess"] <= MAX_LETTER_EXCESS,
               detail=(f"within={overlap['within']:.3f}, "
                       f"length-matched random={overlap['random']:.3f}"))
    return rows


def validate_datasets(pilot: pd.DataFrame, confirmatory: pd.DataFrame
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return canonical items with question IDs and a long validation report.

    Both tables come from ``read_table``, which fixes their columns.
    """
    frames = {"pilot": pilot.copy(), "confirmatory": confirmatory.copy()}
    rows = []
    for split in SPLITS:
        rows.extend(_validate_split(frames[split], split,
                                    EXPECTED_FAMILIES[split]))

    all_items = pd.concat([frames[s] for s in SPLITS], ignore_index=True)
    pilot_frame, confirm_frame = frames["pilot"], frames["confirmatory"]

    family_overlap = sorted(set(pilot_frame.family_id) & set(confirm_frame.family_id))
    _check(rows, split="cross_split", check="family_disjointness",
           value=len(family_overlap), requirement="0 shared family IDs",
           ok=not family_overlap, detail=", ".join(family_overlap))
    for column in ("base", "target"):
        overlap = sorted(set(pilot_frame[column]) & set(confirm_frame[column]))
        _check(rows, split="cross_split", check=f"{column}_disjointness",
               value=len(overlap), requirement=f"0 shared {column}s",
               ok=not overlap, detail=", ".join(overlap))
    equations_p = set(zip(pilot_frame.feature, pilot_frame.base,
                          pilot_frame.target))
    equations_c = set(zip(confirm_frame.feature, confirm_frame.base,
                          confirm_frame.target))
    equation_overlap = sorted(equations_p & equations_c)
    _check(rows, split="cross_split", check="equation_disjointness",
           value=len(equation_overlap), requirement="0 shared equations",
           ok=not equation_overlap,
           detail=", ".join("+".join(x) for x in equation_overlap))

    all_duplicate_targets = sorted(all_items.loc[
        all_items.target.duplicated(keep=False), "target"].unique())
    _check(rows, split="cross_split", check="global_target_uniqueness",
           value=len(all_duplicate_targets), requirement="0 duplicate targets",
           ok=not all_duplicate_targets,
           detail=", ".join(all_duplicate_targets))
    all_cues = set(all_items.feature) | set(all_items.base)
    target_cues = sorted(set(all_items.target) & all_cues)
    _check(rows, split="cross_split", check="global_target_cue_disjointness",
           value=len(target_cues), requirement="0 target/cue role overlaps",
           ok=not target_cues, detail=", ".join(target_cues))

    all_items = add_question_ids(all_items).sort_values(
        ["split", "family_id", "feature", "base"], kind="mergesort"
    ).reset_index(drop=True)
    report = pd.DataFrame(rows)
    return all_items, report


def require_valid(report: pd.DataFrame) -> None:
    errors = report[report.status == "error"]
    if len(errors):
        names = ", ".join(
            f"{row.split}/{row.check}" for row in errors.head(8).itertuples())
        raise TwoClueValidationError(
            f"{len(errors)} validation checks have errors: {names}")


def dataset_fingerprint(items: pd.DataFrame) -> str:
    """SHA-256 of the complete canonical, order-independent dataset table."""
    columns = ["split", *COLUMNS, "question_id"]
    payload = items.sort_values(
        ["split", "family_id", "feature", "base"], kind="mergesort"
    )[columns].to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode()).hexdigest()
