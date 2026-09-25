"""Readers of the word lists and guarded readers of the result tables.

A result table records the fingerprints of the manifests it was computed from.
:func:`read` and :func:`composition_summary` refuse a table whose fingerprints
are not those of the current manifests, so no notebook or table shows a number
from stale inputs.
"""
from __future__ import annotations

import pandas as pd

from .. import wordtext
from ..composition_stimuli import MANIFEST_PATH, load_curated_items, manifest_fingerprint
from ..config import COMPOSITION, LEXICON, RESULTS
from ..models import DEFAULT_SWEEP, PRETTY
from ..render import fingerprint
from ..tau import CONCEPT_IMAGES, CONCEPT_WORDS, MAIN, manifest, word_order

#: The fingerprint columns a result table can carry, and the image set each names.
_MANIFESTS = {"manifest_sha": MAIN, "word_manifest_sha": CONCEPT_WORDS,
              "image_manifest_sha": CONCEPT_IMAGES}


def words() -> pd.DataFrame:
    """``lexicon/words.parquet``: one row per word of the vocabulary, with its role and form."""
    return pd.read_parquet(LEXICON / "words.parquet")


def pairs() -> pd.DataFrame:
    """``lexicon/pairs.parquet``: the word pairs of each relation."""
    return pd.read_parquet(LEXICON / "pairs.parquet")


def audit() -> pd.DataFrame:
    """``lexicon/audit.parquet``: the gates of the vocabulary."""
    return pd.read_parquet(LEXICON / "audit.parquet")


def concept_words() -> pd.DataFrame:
    """``lexicon/concepts.parquet``: the 200 class names, as the class-name stage writes them."""
    return pd.read_parquet(LEXICON / "concepts.parquet")


def concept_audit() -> pd.DataFrame:
    """``lexicon/concepts_audit.parquet``: the gates of the class names."""
    return pd.read_parquet(LEXICON / "concepts_audit.parquet")


def read(name: str) -> pd.DataFrame:
    """``results/<name>.parquet``, refused unless its fingerprints match the current manifests.

    ``manifest_sha`` is checked against the main manifest, ``word_manifest_sha``
    and ``image_manifest_sha`` against the class-name and class-image
    manifests, and ``word_sha`` against the current word order. A table with
    none of the manifest columns is refused.
    """
    path = RESULTS / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"results/{name}.parquet is missing")
    frame = pd.read_parquet(path)
    present = [column for column in _MANIFESTS if column in frame.columns]
    if not present:
        raise RuntimeError(f"results/{name}.parquet records no manifest fingerprint")
    for column in present:
        if set(frame[column]) != {fingerprint(manifest(_MANIFESTS[column]))}:
            raise RuntimeError(f"results/{name}.parquet was computed from a "
                               f"different {_MANIFESTS[column].name} manifest")
    if "word_sha" in frame.columns and (
            set(frame.word_sha) != {wordtext.fingerprint(word_order())}):
        raise RuntimeError(f"results/{name}.parquet was computed for a "
                           f"different word set")
    return frame


def composition_summary() -> pd.DataFrame:
    """``composition/results/summary.parquet``, refused unless it matches the two-clue inputs.

    ``dataset_sha`` is checked against the committed two-clue item tables and
    ``manifest_sha`` against the composition manifest.
    """
    path = COMPOSITION / "results" / "summary.parquet"
    if not path.exists():
        raise FileNotFoundError("composition/results/summary.parquet is missing")
    frame = pd.read_parquet(path)
    current = {"dataset_sha": load_curated_items()[2],
               "manifest_sha": manifest_fingerprint(pd.read_parquet(MANIFEST_PATH))}
    for column, sha in current.items():
        if set(frame[column]) != {sha}:
            raise RuntimeError(f"composition/results/summary.parquet has a "
                               f"different {column} from the current inputs")
    return frame


def sweep_rows(frame: pd.DataFrame, column: str = "model") -> pd.DataFrame:
    """The rows of the 14 image models in the paper's order, with their names in ``label``."""
    order = {key: i for i, key in enumerate(DEFAULT_SWEEP)}
    rows = frame[frame[column].isin(order)]
    return (rows.sort_values(column, key=lambda keys: keys.map(order), kind="stable")
            .assign(label=lambda f: f[column].map(PRETTY)))
