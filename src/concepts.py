"""The class names and the ImageNet-1k validation photographs of their classes.

Each class name names exactly one ImageNet-1k class. The names are drawn on the
96 slots at the recorded centers, so a name that is also in the vocabulary is
the same image file. The photographs are every validation image of every class,
except the 96 used as backgrounds.
"""
from __future__ import annotations

import functools
import hashlib
import io
from pathlib import Path

import pandas as pd
from PIL import Image

from .config import BACKGROUNDS, CLASS_NAMES_TSV, CONCEPTS, DATA_ROOT

#: The committed class-name list, one row per candidate with its decision.
CONCEPT_LIST = CLASS_NAMES_TSV

IMAGES = CONCEPTS / "images"
RENDER_MANIFEST = CONCEPTS / "manifest.parquet"
IMAGE_MANIFEST = CONCEPTS / "images.parquet"

#: The ImageNet-1k parquet export: ``data/validation-*.parquet`` and ``classes.py``.
IMAGENET = DATA_ROOT / "imagenet-1k"

#: Fewest photographs a class needs in a subset, all or text-free, to be
#: retained in it.
MIN_IMAGES = 32


def square_resize(img: Image.Image, size: int) -> Image.Image:
    """Center-crop to a square and resize with Lanczos, as for the backgrounds."""
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2,
                    (w - side) // 2 + side, (h - side) // 2 + side))
    return img.resize((size, size), Image.LANCZOS)


@functools.lru_cache(maxsize=1)
def _classes() -> dict[str, str]:
    """``IMAGENET2012_CLASSES`` from the export's ``classes.py``, in label order."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("imagenet_classes",
                                                  IMAGENET / "classes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.IMAGENET2012_CLASSES)


def wnids() -> tuple[str, ...]:
    """The 1,000 class ids in label order, so ``wnids()[label]`` is the class."""
    return tuple(_classes())


def class_names() -> dict[str, str]:
    """``wnid`` to the class's full comma-separated name."""
    return dict(_classes())


@functools.lru_cache(maxsize=1)
def _lemma_index() -> dict[str, frozenset[str]]:
    """The classes each single-word class name refers to, lowercased and with the
    export's disambiguating trailing digit removed (``crane2`` is ``crane``)."""
    import re

    out: dict[str, set[str]] = {}
    for wnid, names in class_names().items():
        for name in names.split(","):
            lemma = re.sub(r"(?<=[a-z])\d+$", "", name.strip().lower())
            if re.fullmatch(r"[a-z]{3,}", lemma):
                out.setdefault(lemma, set()).add(wnid)
    return {k: frozenset(v) for k, v in out.items()}


def classes_named(word: str) -> frozenset[str]:
    """The classes a word is a name of, under the normalization of :func:`_lemma_index`."""
    return _lemma_index().get(word.strip().lower(), frozenset())


def image_key(payload: bytes) -> str:
    """First sixteen hex digits of the sha256 of a photograph's encoded bytes,
    the ``key`` column of ``backgrounds/backgrounds.parquet``."""
    return hashlib.sha256(payload).hexdigest()[:16]


@functools.lru_cache(maxsize=1)
def background_keys() -> frozenset[str]:
    """Keys of the 96 validation photographs used as backgrounds."""
    p = BACKGROUNDS / "backgrounds.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run scripts/stimuli/backgrounds.py")
    return frozenset(pd.read_parquet(p)["key"].astype(str))


def validation_shards() -> list[Path]:
    found = sorted((IMAGENET / "data").glob("validation-*.parquet"))
    if not found:
        raise SystemExit(
            f"no validation shards under {IMAGENET / 'data'}. The ImageNet-1k "
            f"parquet export is a prerequisite; see README.md.")
    return found


def read_validation(shard: Path):
    """Yield ``(key, wnid, encoded bytes)`` for one validation shard, undecoded
    so a resumed run can skip a photograph already written."""
    import pyarrow.parquet as pq

    table = pq.read_table(shard, columns=["image", "label"])
    images = table.column("image").to_pylist()
    labels = table.column("label").to_pylist()
    names = wnids()
    for record, label in zip(images, labels):
        payload = record["bytes"]
        yield image_key(payload), names[int(label)], payload


def decode(payload: bytes) -> Image.Image:
    return Image.open(io.BytesIO(payload)).convert("RGB")


def image_path(wnid: str, key: str) -> Path:
    return IMAGES / wnid / f"{key}.png"


def concept_words() -> pd.DataFrame:
    """The kept rows of the committed class-name list, one per word."""
    if not CONCEPT_LIST.exists():
        raise FileNotFoundError(
            f"{CONCEPT_LIST} missing. It is the committed word list.")
    table = pd.read_csv(CONCEPT_LIST, sep="\t")
    kept = table[table.decision == "keep"].reset_index(drop=True)
    if kept.empty:
        raise SystemExit(f"{CONCEPT_LIST} holds no kept rows")
    return kept
