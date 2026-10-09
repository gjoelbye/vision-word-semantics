"""Render a word set on the 96 slots, and record what was drawn.

`scripts/stimuli/render.py` renders the lexical vocabulary with it, and
`scripts/stimuli/class_images.py` the class names. The manifest has one row
per image, clean backgrounds first, in a canonical order. Row i of an
embedding cache is the embedding of manifest row i, and `fingerprint` lets a
cache refuse a manifest it was not built from.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from ..config import (BACKGROUNDS, LEXICON, RENDERS, SEED, line_fingerprint,
                      manifest_path)
from .typography import (box_at_centre, crop_safe_window, legible_centre,
                         render_overlay, render_scramble, slot_table,
                         text_extent)

CLEAN = "__clean__"


def fingerprint(manifest: pd.DataFrame) -> str:
    """sha256 of the recorded `image_path` strings, in row order."""
    return line_fingerprint(manifest["image_path"])


def _background_path(index: int) -> Path:
    return BACKGROUNDS / f"bg_{index:03d}.png"


# The text center of each slot, written by the first render and read by every
# later one.
CENTRES_PATH = LEXICON / "centres.parquet"


def slot_centres(words: tuple[str, ...], path: Path | None = None
                 ) -> dict[int, tuple[int, int]]:
    """One text center per slot, shared by every word drawn in it.

    The centers depend on the widest word of the vocabulary, so they are
    computed once from `words` and recorded at `path`. A recorded table is
    reused for any vocabulary whose widest word fits at its centers.
    """
    path = CENTRES_PATH if path is None else path
    table = slot_table().to_dict("records")

    if path.exists():
        rec = pd.read_parquet(path)
        out = {int(r.slot): (int(r.cx), int(r.cy)) for r in rec.itertuples()}
        _check_fits(table, words, out)
        return out

    out = {}
    for s in table:
        extents = [text_extent(s["font"], s["size"], w) for w in words]
        half_w = max(w for w, _ in extents) / 2
        half_h = max(h for _, h in extents) / 2
        with Image.open(_background_path(s["background_index"])) as bg:
            out[int(s["slot"])] = legible_centre(bg.convert("RGB"), s["color"],
                                                 s["size"], half_w, half_h,
                                                 s["placement"])
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(slot=k, cx=v[0], cy=v[1]) for k, v in sorted(out.items())]
                 ).to_parquet(path)
    return out


# Rounding slack: a box of odd width cannot be centered on an integer grid.
_FIT_SLACK = 1


def _check_fits(table: list[dict], words: tuple[str, ...],
                centres: dict[int, tuple[int, int]]) -> None:
    """Raise unless the widest word of each (font, size) fits at its recorded center."""
    x0, y0, side = crop_safe_window()
    faces = {(s["font"], s["size"]) for s in table}
    widest = {f: max(words, key=lambda w: text_extent(*f, w)[0]) for f in faces}
    for s in table:
        word = widest[(s["font"], s["size"])]
        tw, th = text_extent(s["font"], s["size"], word)
        cx, cy = centres[int(s["slot"])]
        inside = (cx - tw / 2 >= x0 - _FIT_SLACK
                  and cx + tw / 2 <= x0 + side + _FIT_SLACK
                  and cy - th / 2 >= y0 - _FIT_SLACK
                  and cy + th / 2 <= y0 + side + _FIT_SLACK)
        if not inside:
            raise ValueError(
                f"{word!r} does not fit at the recorded centre for slot "
                f"{s['slot']}. The recorded table was built for a different "
                f"vocabulary; delete {CENTRES_PATH} to rebuild it, which will "
                f"move existing renders.")


def _control_seed(word: str, slot: int) -> int:
    """Phase seed for the scrambled control, deterministic per (word, slot)."""
    return int(hashlib.sha256(f"scramble/{word}/{slot}/{SEED}".encode())
               .hexdigest()[:12], 16) % (2 ** 31)


def render_word(args) -> tuple[str, list[str], list[str]]:
    """Render one word on every slot, and its phase-scrambled control, as PNG.

    Returns the word and the recorded paths of its word and control images.
    Existing files are kept.
    """
    word, slots, centres, outdir = args
    d = Path(outdir) / word
    d.mkdir(parents=True, exist_ok=True)
    word_paths, ctl_paths = [], []
    for s in slots:
        sl = int(s["slot"])
        out = d / f"{sl:03d}.png"
        ctl = d / f"{sl:03d}_scramble.png"
        word_paths.append(manifest_path(out))
        ctl_paths.append(manifest_path(ctl))
        if out.exists() and ctl.exists():
            continue
        tw, th = text_extent(s["font"], s["size"], word)
        box = box_at_centre(centres[sl], tw, th)
        with Image.open(_background_path(s["background_index"])) as bg:
            bg = bg.convert("RGB")
            if not out.exists():
                render_overlay(bg, word, box, s["font"], s["size"],
                               color_group=s["color"],
                               box_style=s["box"]).save(out)
            if not ctl.exists():
                render_scramble(bg, word, box, s["font"], s["size"],
                                color_group=s["color"], box_style=s["box"],
                                seed=_control_seed(word, sl)).save(ctl)
    return word, word_paths, ctl_paths


def build_manifest(words: list[str], workers: int = 12) -> pd.DataFrame:
    """Render every word on every slot and return the canonical manifest."""
    slots = slot_table().to_dict("records")
    RENDERS.mkdir(parents=True, exist_ok=True)

    rows = [dict(word=CLEAN, slot=int(s["slot"]), condition="clean",
                 image_path=manifest_path(_background_path(s["background_index"])))
            for s in slots]

    centres = slot_centres(tuple(words))
    jobs = [(w, slots, centres, str(RENDERS)) for w in words]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for word, wpaths, ppaths in tqdm(ex.map(render_word, jobs, chunksize=8),
                                         total=len(jobs), desc="rendering",
                                         unit="word"):
            for s, wp, pp in zip(slots, wpaths, ppaths):
                rows.append(dict(word=word, slot=int(s["slot"]),
                                 condition="word", image_path=wp))
                rows.append(dict(word=word, slot=int(s["slot"]),
                                 condition="scramble", image_path=pp))

    return canonical_order(pd.DataFrame(rows))


def canonical_order(man: pd.DataFrame) -> pd.DataFrame:
    """Clean rows first by slot, then words alphabetically, then condition, then slot."""
    man = man.assign(_clean=(man.word == CLEAN).astype(int))
    return (man.sort_values(["_clean", "word", "condition", "slot"],
                            ascending=[False, True, True, True], kind="mergesort")
               .drop(columns="_clean").reset_index(drop=True))
