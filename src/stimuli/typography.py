"""Fonts, the 24 typography settings, the drawing primitives and the 96 slots.

Every word is drawn on the same (background, typography) slots, so both cancel
in a comparison between two words. :func:`build_appearance_design` builds the
settings so that the factors are balanced against each other, and
:func:`balance_report` says which factor pairs are exactly balanced.
"""
from __future__ import annotations

import functools
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from ..config import CROP_SAFE_FRAC, IMAGE_SIZE, SEED

# --- factors ---------------------------------------------------------------

# Six families of different letterform construction: two sans of different
# width, two serifs of different origin, one monospace, and one geometric face
# with a single-story 'a' and circular bowls.
FONT_FAMILIES: tuple[str, ...] = (
    "DejaVu Sans",       # humanist sans, wide
    "PT Sans",           # sans, narrower aperture
    "DejaVu Serif",      # slab-ish serif
    "Nimbus Roman",      # transitional serif, Times lineage
    "DejaVu Sans Mono",  # monospace: every advance width identical
    "URW Gothic",        # geometric sans, single-story a
)

# The family whose advance widths do not vary by glyph, so rendered width is a
# function of character count alone.
MONOSPACE_FAMILIES: frozenset[str] = frozenset({"DejaVu Sans Mono"})

# Font size as a fraction of the image side; font_px converts it to pixels.
SIZE_BANDS: tuple[tuple[str, float], ...] = (
    ("small", 0.05),
    ("medium", 0.08),
    ("large", 0.11),
)
SIZE_BAND_NAMES: tuple[str, ...] = tuple(n for n, _ in SIZE_BANDS)
SIZE_BAND_TO_FRAC: dict[str, float] = dict(SIZE_BANDS)

# One ink color per group, each distinct from the others in hue or lightness.
COLOR_GROUPS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("dark", (15, 15, 15)),
    ("light", (235, 235, 235)),
    ("red", (200, 30, 30)),
    ("blue", (30, 60, 200)),
    ("green", (20, 120, 40)),
    ("purple", (120, 30, 140)),
)
COLOR_GROUP_NAMES: tuple[str, ...] = tuple(n for n, _ in COLOR_GROUPS)
COLOR_GROUP_TO_RGB: dict[str, tuple[int, int, int]] = dict(COLOR_GROUPS)

#: Which of four vertical bands of the crop-safe window holds the text center.
#: A designed factor, balanced against the others; the center is chosen for
#: legibility within the band. The horizontal position is always the middle of
#: the window, because the widest word nearly fills it.
PLACEMENTS: tuple[str, ...] = ("top", "upper_middle", "lower_middle", "bottom")

# Whether a translucent panel sits behind the text.
BOX_STYLES: tuple[str, ...] = ("panel", "bare")

FACTORS: tuple[str, ...] = ("font", "size", "color", "placement", "box")


# --- font resolution -------------------------------------------------------

@functools.lru_cache(maxsize=None)
def font_path_for(family: str) -> str:
    """Absolute path to a TrueType file for ``family``.

    Resolved through matplotlib's font manager, which has already indexed the
    system fonts. A family that is not installed raises LookupError, so no
    fallback face stands in for a level of the font factor.
    """
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    if family not in available:
        raise LookupError(
            f"font family {family!r} is not installed; available families "
            f"include {sorted(available)[:8]}...")
    return font_manager.findfont(
        font_manager.FontProperties(family=family), fallback_to_default=False)


@functools.lru_cache(maxsize=None)
def load_font(family: str, px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path_for(family), px)


def font_px(size_band: str, image_size: int = IMAGE_SIZE) -> int:
    """Pixel size for a band. Depends on the slot only, never on the word."""
    return max(10, int(round(SIZE_BAND_TO_FRAC[size_band] * image_size)))


# --- the appearance design -------------------------------------------------

def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Association between two categorical columns, 0 (independent) to 1."""
    ct = pd.crosstab(a, b).to_numpy(dtype=float)
    n = ct.sum()
    if n == 0:
        return 0.0
    exp = ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / n
    chi2 = float(((ct - exp) ** 2 / np.where(exp == 0, 1, exp)).sum())
    k = min(ct.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


def balance_report(design: pd.DataFrame) -> pd.DataFrame:
    """One row per factor pair: the smallest and largest crosstab cell, whether
    the pair is exactly balanced, and Cramer's V. Exact balance needs the number
    of tuples to be a multiple of the product of the two level counts, so some
    pairs cannot reach it."""
    rows = []
    for i, x in enumerate(FACTORS):
        for y in FACTORS[i + 1:]:
            ct = pd.crosstab(design[x], design[y]).to_numpy()
            rows.append(dict(pair=f"{x} x {y}", cells=ct.size,
                             lo=int(ct.min()), hi=int(ct.max()),
                             exact=bool(ct.min() == ct.max()),
                             cramers_v=round(_cramers_v(design[x], design[y]), 3)))
    return pd.DataFrame(rows).sort_values(["exact", "pair"], ascending=[False, True])


def _latin_rectangle(rows: int, cols: int, rng: random.Random) -> list[list[int]]:
    """A ``rows`` x ``cols`` array whose every row is a permutation of
    ``range(cols)`` and whose every column holds ``rows`` distinct values."""
    for _ in range(2000):
        out, ok = [], True
        for r in range(rows):
            for _ in range(400):
                cand = list(range(cols))
                rng.shuffle(cand)
                if all(cand[c] not in {o[c] for o in out} for c in range(cols)):
                    out.append(cand)
                    break
            else:
                ok = False
                break
        if ok:
            return out
    raise RuntimeError(f"could not build a {rows}x{cols} Latin rectangle")


def _degree_constrained(n_rows: int, n_cols: int, per_row: int,
                        col_total: int, rng: random.Random) -> np.ndarray:
    """A binary ``n_rows`` x ``n_cols`` matrix with ``per_row`` ones in each
    row and ``col_total`` ones in each column. Used to balance a two-level
    factor against a six-level one exactly."""
    for _ in range(4000):
        m = np.zeros((n_rows, n_cols), int)
        for c in range(n_cols):
            for r in rng.sample(range(n_rows), col_total):
                m[r, c] = 1
        if (m.sum(1) == per_row).all():
            return m
    raise RuntimeError("could not build the degree-constrained assignment")


@functools.lru_cache(maxsize=None)
def build_appearance_design(reps: int = 1, seed: int = 0,
                            n_candidates: int = 600) -> pd.DataFrame:
    """The 24 appearance tuples, built by a seeded search.

    Font crosses placement, one tuple per cell. Color is a Latin rectangle over
    (placement, font), so every placement sees all six colors once. Size and
    box style are balanced within each placement, and the search keeps the
    candidate whose largest Cramer's V over the pairs that are not exactly
    balanced is smallest.
    """
    nf, np_, ns, nc, nb = (len(FONT_FAMILIES), len(PLACEMENTS),
                           len(SIZE_BAND_NAMES), len(COLOR_GROUP_NAMES),
                           len(BOX_STYLES))
    rng = random.Random(seed)
    best, best_score = None, np.inf
    for _ in range(n_candidates):
        colour_of = _latin_rectangle(np_, nc, rng)      # [placement][font]
        # box: a 6x4 binary matrix, 2 panels per font and 3 per placement, so
        # that both font x box and placement x box come out exactly balanced.
        box_of = _degree_constrained(nf, np_, per_row=np_ // nb, rng=rng,
                                     col_total=nf // nb)
        # size: 2 of each size per placement, and every font sees every size
        # at least once.
        size_of = None
        for _ in range(400):
            cand_s = []
            for _p in range(np_):
                col = [i % ns for i in range(nf)]
                rng.shuffle(col)
                cand_s.append(col)
            counts = np.zeros((nf, ns), int)
            for _p in range(np_):
                for f in range(nf):
                    counts[f, cand_s[_p][f]] += 1
            if counts.min() >= 1:
                size_of = cand_s
                break
        if size_of is None:
            continue
        rows = []
        for _rep in range(reps):
            for p in range(np_):
                for f in range(nf):
                    rows.append(dict(
                        font=FONT_FAMILIES[f],
                        size=SIZE_BAND_NAMES[size_of[p][f]],
                        color=COLOR_GROUP_NAMES[colour_of[p][f]],
                        placement=PLACEMENTS[p],
                        box=BOX_STYLES[int(box_of[f][p])]))
        cand = pd.DataFrame(rows)
        if cand.duplicated().any():
            continue
        if any(cand[f].value_counts().nunique() != 1 for f in FACTORS):
            continue
        rep = balance_report(cand)
        score = float(rep.loc[~rep.exact, "cramers_v"].max()) if (~rep.exact).any() else 0.0
        if score < best_score:
            best, best_score = cand, score
    if best is None:
        raise RuntimeError("no candidate design satisfied the balance constraints")
    out = best.sort_values(list(FACTORS), kind="mergesort").reset_index(drop=True)
    out.insert(0, "appearance", np.arange(len(out)))
    out["monospace"] = out.font.isin(MONOSPACE_FAMILIES)
    return out


# --- geometry --------------------------------------------------------------

def crop_safe_window(size: int = IMAGE_SIZE,
                     frac: float = CROP_SAFE_FRAC) -> tuple[int, int, int]:
    """``(x0, y0, side)`` of the centered square that survives a center crop."""
    side = int(round(frac * size))
    return (size - side) // 2, (size - side) // 2, side


def text_extent(family: str, size_band: str, text: str) -> tuple[int, int]:
    """``(width, height)`` of the text box: ink width, and the font's ascent
    plus descent, so every word in a slot has the same height and baseline."""
    f = load_font(family, font_px(size_band))
    x0, _, x1, _ = f.getbbox(text)
    ascent, descent = f.getmetrics()
    return int(x1 - x0), int(ascent + descent)


def band_range(band: str, half_h: int, size: int = IMAGE_SIZE) -> tuple[int, int]:
    """The vertical span a slot's text center may occupy, given its band."""
    _, wy, side = crop_safe_window(size)
    lo, hi = wy + half_h, wy + side - half_h
    if hi <= lo:
        mid = (lo + hi) // 2
        return mid, mid
    k = PLACEMENTS.index(band)
    step = (hi - lo) / len(PLACEMENTS)
    return int(round(lo + k * step)), int(round(lo + (k + 1) * step))


def legible_centre(background: "Image.Image", color_group: str, size_band: str,
                   half_w: int, half_h: int, band: str, grid: int = 9,
                   clutter_weight: float = 0.6) -> tuple[int, int]:
    """The text center for one slot, shared by every word drawn there.

    Candidate centers on a vertical grid within ``band`` are scored on the
    contrast between the ink color and the local background minus a clutter
    penalty. ``half_w`` and ``half_h`` are half the extent of the widest and
    tallest word, so every word fits at the chosen center. The horizontal
    position is the middle of the window.
    """
    wx, wy, side = crop_safe_window(background.size[0])
    half_w = max(2, min(int(half_w), side // 2 - 1))
    half_h = max(2, min(int(half_h), side // 2 - 1))
    cx = wx + side // 2
    y_lo, y_hi = band_range(band, half_h, background.size[0])

    a = np.asarray(background.convert("RGB"), np.float32)
    lum = a.mean(2)
    lap = (-4 * lum + np.roll(lum, 1, 0) + np.roll(lum, -1, 0)
           + np.roll(lum, 1, 1) + np.roll(lum, -1, 1))
    ink = np.array(COLOR_GROUP_TO_RGB[color_group], np.float32)

    x0, x1 = max(0, cx - half_w), min(a.shape[1], cx + half_w)
    best, best_score = (cx, (y_lo + y_hi) // 2), -np.inf
    for cy in np.linspace(y_lo, y_hi, grid).astype(int):
        y0, y1 = max(0, cy - half_h), min(a.shape[0], cy + half_h)
        if y1 <= y0 or x1 <= x0:
            continue
        contrast = float(np.abs(a[y0:y1, x0:x1].mean((0, 1)) - ink).mean()) / 255.0
        clutter = min(float(np.abs(lap[y0:y1, x0:x1]).mean()) / 64.0, 1.0)
        score = contrast - clutter_weight * clutter
        if score > best_score:
            best, best_score = (int(cx), int(cy)), score
    return best


def box_at_centre(centre: tuple[int, int], text_w: int, text_h: int,
                  size: int = IMAGE_SIZE) -> Box:
    """A word's box centered on the slot's fixed point, clamped into the window."""
    wx, wy, side = crop_safe_window(size)
    x = int(round(centre[0] - text_w / 2))
    y = int(round(centre[1] - text_h / 2))
    x = max(wx, min(x, wx + side - text_w))
    y = max(wy, min(y, wy + side - text_h))
    return Box(x, y, text_w, text_h)


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int


# --- drawing ---------------------------------------------------------------

def _draw_panel(draw: ImageDraw.ImageDraw, box: Box, pad: int | None = None,
                fill: tuple[int, int, int, int] = (255, 255, 255, 140)) -> None:
    """Translucent panel behind the text, padded by a sixth of the box height (at least 3 px)."""
    if pad is None:
        pad = max(3, box.h // 6)
    draw.rectangle(
        [box.x - pad, box.y - pad, box.x + box.w + pad, box.y + box.h + pad],
        fill=fill)


def panel_fill(color_group: str) -> tuple[int, int, int, int]:
    """Panel color on the opposite side of mid-gray from the ink."""
    r, g, b = COLOR_GROUP_TO_RGB[color_group]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return (20, 20, 20, 150) if luma > 140 else (255, 255, 255, 140)


def render_overlay(background: Image.Image, text: str, box: Box,
                   family: str, size_band: str, *,
                   color_group: str, box_style: str) -> Image.Image:
    """Draw ``text`` onto a copy of ``background`` and return it."""
    base = background.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if box_style == "panel":
        _draw_panel(draw, box, fill=panel_fill(color_group))
    font = load_font(family, font_px(size_band))
    bx = font.getbbox(text)
    # Horizontal from the ink's left edge, so the ink fills the box width;
    # vertical from the ascender line, so a descender does not move the word.
    draw.text((box.x - bx[0], box.y), text, font=font, anchor="la",
              fill=(*COLOR_GROUP_TO_RGB[color_group], 255))
    return Image.alpha_composite(base, layer).convert("RGB")


def text_inkmap(family: str, size_band: str, text: str,
                w: int, h: int) -> np.ndarray:
    """Grayscale ink coverage of the word inside a ``w`` by ``h`` box."""
    font = load_font(family, font_px(size_band))
    canvas = Image.new("L", (max(1, w), max(1, h)), 0)
    bx = font.getbbox(text)
    ImageDraw.Draw(canvas).text((-bx[0], 0), text, font=font, anchor="la", fill=255)
    return np.asarray(canvas, np.float64) / 255.0


def phase_scramble(ink: np.ndarray, seed: int) -> np.ndarray:
    """The ink map with its Fourier phase randomized and its magnitude spectrum,
    mean and spread kept: the same ink and spatial frequencies without the glyphs."""
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(ink)
    mag = np.abs(F)
    ph = rng.uniform(-np.pi, np.pi, size=ink.shape)
    ph[0, 0] = np.angle(F[0, 0])          # keep DC so the mean ink survives
    out = np.fft.ifft2(mag * np.exp(1j * ph)).real
    out = (out - out.mean()) / (out.std() + 1e-8) * (ink.std() + 1e-8) + ink.mean()
    return np.clip(out, 0.0, 1.0)


def render_scramble(background: Image.Image, text: str, box: Box, family: str,
                    size_band: str, *, color_group: str, box_style: str,
                    seed: int) -> Image.Image:
    """Draw the phase-scrambled counterpart of ``text`` in the same box."""
    ink = phase_scramble(text_inkmap(family, size_band, text, box.w, box.h), seed)
    base = background.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    if box_style == "panel":
        _draw_panel(ImageDraw.Draw(layer), box, fill=panel_fill(color_group))
    h, w = ink.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = COLOR_GROUP_TO_RGB[color_group]
    rgba[..., 3] = (ink * 255).round().astype(np.uint8)
    layer.alpha_composite(Image.fromarray(rgba, "RGBA"), (box.x, box.y))
    return Image.alpha_composite(base, layer).convert("RGB")


# --- the slot table ----------------------------------------------------------
# A slot is one background paired with one typography setting. Every word is
# drawn on all of them, so background and typography cancel in any comparison
# between words. Each typography setting is used on ``REPLICATES`` different
# backgrounds, and ``split_halves`` puts half of the replicates on each side, so
# the two halves carry the same typography and differ only in the photographs.

# Backgrounds each appearance tuple is drawn on. Four gives 96 slots from 24
# tuples and two replicates per half of the matched-appearance split. The
# background count is n_slots().
REPLICATES = 4


@functools.lru_cache(maxsize=None)
def slot_table() -> pd.DataFrame:
    """One row per slot: ``slot, appearance, replicate, background_index`` plus
    every appearance factor. Deterministic.

    Each slot has its own background, and the background a given appearance
    receives in replicate 0 is unrelated to the one it receives in replicate 1.
    """
    appearance = build_appearance_design()
    order = list(range(len(appearance) * REPLICATES))
    random.Random(SEED).shuffle(order)
    rows = []
    for rep in range(REPLICATES):
        for _, a in appearance.iterrows():
            s = len(rows)
            rows.append(dict(slot=s, appearance=int(a.appearance), replicate=rep,
                             background_index=order[s],
                             **{f: a[f] for f in FACTORS},
                             monospace=bool(a.monospace)))
    return pd.DataFrame(rows)


def n_slots() -> int:
    return len(slot_table())


def slots_for(appearances) -> tuple[int, ...]:
    """Every slot carrying one of these appearance tuples, all replicates."""
    t = slot_table()
    return tuple(sorted(t.loc[t.appearance.isin(list(appearances)), "slot"]))


def split_halves(appearances=None) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Matched-appearance halves: replicates {0, 1} against {2, 3}.

    Each half carries every appearance tuple twice, and the halves differ only
    in which photographs they were drawn on. ``appearances`` restricts both
    halves to those appearance tuples; ``None`` keeps all of them. Slots are
    returned in ascending order.
    """
    t = slot_table()
    if appearances is not None:
        t = t[t.appearance.isin(list(appearances))]
    half = REPLICATES // 2
    lo, hi = t.replicate < half, t.replicate >= half
    # Sort on (appearance, replicate), which is a unique key within a half.
    key = ["appearance", "replicate"]
    ta = t.loc[lo].sort_values(key, kind="mergesort")[list(FACTORS)].to_numpy()
    tb = t.loc[hi].sort_values(key, kind="mergesort")[list(FACTORS)].to_numpy()
    assert ta.shape == tb.shape and (ta == tb).all(), \
        "the two halves do not carry identical appearance tuples"
    return (tuple(sorted(t.loc[lo, "slot"].tolist())),
            tuple(sorted(t.loc[hi, "slot"].tolist())))
