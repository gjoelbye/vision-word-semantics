"""The figure style: page width, type sizes, colors, scales, the bold face, saving and the notebook setup.

Every notebook calls `setup_notebook` first and writes its figures through
`save`. Figures are built at their final physical size and saved without a
tight bounding box, so the PDF is the size the paper prints.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from ..config import FIGURES, FONTS, REPO_ROOT

PT_PER_INCH = 72.0
TEXT_WIDTH_IN = 5.5

LABEL_SIZE = 8.5
ANNOTATION_SIZE = 8.0
TICK_SIZE = 7.5

# Figure 1b type sizes, in points.
FIGURE1B_COUNTRY_SIZE = 6.75
FIGURE1B_CAPITAL_SIZE = 7.25

# Figure 1b origin marker size, in points squared.
FIGURE1B_ORIGIN_SIZE = 5.0

# Figure 1b horizontal axis arrow: line width and head scale, in points.
FIGURE1B_AXIS = {"width": 0.72, "head_scale": 6.80}

# Type sizes shared by Figures 2, 3, 5 and 6, in points: panel letters, titles,
# panel subtitles and other text.
PANEL_SIZE = 8.0
TITLE_SIZE = 7.25
SUBTITLE_SIZE = 6.5
TEXT_SIZE = 6.5

# Marker size in Figure 6, in points squared.
MARK_SIZE = 17.0

# Width of the hairline that runs from a model's name to its marks in Figures 5 and 6.
LEADER_WIDTH = 0.42

SEMANTIC = "#0C6E5E"

# Comparators and reference rules are achromatic.
INK = "#262B29"
CONTROL = "#69716E"
REFERENCE = "#3E4643"
CHANCE = "#8A928F"
GRID = "#DDE2E0"
FAINT = "#EEF1F0"
WHITE = "#FFFFFF"
RULE = "#C5CCC9"

# Dash per line role.
DASH = {"claim": "-", "control": (0, (3, 2)),
        "reference": (0, (1, 1)), "chance": (0, (4, 3)),
        "construction": (0, (1.25, 1.9))}


def use_style() -> None:
    """Apply the figure style to Matplotlib's settings."""
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        # Noto Sans has a TrueType file, so the PDF embeds Type 42 fonts.
        "font.sans-serif": ["Noto Sans", "Nimbus Sans", "Helvetica",
                            "Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Noto Sans",
        "mathtext.it": "Noto Sans:italic",
        "mathtext.bf": "Noto Sans:bold",
        "mathtext.bfit": "Noto Sans:bold:italic",
        "mathtext.cal": "Noto Sans",
        "mathtext.tt": "Noto Sans",
        "mathtext.sf": "Noto Sans",
        "mathtext.fallback": "None",
        "font.size": LABEL_SIZE,
        "axes.titlesize": LABEL_SIZE,
        "axes.titleweight": "normal",
        "axes.labelsize": LABEL_SIZE,
        "axes.labelcolor": INK,
        "axes.edgecolor": CONTROL,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "legend.fontsize": ANNOTATION_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "xtick.color": CONTROL,
        "ytick.color": CONTROL,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.8,
    })


def inches(height_pt: float, width_in: float = TEXT_WIDTH_IN
           ) -> tuple[float, float]:
    """Final physical figure size as `(width inches, height inches)`."""
    return float(width_in), float(height_pt) / PT_PER_INCH


# The two alignment measures of Figure 3 share one cell, so their ramps use the
# same OKLCH lightness steps and differ in hue. Chroma tapers toward both ends.
ALIGNMENT_LIGHTNESS = (0.975, 0.91, 0.83, 0.74, 0.64, 0.54, 0.44)
ALIGNMENT_CHROMA_TAPER = (0.18, 0.45, 0.75, 0.95, 1.0, 0.95, 0.80)
# `(hue in degrees, peak chroma)` in OKLCH for each measure's ramp.
NEIGHBOR_HUE = (254.7, 0.117)
CKA_HUE = (68.8, 0.133)


def oklch_to_hex(lightness: float, chroma: float, hue: float) -> str:
    """An OKLCH color as sRGB hex, lowering chroma until it is in gamut."""
    for scale in np.linspace(1.0, 0.0, 41):
        a = chroma * scale * np.cos(np.radians(hue))
        b = chroma * scale * np.sin(np.radians(hue))
        lms = np.array([lightness + 0.3963377774 * a + 0.2158037573 * b,
                        lightness - 0.1055613458 * a - 0.0638541728 * b,
                        lightness - 0.0894841775 * a - 1.2914855480 * b]) ** 3
        linear = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                           [-1.2684380046, 2.6097574011, -0.3413193965],
                           [-0.0041960863, -0.7034186147, 1.7076147010]]) @ lms
        if linear.min() >= -1e-4 and linear.max() <= 1 + 1e-4:
            linear = np.clip(linear, 0.0, 1.0)
            srgb = np.where(linear <= 0.0031308, 12.92 * linear,
                            1.055 * linear ** (1 / 2.4) - 0.055)
            return mpl.colors.to_hex(srgb)
    raise ValueError(f"lightness {lightness} is outside sRGB")


def oklch_ramp(hue: float, chroma: float) -> tuple[str, ...]:
    """One sequential ramp at the shared alignment lightness steps."""
    return tuple(oklch_to_hex(lightness, chroma * taper, hue)
                 for lightness, taper in zip(ALIGNMENT_LIGHTNESS,
                                             ALIGNMENT_CHROMA_TAPER))


def _log_scale(name: str, hue: tuple[float, float], low: float, high: float
               ) -> tuple[LinearSegmentedColormap, LogNorm]:
    """An OKLCH ramp on a logarithmic norm, so equal ratios take equal color steps."""
    return (LinearSegmentedColormap.from_list(name, list(oklch_ramp(*hue))),
            LogNorm(vmin=low, vmax=high, clip=True))


def neighbor_alignment_scale(low: float, high: float
                             ) -> tuple[LinearSegmentedColormap, LogNorm]:
    """One logarithmic map for mutual k-nearest-neighbor alignment."""
    return _log_scale("neighbor_alignment", NEIGHBOR_HUE, low, high)


def cka_scale(low: float, high: float
              ) -> tuple[LinearSegmentedColormap, LogNorm]:
    """One logarithmic map for linear CKA between two similarity matrices."""
    return _log_scale("cka", CKA_HUE, low, high)


def luminance(colour) -> float:
    """WCAG relative luminance, for choosing ink or white text on a fill."""
    rgb = np.asarray(mpl.colors.to_rgb(colour))
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(linear @ (0.2126, 0.7152, 0.0722))


def stroke_text(text, colour: str, linewidth: float) -> None:
    """Draw `text` over a stroke of `colour`.

    A white stroke keeps a label legible over lines. A hairline stroke in the
    text's own color emphasizes small text, since the variable Noto face renders
    medium and semibold requests at regular weight.
    """
    text.set_path_effects([
        patheffects.Stroke(linewidth=linewidth, foreground=colour),
        patheffects.Normal(),
    ])


# The cut bold face is cached in `FONTS`. Matplotlib matches a face on the
# weight in its OS/2 table and does not instance the weight axis of a variable
# font, so a bold request on the installed variable sans gets the regular outline.


def cut_static_face(weight: int = 700) -> Path:
    """The installed variable sans, instanced at `weight` and cached in `FONTS`."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont
    from matplotlib import font_manager as fm

    family = mpl.rcParams["font.sans-serif"][0]
    source = Path(fm.findfont(fm.FontProperties(family=family)))
    out = FONTS / f"{source.stem.partition('[')[0]}-{weight}.ttf"
    if out.exists():
        return out
    font = TTFont(source)
    if "fvar" not in font:
        raise RuntimeError(f"{source.name} has no weight axis to instance; the "
                           f"figure asked for {family} and got another face")
    instantiateVariableFont(font, {"wght": weight}, updateFontNames=True,
                            inplace=True)
    FONTS.mkdir(parents=True, exist_ok=True)
    # Written aside and moved, so an interrupted run leaves no partial face.
    part = out.with_suffix(".part")
    font.save(part)
    part.replace(out)
    return out


def use_real_bold(weight: int = 700) -> None:
    """Register the cut face as the bold of the sans, and raise if bold does not resolve to it.

    Call it after `use_style` and before the first text is drawn. It
    changes what bold resolves to for every later figure in the process.
    """
    from matplotlib import font_manager as fm

    path = cut_static_face(weight)
    fm.fontManager.addfont(path)
    family = mpl.rcParams["font.sans-serif"][0]
    got = Path(fm.fontManager.findfont(
        fm.FontProperties(family=family, weight="bold")))
    if got.resolve() != path.resolve():
        raise RuntimeError(f"bold resolves to {got.name}, not {path.name}")


def save(fig, name: str, *, dpi: float | None = None) -> None:
    """Save the figure at its exact size as `figures/<name>.pdf`.

    `dpi` overrides the style's `savefig.dpi` for the raster parts.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    # Constrained layout can move artists on its first draw; settle it before saving.
    fig.canvas.draw()
    fig.canvas.draw()
    path = FIGURES / f"{name}.pdf"
    fig.savefig(path, bbox_inches=None, dpi=dpi,
                metadata={"Creator": "ViT-Semantic",
                          "CreationDate": None, "ModDate": None})
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def _number(value: float) -> str:
    """Four decimals, or three significant digits for a nonzero value below 5e-5 in size."""
    return f"{value:.2e}" if 0 < abs(value) < 5e-5 else f"{value:.4f}"


def setup_notebook() -> None:
    """Apply the figure style, show whole tables at four decimals, and silence library warnings."""
    import logging
    import warnings

    import pandas as pd

    use_style()
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", _number)
    for category in (FutureWarning, DeprecationWarning, UserWarning):
        warnings.filterwarnings("ignore", category=category)

    logging.getLogger("fontTools").setLevel(logging.ERROR)

    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
