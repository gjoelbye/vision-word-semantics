"""Formatting primitives for the LaTeX tables: row groups, bands, marks, numbers and layout.

A table body is written as a ``tabular*`` and rewritten by :func:`fit` into a
tabular that fills the line with its own column separation, so a gray band runs
across a row without gaps.
"""
from __future__ import annotations

import re

import pandas as pd

from .. import wordspeech, wordtext
from ..models import PRETTY

#: Row groups of every model table, separated by white space: DINOv3, Web-DINO,
#: DINOv2, MAE, the caption-trained pair and the untrained network.
GROUPS = (("dinov3-b", "dinov3-l", "dinov3-h", "dinov3-7b"),
          ("webssl-dino-1b",),
          ("dinov2-b", "dinov2-l", "dinov2-g"),
          ("mae-b", "mae-l", "mae-h"),
          ("clip-b", "siglip2-b"),
          ("vit-random-b",))

#: The encoders printed under the models as references, with their table labels.
REFERENCE = {key: PRETTY[key].replace(" ", "~")
             for key in wordtext.TEXT_MODELS + wordspeech.SPEECH_MODELS}

#: The untrained network every model is read against.
BASELINE = "vit-random-b"
#: Standard errors of the difference a value must exceed the baseline by to be underlined.
MARK_SIGMA = 2


def above_baseline(value: float, se: float, base: float, base_se: float) -> bool:
    """True where ``value`` exceeds ``base`` by more than MARK_SIGMA standard errors of the difference.

    The standard error of the difference is ``sqrt(se^2 + base_se^2)``, which
    treats the two estimates as independent.
    """
    return value - base > MARK_SIGMA * (se ** 2 + base_se ** 2) ** 0.5


#: The colour name and gray level of a banded row group.
BAND = ("tabband", 0.94)

#: The sentence every table file adds to its note about the bands.
BAND_NOTE = (" The rows are grouped by model family and every second group "
             "carries a gray band.")


def band(index: int) -> str:
    """The row fill of the ``index``-th row group: every even group is banded."""
    return f"\\rowcolor{{{BAND[0]}}}" if index % 2 == 0 else ""


def band_colour() -> str:
    """The line that defines the band colour."""
    return f"\\definecolor{{{BAND[0]}}}{{gray}}{{{BAND[1]}}}"


def fitted(spec: str, body: list[str]) -> list[str]:
    """A tabular with column separation set so that it fills the line width.

    The body is measured with no separation, and ``\\tabcolsep`` is set to share
    the remaining width evenly over the columns.
    """
    columns = sum(c in "lrc" for c in spec)
    return ["\\begingroup",
            "\\def\\tablebody{%",
            *body,
            "}%",
            "\\setlength{\\tabcolsep}{0pt}%",
            f"\\sbox0{{\\begin{{tabular}}{{{spec}}}\\tablebody\\end{{tabular}}}}%",
            "\\ifdim\\wd0<\\linewidth%",
            "\\setlength{\\tabcolsep}{\\dimexpr(\\linewidth-\\wd0)"
            f"/\\numexpr2*{columns}\\relax\\relax}}%",
            "\\else\\setlength{\\tabcolsep}{0pt}\\fi%",
            f"\\begin{{tabular}}{{{spec}}}\\tablebody\\end{{tabular}}",
            "\\endgroup"]


#: A ``tabular*`` opening line, capturing its column specification.
TABULAR = re.compile(r"\\begin\{tabular\*\}\{[^}]*\}"
                     r"\{@\{\\extracolsep\{\\fill\}\}(.*?)(?:@\{\})?\}$")


def fit(lines: list[str]) -> list[str]:
    """Every ``tabular*`` in ``lines`` rewritten by :func:`fitted`."""
    out, i = [], 0
    while i < len(lines):
        match = TABULAR.match(lines[i])
        if match is None:
            out.append(lines[i])
            i += 1
            continue
        close = lines.index("\\end{tabular*}", i)
        out += fitted(match.group(1), lines[i + 1:close])
        i = close + 1
    return out


#: Type size of a standard error printed after its value.
SE_SIZE = "\\scriptsize"


def without_zero(text: str) -> str:
    """A number below one without its leading zero, as ``.034``."""
    return text[1:] if text.startswith("0.") else text


def se_text(se: float, digits: int = 3) -> str:
    """A standard error without its leading zero; raises ValueError if missing or negative."""
    if not se >= 0:
        raise ValueError(f"standard error {se!r} is missing or negative")
    return without_zero(f"{se:.{digits}f}")


def se_suffix(se: float, digits: int = 3, size: str = SE_SIZE) -> str:
    """The small ``±.034`` printed after a value."""
    return f"\\,{{{size}$\\pm${se_text(se, digits)}}}"


def bare(text: str, digits: int = 3, size: str = SE_SIZE) -> str:
    """A value without a standard error, padded to align with values that have one."""
    return f"{text}\\hphantom{{{se_suffix(0.0, digits, size)}}}"


def cell(value: float, bold: bool, mark: bool, se: float | None = None,
         digits: int = 3, se_size: str = SE_SIZE,
         strip_zero: bool = False) -> str:
    """A value, bold if ``bold``, underlined if ``mark``, followed by its standard error."""
    text = f"{value:.{digits}f}"
    if strip_zero:
        text = without_zero(text)
    if bold:
        text = f"\\textbf{{{text}}}"
    if mark:
        text = f"\\underline{{{text}}}"
    if se is not None:
        text = f"{text}{se_suffix(se, digits, se_size)}"
    return text


def pval(p: float) -> str:
    """A p-value to three places, ``$<$0.001`` below 0.001, and a dash when missing."""
    if p != p:
        return "-"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def rank_text(value: float) -> str:
    """A median rank in its shortest form."""
    return f"{value:g}"


def zero(value: float, digits: int = 3) -> str:
    """A value to ``digits`` places, with a negative zero printed as zero."""
    text = f"{value:.{digits}f}"
    return text.replace("-0." + "0" * digits, "0." + "0" * digits)


def _sources(sources: list[str]) -> str:
    return ", ".join(f"data/results/{s}.parquet" for s in sources)


def header(notebook: str, sources: list[str], manifest_sha: str) -> str:
    """Line 1 of a table from the main word set: its notebook, inputs and manifest fingerprint."""
    return (f"% Generated by notebooks/{notebook}.ipynb from {_sources(sources)}, "
            f"manifest {manifest_sha}. Do not edit.\n")


def class_header(notebook: str, sources: list[str], word_sha: str,
                 image_sha: str) -> str:
    """Line 1 of a table from the class-name experiment, pinned by both of its manifests."""
    return (f"% Generated by notebooks/{notebook}.ipynb from {_sources(sources)}, "
            f"class names {word_sha}, class images {image_sha}. Do not edit.\n")


def reference_rows(lines: list[str], printed: list, render, *,
                   rule_before_speech: bool = False,
                   start: int = len(GROUPS)) -> None:
    """Append the encoder rows under a rule, continuing the bands of the model rows.

    ``render`` returns the cells of one encoder. With ``rule_before_speech`` a
    second rule separates the speech encoder from the text encoders.
    """
    lines.append("\\midrule")
    for model, label in REFERENCE.items():
        speech = model in wordspeech.SPEECH_MODELS
        if rule_before_speech and model == wordspeech.SPEECH_MODELS[0]:
            lines.append("\\midrule")
        cells = render(model)
        printed.append((label, cells))
        lines.append(band(start + int(speech)) + f"{label} & "
                     + " & ".join(cells) + " \\\\")


def model_rows(lines: list[str], printed: list, models, render) -> None:
    """Append one row per model in the order of GROUPS, banding the groups that appear."""
    shown = 0
    for group in GROUPS:
        listed = [m for m in group if m in models]
        if not listed:
            continue
        if shown:
            lines.append("\\addlinespace")
        for model in listed:
            cells = render(model)
            printed.append((PRETTY[model], cells))
            lines.append(band(shown) + f"{PRETTY[model]} & "
                         + " & ".join(cells) + " \\\\")
        shown += 1


def _unwrap(text: str, macro: str, keep: bool) -> str:
    """Replace every ``\\macro{...}`` by its argument, or remove it."""
    opening = f"\\{macro}{{"
    while (start := text.find(opening)) >= 0:
        depth, i = 1, start + len(opening)
        while depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        inner = text[start + len(opening):i - 1]
        text = text[:start] + (inner if keep else "") + text[i:]
    return text


def plain(text: str) -> str:
    """A table cell without its LaTeX markup, as ``0.600 ±.063``."""
    text = _unwrap(text, "hphantom", keep=False)
    for macro in ("textbf", "underline", "emph", "shortstack"):
        text = _unwrap(text, macro, keep=True)
    text = re.sub(r"\\,\{\\(?:scriptsize|tiny)\$\\pm\$([^}]*)\}", r" ±\1", text)
    return (text.replace("$^{*}$", "*").replace("$<$", "<").replace("$p$", "p")
            .replace("\\\\", " ").replace("~", " ").strip())


def printed_frame(printed: list, columns: list[str],
                  index_name: str = "Model") -> pd.DataFrame:
    """The printed cells of a table as plain text, one row per ``(label, cells)``."""
    frame = pd.DataFrame([[plain(c) for c in cells] for _, cells in printed],
                         columns=columns,
                         index=pd.Index([plain(label) for label, _ in printed],
                                        name=index_name))
    return frame
