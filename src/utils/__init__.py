"""The figure style, the result readers, the LaTeX primitives and the notebook setup."""
from __future__ import annotations


def _number(value: float) -> str:
    """Four decimals, or three significant digits for a nonzero value below 5e-5 in size."""
    return f"{value:.2e}" if 0 < abs(value) < 5e-5 else f"{value:.4f}"


def setup_notebook() -> None:
    """Apply the figure style, show whole tables at four decimals, and silence library warnings."""
    import logging
    import warnings

    import pandas as pd

    from . import viz

    viz.use_style()
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
