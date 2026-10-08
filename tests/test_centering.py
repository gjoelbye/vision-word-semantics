"""Cosine similarity separates words only once word directions are centered.

Cosine on uncentered word directions puts every word near every other when most
of the energy sits in the direction all words share, so the similarity reports
that text is present rather than which word it is. Centering removes that
direction.
"""
import numpy as np
import pandas as pd
import pytest

from src.analysis import directions
from src.stimuli.render import CLEAN

N_SLOTS, D = 4, 6
WORDS = ["alpha", "beta", "gamma"]
BG = np.arange(N_SLOTS * D, dtype=np.float32).reshape(N_SLOTS, D) * 0.1
SHARED = np.array([10.0, 10.0, 10.0, 0, 0, 0], np.float32)   # "text is present"
SPECIFIC = {"alpha": np.array([0, 0, 0, 1.0, 0, 0], np.float32),
            "beta": np.array([0, 0, 0, 0, 1.0, 0], np.float32),
            "gamma": np.array([0, 0, 0, 0, 0, 1.0], np.float32)}


def _install(monkeypatch, control: str | None):
    """Install a synthetic manifest and cache whose word rows carry the shared
    and the word-specific part. ``control`` sets what the scrambled rows carry
    on top of the background: ``"background"`` nothing, ``"shared"`` the shared
    part, ``None`` no scrambled rows at all."""
    rows = [dict(word=CLEAN, slot=s, condition="clean", image_path=f"/bg/{s}")
            for s in range(N_SLOTS)]
    for w in WORDS:
        for s in range(N_SLOTS):
            rows.append(dict(word=w, slot=s, condition="word", image_path=f"/w/{w}/{s}"))
    if control is not None:
        for w in WORDS:
            for s in range(N_SLOTS):
                rows.append(dict(word=w, slot=s, condition="scramble",
                                 image_path=f"/p/{w}/{s}"))
    man = pd.DataFrame(rows)

    feats = []
    for r in rows:
        v = BG[r["slot"]].copy()
        if r["condition"] == "word":
            v = v + SHARED + SPECIFIC[r["word"]]
        elif r["condition"] == "scramble" and control == "shared":
            v = v + SHARED
        feats.append(v)
    x = np.stack(feats)[:, None, :].astype(np.float16)

    monkeypatch.setattr(directions, "manifest", lambda ds=None: man)
    monkeypatch.setattr(directions, "load_cache", lambda k, *, ds=None: (x, np.array([0]), 1))
    monkeypatch.setattr(directions, "word_order", lambda ds=None: tuple(WORDS))


def test_uncentred_tau_is_dominated_by_what_the_words_share(monkeypatch):
    _install(monkeypatch, control="background")
    T = directions.word_directions("fake", layer=0, center=False)
    assert directions.shared_mean_share(T) > 0.9
    Tn = directions.rownorm(T)
    off = (Tn @ Tn.T)[np.triu_indices(len(Tn), 1)]
    assert off.min() > 0.9, "arbitrary words should look near-identical here"


def test_centring_removes_it(monkeypatch):
    _install(monkeypatch, control="background")
    T = directions.word_directions("fake", layer=0, center=True)
    assert directions.shared_mean_share(T) < 1e-6
    Tn = directions.rownorm(T)
    off = (Tn @ Tn.T)[np.triu_indices(len(Tn), 1)]
    assert off.max() < 0.1, "distinct words should now be distinguishable"


def test_word_directions_recover_the_word_specific_part(monkeypatch):
    _install(monkeypatch, control="shared")
    T = directions.word_directions("fake", layer=0, center=False)
    for i, w in enumerate(WORDS):
        np.testing.assert_allclose(T[i], SPECIFIC[w], atol=1e-2)


def test_word_directions_need_the_control_condition(monkeypatch):
    _install(monkeypatch, control=None)
    with pytest.raises(directions.MissingCondition):
        directions.word_directions("fake", layer=0, center=False)
