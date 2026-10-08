"""The paired difference subtracts from each word image the scrambled control
of the same word and slot. An indexing error leaves no trace in the result, so
the difference is checked on synthetic features with a known answer."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import directions
from src.stimuli.render import CLEAN

N_SLOTS, D = 4, 3
WORDS = ["alpha", "beta"]
# Synthetic features: each background contributes its own vector, the word
# adds its effect on top, and the scrambled word adds its own ink. The paired
# difference recovers the word effect minus the ink.
BG = np.arange(N_SLOTS * D, dtype=np.float32).reshape(N_SLOTS, D)
WORD_EFFECT = {"alpha": np.array([1.0, -2.0, 0.5], np.float32),
               "beta": np.array([-3.0, 0.25, 4.0], np.float32)}
INK = {"alpha": np.array([0.5, 0.5, 0.0], np.float32),
       "beta": np.array([0.0, -1.0, 1.0], np.float32)}
EXPECTED = {w: WORD_EFFECT[w] - INK[w] for w in WORDS}


def _rows() -> list[dict]:
    rows = [dict(word=CLEAN, slot=s, condition="clean", image_path=f"/bg/{s}")
            for s in range(N_SLOTS)]
    for condition in ("word", "scramble"):
        rows += [dict(word=w, slot=s, condition=condition,
                      image_path=f"/{condition}/{w}/{s}")
                 for w in WORDS for s in range(N_SLOTS)]
    return rows


def _features(rows: list[dict]) -> np.ndarray:
    feats = []
    for r in rows:
        v = BG[r["slot"]].copy()
        if r["condition"] == "word":
            v = v + WORD_EFFECT[r["word"]]
        elif r["condition"] == "scramble":
            v = v + INK[r["word"]]
        feats.append(v)
    return np.stack(feats)[:, None, :].astype(np.float16)      # one layer


def _install(mp, man: pd.DataFrame, x: np.ndarray) -> None:
    mp.setattr(directions, "manifest", lambda ds=None: man)
    mp.setattr(directions, "load_cache",
               lambda key, *, ds=None: (x, np.array([0]), 1))
    mp.setattr(directions, "word_order", lambda ds=None: tuple(WORDS))


@pytest.fixture
def synthetic(monkeypatch):
    rows = _rows()
    man = pd.DataFrame(rows)
    _install(monkeypatch, man, _features(rows))
    return man


def test_paired_difference_recovers_the_word_effect(synthetic):
    t = directions.residual_tau("fake", layer=0)
    for i, w in enumerate(WORDS):
        np.testing.assert_allclose(t[i], EXPECTED[w], atol=1e-2)


def test_control_is_paired_by_word_and_slot_not_by_row_position(synthetic):
    """Word and scramble rows are paired by their (word, slot) key. Rows that
    arrive interleaved still pair correctly, and a control whose rows arrive in
    a different order from the word rows is refused rather than subtracted
    from the wrong slot."""
    man = synthetic
    word = man[man.condition.eq("word")]
    scramble = man[man.condition.eq("scramble")]
    clean = man[man.condition.eq("clean")]
    interleaved = pd.concat([word.iloc[::2], scramble.iloc[::2],
                             clean.iloc[::-1], word.iloc[1::2],
                             scramble.iloc[1::2]]).reset_index(drop=True)
    with pytest.MonkeyPatch.context() as mp:
        _install(mp, interleaved, _features(interleaved.to_dict("records")))
        t = directions.residual_tau("fake", layer=0)
    for i, w in enumerate(WORDS):
        np.testing.assert_allclose(t[i], EXPECTED[w], atol=1e-2)

    mispaired = pd.concat([clean, word, scramble.iloc[::-1]]
                          ).reset_index(drop=True)
    with pytest.MonkeyPatch.context() as mp:
        _install(mp, mispaired, _features(mispaired.to_dict("records")))
        with pytest.raises(directions.MissingCondition, match="not paired"):
            directions.residual_tau("fake", layer=0)


def test_slot_subset_still_recovers_it(synthetic):
    t = directions.residual_tau("fake", layer=0, slots=(0, 2))
    np.testing.assert_allclose(t[0], EXPECTED["alpha"], atol=1e-2)
    np.testing.assert_allclose(t[1], EXPECTED["beta"], atol=1e-2)


def test_degenerate_layer_raises_instead_of_scoring_zero(monkeypatch):
    """At hidden-state index 0 the class token is the learned initial
    parameter, identical for a word and its scrambled control, so the paired
    difference is exactly zero. Scoring it would report an accuracy of 0.000
    where the value is undefined."""
    rows = _rows()
    const = np.ones((len(rows), 1, D), np.float16)      # same vector everywhere
    _install(monkeypatch, pd.DataFrame(rows), const)
    with pytest.raises(directions.DegenerateTau):
        directions.residual_tau("fake", layer=0)


def test_the_untrained_floor_is_reproducible():
    """The untrained ViT-B is built from a fixed seed, so two loads return the
    same weights. Unseeded, ``AutoModel.from_config`` would build a different
    network on every call, and the untrained baseline would change between
    runs."""
    from src.models import load

    first = next(load("vit-random-b", "cpu")[0].parameters())
    second = next(load("vit-random-b", "cpu")[0].parameters())
    assert np.array_equal(first.detach().numpy(), second.detach().numpy())
