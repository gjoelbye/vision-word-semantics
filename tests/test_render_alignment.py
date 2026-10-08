"""An embedding cache is bound to its manifest by row position, so the manifest
order must be fixed and the fingerprint must see any reordering. The recorded
text centers of the slots and the resolution of recorded paths are checked
here too."""
import pandas as pd

from src.stimuli.render import CLEAN, fingerprint


def _manifest(words, slots):
    rows = [dict(word=CLEAN, slot=s, image_path=f"/bg/{s}.png") for s in slots]
    rows += [dict(word=w, slot=s, image_path=f"/r/{w}/{s}.jpg")
             for w in words for s in slots]
    man = pd.DataFrame(rows)
    man["_clean"] = (man.word == CLEAN).astype(int)
    return (man.sort_values(["_clean", "word", "slot"],
                            ascending=[False, True, True], kind="mergesort")
               .drop(columns="_clean").reset_index(drop=True))


def test_order_is_canonical_regardless_of_arrival_order():
    a = _manifest(["beta", "alpha"], [1, 0])
    b = _manifest(["alpha", "beta"], [0, 1])
    pd.testing.assert_frame_equal(a, b)


def test_clean_rows_come_first():
    m = _manifest(["alpha"], [0, 1])
    assert (m.head(2).word == CLEAN).all()


def test_fingerprint_changes_when_order_changes():
    m = _manifest(["alpha", "beta"], [0, 1])
    shuffled = m.iloc[::-1].reset_index(drop=True)
    assert fingerprint(m) != fingerprint(shuffled)


def test_fingerprint_is_stable():
    m = _manifest(["alpha", "beta"], [0, 1])
    assert fingerprint(m) == fingerprint(m.copy())


def test_the_centre_table_is_recorded_and_reused(tmp_path, monkeypatch):
    """The text centers depend on the widest word, so they are recorded once and
    a wider vocabulary that still fits reuses them."""
    from src.stimuli import render as R

    path = tmp_path / "centres.parquet"
    words = ("cat", "lemon", "structure")
    first = R.slot_centres(words, path=path)
    assert path.exists()
    assert R.slot_centres(words, path=path) == first
    assert R.slot_centres(words + ("management",), path=path) == first


def test_a_word_that_cannot_fit_the_recorded_centre_is_refused(tmp_path):
    """A vocabulary the recorded table cannot hold raises rather than moving
    existing renders."""
    import pytest

    from src.stimuli import render as R

    path = tmp_path / "centres.parquet"
    R.slot_centres(("cat", "lemon"), path=path)
    with pytest.raises(ValueError, match="does not fit at the recorded centre"):
        R.slot_centres(("cat", "mmmmmmmmmm"), path=path)


def test_a_recorded_path_resolves_under_this_machines_data_root(monkeypatch):
    """A recorded absolute prefix is replaced by this machine's data root."""
    from pathlib import Path

    from src import config as C

    monkeypatch.setattr(C, "DATA_ROOT", Path("/elsewhere/data"))
    got = C.local_path("/whatever/was/here/renders/zynga/095.png")
    assert got == Path("/elsewhere/data/renders/zynga/095.png")


def test_the_split_takes_the_leftmost_data_directory():
    """A word directory named after a data directory is a word directory."""
    from src.config import data_relative

    got = data_relative("/any/root/renders/results/095.png")
    assert str(got) == "renders/results/095.png"


def test_a_path_below_no_data_directory_is_refused():
    """A path with no data directory in it raises."""
    import pytest

    from src.config import data_relative

    with pytest.raises(ValueError, match="no known data directory"):
        data_relative("/tmp/loose/095.png")
