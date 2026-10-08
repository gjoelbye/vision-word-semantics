"""The matched-appearance split behind the split-half cosine varies the background and nothing else."""
from src import slots as S
from src.typography import FACTORS, build_appearance_design

#: Appearance tuples for the subset case, as a measurement on fewer images draws them.
SUBSET = (0, 5, 11, 17)


def test_slot_count_follows_the_appearance_design():
    """96 slots is 24 appearance tuples on 4 backgrounds each. The background
    selection reads this number rather than a count of its own, so the two
    agree."""
    assert S.REPLICATES % 2 == 0, "the split needs even replicates"
    assert S.n_slots() == len(build_appearance_design()) * S.REPLICATES
    assert S.n_slots() == 96


def test_every_slot_has_its_own_background():
    t = S.slot_table()
    assert t.background_index.nunique() == len(t)


def test_split_halves_is_appearance_matched():
    a, b = S.split_halves()
    t = S.slot_table().set_index("slot")
    left = t.loc[list(a), list(FACTORS)].sort_values(list(FACTORS)).reset_index(drop=True)
    right = t.loc[list(b), list(FACTORS)].sort_values(list(FACTORS)).reset_index(drop=True)
    assert left.equals(right)


def test_split_halves_uses_every_slot_once():
    """At four replicates, a split of replicate 0 against replicate 1 would
    silently discard half the data while still returning two valid-looking
    halves."""
    a, b = S.split_halves()
    t = S.slot_table()
    assert len(a) == len(b) == len(t) // 2
    assert set(a) | set(b) == set(t.slot)
    assert not set(a) & set(b)


def test_each_half_carries_every_appearance_the_same_number_of_times():
    a, b = S.split_halves()
    t = S.slot_table().set_index("slot")
    ca = t.loc[list(a), "appearance"].value_counts()
    cb = t.loc[list(b), "appearance"].value_counts()
    assert ca.equals(cb.reindex(ca.index))
    assert ca.nunique() == 1


def test_split_halves_is_not_confounded_with_box_style():
    """Both halves carry every box style and every font. A split with every
    paneled slot on one side would compare panel against no panel, and the
    split-half cosine would then reflect the translucent rectangle as well as
    the word."""
    a, b = S.split_halves()
    t = S.slot_table().set_index("slot")
    assert set(t.loc[list(a), "box"]) == set(t.loc[list(b), "box"])
    assert set(t.loc[list(a), "font"]) == set(t.loc[list(b), "font"])


def test_split_halves_restricted_to_an_appearance_subset():
    """Restricted to some appearance tuples, each half holds exactly those
    tuples, each twice, and together the halves hold every slot they are drawn
    on and nothing else."""
    a, b = S.split_halves(SUBSET)
    t = S.slot_table().set_index("slot")
    assert set(a) | set(b) == set(S.slots_for(SUBSET))
    assert not set(a) & set(b)
    assert len(a) == len(b) == len(SUBSET) * S.REPLICATES // 2
    for half in (a, b):
        counts = t.loc[list(half), "appearance"].value_counts()
        assert set(counts.index) == set(SUBSET)
        assert set(counts) == {S.REPLICATES // 2}
    assert (t.loc[list(a), "replicate"] < S.REPLICATES // 2).all()
    assert (t.loc[list(b), "replicate"] >= S.REPLICATES // 2).all()
    assert list(a) == sorted(a) and list(b) == sorted(b)
