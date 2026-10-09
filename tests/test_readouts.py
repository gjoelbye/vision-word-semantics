"""Neighbor readouts: chance levels, the tie rule, the top-10 boundary, the permutation null and the subsampling interval."""
import numpy as np
import pytest

from src.analysis import lexical
from src.stimuli.vocabulary import letter_overlap


def _blocked(n_per=25, n_class=4, d=32, seed=0, sep=3.0):
    """Points drawn around `n_class` random centers, and their labels.

    At the default `sep` of 3.0 purity is exactly 1.0 for every seed, so any
    interval is [1.0, 1.0]. Tests of an interval pass a smaller `sep`.
    """
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_class), n_per)
    centres = rng.normal(size=(n_class, d))
    return rng.normal(size=(len(labels), d)) + sep * centres[labels], labels


def test_chance_purity_is_the_closed_form():
    """Four balanced classes of 25 give (25-1)/(100-1)."""
    labels = np.repeat(np.arange(4), 25)
    assert lexical.chance_purity(labels) == pytest.approx(24 / 99)


def test_chance_purity_handles_unequal_classes():
    labels = np.array([0, 0, 0, 1, 1, 2])
    n = len(labels)
    expected = (3 * 2 + 2 * 1 + 1 * 0) / (n * (n - 1))
    assert lexical.chance_purity(labels) == pytest.approx(expected)


def test_restricted_chance_uses_only_admissible_candidates():
    """Two labels crossed evenly with two four-item clusters leave one
    same-label candidate among the three admissible neighbors."""
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    clusters = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert (lexical.restricted_chance_purity(labels, clusters)
            == pytest.approx(1 / 3))


def test_a_tie_counts_for_the_partner_not_against_it():
    """An argmax would break ties by lowest index, which is alphabetical order,
    and score an exact tie as a loss."""
    S = np.zeros((4, 4))
    np.fill_diagonal(S, -np.inf)
    assert lexical.partner_rank(S, 0, 3) == 0
    assert lexical.partner_rank(S, 0, 1) == 0


def test_a_within_item_tie_counts_for_neither_competitor():
    """The comparison within an item and `partner_rank` treat a tie the same
    way, so the win count and the top-10 count agree about the same item. Four
    words lie in a plane, with the anchor exactly between its synonym and its
    look-alike. The difference of cosines is zero, and an exact sign test drops
    the item rather than award it to either side."""
    import numpy as np

    anchor = np.array([1.0, 0.0])
    synonym = np.array([np.cos(0.4), np.sin(0.4)])
    twin = np.array([np.cos(0.4), -np.sin(0.4)])
    far = np.array([-1.0, 0.0])
    T = lexical.rownorm(np.stack([anchor, synonym, twin, far]))

    cosine = T @ T.T
    difference = cosine[0, 1] - cosine[0, 2]
    assert difference == pytest.approx(0.0, abs=1e-12)
    assert int(difference > 0) == 0 and int(difference < 0) == 0

    S = lexical.similarity(T)
    assert lexical.partner_rank(S, 0, 1) == lexical.partner_rank(S, 0, 2) == 0


def test_partner_rank_counts_only_strictly_better_words():
    S = np.array([[-np.inf, 0.1, 0.9, 0.5]])
    S = np.vstack([S, np.zeros((3, 4))])
    assert lexical.partner_rank(S, 0, 1) == 2      # 0.9 and 0.5 beat it
    assert lexical.partner_rank(S, 0, 2) == 0      # nothing beats it


def test_within_ten_admits_nine_words_ahead_and_not_ten():
    """A partner beaten by exactly nine words is among the ten nearest and
    one beaten by ten is not, so the boundary is `rank < 10` on the
    zero-based rank and not `rank <= 10`. A tie at tenth place counts for
    the partner, as every tie here does."""
    n = 12
    S = np.zeros((n, n))
    np.fill_diagonal(S, -np.inf)
    S[0, 1] = 0.5                                   # nine words ahead
    S[0, 2:11] = 0.9
    S[2, 3] = 0.5                                   # ten words ahead
    S[2, [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]] = 0.9
    S[4, 5] = 0.5                                   # nine ahead, one tied
    S[4, [0, 1, 2, 3, 6, 7, 8, 9, 10]] = 0.9
    S[4, 11] = 0.5
    words = [f"w{i}" for i in range(n)]
    index = {w: i for i, w in enumerate(words)}
    rows = lexical.retrieval(S, index,
                             [("w0", "w1"), ("w2", "w3"), ("w4", "w5")],
                             "test")
    fwd = {r["direction"]: r for r in rows}["forward"]
    assert fwd["hits"] == 0
    assert fwd["hits10"] == 2
    assert fwd["rate10"] == pytest.approx(2 / 3)
    assert fwd["chance10"] == pytest.approx(10 / 11)


def test_the_permutation_null_lands_on_chance():
    T, labels = _blocked()
    S = lexical.similarity(T)
    out = lexical.purity_permutation(S, labels, n_perm=200)
    assert out["purity"] > 0.9
    assert out["null_mean"] == pytest.approx(out["chance"], abs=0.03)
    assert out["p"] < 0.01


def test_the_p_value_cannot_be_exactly_zero():
    """A finite number of shuffles cannot support p = 0, so the
    (count+1)/(n+1) correction floors it at 1/(n+1), here 1/201."""
    T, labels = _blocked()
    out = lexical.purity_permutation(lexical.similarity(T), labels, n_perm=200)
    assert out["p"] >= 1 / 201


def test_purity_interval_brackets_its_own_point_estimate():
    """The interval contains its own point estimate. A bootstrap would place it
    below the estimate by duplicating words, and an uncorrected subsample would
    too, because a smaller pool lowers purity."""
    for seed in range(6):
        T, labels = _blocked(seed=seed, sep=0.45, n_per=30)
        S = lexical.similarity(T)
        point = lexical.knn_purity(S, labels)
        out = lexical.purity_subsample(S, labels, n=200)
        lo, hi = out["ci_lo"], out["ci_hi"]
        assert lo <= point <= hi, f"seed {seed}: {point} outside [{lo}, {hi}]"


def test_purity_interval_is_wide_enough_to_cover_the_truth():
    """An uncorrected subsample of 80% varies about half as much as the
    estimate does, so its percentiles give an interval half the right width.
    Across 20 samples of words from one fixed population, the reported
    half-width reaches at least 0.6 of 1.96 standard deviations of the
    estimate."""
    rng = np.random.default_rng(0)
    n_class, d, sep, pop_n = 4, 64, 0.45, 4000
    centres = rng.normal(size=(n_class, d))
    pop_labels = np.repeat(np.arange(n_class), pop_n // n_class)
    pop = rng.normal(size=(pop_n, d)) + sep * centres[pop_labels]

    def draw(seed):
        r = np.random.default_rng(100 + seed)
        idx = np.concatenate([r.choice(np.where(pop_labels == c)[0], 30,
                                       replace=False) for c in range(n_class)])
        return pop[idx], pop_labels[idx]

    points = [lexical.knn_purity(lexical.similarity(*draw(s)[:1]), draw(s)[1])
              for s in range(20)]
    spread = 1.96 * np.std(points, ddof=1)
    T, labels = draw(0)
    out = lexical.purity_subsample(lexical.similarity(T), labels, n=200)
    lo, hi = out["ci_lo"], out["ci_hi"]
    assert (hi - lo) / 2 >= 0.6 * spread, \
        f"half-width {(hi - lo) / 2:.4f} against 1.96 SD {spread:.4f}"


def test_subsampling_never_repeats_a_word():
    from src.stats import subsample_indices
    for idx in list(subsample_indices(50, draws=20)):
        assert len(set(idx)) == len(idx)


def test_restricting_to_a_cluster_changes_who_the_neighbours_are():
    """The letter-cluster control. With the restriction on, a word can only be
    scored against words in its own cluster."""
    T, labels = _blocked()
    clusters = np.tile(np.arange(4), len(labels) // 4)
    S = lexical.similarity(T)
    free = lexical.knn_purity(S, labels)
    tied = lexical.knn_purity(S, labels, restrict=clusters)
    assert free != tied


def test_public_neighbour_indices_obey_the_restriction():
    T, _ = _blocked(n_per=6, n_class=4)
    clusters = np.repeat(np.arange(4), 6)
    nn = lexical.neighbour_indices(lexical.similarity(T), k=3,
                                   restrict=clusters)
    assert np.all(clusters[nn] == clusters[:, None])


def test_a_restriction_too_tight_for_k_is_an_error_not_a_wrong_number():
    """Without the error, a hard top-k would fill the missing places with -inf
    rows in index order, including the word itself, and return a purity that
    partly counts words the restriction excludes."""
    T, labels = _blocked(n_per=5, n_class=4)
    clusters = np.arange(len(labels)) % 10        # clusters of two
    with pytest.raises(ValueError, match="admissible neighbours"):
        lexical.knn_purity(lexical.similarity(T), labels, k=10,
                           restrict=clusters)


def test_retrieval_reports_each_direction_with_its_own_denominator():
    """`retrieval` returns one row per query direction, each counted over its own pairs."""
    T, labels = _blocked(n_per=10, n_class=2)
    words = [f"w{i}" for i in range(len(labels))]
    index = {w: i for i, w in enumerate(words)}
    pairs = [("w0", "w1"), ("w2", "w3")]
    rows = lexical.retrieval(lexical.similarity(T), index, pairs, "test")
    by = {r["direction"]: r for r in rows}
    assert set(by) == {"forward", "backward"}
    assert by["forward"]["n"] == by["backward"]["n"] == 2
    assert by["forward"]["chance10"] == pytest.approx(10 / (len(words) - 1))


def test_letter_overlap_compares_against_random_pairs():
    vocab = ["france", "paris", "japan", "tokyo", "lemon", "shirt"]
    same = letter_overlap([("lemon", "lemon")], vocab, n_random=200)
    assert same["within"] == pytest.approx(1.0)
    assert same["excess"] > 0
