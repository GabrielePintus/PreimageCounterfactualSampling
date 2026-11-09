import numpy as np
import pytest

from certcf.atlas import CertCFAtlas
from experiments.atlas_coverage import membership_matrix, nested_curves, row_overlap_count


def toy_atlas(norm=1):
    atlas = CertCFAtlas.__new__(CertCFAtlas)
    atlas.norm = norm
    atlas.classification_margin = 0.1
    atlas.input_bounds = (np.array([-2., -2.]), np.array([2., 2.]))
    atlas.bounds = {0: {"X": np.array([[0., 0.], [1., 0.]]),
                        "eps": np.array([1., .5]),
                        "lA": np.array([[[1., 0.]], [[-1., 0.]]]),
                        "lbias": np.array([[.5], [1.5]])}}
    return atlas


@pytest.mark.parametrize("norm", [1, 2, np.inf])
def test_membership_matches_production(norm):
    atlas = toy_atlas(norm)
    points = np.concatenate([np.random.default_rng(2).uniform(-2, 2, (100, 2)),
                             [[0, 0], [1, 0], [-.4, 0], [-.401, 0], [0, 1], [0, 1.001]]])
    actual, balls = membership_matrix(atlas, points, 0, batch_size=7)
    expected = np.array([[atlas._polytope_membership_for_anchor(
        x, atlas.bounds[0], i, delta=0, robust_norm=norm)[0]
        for i in range(2)] for x in points])
    np.testing.assert_array_equal(actual, expected)
    assert actual.any() and (~actual).any()
    assert np.all(~actual | balls)
    assert actual[-6, 0] and actual[-5, 1]
    assert actual[-4, 0] and not actual[-3, 0]


def test_input_bounds_and_empty_region():
    atlas = toy_atlas()
    atlas.input_bounds = (np.array([0., -2.]), np.array([.2, 2.]))
    atlas.bounds[0]["eps"][1] = 0
    membership, _ = membership_matrix(atlas, [[-.1, 0], [.1, 0]], 0)
    np.testing.assert_array_equal(membership, [[False, False], [True, False]])


def test_nested_prefixes_reproducible_and_monotone():
    matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=bool)
    kwargs = dict(fractions=[.25, .5, .75, 1.], orderings=10, seed=42, label=0)
    frame, orders = nested_curves(matrix, **kwargs)
    repeated, orders2 = nested_curves(matrix, **kwargs)
    assert frame.equals(repeated)
    np.testing.assert_array_equal(orders, orders2)
    assert (frame[frame.fraction == 1].coverage == .75).all()
    for repeat, group in frame.groupby("ordering"):
        assert (np.diff(group.n_covered) >= 0).all()
        for row in group.itertuples():
            assert row.n_covered == matrix[:, orders[repeat, :row.n_regions]].any(axis=1).sum()


def test_empty_cases_and_zero_hits():
    kwargs = dict(fractions=[.1, 1.], orderings=2, seed=42, label=1)
    frame, _ = nested_curves(np.zeros((3, 0), dtype=bool), **kwargs)
    assert (frame.coverage == 0).all()
    frame, _ = nested_curves(np.zeros((0, 4), dtype=bool), **kwargs)
    assert frame.coverage.isna().all()
    frame, _ = nested_curves(np.zeros((3, 4), dtype=bool), **kwargs)
    assert (frame.coverage == 0).all()


def test_invalid_fraction_and_overlaps():
    with pytest.raises(ValueError):
        nested_curves(np.zeros((2, 3)), fractions=[1.1], orderings=2, seed=42, label=0)
    assert row_overlap_count([[1, 2], [3, 4]], [[1, 2], [5, 6]]) == 1
