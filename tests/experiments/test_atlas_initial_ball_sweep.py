import numpy as np
import pytest

from experiments.atlas_initial_ball_sweep import entry_thresholds


def test_entry_threshold_matches_union_not_nearest_anchor():
    # The farther anchor can offer the first entry because its radius grows faster.
    distances = np.array([[2., 3.], [4., 8.]])
    clearance = np.array([2., 6.])
    threshold = entry_thresholds(distances, clearance)
    np.testing.assert_allclose(threshold, [.5, 8 / 6])
    for alpha in [.2, .5, .7, 1., 2.]:
        np.testing.assert_array_equal(threshold <= alpha,
                                      (distances <= alpha * clearance).any(axis=1))


def test_zero_clearance_and_empty_regions():
    np.testing.assert_array_equal(entry_thresholds([[0.], [1.]], [0.]), [0., np.inf])
    assert np.isinf(entry_thresholds(np.empty((3, 0)), [])).all()
    assert entry_thresholds(np.empty((0, 2)), [1., 2.]).size == 0


def test_tolerance_and_invalid_inputs():
    assert entry_thresholds([[1.000001]], [2.], tolerance=.000002)[0] < .5
    with pytest.raises(ValueError):
        entry_thresholds([[1.]], [-2.])
    with pytest.raises(ValueError):
        entry_thresholds([[float("nan")]], [2.])
