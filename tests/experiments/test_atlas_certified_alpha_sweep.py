import numpy as np
import pytest

from experiments.atlas_certified_alpha_sweep import coverage_stages


def test_loss_partition():
    result = coverage_stages([1, 1, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0])
    assert result["initial_covered"] == 3
    assert result["final_ball_covered"] == 2
    assert result["certified_covered"] == 1
    assert result["certified_coverage"] == .25
    assert sum(result[k] for k in ("outside_initial", "lost_shrink_or_discard",
                                  "lost_affine_constraints", "certified_covered")) == 4


def test_non_nested_rejected():
    with pytest.raises(AssertionError):
        coverage_stages([0], [1], [0])
    with pytest.raises(AssertionError):
        coverage_stages([1], [0], [1])
    with pytest.raises(ValueError):
        coverage_stages([1, 0], [1], [1])


def test_empty_and_zero_hits():
    assert coverage_stages([], [], [])["certified_coverage"] is None
    assert coverage_stages(np.ones(5), np.zeros(5), np.zeros(5))["lost_shrink_or_discard"] == 5
