import numpy as np
import pytest
from scipy.optimize import linprog

from experiments.atlas_nonempty_policy import binary_l1_feasibility


def test_excluded_anchor_but_nonempty_region():
    good, witness, center, maximum = binary_l1_feasibility([[0., 0.]], [1.], [[[2., 0.]]], [[-1.]], .1)
    assert good[0] and center[0] < 0 and maximum[0] > 0
    assert 2 * witness[0, 0] - 1 >= .1
    assert np.abs(witness[0]).sum() <= 1


def test_empty_and_degenerate():
    good, _, _, _ = binary_l1_feasibility([[0.], [0.], [0.]], [1., 1., 1.],
        [[[1.]], [[0.]], [[1.]]], [[-2.], [1.], [-1.]], 0.)
    np.testing.assert_array_equal(good, [False, True, True])


def test_against_linear_program():
    rng = np.random.default_rng(42)
    n, d = 30, 3
    centers, a = rng.normal(size=(n, d)), rng.normal(size=(n, 1, d))
    radii, bias = rng.uniform(.1, 2, n), rng.normal(size=(n, 1))
    actual, _, _, _ = binary_l1_feasibility(centers, radii, a, bias, .1)
    for i in range(n):
        # x = center + u - v, u,v >= 0, sum(u+v) <= radius.
        constraints = np.stack([np.ones(2*d), np.r_[-a[i, 0], a[i, 0]]])
        limits = [radii[i], float(a[i, 0] @ centers[i] + bias[i, 0] - .1)]
        solved = linprog(np.zeros(2*d), A_ub=constraints, b_ub=limits, bounds=(0, None), method="highs")
        assert actual[i] == solved.success


def test_reject_multiclass():
    with pytest.raises(ValueError):
        binary_l1_feasibility([[0., 0.]], [1.], [[[1., 0.], [0., 1.]]], [[0., 0.]], .1)
