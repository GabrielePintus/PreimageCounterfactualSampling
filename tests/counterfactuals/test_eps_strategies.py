import numpy as np

from certcf.eps_strategies import ConstantEpsStrategy, NearestOppositeClassClearanceStrategy


def test_constant_eps_strategy_ignores_norm():
    strategy = ConstantEpsStrategy(0.3)
    X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y = np.array([0, 1], dtype=np.int64)

    eps = strategy.compute_eps(X, y, norm=1)

    assert np.allclose(eps, np.array([0.3, 0.3], dtype=np.float64))


def test_nearest_opposite_class_clearance_strategy_uses_l1_norm():
    strategy = NearestOppositeClassClearanceStrategy(alpha=0.5)
    X = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, 1.0],
            [4.0, 0.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1], dtype=np.int64)

    eps = strategy.compute_eps(X, y, norm=1)

    expected = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    assert np.allclose(eps, expected)


def test_nearest_opposite_class_clearance_strategy_uses_linf_norm_by_default():
    strategy = NearestOppositeClassClearanceStrategy(alpha=0.5)
    X = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, 1.0],
            [4.0, 0.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1], dtype=np.int64)

    eps = strategy.compute_eps(X, y)

    expected = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float64)
    assert np.allclose(eps, expected)


def test_nearest_opposite_class_clearance_strategy_uses_l2_norm():
    strategy = NearestOppositeClassClearanceStrategy(alpha=1.0)
    X = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, 1.0],
            [4.0, 0.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1], dtype=np.int64)

    eps = strategy.compute_eps(X, y, norm=2)

    expected = np.array(
        [
            np.sqrt(2.0),
            np.sqrt(2.0),
            np.sqrt(2.0),
            2.0,
        ],
        dtype=np.float64,
    )
    assert np.allclose(eps, expected)
