import numpy as np

from counterfactuals.metrics.plausibility import KNNPlausibility
from counterfactuals.metrics.proximity import L1Proximity, L2Proximity
from counterfactuals.metrics.sparsity import SparsityMetric


def test_basic_metrics_values():
    x_orig = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    x_cf = np.array([1.0, 0.0, 1.0], dtype=np.float32)

    assert L2Proximity().evaluate(x_orig, x_cf) == 1.0
    assert L1Proximity().evaluate(x_orig, x_cf) == 1.0
    assert SparsityMetric().evaluate(x_orig, x_cf) == (1.0 / 3.0)


def test_plausibility_runs():
    x_train = np.array([[0.0, 0.0], [1.0, 1.0], [0.8, 0.9], [-1.0, -1.0], [-0.8, -0.9]], dtype=np.float32)
    metric = KNNPlausibility(x_train=x_train, n_neighbors=3)
    score = metric.evaluate(x_orig=x_train[0], x_cf=np.array([0.9, 1.1], dtype=np.float32))
    assert score >= 0.0
