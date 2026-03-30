"""Redundancy metric for counterfactual explanations."""

from __future__ import annotations

import numpy as np

from counterfactuals.core.interfaces import MetricInterface


class RedundancyMetric(MetricInterface):
    """Fraction of changed features that are unnecessary.

    For each feature that differs between *x_orig* and *x_cf*, the feature is
    temporarily reverted to its original value and the classifier is queried.
    If the CF is still valid (target class predicted), the change was redundant.

    ``redundancy = n_redundant_changes / n_changed_features``

    Returns 0.0 when no features are changed.

    Requires ``context`` to contain:
    - ``target_class`` (int): the desired prediction class.
    - ``predict_fn`` (callable): maps a 2-D array (n, d) → 1-D int predictions.
    """

    name = "redundancy"

    def __init__(self, atol: float = 1e-6) -> None:
        self.atol = atol

    def evaluate(self, x_orig: np.ndarray, x_cf: np.ndarray, context=None) -> float:
        if context is None or "target_class" not in context or "predict_fn" not in context:
            raise ValueError(
                "RedundancyMetric requires context with 'target_class' and 'predict_fn'."
            )
        target_class = int(context["target_class"])
        predict_fn = context["predict_fn"]  # callable: (n, d) → (n,) int array

        x_orig = np.asarray(x_orig, dtype=np.float32)
        x_cf = np.asarray(x_cf, dtype=np.float32)

        changed = np.where(np.abs(x_cf - x_orig) > self.atol)[0]
        if len(changed) == 0:
            return 0.0

        n_redundant = 0
        for k in changed:
            x_test = x_cf.copy()
            x_test[k] = x_orig[k]
            if int(predict_fn(x_test[None, :])[0]) == target_class:
                n_redundant += 1

        return float(n_redundant / len(changed))
