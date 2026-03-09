"""Validity metric for counterfactual explanations."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from counterfactuals.core.interfaces import MetricInterface, ModelInterface


class ValidityMetric(MetricInterface):
    """Returns 1.0 when the counterfactual reaches the target class, else 0.0."""

    name = "validity"

    def __init__(self, model: ModelInterface):
        self.model = model

    def evaluate(
        self,
        x_orig: np.ndarray,
        x_cf: np.ndarray,
        context: Optional[Dict[str, np.ndarray]] = None,
    ) -> float:
        del x_orig
        if context is None or "target_class" not in context:
            raise ValueError("ValidityMetric requires context['target_class']")
        target_class = int(context["target_class"])
        pred = int(self.model.predict(np.asarray(x_cf, dtype=np.float32))[0])
        return 1.0 if pred == target_class else 0.0
