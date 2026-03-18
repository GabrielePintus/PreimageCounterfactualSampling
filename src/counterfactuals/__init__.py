"""Research-grade modular framework for counterfactual benchmarking."""

from counterfactuals.core.base_classes import (
    BaseCounterfactualMethod,
    CounterfactualResult,
)
from counterfactuals.experiments.runner import run_experiment, run_from_config_path

__all__ = [
    "BaseCounterfactualMethod",
    "CounterfactualResult",
    "run_experiment",
    "run_from_config_path",
]
