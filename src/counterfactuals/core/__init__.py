"""Core APIs for the counterfactual framework."""

from .base_classes import BaseCounterfactualMethod, CounterfactualResult
from .interfaces import DatasetInterface, MetricInterface, ModelInterface
from .registry import Registry

__all__ = [
    "BaseCounterfactualMethod",
    "CounterfactualResult",
    "DatasetInterface",
    "MetricInterface",
    "ModelInterface",
    "Registry",
]
