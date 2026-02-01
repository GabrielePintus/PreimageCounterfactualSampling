"""
Preimage Sampling Library for Counterfactual Explanation Generation

This library provides tools for:
- Neural network certification using LiRPA
- Preimage approximation via certified polytopes
- Counterfactual generation through Certified Polyhedral Projection (CPP)
"""

__version__ = "0.1.0"

from .models import SimpleClassifier, MNISTClassifier
from .certification import WrappedModel, PreimageApproximation
from .sampling import CounterfactualSampler

__all__ = [
    "SimpleClassifier",
    "WrappedModel",
    "PreimageApproximation",
    "CounterfactualSampler",
]
