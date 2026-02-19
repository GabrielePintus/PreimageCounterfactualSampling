"""
Preimage Sampling Library for Counterfactual Explanation Generation

This library provides tools for:
- Neural network certification using LiRPA
- Preimage approximation via certified polytopes
- Counterfactual generation through Certified Polyhedral Projection (CPP)
- Efficient spatial indexing via Bounding Volume Hierarchies (BVH)

Quick Start
-----------
>>> from preimage_sampling import CertifiedAtlas
>>>
>>> # Build the atlas (offline phase)
>>> atlas = CertifiedAtlas(model, dataset, device='cuda')
>>> atlas.build(eps=0.1, norm=2)
>>>
>>> # Generate counterfactual (online phase)
>>> result = atlas.find_counterfactual(x_query, target_class=3)
>>> print(f"Counterfactual found: {result.success}, distance: {result.distance:.4f}")
"""

__version__ = "0.2.0"

# High-level API (recommended)
from .atlas import CertifiedAtlas, CounterfactualResult

# Lower-level components (for advanced users)
from .certification import WrappedModel, PreimageApproximation
from .sampling import CounterfactualSampler
from .indexing import BVHIndex, BVHNode

__all__ = [
    # High-level API
    "CertifiedAtlas",
    "CounterfactualResult",
    # Lower-level components
    "WrappedModel",
    "PreimageApproximation",
    "CounterfactualSampler",
    "BVHIndex",
    "BVHNode",
]
