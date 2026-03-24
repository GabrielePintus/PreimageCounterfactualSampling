"""Preprocessing primitives for shared benchmark representation spaces."""

from .transforms import (
    IdentityTransform,
    InverseTransformModel,
    OHEBlockSpec,
    PCATransform,
    RepresentationTransform,
    adult_ohe_blocks,
    compas_ohe_blocks,
    german_credit_ohe_blocks,
    snap_ohe_blocks,
)

__all__ = [
    "RepresentationTransform",
    "IdentityTransform",
    "PCATransform",
    "InverseTransformModel",
    "OHEBlockSpec",
    "snap_ohe_blocks",
    "adult_ohe_blocks",
    "compas_ohe_blocks",
    "german_credit_ohe_blocks",
]
