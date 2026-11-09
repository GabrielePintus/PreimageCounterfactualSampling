"""LiRPA-based certification and bound computation."""

from .wrapping import WrappedModel
from .bounds import get_lower_bound, get_upper_bound
from .backend import BuildRequest, BuildResult, CertificationBackend, InputDomain


def __getattr__(name):
    if name in {"run_lirpa", "PreimageApproximation"}:
        from . import lirpa
        return getattr(lirpa, name)
    raise AttributeError(name)

__all__ = [
    "WrappedModel",
    "get_lower_bound",
    "get_upper_bound",
    "run_lirpa",
    "PreimageApproximation",
    "BuildRequest", "BuildResult", "CertificationBackend", "InputDomain",
]
