"""Contracts for offline certification strategies (no LiRPA dependency)."""
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from certcf.geometry.regions import CertifiedRegion, RegionGeometry


@dataclass(frozen=True)
class InputDomain:
    domain_id: str
    target_class: int
    geometry: RegionGeometry


@dataclass(frozen=True)
class BuildRequest:
    model: Any
    domains: tuple[InputDomain, ...]
    class_labels: tuple[int, ...]
    input_shape: tuple[int, ...]
    classification_margin: float = 0.0

    def __post_init__(self):
        import numpy as np
        if not self.class_labels or len(set(self.class_labels)) != len(self.class_labels):
            raise ValueError("Class labels must be nonempty and unique, in model-output order")
        if not self.input_shape or any(d <= 0 for d in self.input_shape):
            raise ValueError("A positive model input shape is required")
        if not np.isfinite(self.classification_margin) or self.classification_margin < 0:
            raise ValueError("Invalid classification margin")
        if len({d.domain_id for d in self.domains}) != len(self.domains):
            raise ValueError("Duplicate domain IDs")
        for domain in self.domains:
            if not domain.domain_id or domain.target_class not in self.class_labels:
                raise ValueError("Invalid domain ID or target class")
            if domain.geometry.dimension != int(np.prod(self.input_shape)):
                raise ValueError("Input domain dimension mismatch")


@dataclass(frozen=True)
class BuildResult:
    regions: tuple[CertifiedRegion, ...]
    class_labels: tuple[int, ...]
    input_shape: tuple[int, ...]
    status: Literal["complete", "budget_exhausted", "error"] = "complete"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        import numpy as np
        if self.status not in ("complete", "budget_exhausted", "error"):
            raise ValueError("Invalid build status")
        if not self.class_labels or len(set(self.class_labels)) != len(self.class_labels):
            raise ValueError("Invalid class labels")
        if not self.input_shape or any(d <= 0 for d in self.input_shape):
            raise ValueError("Invalid input shape")
        if len({r.region_id for r in self.regions}) != len(self.regions):
            raise ValueError("Duplicate region IDs")
        for region in self.regions:
            if region.target_class not in self.class_labels or region.geometry.dimension != int(np.prod(self.input_shape)):
                raise ValueError("Region does not match build metadata")


class CertificationBackend(Protocol):
    def build(self, request: BuildRequest) -> BuildResult:
        """Return sound regions; incomplete construction is not an emptiness proof."""
        ...
