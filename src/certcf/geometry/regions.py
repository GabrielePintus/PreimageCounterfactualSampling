"""Backend-independent convex regions. Linear constraints use A @ x + b >= 0.

These containers validate geometry, not the truth of a neural-network certificate.
Only trusted certification backends should construct certified atlas artifacts.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def lp_norm(value: float) -> float:
    value = float(value)
    if np.isnan(value) or value < 1:
        raise ValueError("An Lp exponent must be >= 1")
    return value


def _array(value, ndim):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError("Geometry must have the expected rank and finite coefficients")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class NormBall:
    center: np.ndarray
    radius: float
    p: float = 1

    def __post_init__(self):
        object.__setattr__(self, "center", _array(self.center, 1))
        object.__setattr__(self, "p", lp_norm(self.p))
        if not np.isfinite(self.radius) or self.radius < 0:
            raise ValueError("Ball radius must be finite and nonnegative")


@dataclass(frozen=True)
class RegionGeometry:
    A: np.ndarray
    b: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    ball: NormBall | None = None

    def __post_init__(self):
        for name, rank in (("A", 2), ("b", 1), ("lower", 1), ("upper", 1)):
            object.__setattr__(self, name, _array(getattr(self, name), rank))
        d = len(self.lower)
        if not d or self.upper.shape != (d,) or self.A.shape != (len(self.b), d):
            raise ValueError("Inconsistent region dimensions")
        if np.any(self.lower > self.upper):
            raise ValueError("Region lower bounds exceed upper bounds")
        if self.ball is not None and self.ball.center.shape != (d,):
            raise ValueError("Ball dimension does not match region")

    @property
    def dimension(self):
        return len(self.lower)

    def contains(self, points, *, tolerance=1e-6):
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Invalid membership tolerance")
        x = np.asarray(points, dtype=np.float64)
        if x.ndim not in (1, 2) or x.shape[-1] != self.dimension:
            raise ValueError("Point dimension does not match region")
        ok = np.isfinite(x).all(axis=-1)
        ok &= (x >= self.lower - tolerance).all(axis=-1)
        ok &= (x <= self.upper + tolerance).all(axis=-1)
        ok &= (x @ self.A.T + self.b >= -tolerance).all(axis=-1)
        if self.ball is not None:
            ok &= np.linalg.norm(x - self.ball.center, ord=self.ball.p, axis=-1) <= self.ball.radius + tolerance
        return ok

    def eroded(self, delta: float, p: float):
        """Erode halfspaces/box exactly; mixed-norm ball erosion is conservative."""
        p = lp_norm(p)
        if not np.isfinite(delta) or delta < 0:
            raise ValueError("Robustness radius must be finite and nonnegative")
        dual = np.inf if p == 1 else (1 if p == np.inf else p / (p - 1))
        lower, upper = self.lower + delta, self.upper - delta
        if np.any(lower > upper):
            return None
        ball = self.ball
        if ball is not None:
            factor = self.dimension ** max(0.0, 1 / ball.p - 1 / p)
            radius = ball.radius - delta * factor
            if radius < 0:
                return None
            ball = NormBall(ball.center, radius, ball.p)
        return RegionGeometry(self.A, self.b - delta * np.linalg.norm(self.A, ord=dual, axis=1), lower, upper, ball)

    def projection_parameters(self):
        """Exact bridge to the existing halfspace + enclosing cube/ball solver.

        The cube is only a redundant bound. All actual box faces are retained.
        Its center is NOT the retrieval representative.
        """
        d = self.dimension
        A = np.vstack((self.A, np.eye(d), -np.eye(d)))
        b = np.concatenate((self.b, -self.lower, self.upper))
        if self.ball is None:
            center = (self.lower + self.upper) / 2
            radius = max(float(np.max((self.upper - self.lower) / 2)), 1e-12)
            p = np.inf
        else:
            center, radius, p = self.ball.center, self.ball.radius, self.ball.p
            if radius == 0:
                A = np.vstack((A, np.eye(d), -np.eye(d)))
                b = np.concatenate((b, -center, center))
                radius, p = 1e-12, np.inf
        return A, b, center, radius, p


@dataclass(frozen=True)
class CertifiedRegion:
    region_id: str
    target_class: int
    geometry: RegionGeometry
    source_domain_id: str
    representative: np.ndarray | None = None
    certificate: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.region_id or not self.source_domain_id:
            raise ValueError("Region and source domain IDs are required")
        representative = self.representative
        if representative is None:
            representative = (self.geometry.lower + self.geometry.upper) / 2
        representative = _array(representative, 1)
        if representative.shape != (self.geometry.dimension,):
            raise ValueError("Representative dimension does not match geometry")
        object.__setattr__(self, "representative", representative)


def regions_from_legacy(bounds, norm, margin=0.0):
    """Lossless geometry conversion; legacy diagnostic arrays remain untouched."""
    regions = []
    for label, values in bounds.items():
        for i, center in enumerate(values["X"]):
            radius = float(values["eps"][i])
            center = np.asarray(center, dtype=np.float64).reshape(-1)
            geometry = RegionGeometry(
                np.asarray(values["lA"][i]).reshape(-1, len(center)),
                np.asarray(values["lbias"][i]).reshape(-1) - margin,
                center - radius, center + radius,
                NormBall(center, radius, norm),
            )
            regions.append(CertifiedRegion(
                f"{label}:{i}", int(label), geometry, f"{label}:{i}", center,
                {"backend": "local_lirpa", "classification_margin": float(margin)},
            ))
    return tuple(regions)
