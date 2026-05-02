"""Growing Spheres baseline for model-agnostic counterfactual search.

Laugel et al. (2017): "Inverse Classification for Comparison-based
Interpretability in Machine Learning". FUZZ-IEEE 2017.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from counterfactuals.core.base_classes import BaseCounterfactualMethod, CounterfactualResult
from counterfactuals.core.interfaces import ModelInterface  # noqa: F401 – used by type annotation in __init__
from counterfactuals.methods._constraints import (
    apply_directional_constraints,
    constraint_metadata,
    normalize_directional_dims,
    normalize_fixed_dims,
    validate_disjoint_directional_dims,
)


class GrowingSpheresMethod(BaseCounterfactualMethod):
    """Growing Spheres with paper-style enemy search + feature selection."""

    def __init__(
        self,
        model: ModelInterface,
        n_in_layer: int = 512,
        max_radius: float = 3.0,
        radius_step: float = 0.15,
        norm: int | float | str = 2,
        fixed_dims: Optional[Sequence[int]] = None,
        immutable_features: Optional[Sequence[str]] = None,
        nondecreasing_dims: Optional[Sequence[int]] = None,
        nonincreasing_dims: Optional[Sequence[int]] = None,
        nondecreasing_features: Optional[Sequence[str]] = None,
        nonincreasing_features: Optional[Sequence[str]] = None,
        random_seed: int = 42,
    ):
        super().__init__(model=model, random_seed=random_seed)
        self.n_in_layer = n_in_layer
        self.max_radius = max_radius
        self.radius_step = radius_step
        self.norm = self._normalize_norm(norm)
        self.fixed_dims = normalize_fixed_dims(fixed_dims)
        self.immutable_features = tuple(str(name) for name in (immutable_features or ()))
        self.nondecreasing_dims = normalize_directional_dims(
            nondecreasing_dims,
            name="nondecreasing_dims",
        )
        self.nonincreasing_dims = normalize_directional_dims(
            nonincreasing_dims,
            name="nonincreasing_dims",
        )
        validate_disjoint_directional_dims(self.nondecreasing_dims, self.nonincreasing_dims)
        self.nondecreasing_features = tuple(str(name) for name in (nondecreasing_features or ()))
        self.nonincreasing_features = tuple(str(name) for name in (nonincreasing_features or ()))
        self._feature_scale: Optional[np.ndarray] = None

    @staticmethod
    def _normalize_norm(norm: int | float | str) -> int | float:
        if isinstance(norm, str):
            if norm.lower() in {"inf", "infinity"}:
                return np.inf
            return float(norm)
        return norm

    def _lp_distance(self, x: np.ndarray, y: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        return np.linalg.norm(x - y, ord=self.norm, axis=axis)

    def _fit(self) -> None:
        assert self._x_train is not None
        std = np.std(self._x_train, axis=0)
        std[std == 0.0] = 1.0
        # The spherical search samples isotropic directions, then rescales them
        # feature-wise so exploration respects observed feature dispersion.
        self._feature_scale = std

    def generate(self, x: np.ndarray, target_class: Optional[int] = None) -> CounterfactualResult:
        if not self._is_fitted or self._feature_scale is None:
            raise RuntimeError("Method is not fitted. Call fit() before generate().")

        x_query = np.asarray(x, dtype=np.float32).reshape(-1)
        target_class = self.resolve_target_class(x=x_query, target_class=target_class)

        radius = self.radius_step
        best = None
        best_dist = float("inf")

        # Expand the search shell until we hit the first layer containing at least
        # one valid "enemy" point from the target class.
        while radius <= self.max_radius:
            inner_radius = max(0.0, radius - self.radius_step)
            candidates = self._sample_layer(x0=x_query, inner_radius=inner_radius, outer_radius=radius)
            preds = self.model.predict(candidates)
            valid = candidates[preds == target_class]
            if len(valid) > 0:
                # Among the first valid shell, keep the closest enemy and then apply
                # Growing Spheres' post-hoc feature selection step for sparsity.
                dists = self._lp_distance(valid, x_query[None, :], axis=1)
                idx = int(np.argmin(dists))
                enemy = valid[idx]
                best = self._feature_selection(
                    x0=x_query,
                    enemy=enemy,
                    target_class=target_class,
                )
                best_dist = float(self._lp_distance(best, x_query))
                break
            radius += self.radius_step

        if best is None:
            return CounterfactualResult(
                x_cf=x_query.copy(),
                success=False,
                distance=0.0,
                metadata={
                    "target_class": target_class,
                    "searched_radius": self.max_radius,
                    **self._constraint_metadata(),
                },
            )

        best = self._apply_constraints(best, x_query)
        best_dist = float(self._lp_distance(best, x_query))

        return CounterfactualResult(
            x_cf=best.astype(np.float32),
            success=True,
            distance=best_dist,
            metadata={
                "target_class": target_class,
                "searched_radius": radius,
                "norm": self.norm,
                **self._constraint_metadata(),
            },
        )

    def _apply_fixed_dims(self, x: np.ndarray, x0: np.ndarray) -> np.ndarray:
        if self.fixed_dims is None:
            return x
        constrained = np.asarray(x).copy()
        constrained[..., self.fixed_dims] = np.asarray(x0)[..., self.fixed_dims]
        return constrained

    def _apply_constraints(self, x: np.ndarray, x0: np.ndarray) -> np.ndarray:
        constrained = self._apply_fixed_dims(x, x0)
        return apply_directional_constraints(
            constrained,
            x0,
            self.nondecreasing_dims,
            self.nonincreasing_dims,
        )

    def _constraint_metadata(self) -> dict[str, int | str]:
        return constraint_metadata(
            self.fixed_dims,
            self.immutable_features,
            self.nondecreasing_dims,
            self.nonincreasing_dims,
            self.nondecreasing_features,
            self.nonincreasing_features,
        )

    def _sample_layer(self, x0: np.ndarray, inner_radius: float, outer_radius: float) -> np.ndarray:
        # Sample uniformly over directions in the chosen lp geometry, then sample
        # radii so points are uniform over the shell volume rather than concentrated
        # near the inner boundary.
        unit = self._sample_unit_directions(dim=x0.shape[0])

        if outer_radius <= inner_radius:
            radii = np.full((self.n_in_layer, 1), outer_radius, dtype=np.float32)
        else:
            dim = float(x0.shape[0])
            u = self.rng.uniform(0.0, 1.0, size=(self.n_in_layer, 1))
            # Exact shell sampling formula, rewritten to avoid underflow/overflow
            # in high dimension:
            #   r = (u * (R^d - r0^d) + r0^d)^(1/d)
            #     = R * (u * (1 - (r0/R)^d) + (r0/R)^d)^(1/d)
            inner_fraction = (inner_radius / outer_radius) ** dim
            radii = (
                outer_radius
                * np.power(u * (1.0 - inner_fraction) + inner_fraction, 1.0 / dim)
            ).astype(np.float32)

        scaled = unit * radii * self._feature_scale[None, :]
        return self._apply_constraints(x0[None, :] + scaled, x0)

    def _sample_unit_directions(self, dim: int) -> np.ndarray:
        if self.norm == 1:
            magnitudes = self.rng.exponential(scale=1.0, size=(self.n_in_layer, dim))
            signs = self.rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(self.n_in_layer, dim))
            directions = magnitudes * signs
        elif self.norm == 2:
            directions = self.rng.normal(size=(self.n_in_layer, dim))
        elif self.norm == np.inf:
            directions = self.rng.uniform(-1.0, 1.0, size=(self.n_in_layer, dim))
            max_abs = np.max(np.abs(directions), axis=1, keepdims=True)
            max_abs[max_abs == 0.0] = 1.0
            directions = directions / max_abs
            return directions.astype(np.float32)
        else:
            raise ValueError(
                "GrowingSpheresMethod currently supports norm in {1, 2, 'inf'}"
            )

        norms = np.linalg.norm(directions, ord=self.norm, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (directions / norms).astype(np.float32)

    def _feature_selection(
        self,
        x0: np.ndarray,
        enemy: np.ndarray,
        target_class: int,
    ) -> np.ndarray:
        """Greedy post-hoc sparsification as in the original Growing Spheres procedure."""
        x_cf = enemy.copy()
        x_cf = self._apply_constraints(x_cf, x0)
        changed = np.where(np.abs(x_cf - x0) > 1e-8)[0]
        if changed.size == 0:
            return x_cf

        # Try reverting smallest-magnitude changes first, keeping target prediction.
        order = changed[np.argsort(np.abs(x_cf[changed] - x0[changed]))]
        for idx in order:
            trial = x_cf.copy()
            trial[idx] = x0[idx]
            if int(self.model.predict(trial[None, :])[0]) == target_class:
                x_cf = trial
        return self._apply_constraints(x_cf, x0)
