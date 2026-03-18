"""Growing Spheres baseline for model-agnostic counterfactual search."""

from __future__ import annotations

from typing import Optional

import numpy as np

from counterfactuals.core.base_classes import BaseCounterfactualMethod, CounterfactualResult
from counterfactuals.core.interfaces import ModelInterface  # noqa: F401 – used by type annotation in __init__


class GrowingSpheresMethod(BaseCounterfactualMethod):
    """Growing Spheres with paper-style enemy search + feature selection."""

    def __init__(
        self,
        model: ModelInterface,
        n_in_layer: int = 512,
        max_radius: float = 3.0,
        radius_step: float = 0.15,
        random_seed: int = 42,
    ):
        super().__init__(model=model, random_seed=random_seed)
        self.n_in_layer = n_in_layer
        self.max_radius = max_radius
        self.radius_step = radius_step
        self._feature_scale: Optional[np.ndarray] = None

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
        target_class = self._resolve_target_class(x=x_query, target_class=target_class)

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
                dists = np.linalg.norm(valid - x_query[None, :], axis=1)
                idx = int(np.argmin(dists))
                enemy = valid[idx]
                best = self._feature_selection(
                    x0=x_query,
                    enemy=enemy,
                    target_class=target_class,
                )
                best_dist = float(np.linalg.norm(best - x_query, ord=2))
                break
            radius += self.radius_step

        if best is None:
            return CounterfactualResult(
                x_cf=x_query.copy(),
                success=False,
                distance=0.0,
                metadata={"target_class": target_class, "searched_radius": self.max_radius},
            )

        return CounterfactualResult(
            x_cf=best.astype(np.float32),
            success=True,
            distance=best_dist,
            metadata={"target_class": target_class, "searched_radius": radius},
        )

    def _sample_layer(self, x0: np.ndarray, inner_radius: float, outer_radius: float) -> np.ndarray:
        # Sample uniformly over directions, then sample radii so points are uniform
        # over the shell volume rather than concentrated near the inner boundary.
        directions = self.rng.normal(size=(self.n_in_layer, x0.shape[0]))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        unit = directions / norms

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
        return x0[None, :] + scaled

    def _feature_selection(
        self,
        x0: np.ndarray,
        enemy: np.ndarray,
        target_class: int,
    ) -> np.ndarray:
        """Greedy post-hoc sparsification as in the original Growing Spheres procedure."""
        x_cf = enemy.copy()
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
        return x_cf

    def _resolve_target_class(self, x: np.ndarray, target_class: Optional[int]) -> int:
        if target_class is not None:
            return int(target_class)
        pred = int(self.model.predict(x)[0])
        if self.model.predict_proba(x).shape[1] != 2:
            raise ValueError("target_class is required for non-binary tasks")
        return 1 - pred
