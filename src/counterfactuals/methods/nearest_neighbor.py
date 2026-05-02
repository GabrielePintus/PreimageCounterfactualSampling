"""Nearest-neighbour baseline counterfactual method.

Returns the closest opposite-class training point as a counterfactual.
Used as a lower-bound baseline for distance metrics.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from counterfactuals.core.base_classes import BaseCounterfactualMethod, CounterfactualResult
from counterfactuals.core.interfaces import ModelInterface
from counterfactuals.methods._constraints import (
    constraint_metadata,
    directional_compatibility_mask,
    normalize_directional_dims,
    normalize_fixed_dims,
    validate_disjoint_directional_dims,
)


class NearestNeighborMethod(BaseCounterfactualMethod):
    """Return the closest training sample from the requested target class."""

    def __init__(
        self,
        # The trained classifier
        model: ModelInterface,

        # Norm used to measure distance to candidates (e.g. 1, 2, "inf")
        norm: int | float | str = 2,

        # Custom downsampling strategy
        subsample_method: str = "kmedoids",
        k_per_class: int | None = None,
        fixed_dims: Optional[Sequence[int]] = None,
        immutable_features: Optional[Sequence[str]] = None,
        nondecreasing_dims: Optional[Sequence[int]] = None,
        nonincreasing_dims: Optional[Sequence[int]] = None,
        nondecreasing_features: Optional[Sequence[str]] = None,
        nonincreasing_features: Optional[Sequence[str]] = None,

        # Random seed for reproducibility (e.g., in subsampling)
        random_seed: int = 42,
    ):
        super().__init__(model=model, random_seed=random_seed, k_per_class=k_per_class, subsample_method=subsample_method)
        self.norm = float(norm) if isinstance(norm, str) else norm
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

    def _fit(self) -> None:
        self.train_pred = self.model.predict(self._x_train)

    def generate(self, x: np.ndarray, target_class: int) -> CounterfactualResult:
        if not self._is_fitted:
            raise RuntimeError("Method is not fitted. Call fit() before generate().")

        x_query = np.asarray(x, dtype=np.float32).reshape(-1)

        target_mask = self.train_pred == target_class
        if not np.any(target_mask):
            raise ValueError(f"No training samples predicted as target_class={target_class}")

        candidates = self._x_train[target_mask]
        candidate_indices = np.flatnonzero(target_mask)
        n_target_candidates = int(np.sum(target_mask))
        if self.fixed_dims is not None:
            compatible_mask = np.all(
                np.isclose(candidates[:, self.fixed_dims], x_query[self.fixed_dims][None, :]),
                axis=1,
            )
            candidates = candidates[compatible_mask]
            candidate_indices = candidate_indices[compatible_mask]
            if len(candidates) == 0:
                return CounterfactualResult(
                    x_cf=x_query.copy(),
                    success=False,
                    distance=0.0,
                    metadata={
                        "target_class": int(target_class),
                        "reason": "no_immutable_compatible_candidate",
                        "n_target_candidates": n_target_candidates,
                        "n_immutable_compatible_candidates": 0,
                        **self._constraint_metadata(),
                    },
                )
        n_immutable_compatible = int(len(candidates))

        directional_mask = directional_compatibility_mask(
            candidates,
            x_query,
            self.nondecreasing_dims,
            self.nonincreasing_dims,
        )
        candidates = candidates[directional_mask]
        candidate_indices = candidate_indices[directional_mask]
        if len(candidates) == 0:
            return CounterfactualResult(
                x_cf=x_query.copy(),
                success=False,
                distance=0.0,
                metadata={
                    "target_class": int(target_class),
                    "reason": "no_directional_compatible_candidate",
                    "n_target_candidates": n_target_candidates,
                    "n_immutable_compatible_candidates": n_immutable_compatible,
                    "n_directional_compatible_candidates": 0,
                    **self._constraint_metadata(),
                },
            )

        distances = np.linalg.norm(candidates - x_query[None, :], ord=self.norm, axis=1)
        best_idx = int(np.argmin(distances))
        x_cf = candidates[best_idx].astype(np.float32)

        return CounterfactualResult(
            x_cf=x_cf,
            success=True,
            distance=float(distances[best_idx]),
            metadata={
                "target_class": int(target_class),
                "nearest_train_index": int(candidate_indices[best_idx]),
                "nearest_train_distance": float(distances[best_idx]),
                "n_target_candidates": n_target_candidates,
                "n_immutable_compatible_candidates": n_immutable_compatible,
                "n_directional_compatible_candidates": int(len(candidates)),
                **self._constraint_metadata(),
            },
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
