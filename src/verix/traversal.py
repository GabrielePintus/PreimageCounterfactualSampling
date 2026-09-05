"""Traversal strategies used by the original VERIX implementation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

FeatureTransform = Callable[[np.ndarray, np.ndarray], np.ndarray]
ScoreFunction = Callable[[np.ndarray], np.ndarray]


def deletion_transform(values: np.ndarray, _: np.ndarray) -> np.ndarray:
    """Delete a feature by replacing all of its coordinates with zero."""

    return np.zeros_like(values)


def reversal_transform(
    upper_bounds: float | Sequence[float] | np.ndarray = 1.0,
) -> FeatureTransform:
    """Return the paper's reversal transform ``upper_bound - value``."""

    bounds = np.asarray(upper_bounds, dtype=np.float64)

    def transform(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
        if bounds.ndim == 0:
            selected_bounds = bounds
        else:
            selected_bounds = bounds.reshape(-1)[indices]
        return np.asarray(selected_bounds - values, dtype=values.dtype)

    return transform


def random_order(feature_count: int, *, seed: int = 0) -> tuple[int, ...]:
    """Return the seeded random traversal used by the reference code."""

    if feature_count <= 0:
        raise ValueError("feature_count must be positive")
    order = list(range(feature_count))
    # The reference implementation uses Python's ``random`` module.
    import random

    random.Random(seed).shuffle(order)
    return tuple(order)


def occlusion_sensitivity_order(
    x: np.ndarray,
    *,
    score_function: ScoreFunction,
    predicted_class: int,
    transform: FeatureTransform,
    feature_groups: Sequence[Sequence[int]] | None = None,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Rank features from least to most sensitive, as in Definition 3.6.

    Sensitivity is the original predicted-class score minus the score after
    transforming one feature. A stable ascending sort reproduces the
    traversal direction used by the official implementation.
    """

    x_flat = np.asarray(x, dtype=np.float64).reshape(-1)
    if x_flat.size == 0:
        raise ValueError("x must contain at least one input coordinate")
    groups = _normalize_groups(feature_groups, input_dimension=x_flat.size)

    original_scores = np.asarray(score_function(x_flat[None, :]))
    if original_scores.ndim != 2:
        raise ValueError("score_function must return a 2D [batch, class] array")
    if not 0 <= int(predicted_class) < original_scores.shape[1]:
        raise ValueError("predicted_class is outside the score array")

    manipulated = np.repeat(x_flat[None, :], len(groups), axis=0)
    for feature, group in enumerate(groups):
        indices = np.asarray(group, dtype=np.int64)
        manipulated[feature, indices] = transform(
            manipulated[feature, indices].copy(),
            indices,
        )

    manipulated_scores = np.asarray(score_function(manipulated))
    expected_shape = (len(groups), original_scores.shape[1])
    if manipulated_scores.shape != expected_shape:
        raise ValueError(
            f"score_function returned {manipulated_scores.shape}, expected {expected_shape}"
        )

    sensitivity = (
        float(original_scores[0, int(predicted_class)])
        - manipulated_scores[:, int(predicted_class)]
    )
    order = tuple(
        int(index)
        for index in np.argsort(sensitivity, kind="stable")
    )
    return order, np.asarray(sensitivity, dtype=np.float64)


def _normalize_groups(
    groups: Sequence[Sequence[int]] | None,
    *,
    input_dimension: int,
) -> tuple[tuple[int, ...], ...]:
    if groups is None:
        return tuple((index,) for index in range(input_dimension))
    normalized = tuple(tuple(int(index) for index in group) for group in groups)
    flattened = [index for group in normalized for index in group]
    if (
        not normalized
        or any(not group for group in normalized)
        or sorted(flattened) != list(range(input_dimension))
    ):
        raise ValueError(
            "feature_groups must partition every flattened input coordinate exactly once"
        )
    return normalized
