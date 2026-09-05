"""Core implementation of Algorithm 1 from the VERIX paper.

The algorithm is deliberately independent of a particular neural-network
verifier. A checker supplies the paper's ``CHECK`` sub-procedure, while this
module implements the exact iterative construction of the explanation,
irrelevant-feature set, and counterfactual witnesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


class CheckStatus(str, Enum):
    """Possible outcomes of the paper's ``CHECK`` sub-procedure."""

    HOLDS = "holds"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CheckRequest:
    """One prediction-invariance specification passed to ``CHECK``."""

    x: np.ndarray
    reference_output: int | float
    free_features: tuple[int, ...]
    free_input_indices: tuple[int, ...]
    epsilon: float
    norm: float
    discrepancy: float


@dataclass(frozen=True)
class CheckResult:
    """Result returned by an invariance checker."""

    status: CheckStatus
    counterexample: np.ndarray | None = None
    counterexample_output: int | float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class InvarianceChecker(Protocol):
    """Black-box verifier interface corresponding to ``CHECK`` in the paper."""

    is_complete: bool

    def predict(self, x: np.ndarray) -> int | float:
        """Return the classification label or regression output for ``x``."""

    def check(self, request: CheckRequest) -> CheckResult:
        """Check invariance over the perturbations described by ``request``."""


@dataclass(frozen=True)
class CounterfactualWitness:
    """Concrete counterexample returned when one feature check is violated."""

    feature: int
    free_features: tuple[int, ...]
    free_input_indices: tuple[int, ...]
    x: np.ndarray
    output: int | float | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VeriXStep:
    """Trace of one iteration of Algorithm 1."""

    feature: int
    free_features: tuple[int, ...]
    free_input_indices: tuple[int, ...]
    status: CheckStatus
    elapsed_seconds: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VeriXResult:
    """Explanation, irrelevant features, witnesses, and execution trace."""

    reference_output: int | float
    explanation: tuple[int, ...]
    irrelevant: tuple[int, ...]
    unknown: tuple[int, ...]
    counterfactuals: tuple[CounterfactualWitness, ...]
    traversal_order: tuple[int, ...]
    feature_groups: tuple[tuple[int, ...], ...]
    steps: tuple[VeriXStep, ...]
    checker_is_complete: bool
    elapsed_seconds: float

    @property
    def locally_minimal(self) -> bool:
        """Whether the paper's local optimality conditions were established."""

        return self.checker_is_complete and not self.unknown


class VeriX:
    """Compute a VERIX robust explanation and its native counterfactuals.

    ``feature_groups`` maps each semantic feature to one or more flattened
    input coordinates. With no mapping, every input coordinate is treated as
    one feature. Grouping preserves the paper's treatment of an RGB pixel as
    one feature while still allowing its three channels to vary together.
    """

    def __init__(
        self,
        checker: InvarianceChecker,
        *,
        epsilon: float,
        norm: int | float = np.inf,
        discrepancy: float = 0.0,
        feature_groups: Sequence[Sequence[int]] | None = None,
    ) -> None:
        if epsilon < 0.0:
            raise ValueError("epsilon must be non-negative")
        normalized_norm = float(norm)
        if normalized_norm not in {1.0, 2.0, float("inf")}:
            raise ValueError("norm must be one of {1, 2, np.inf}")
        if discrepancy < 0.0:
            raise ValueError("discrepancy must be non-negative")
        if not isinstance(checker, InvarianceChecker):
            raise TypeError("checker must implement the InvarianceChecker protocol")

        self.checker = checker
        self.epsilon = float(epsilon)
        self.norm = normalized_norm
        self.discrepancy = float(discrepancy)
        self._feature_groups_arg = feature_groups

    def explain(
        self,
        x: np.ndarray,
        *,
        traversal_order: Sequence[int] | None = None,
        step_callback: Callable[[VeriXStep], None] | None = None,
    ) -> VeriXResult:
        """Run the single feature traversal in Algorithm 1."""

        started = perf_counter()
        x_flat = np.asarray(x, dtype=np.float64).reshape(-1)
        if x_flat.size == 0:
            raise ValueError("x must contain at least one input coordinate")

        feature_groups = self._normalize_feature_groups(x_flat.size)
        order = self._normalize_traversal_order(
            traversal_order,
            feature_count=len(feature_groups),
        )
        reference_output = self.checker.predict(x_flat.copy())

        explanation: list[int] = []
        irrelevant: list[int] = []
        unknown: list[int] = []
        counterfactuals: list[CounterfactualWitness] = []
        steps: list[VeriXStep] = []

        for feature in order:
            free_features = tuple((*irrelevant, feature))
            free_input_indices = tuple(
                coordinate
                for free_feature in free_features
                for coordinate in feature_groups[free_feature]
            )
            request = CheckRequest(
                x=x_flat.copy(),
                reference_output=reference_output,
                free_features=free_features,
                free_input_indices=free_input_indices,
                epsilon=self.epsilon,
                norm=self.norm,
                discrepancy=self.discrepancy,
            )

            check_started = perf_counter()
            check_result = self.checker.check(request)
            check_elapsed = perf_counter() - check_started
            if not isinstance(check_result, CheckResult):
                raise TypeError("checker.check() must return a CheckResult")

            if check_result.status is CheckStatus.HOLDS:
                irrelevant.append(feature)
            elif check_result.status is CheckStatus.VIOLATED:
                if check_result.counterexample is None:
                    raise ValueError(
                        "a violated CHECK must include the verifier's concrete counterexample"
                    )
                witness = np.asarray(
                    check_result.counterexample,
                    dtype=np.float64,
                ).reshape(-1)
                if witness.shape != x_flat.shape:
                    raise ValueError(
                        "counterexample must have the same flattened shape as x"
                    )
                explanation.append(feature)
                counterfactuals.append(
                    CounterfactualWitness(
                        feature=feature,
                        free_features=free_features,
                        free_input_indices=free_input_indices,
                        x=witness.copy(),
                        output=check_result.counterexample_output,
                        metadata=dict(check_result.metadata),
                    )
                )
            elif check_result.status is CheckStatus.UNKNOWN:
                # Section 4.6: only proven-invariant features enter B. Unknown
                # features remain in the explanation, making it sound but
                # potentially larger than a locally minimal explanation.
                explanation.append(feature)
                unknown.append(feature)
            else:
                raise ValueError(f"unsupported CHECK status: {check_result.status!r}")

            step = VeriXStep(
                feature=feature,
                free_features=free_features,
                free_input_indices=free_input_indices,
                status=check_result.status,
                elapsed_seconds=check_elapsed,
                metadata=dict(check_result.metadata),
            )
            steps.append(step)
            if step_callback is not None:
                step_callback(step)

        return VeriXResult(
            reference_output=reference_output,
            explanation=tuple(explanation),
            irrelevant=tuple(irrelevant),
            unknown=tuple(unknown),
            counterfactuals=tuple(counterfactuals),
            traversal_order=order,
            feature_groups=feature_groups,
            steps=tuple(steps),
            checker_is_complete=bool(self.checker.is_complete),
            elapsed_seconds=perf_counter() - started,
        )

    def _normalize_feature_groups(
        self,
        input_dimension: int,
    ) -> tuple[tuple[int, ...], ...]:
        if self._feature_groups_arg is None:
            return tuple((index,) for index in range(input_dimension))

        groups = tuple(
            tuple(int(index) for index in group)
            for group in self._feature_groups_arg
        )
        if not groups:
            raise ValueError("feature_groups must contain at least one feature")
        if any(not group for group in groups):
            raise ValueError("feature_groups cannot contain an empty feature")

        flattened = [index for group in groups for index in group]
        if sorted(flattened) != list(range(input_dimension)):
            raise ValueError(
                "feature_groups must partition every flattened input coordinate exactly once"
            )
        return groups

    @staticmethod
    def _normalize_traversal_order(
        traversal_order: Sequence[int] | None,
        *,
        feature_count: int,
    ) -> tuple[int, ...]:
        if traversal_order is None:
            return tuple(range(feature_count))
        order = tuple(int(index) for index in traversal_order)
        if sorted(order) != list(range(feature_count)):
            raise ValueError(
                "traversal_order must be a permutation of all feature indices"
            )
        return order
