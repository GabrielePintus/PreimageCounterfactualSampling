"""MNIST protocol helpers for the paper-faithful VERIX implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .core import CounterfactualWitness, VeriX, VeriXResult, VeriXStep
from .traversal import occlusion_sensitivity_order, reversal_transform

MNIST_PIXELS = 28 * 28


class ONNXMNISTScorer:
    """Batch logits adapter used by the original sensitivity heuristic."""

    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name

    def logits(self, x: np.ndarray) -> np.ndarray:
        array = np.asarray(x, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim == 2:
            if array.shape[1] != MNIST_PIXELS:
                raise ValueError(f"expected {MNIST_PIXELS} flattened pixels")
            array = array.reshape(-1, 1, 28, 28)
        if array.ndim != 4 or tuple(array.shape[1:]) != (1, 28, 28):
            raise ValueError("MNIST inputs must have shape [N, 784] or [N, 1, 28, 28]")
        return np.asarray(
            self.session.run(None, {self.input_name: array})[0],
            dtype=np.float64,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.logits(x).argmax(axis=1)


@dataclass(frozen=True)
class WitnessValidation:
    """Canonical validation and metrics for one native VERIX witness."""

    witness_index: int
    feature: int
    predicted_class: int
    class_changed: bool
    in_bounds: bool
    free_linf_distance: float
    fixed_max_abs_difference: float
    l1_distance: float
    l2_distance: float
    l0_changed: int
    target_logit_margin: float
    canonical_valid: bool


@dataclass(frozen=True)
class SelectedWitness:
    """Nearest canonically valid native witness."""

    witness_index: int
    feature: int
    target_class: int
    x: np.ndarray
    validation: WitnessValidation


def run_mnist_verix(
    checker,
    scorer: ONNXMNISTScorer,
    x: np.ndarray,
    *,
    epsilon: float = 0.05,
    step_callback: Callable[[VeriXStep], None] | None = None,
) -> tuple[VeriXResult, np.ndarray]:
    """Run the original MNIST reversal traversal and Algorithm 1."""

    x_flat = np.asarray(x, dtype=np.float64).reshape(-1)
    if x_flat.size != MNIST_PIXELS:
        raise ValueError(f"expected {MNIST_PIXELS} MNIST pixels")
    predicted_class = int(scorer.predict(x_flat)[0])
    traversal_order, sensitivity = occlusion_sensitivity_order(
        x_flat,
        score_function=scorer.logits,
        predicted_class=predicted_class,
        transform=reversal_transform(1.0),
    )
    result = VeriX(
        checker,
        epsilon=float(epsilon),
        norm=np.inf,
        discrepancy=0.0,
    ).explain(
        x_flat,
        traversal_order=traversal_order,
        step_callback=step_callback,
    )
    if int(result.reference_output) != predicted_class:
        raise RuntimeError(
            "ONNX sensitivity scorer and Marabou checker disagree on the source class"
        )
    return result, sensitivity


def validate_mnist_witnesses(
    result: VeriXResult,
    x_query: np.ndarray,
    *,
    logits_function: Callable[[np.ndarray], np.ndarray],
    epsilon: float,
    bounds: tuple[float, float] = (0.0, 1.0),
    tolerance: float = 1.0e-6,
    l0_tolerance: float = 1.0e-6,
) -> tuple[WitnessValidation, ...]:
    """Validate every solver witness against the canonical classifier."""

    x_flat = np.asarray(x_query, dtype=np.float64).reshape(-1)
    if x_flat.size != MNIST_PIXELS:
        raise ValueError(f"expected {MNIST_PIXELS} MNIST pixels")
    if not result.counterfactuals:
        return ()

    witnesses = np.stack(
        [np.asarray(witness.x, dtype=np.float64).reshape(-1) for witness in result.counterfactuals]
    )
    logits = np.asarray(logits_function(witnesses), dtype=np.float64)
    if logits.shape[0] != len(witnesses):
        raise ValueError("logits_function returned the wrong number of rows")
    predicted = logits.argmax(axis=1)
    validations: list[WitnessValidation] = []

    for index, (witness, x_cf, witness_logits, target) in enumerate(
        zip(result.counterfactuals, witnesses, logits, predicted)
    ):
        diff = np.abs(x_cf - x_flat)
        free_indices = np.asarray(witness.free_input_indices, dtype=np.int64)
        fixed_mask = np.ones(MNIST_PIXELS, dtype=bool)
        fixed_mask[free_indices] = False
        free_linf = float(np.max(diff[free_indices])) if free_indices.size else 0.0
        fixed_max = float(np.max(diff[fixed_mask])) if np.any(fixed_mask) else 0.0
        target = int(target)
        other_logits = witness_logits.copy()
        other_logits[target] = -np.inf
        margin = float(witness_logits[target] - np.max(other_logits))
        in_bounds = bool(
            np.all(x_cf >= float(bounds[0]) - tolerance)
            and np.all(x_cf <= float(bounds[1]) + tolerance)
        )
        class_changed = target != int(result.reference_output)
        canonical_valid = bool(
            np.isfinite(x_cf).all()
            and in_bounds
            and class_changed
            and free_linf <= float(epsilon) + tolerance
            and fixed_max <= tolerance
        )
        validations.append(
            WitnessValidation(
                witness_index=index,
                feature=int(witness.feature),
                predicted_class=target,
                class_changed=class_changed,
                in_bounds=in_bounds,
                free_linf_distance=free_linf,
                fixed_max_abs_difference=fixed_max,
                l1_distance=float(np.linalg.norm(x_cf - x_flat, ord=1)),
                l2_distance=float(np.linalg.norm(x_cf - x_flat, ord=2)),
                l0_changed=int(np.count_nonzero(diff > l0_tolerance)),
                target_logit_margin=margin,
                canonical_valid=canonical_valid,
            )
        )
    return tuple(validations)


def select_nearest_valid_witness(
    witnesses: Sequence[CounterfactualWitness],
    validations: Sequence[WitnessValidation],
) -> SelectedWitness | None:
    """Select by L1, then traversal-produced witness order and feature index."""

    if len(witnesses) != len(validations):
        raise ValueError("witnesses and validations must have matching lengths")
    valid = [validation for validation in validations if validation.canonical_valid]
    if not valid:
        return None
    best = min(
        valid,
        key=lambda item: (
            item.l1_distance,
            item.witness_index,
            item.feature,
            item.predicted_class,
        ),
    )
    witness = witnesses[best.witness_index]
    return SelectedWitness(
        witness_index=best.witness_index,
        feature=best.feature,
        target_class=best.predicted_class,
        x=np.asarray(witness.x, dtype=np.float64).reshape(-1).copy(),
        validation=best,
    )
