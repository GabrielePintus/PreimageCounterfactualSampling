"""Marabou/ONNX implementation of VERIX's classification ``CHECK``.

This follows the public reference implementation: free input coordinates are
bounded independently in an L-infinity box, all remaining coordinates are
fixed, and each competing logit is checked against the original class logit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verix.core import CheckRequest, CheckResult, CheckStatus


@dataclass(frozen=True)
class MarabouOptions:
    """Solver settings for the Marabou backend.

    Worker count, timeout, and margin follow the official VERIX repository.
    The native solver is the safe default because Gurobi is optional.
    """

    num_workers: int = 16
    timeout_seconds: int = 300
    verbosity: int = 0
    solve_with_milp: bool = False
    classification_margin: float = 1.0e-6


class MarabouClassificationChecker:
    """Sound and complete classification checker, subject to solver timeout."""

    is_complete = True

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_lower_bounds: float | np.ndarray = 0.0,
        input_upper_bounds: float | np.ndarray = 1.0,
        options: MarabouOptions | None = None,
    ) -> None:
        try:
            import onnx
            import onnxruntime as ort
            from maraboupy import Marabou
        except ImportError as exc:
            raise ImportError(
                "The Marabou VERIX backend requires the optional dependencies "
                "maraboupy, onnx, and onnxruntime. Install the project with "
                "`pip install -e '.[verix]'` on Python 3.8-3.11."
            ) from exc

        self._onnx = onnx
        self._ort = ort
        self._marabou = Marabou
        self.model_path = str(Path(model_path))
        self.options = options or MarabouOptions()

        self.onnx_model = onnx.load(self.model_path)
        self.onnx_session = ort.InferenceSession(self.model_path)
        self.input_name = self.onnx_model.graph.input[0].name

        output_names = None
        if (
            self.onnx_model.graph.node
            and self.onnx_model.graph.node[-1].op_type == "Softmax"
        ):
            output_names = list(self.onnx_model.graph.node[-1].input)
        self.network = Marabou.read_onnx(
            filename=self.model_path,
            outputNames=output_names,
        )
        self.input_variables = self.network.inputVars[0].flatten()
        self.output_variables = self.network.outputVars[0].flatten()
        self.input_shape = tuple(int(size) for size in self.network.inputVars[0].shape)
        self.lower_bounds = self._broadcast_bounds(
            input_lower_bounds,
            name="input_lower_bounds",
        )
        self.upper_bounds = self._broadcast_bounds(
            input_upper_bounds,
            name="input_upper_bounds",
        )
        if np.any(self.lower_bounds > self.upper_bounds):
            raise ValueError("input_lower_bounds cannot exceed input_upper_bounds")

        self.solver_options = self._create_solver_options(
            int(self.options.timeout_seconds)
        )

    def _create_solver_options(self, timeout_seconds: int) -> Any:
        return self._marabou.createOptions(
            numWorkers=int(self.options.num_workers),
            timeoutInSeconds=int(timeout_seconds),
            verbosity=int(self.options.verbosity),
            solveWithMILP=bool(self.options.solve_with_milp),
        )

    def set_timeout_seconds(self, timeout_seconds: int) -> None:
        """Update the native timeout before a budgeted VERIX check."""

        if int(timeout_seconds) <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.solver_options = self._create_solver_options(int(timeout_seconds))

    def predict(self, x: np.ndarray) -> int:
        scores = self._predict_scores(x)
        return int(np.argmax(scores))

    def check(self, request: CheckRequest) -> CheckResult:
        if request.norm != float("inf"):
            raise NotImplementedError(
                "The official VERIX Marabou implementation realizes only "
                "L-infinity perturbations through coordinate-wise bounds."
            )
        if request.discrepancy != 0.0:
            raise NotImplementedError(
                "MarabouClassificationChecker supports classification with discrepancy=0"
            )

        x_flat = np.asarray(request.x, dtype=np.float64).reshape(-1)
        if x_flat.size != self.input_variables.size:
            raise ValueError(
                f"x has {x_flat.size} coordinates, expected {self.input_variables.size}"
            )
        free = np.zeros(x_flat.size, dtype=bool)
        free[np.asarray(request.free_input_indices, dtype=np.int64)] = True

        for index, variable in enumerate(self.input_variables):
            if free[index]:
                lower = max(
                    float(self.lower_bounds[index]),
                    float(x_flat[index] - request.epsilon),
                )
                upper = min(
                    float(self.upper_bounds[index]),
                    float(x_flat[index] + request.epsilon),
                )
            else:
                lower = upper = float(x_flat[index])
            self.network.setLowerBound(int(variable), lower)
            self.network.setUpperBound(int(variable), upper)

        reference_class = int(request.reference_output)
        checked_classes: list[int] = []
        try:
            for other_class, other_variable in enumerate(self.output_variables):
                if other_class == reference_class:
                    continue
                checked_classes.append(other_class)
                self.network.addInequality(
                    [
                        int(self.output_variables[reference_class]),
                        int(other_variable),
                    ],
                    [1.0, -1.0],
                    -float(self.options.classification_margin),
                    isProperty=True,
                )
                exit_code, values, stats = self.network.solve(
                    options=self.solver_options,
                    verbose=False,
                )
                self.network.additionalEquList.clear()
                normalized_exit = str(exit_code).lower()

                if normalized_exit == "sat":
                    counterexample = np.asarray(
                        [
                            values.get(int(variable))
                            for variable in self.input_variables
                        ],
                        dtype=np.float64,
                    )
                    output_values = np.asarray(
                        [
                            values.get(int(variable))
                            for variable in self.output_variables
                        ],
                        dtype=np.float64,
                    )
                    return CheckResult(
                        status=CheckStatus.VIOLATED,
                        counterexample=counterexample,
                        counterexample_output=int(np.argmax(output_values)),
                        metadata=self._metadata(
                            checked_classes=checked_classes,
                            exit_code=exit_code,
                            stats=stats,
                        ),
                    )
                if normalized_exit in {"timeout", "unknown"}:
                    return CheckResult(
                        status=CheckStatus.UNKNOWN,
                        metadata=self._metadata(
                            checked_classes=checked_classes,
                            exit_code=exit_code,
                            stats=stats,
                        ),
                    )
                if normalized_exit != "unsat":
                    return CheckResult(
                        status=CheckStatus.UNKNOWN,
                        metadata=self._metadata(
                            checked_classes=checked_classes,
                            exit_code=exit_code,
                            stats=stats,
                        ),
                    )

            return CheckResult(
                status=CheckStatus.HOLDS,
                metadata={
                    "checked_classes": tuple(checked_classes),
                    "exit_code": "unsat",
                },
            )
        finally:
            self.network.clearProperty()

    def _predict_scores(self, x: np.ndarray) -> np.ndarray:
        x_flat = np.asarray(x, dtype=np.float32).reshape(-1)
        if x_flat.size != self.input_variables.size:
            raise ValueError(
                f"x has {x_flat.size} coordinates, expected {self.input_variables.size}"
            )
        # Marabou preserves the full ONNX input tensor shape, including the
        # batch dimension (normally fixed to one).
        model_input = x_flat.reshape(self.input_shape)
        prediction = self.onnx_session.run(
            None,
            {self.input_name: model_input},
        )
        return np.asarray(prediction[0])[0]

    def _broadcast_bounds(
        self,
        bounds: float | np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        array = np.asarray(bounds, dtype=np.float64)
        try:
            return np.broadcast_to(
                array,
                (self.input_variables.size,),
            ).copy()
        except ValueError as exc:
            raise ValueError(
                f"{name} must be scalar or broadcast to {self.input_variables.size} inputs"
            ) from exc

    @staticmethod
    def _metadata(
        *,
        checked_classes: list[int],
        exit_code: Any,
        stats: Any,
    ) -> dict[str, Any]:
        return {
            "checked_classes": tuple(checked_classes),
            "exit_code": str(exit_code),
            "stats": stats,
        }
