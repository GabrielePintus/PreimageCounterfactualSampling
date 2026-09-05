"""Feature-level checkpointing for long-running VERIX traversals."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .core import CheckRequest, CheckResult, CheckStatus, InvarianceChecker


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def request_signature(request: CheckRequest) -> str:
    """Fingerprint the state-defining parts of a verification request."""

    payload = {
        "reference_output": _json_value(request.reference_output),
        "free_features": list(request.free_features),
        "free_input_indices": list(request.free_input_indices),
        "epsilon": float(request.epsilon),
        "norm": float(request.norm),
        "discrepancy": float(request.discrepancy),
        "x_sha256": hashlib.sha256(
            np.asarray(request.x, dtype=np.float64).reshape(-1).tobytes()
        ).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CachedCheck:
    """Serializable result of one completed feature check."""

    signature: str
    status: CheckStatus
    counterexample: np.ndarray | None
    counterexample_output: int | float | None
    metadata: dict[str, Any]

    @classmethod
    def from_result(cls, request: CheckRequest, result: CheckResult) -> "CachedCheck":
        return cls(
            signature=request_signature(request),
            status=result.status,
            counterexample=(
                None
                if result.counterexample is None
                else np.asarray(result.counterexample, dtype=np.float64).reshape(-1).copy()
            ),
            counterexample_output=result.counterexample_output,
            metadata=_json_value(dict(result.metadata)),
        )

    def to_result(self) -> CheckResult:
        return CheckResult(
            status=self.status,
            counterexample=(
                None if self.counterexample is None else self.counterexample.copy()
            ),
            counterexample_output=self.counterexample_output,
            metadata=dict(self.metadata),
        )


def save_check_cache(
    path: str | Path,
    records: Sequence[CachedCheck],
    *,
    input_dimension: int,
    metadata: dict[str, Any],
) -> None:
    """Atomically save completed checks in a pickle-free compressed NPZ."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = len(records)
    counterexamples = np.full(
        (count, int(input_dimension)),
        np.nan,
        dtype=np.float64,
    )
    has_counterexample = np.zeros(count, dtype=bool)
    outputs = np.full(count, np.nan, dtype=np.float64)
    output_present = np.zeros(count, dtype=bool)
    for index, record in enumerate(records):
        if record.counterexample is not None:
            value = np.asarray(record.counterexample, dtype=np.float64).reshape(-1)
            if value.size != int(input_dimension):
                raise ValueError("cached counterexample has the wrong input dimension")
            counterexamples[index] = value
            has_counterexample[index] = True
        if record.counterexample_output is not None:
            outputs[index] = float(record.counterexample_output)
            output_present[index] = True

    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(
                json.dumps(_json_value(metadata), sort_keys=True)
            ),
            signatures=np.asarray([record.signature for record in records]),
            statuses=np.asarray([record.status.value for record in records]),
            counterexamples=counterexamples,
            has_counterexample=has_counterexample,
            counterexample_outputs=outputs,
            output_present=output_present,
            result_metadata_json=np.asarray(
                [json.dumps(record.metadata, sort_keys=True) for record in records]
            ),
        )
    os.replace(temporary, destination)


def load_check_cache(
    path: str | Path,
    *,
    expected_metadata: dict[str, Any],
) -> list[CachedCheck]:
    """Load a cache only when all supplied identity fields match."""

    source = Path(path)
    if not source.exists():
        return []
    try:
        with np.load(source, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            if any(metadata.get(key) != value for key, value in expected_metadata.items()):
                return []
            signatures = data["signatures"].astype(str)
            statuses = data["statuses"].astype(str)
            counterexamples = data["counterexamples"]
            has_counterexample = data["has_counterexample"]
            outputs = data["counterexample_outputs"]
            output_present = data["output_present"]
            result_metadata = data["result_metadata_json"].astype(str)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return []

    lengths = {
        len(signatures),
        len(statuses),
        len(counterexamples),
        len(has_counterexample),
        len(outputs),
        len(output_present),
        len(result_metadata),
    }
    if len(lengths) != 1:
        return []
    records: list[CachedCheck] = []
    try:
        for index in range(len(signatures)):
            records.append(
                CachedCheck(
                    signature=str(signatures[index]),
                    status=CheckStatus(str(statuses[index])),
                    counterexample=(
                        counterexamples[index].copy()
                        if bool(has_counterexample[index])
                        else None
                    ),
                    counterexample_output=(
                        float(outputs[index]) if bool(output_present[index]) else None
                    ),
                    metadata=json.loads(str(result_metadata[index])),
                )
            )
    except (ValueError, json.JSONDecodeError):
        return []
    return records


class CheckpointingChecker:
    """Replay cached checks, then persist each new delegated check."""

    def __init__(
        self,
        checker: InvarianceChecker,
        *,
        cached: Sequence[CachedCheck] = (),
        on_update: Callable[[Sequence[CachedCheck]], None] | None = None,
    ) -> None:
        self.checker = checker
        self.records = list(cached)
        self.on_update = on_update
        self.cursor = 0
        self.replayed_count = 0
        self.is_complete = bool(checker.is_complete)

    def predict(self, x: np.ndarray) -> int | float:
        return self.checker.predict(x)

    def check(self, request: CheckRequest) -> CheckResult:
        signature = request_signature(request)
        if self.cursor < len(self.records):
            record = self.records[self.cursor]
            if record.signature != signature:
                raise RuntimeError(
                    "cached VERIX feature checks do not match the current traversal"
                )
            self.cursor += 1
            self.replayed_count += 1
            return record.to_result()

        started = time.perf_counter()
        result = self.checker.check(request)
        elapsed_seconds = time.perf_counter() - started
        result = CheckResult(
            status=result.status,
            counterexample=result.counterexample,
            counterexample_output=result.counterexample_output,
            metadata={
                **dict(result.metadata),
                "verix_check_elapsed_seconds": elapsed_seconds,
            },
        )
        record = CachedCheck.from_result(request, result)
        self.records.append(record)
        self.cursor += 1
        if self.on_update is not None:
            self.on_update(tuple(self.records))
        return result


class WallClockBudgetChecker:
    """Bound fresh checker work by one cumulative wall-clock query budget.

    ``spent_seconds`` accounts for solver checks restored from a partial
    checkpoint.  When the budget is exhausted, subsequent checks return
    ``UNKNOWN`` immediately.  This preserves VERIX soundness while allowing a
    grid run to continue to the next query.
    """

    def __init__(
        self,
        checker: InvarianceChecker,
        *,
        timeout_seconds: float,
        spent_seconds: float = 0.0,
        max_solver_timeout_seconds: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if spent_seconds < 0.0:
            raise ValueError("spent_seconds must be non-negative")
        if (
            max_solver_timeout_seconds is not None
            and max_solver_timeout_seconds <= 0.0
        ):
            raise ValueError("max_solver_timeout_seconds must be positive")
        self.checker = checker
        self.timeout_seconds = float(timeout_seconds)
        self.spent_seconds = float(spent_seconds)
        self.max_solver_timeout_seconds = (
            None
            if max_solver_timeout_seconds is None
            else float(max_solver_timeout_seconds)
        )
        self._clock = clock
        self._started = float(clock())
        remaining = max(0.0, self.timeout_seconds - self.spent_seconds)
        self._deadline = self._started + remaining
        self.budget_exhausted = remaining <= 0.0

    @property
    def is_complete(self) -> bool:
        return bool(self.checker.is_complete)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self._deadline - float(self._clock()))

    def predict(self, x: np.ndarray) -> int | float:
        return self.checker.predict(x)

    def check(self, request: CheckRequest) -> CheckResult:
        remaining_before = self.remaining_seconds
        if remaining_before <= 0.0:
            self.budget_exhausted = True
            return CheckResult(
                status=CheckStatus.UNKNOWN,
                metadata={
                    "query_budget_exhausted": True,
                    "query_budget_seconds": self.timeout_seconds,
                    "query_budget_remaining_before_check_seconds": 0.0,
                    "query_budget_solver_skipped": True,
                },
            )

        solver_timeout = remaining_before
        if self.max_solver_timeout_seconds is not None:
            solver_timeout = min(
                solver_timeout,
                self.max_solver_timeout_seconds,
            )
        set_timeout = getattr(self.checker, "set_timeout_seconds", None)
        if callable(set_timeout):
            # Marabou accepts an integer number of seconds.  Rounding upward
            # avoids discarding the final fractional second of the budget.
            set_timeout(max(1, int(math.ceil(solver_timeout))))

        result = self.checker.check(request)
        remaining_after = self.remaining_seconds
        exhausted_after = remaining_after <= 0.0
        if exhausted_after:
            self.budget_exhausted = True
        return CheckResult(
            status=result.status,
            counterexample=result.counterexample,
            counterexample_output=result.counterexample_output,
            metadata={
                **dict(result.metadata),
                "query_budget_exhausted": exhausted_after,
                "query_budget_seconds": self.timeout_seconds,
                "query_budget_remaining_before_check_seconds": remaining_before,
                "query_budget_remaining_after_check_seconds": remaining_after,
                "query_budget_solver_skipped": False,
            },
        )
