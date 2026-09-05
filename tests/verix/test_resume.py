from pathlib import Path

import numpy as np

from verix import CheckRequest, CheckResult, CheckStatus
from verix.resume import (
    CachedCheck,
    CheckpointingChecker,
    WallClockBudgetChecker,
    load_check_cache,
    save_check_cache,
)


def request(feature=0):
    return CheckRequest(
        x=np.array([0.2, 0.8]),
        reference_output=0,
        free_features=(feature,),
        free_input_indices=(feature,),
        epsilon=0.05,
        norm=np.inf,
        discrepancy=0.0,
    )


class Delegate:
    is_complete = True

    def __init__(self):
        self.calls = 0

    def predict(self, x):
        return 0

    def check(self, value):
        self.calls += 1
        return CheckResult(
            CheckStatus.VIOLATED,
            counterexample=np.array([0.25, 0.8]),
            counterexample_output=1,
            metadata={"solver": "native"},
        )


def test_cache_roundtrip_and_fingerprint_guard(tmp_path: Path):
    result = Delegate().check(request())
    record = CachedCheck.from_result(request(), result)
    path = tmp_path / "partial.npz"
    save_check_cache(
        path,
        [record],
        input_dimension=2,
        metadata={"run": "abc", "query": 7},
    )

    loaded = load_check_cache(
        path,
        expected_metadata={"run": "abc", "query": 7},
    )
    assert len(loaded) == 1
    assert loaded[0].status is CheckStatus.VIOLATED
    assert np.array_equal(loaded[0].counterexample, np.array([0.25, 0.8]))
    assert loaded[0].metadata == {"solver": "native"}
    assert load_check_cache(path, expected_metadata={"run": "changed"}) == []


def test_checkpointing_checker_replays_without_solver_call_and_then_appends():
    first_result = CheckResult(CheckStatus.HOLDS, metadata={"cached": True})
    cached = [CachedCheck.from_result(request(0), first_result)]
    updates = []
    delegate = Delegate()
    checker = CheckpointingChecker(
        delegate,
        cached=cached,
        on_update=lambda records: updates.append(len(records)),
    )

    replayed = checker.check(request(0))
    fresh = checker.check(request(1))

    assert replayed.status is CheckStatus.HOLDS
    assert fresh.status is CheckStatus.VIOLATED
    assert fresh.metadata["verix_check_elapsed_seconds"] >= 0.0
    assert delegate.calls == 1
    assert checker.replayed_count == 1
    assert updates == [2]


class BudgetDelegate(Delegate):
    def __init__(self):
        super().__init__()
        self.timeouts = []

    def set_timeout_seconds(self, value):
        self.timeouts.append(value)


def test_wall_clock_budget_uses_remaining_time_for_native_solver():
    now = [10.0]
    delegate = BudgetDelegate()
    checker = WallClockBudgetChecker(
        delegate,
        timeout_seconds=120.0,
        spent_seconds=30.5,
        max_solver_timeout_seconds=300.0,
        clock=lambda: now[0],
    )

    result = checker.check(request())

    assert result.status is CheckStatus.VIOLATED
    assert delegate.calls == 1
    assert delegate.timeouts == [90]
    assert result.metadata["query_budget_exhausted"] is False


def test_wall_clock_budget_returns_unknown_without_solver_when_exhausted():
    delegate = BudgetDelegate()
    checker = WallClockBudgetChecker(
        delegate,
        timeout_seconds=60.0,
        spent_seconds=60.0,
        clock=lambda: 5.0,
    )

    result = checker.check(request())

    assert result.status is CheckStatus.UNKNOWN
    assert result.metadata["query_budget_exhausted"] is True
    assert result.metadata["query_budget_solver_skipped"] is True
    assert delegate.calls == 0
    assert delegate.timeouts == []
