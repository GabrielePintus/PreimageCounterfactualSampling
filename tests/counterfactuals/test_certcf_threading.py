from __future__ import annotations

import time
from typing import Any

import numpy as np

from certcf.atlas import CertCFAtlas, CounterfactualResult as AtlasCounterfactualResult
from counterfactuals.methods.certcf import CertCF


def _atlas_for_threading(query_parallelism: int = 1) -> CertCFAtlas:
    atlas = CertCFAtlas.__new__(CertCFAtlas)
    atlas.query_parallelism = int(query_parallelism)
    atlas.default_query_method = "nearest_anchor"
    atlas.norm = 1
    return atlas


def _atlas_result(x_cf: np.ndarray, target_class: int, *, success: bool = True) -> AtlasCounterfactualResult:
    return AtlasCounterfactualResult(
        x_cf=np.asarray(x_cf, dtype=np.float32) if x_cf is not None else None,
        distance=float(np.linalg.norm(x_cf)) if x_cf is not None else float("inf"),
        target_class=int(target_class),
        anchor_idx=0 if success else None,
        n_qp_solved=1 if success else 0,
        success=bool(success),
        profiling={"target_class": int(target_class)},
    )


class _ImmediateFuture:
    def __init__(self, result: Any):
        self._result = result

    def result(self):
        return self._result

    def __hash__(self):
        return id(self)


class _RecordingExecutor:
    def __init__(self, max_workers: int, thread_name_prefix: str):
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self.submissions: list[tuple[np.ndarray, int]] = []

    def submit(self, fn, **kwargs):
        self.submissions.append(
            (np.asarray(kwargs["x_query"], dtype=np.float32).copy(), int(kwargs["target_class"]))
        )
        return _ImmediateFuture(fn(**kwargs))

    def shutdown(self, wait: bool = True, cancel_futures: bool = False):
        del wait, cancel_futures


def test_atlas_find_counterfactual_batch_preserves_order_and_groups_targets(monkeypatch):
    atlas = _atlas_for_threading(query_parallelism=3)
    recording_executor = _RecordingExecutor(max_workers=3, thread_name_prefix="certcf-query")

    monkeypatch.setattr("certcf.atlas.ThreadPoolExecutor", lambda max_workers, thread_name_prefix: recording_executor)
    monkeypatch.setattr("certcf.atlas.wait", lambda futures, timeout=None, return_when=None: (set(futures), set()))

    def fake_find_counterfactual(
        *,
        x_query,
        target_class,
        method=None,
        delta=0.0,
        robust_norm=None,
        solver_maxiter=None,
        fixed_dims=None,
        query_k_candidates=1,
    ):
        del method, delta, robust_norm, solver_maxiter, fixed_dims, query_k_candidates
        return _atlas_result(
            x_cf=np.array([x_query[0], target_class], dtype=np.float32),
            target_class=target_class,
        )

    atlas.find_counterfactual = fake_find_counterfactual  # type: ignore[method-assign]
    queries = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    targets = np.array([1, 0, 1, 0], dtype=np.int64)

    results = atlas.find_counterfactual_batch(X_query=queries, target_class=targets)

    assert [int(r.target_class) for r in results] == [1, 0, 1, 0]
    assert [float(r.x_cf[0]) for r in results] == [1.0, 2.0, 3.0, 4.0]
    assert [submission[1] for submission in recording_executor.submissions] == [1, 1, 0, 0]


def test_atlas_find_counterfactual_batch_parallel_matches_serial_outputs():
    queries = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    targets = np.array([1, 0, 1, 0], dtype=np.int64)

    def fake_find_counterfactual(
        *,
        x_query,
        target_class,
        method=None,
        delta=0.0,
        robust_norm=None,
        solver_maxiter=None,
        fixed_dims=None,
        query_k_candidates=1,
    ):
        del method, delta, robust_norm, solver_maxiter, fixed_dims, query_k_candidates
        return _atlas_result(
            x_cf=np.array([x_query[0], target_class], dtype=np.float32),
            target_class=target_class,
        )

    atlas_serial = _atlas_for_threading(query_parallelism=1)
    atlas_serial.find_counterfactual = fake_find_counterfactual  # type: ignore[method-assign]
    atlas_parallel = _atlas_for_threading(query_parallelism=2)
    atlas_parallel.find_counterfactual = fake_find_counterfactual  # type: ignore[method-assign]

    serial_results = atlas_serial.find_counterfactual_batch(X_query=queries, target_class=targets)
    parallel_results = atlas_parallel.find_counterfactual_batch(X_query=queries, target_class=targets)

    assert [int(r.target_class) for r in parallel_results] == [int(r.target_class) for r in serial_results]
    assert [r.success for r in parallel_results] == [r.success for r in serial_results]
    assert [r.x_cf.tolist() for r in parallel_results] == [r.x_cf.tolist() for r in serial_results]


def test_atlas_find_counterfactual_batch_uses_real_thread_parallelism():
    queries = np.arange(4, dtype=np.float32).reshape(-1, 1)
    targets = np.array([1, 1, 1, 1], dtype=np.int64)

    def sleeping_find_counterfactual(
        *,
        x_query,
        target_class,
        method=None,
        delta=0.0,
        robust_norm=None,
        solver_maxiter=None,
        fixed_dims=None,
        query_k_candidates=1,
    ):
        del method, delta, robust_norm, solver_maxiter, fixed_dims, query_k_candidates
        time.sleep(0.10)
        return _atlas_result(
            x_cf=np.array([x_query[0]], dtype=np.float32),
            target_class=target_class,
        )

    atlas_serial = _atlas_for_threading(query_parallelism=1)
    atlas_serial.find_counterfactual = sleeping_find_counterfactual  # type: ignore[method-assign]
    atlas_parallel = _atlas_for_threading(query_parallelism=4)
    atlas_parallel.find_counterfactual = sleeping_find_counterfactual  # type: ignore[method-assign]

    t0 = time.perf_counter()
    atlas_serial.find_counterfactual_batch(X_query=queries, target_class=targets)
    serial_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    atlas_parallel.find_counterfactual_batch(X_query=queries, target_class=targets)
    parallel_time = time.perf_counter() - t0

    assert parallel_time < serial_time * 0.7


def test_atlas_find_counterfactual_batch_soft_timeout_marks_only_slow_queries():
    atlas = _atlas_for_threading(query_parallelism=2)
    queries = np.array([[0.0], [1.0]], dtype=np.float32)
    targets = np.array([1, 0], dtype=np.int64)

    def sleeping_find_counterfactual(
        *,
        x_query,
        target_class,
        method=None,
        delta=0.0,
        robust_norm=None,
        solver_maxiter=None,
        fixed_dims=None,
        query_k_candidates=1,
    ):
        del method, delta, robust_norm, solver_maxiter, fixed_dims, query_k_candidates
        time.sleep(0.15 if int(target_class) == 1 else 0.01)
        return _atlas_result(
            x_cf=np.array([x_query[0]], dtype=np.float32),
            target_class=target_class,
        )

    atlas.find_counterfactual = sleeping_find_counterfactual  # type: ignore[method-assign]

    t0 = time.perf_counter()
    results = atlas.find_counterfactual_batch(
        X_query=queries,
        target_class=targets,
        timeout_s_per_query=0.05,
    )
    elapsed = time.perf_counter() - t0

    assert [result.success for result in results] == [False, True]
    assert results[0].profiling["reason"] == "timeout"
    assert bool(results[0].profiling["timed_out"]) is True
    assert results[1].success is True
    assert elapsed < 0.14

    time.sleep(0.20)


def test_certcf_generate_batch_wraps_atlas_results_and_accepts_vector_targets():
    class FakeAtlas:
        def __init__(self):
            self.calls = []

        def find_counterfactual_batch(self, **kwargs):
            self.calls.append(kwargs)
            return [
                _atlas_result(np.array([1.0, 0.0], dtype=np.float32), target_class=1),
                AtlasCounterfactualResult(
                    x_cf=None,
                    distance=float("inf"),
                    target_class=0,
                    anchor_idx=None,
                    n_qp_solved=0,
                    success=False,
                    profiling={"reason": "timeout", "timed_out": True},
                ),
            ]

    method = CertCF.__new__(CertCF)
    method._is_fitted = True
    method.atlas = FakeAtlas()
    method.delta = 0.25
    method.robust_norm = None
    method.query_k_candidates = 3

    results = method.generate_batch(
        x=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        target_class=np.array([1, 0], dtype=np.int64),
        timeout_s_per_query=0.5,
    )

    assert len(results) == 2
    assert results[0].success is True
    assert np.allclose(results[0].x_cf, np.array([1.0, 0.0], dtype=np.float32))
    assert results[1].success is False
    assert results[1].metadata["reason"] == "timeout"

    atlas_call = method.atlas.calls[0]
    assert np.array_equal(atlas_call["target_class"], np.array([1, 0], dtype=np.int64))
    assert np.isclose(atlas_call["timeout_s_per_query"], 0.5)
