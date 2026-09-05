from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from experiments.topk_heuristic_ablation import (
    DEFAULT_CONFIG,
    TopKHeuristicAblationRunner,
    add_minimal_k_columns,
    one_sided_binomial_upper_bound,
    strict_prefix_search,
    summarize_results,
    validate_config,
)


class _FakeAtlas:
    distance_norm = 1
    norm = 1
    solver_maxiter = 100

    def __init__(self):
        self.bounds = {
            1: {
                "X": np.array([[2.0], [3.0], [4.0]]),
                "eps": np.ones(3),
            }
        }
    @staticmethod
    def _anchor_bbox_lower_bounds(x, centers, eps, distance_norm):
        del x, centers, eps, distance_norm
        return np.zeros(3)

    @staticmethod
    def _polytope_membership_for_anchor(center, bounds, index, delta, robust_norm):
        del center, bounds, index, delta, robust_norm
        return True, True

    @staticmethod
    def _make_project_fn_for_constraints(*args, **kwargs):
        del args, kwargs
        elapsed = [0.0]
        distances = {0: 1.8, 1: 0.7, 2: 0.2}

        def project(index, incumbent):
            elapsed[0] += 0.001
            distance = distances[index]
            if distance >= incumbent:
                return None, math.inf
            return np.array([distance]), distance

        return project, elapsed, [{}]


def test_strict_prefix_search_reuses_candidates_and_is_monotone():
    states = strict_prefix_search(_FakeAtlas(), np.array([0.0]), 1, [1, 2, 3])

    assert [state.k for state in states] == [1, 2, 3]
    assert [state.n_projections for state in states] == [1, 2, 3]
    assert [state.distance for state in states] == pytest.approx([1.8, 0.7, 0.2])
    assert [state.anchor_rank for state in states] == [1, 2, 3]
    assert all(state.success for state in states)


def _result_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_position": query,
                "query_idx": 100 + query,
                "k": k,
                "strict_success": True,
                "strict_target_valid": True,
                "exact_success": True,
                "exact_recovery": k >= query + 1,
                "within_1pct": k >= query + 1,
                "within_5pct": True,
                "validity_fallback_needed": False,
                "exhaustive_fallback_needed": k < query + 1,
                "gap_absolute": 0.0 if k >= query + 1 else 1.0,
                "gap_relative": 0.0 if k >= query + 1 else 0.1,
                "strict_runtime_ms": float(k),
                "strict_projection_time_ms": float(k) / 2,
                "strict_n_projections": k,
                "exact_runtime_ms": 10.0,
                "exact_n_projections": 5,
            }
            for query in range(3)
            for k in range(1, 4)
        ]
    )


def test_minimal_k_and_summary_are_computed_per_query():
    combined, ranks = add_minimal_k_columns(_result_frame(), maximum_k=3)
    summary = summarize_results(combined, confidence=0.95)

    assert ranks["minimum_k_exact"].tolist() == [1, 2, 3]
    assert combined.groupby("query_position")["minimum_k_exact"].first().tolist() == [1, 2, 3]
    assert summary["exact_recovery_rate"].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert summary.iloc[-1]["miss_probability_upper_confidence"] == pytest.approx(
        1.0 - 0.05 ** (1.0 / 3.0)
    )


def test_zero_failure_bound_matches_rule_of_three_asymptotically():
    bound = one_sided_binomial_upper_bound(0, 1000, confidence=0.95)
    assert bound == pytest.approx(1.0 - 0.05 ** (1.0 / 1000.0))
    assert bound == pytest.approx(0.003, rel=0.01)


def test_config_rejects_sparsity_objective_for_distance_ablation():
    config = {
        **DEFAULT_CONFIG,
        "experiment": dict(DEFAULT_CONFIG["experiment"]),
        "support": dict(DEFAULT_CONFIG["support"]),
        "certcf": dict(DEFAULT_CONFIG["certcf"]),
        "artifacts": dict(DEFAULT_CONFIG["artifacts"]),
    }
    validate_config(config)
    config["certcf"]["sparsity_penalty"] = "reweighted_l1"
    with pytest.raises(ValueError, match="sparsity_penalty"):
        validate_config(config)


def test_parallel_results_are_compared_with_matching_serial_query_and_k():
    frame = pd.DataFrame(
        [
            {
                "query_position": query,
                "k": k,
                "candidate_workers_requested": workers,
                "runtime_ms": runtime,
                "distance": distance,
                "anchor_idx": anchor,
            }
            for query, k, workers, runtime, distance, anchor in [
                (0, 1, 1, 8.0, 2.0, 4),
                (0, 1, 4, 3.0, 2.0, 4),
                (0, 2, 1, 12.0, 1.5, 7),
                (0, 2, 4, 4.0, 1.5 + 1.0e-8, 7),
            ]
        ]
    )

    compared = TopKHeuristicAblationRunner._attach_parallel_serial_reference(frame)

    parallel = compared.loc[compared["candidate_workers_requested"].eq(4)].sort_values("k")
    assert parallel["speedup_vs_serial"].tolist() == pytest.approx([8.0 / 3.0, 3.0])
    assert parallel["matches_serial_distance"].all()
    assert parallel["matches_serial_anchor"].all()
