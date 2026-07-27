from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notebooks.utils.norm_ablation import (
    NORM_ABLATION_RUNS,
    attach_l0_count,
    load_norm_ablation_results,
    norm_ablation_aggregate_summary,
    norm_ablation_completion_table,
    norm_ablation_dataset_summary,
    norm_ablation_paired_deltas,
    shared_success_norm_subset,
    validate_norm_ablation_results,
)


def _ablation_df() -> pd.DataFrame:
    rows = []
    for dataset_idx, dataset in enumerate(["adult", "compas"]):
        for query_idx in [0, 1]:
            for run_idx, run_name in enumerate(NORM_ABLATION_RUNS):
                l1 = 1.0 + dataset_idx + query_idx + 0.5 * run_idx
                l2 = 2.0 + dataset_idx + query_idx - 0.5 * run_idx
                success = not (
                    dataset == "adult" and query_idx == 1 and run_idx == 1
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "method": "certcf",
                        "run_name": run_name,
                        "query_idx": query_idx,
                        "target_class": 1,
                        "success": success,
                        "l1_distance": l1 if success else np.nan,
                        "l2_distance": l2 if success else np.nan,
                        "l0_sparsity": 0.5 if success else np.nan,
                        "runtime_s": 0.1 + 0.02 * run_idx,
                        "build_time_s": 3.0,
                        "meta__distance_norm": float(run_idx + 1),
                        "meta__sparsity_penalty": "none",
                        "x_orig_0": 0.0,
                        "x_orig_1": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_completion_table_reports_every_dataset_run():
    df = _ablation_df()

    completion = norm_ablation_completion_table(
        df,
        datasets=["adult", "compas"],
        expected_tasks=2,
    )

    assert len(completion) == 4
    assert completion["complete"].all()
    assert set(completion["unique_tasks"]) == {2}


def test_validation_accepts_matched_tasks_even_when_one_run_fails():
    validate_norm_ablation_results(
        _ablation_df(),
        datasets=["adult", "compas"],
        expected_tasks=2,
    )


def test_validation_rejects_wrong_distance_norm():
    df = _ablation_df()
    df.loc[df["run_name"].eq(NORM_ABLATION_RUNS[1]), "meta__distance_norm"] = 1.0

    with pytest.raises(ValueError, match="distance norms"):
        validate_norm_ablation_results(
            df,
            datasets=["adult", "compas"],
            expected_tasks=2,
        )


def test_shared_success_subset_uses_dataset_query_and_target_key():
    shared = shared_success_norm_subset(_ablation_df())

    assert len(shared) == 6
    retained = set(
        shared[["dataset", "query_idx"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    assert retained == {("adult", 0), ("compas", 0), ("compas", 1)}


def test_dataset_and_aggregate_summaries_use_paired_successes():
    dataset_summary = norm_ablation_dataset_summary(_ablation_df())
    aggregate = norm_ablation_aggregate_summary(dataset_summary)

    adult_l1 = dataset_summary[
        dataset_summary["dataset"].eq("adult")
        & dataset_summary["run_name"].eq(NORM_ABLATION_RUNS[0])
    ].iloc[0]
    adult_l2 = dataset_summary[
        dataset_summary["dataset"].eq("adult")
        & dataset_summary["run_name"].eq(NORM_ABLATION_RUNS[1])
    ].iloc[0]
    assert adult_l1["validity_pct"] == 100.0
    assert adult_l2["validity_pct"] == 50.0
    assert adult_l1["n_shared_success"] == adult_l2["n_shared_success"] == 1
    assert adult_l1["l0_count_mean"] == 1.0
    assert set(aggregate["n_datasets"]) == {2}


def test_paired_deltas_are_l2_minus_l1():
    query_deltas, dataset_deltas = norm_ablation_paired_deltas(_ablation_df())

    assert len(query_deltas) == 3
    np.testing.assert_allclose(
        query_deltas["l1_distance_delta_l2_minus_l1"],
        0.5,
    )
    np.testing.assert_allclose(
        query_deltas["l2_distance_delta_l2_minus_l1"],
        -0.5,
    )
    assert len(dataset_deltas) == 2
    adult = dataset_deltas[dataset_deltas["dataset"].eq("adult")].iloc[0]
    # On Adult's one shared task, L1 distance changes from 1.0 to 1.5.
    assert np.isclose(adult["l1_distance_pct_change"], 50.0)


def test_attach_l0_count_infers_encoded_dimension():
    enriched = attach_l0_count(_ablation_df())

    assert set(enriched["n_features"]) == {2}
    assert set(enriched.loc[enriched["success"], "l0_count"]) == {1.0}


def test_progress_loader_reads_per_dataset_results(tmp_path):
    combined = tmp_path / "norm.parquet"
    adult = tmp_path / "norm_adult.parquet"
    frame = _ablation_df().query("dataset == 'adult'")
    frame.to_parquet(adult, index=False)

    loaded, status = load_norm_ablation_results(
        combined,
        datasets=["adult", "compas"],
    )

    assert len(loaded) == len(frame)
    assert set(loaded["dataset"]) == {"adult"}
    assert bool(status.loc[status["source"].eq("adult"), "readable"].iloc[0])
    assert not bool(status.loc[status["source"].eq("compas"), "available"].iloc[0])
