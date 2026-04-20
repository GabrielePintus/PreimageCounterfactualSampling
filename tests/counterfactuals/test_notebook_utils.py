from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_specs import get_tabular_dataset_spec
from notebooks.utils.constraints import (
    _ohe_block_valid_mask,
    constraint_quality_rows,
    constraint_quality_summary,
    constraint_quality_tables,
)
from notebooks.utils.filters import (
    annotate_common_success_subset,
    shared_success_retention_summary,
    shared_success_subset,
)
from notebooks.utils.io import (
    _resolve_existing_path,
    load_result_set,
    normalize_benchmark_df,
    prepare_benchmark_df,
)
from notebooks.utils.manifoldness import load_tabular_manifold_resources
from notebooks.utils.plots import (
    plot_conditioned_proximity_kdes,
    plot_proximity_kdes_by_dataset,
    plot_validity_distance_curves,
)
from notebooks.utils.robustness import evaluate_empirical_robustness_curves, sample_lp_ball
from notebooks.utils.style import method_palette, style_method_table
from notebooks.utils.summaries import (
    curve_endpoint_table,
    manifoldness_summary,
    matched_success_summary,
    method_order,
    method_comparison_summary,
    validity_proximity_curve,
    validity_summary,
)


def _demo_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["certcf", "dice", "certcf", "dice"],
            "query_idx": [0, 0, 1, 1],
            "success": [True, True, True, False],
            "space": ["raw", "raw", "raw", "raw"],
            "build_time_s": [1.0, 2.0, 1.0, 2.0],
            "runtime_s": [0.1, 0.2, 0.1, 0.3],
            "l2_distance": [0.5, 0.7, 0.4, np.nan],
            "l1_distance": [1.0, 1.2, 0.9, np.nan],
            "mad_l1_distance": [0.6, 0.8, 0.5, np.nan],
            "l0_sparsity": [0.2, 0.3, 0.1, np.nan],
            "redundancy": [0.1, 0.2, 0.05, np.nan],
            "y_orig": [0, 0, 1, 1],
            "target_class": [1, 1, 0, 0],
            "x_orig_0": [0.0, 0.0, 1.0, 1.0],
            "x_cf_0": [0.4, 0.6, 0.7, np.nan],
        }
    )


def test_normalize_benchmark_df_adds_standard_columns():
    df = normalize_benchmark_df(_demo_df())

    assert "method_label" in df.columns
    assert "source_class" in df.columns
    assert df.loc[0, "method_label"] == "CertCF"
    assert int(df.loc[0, "source_class"]) == 0


def test_normalize_benchmark_df_preserves_distinct_run_name_variants():
    df = normalize_benchmark_df(
        pd.DataFrame(
            {
                "method": ["certcf", "certcf", "dice"],
                "run_name": ["certcf_input_kmedoids", "certcf_latent_kmedoids", "dice"],
                "query_idx": [0, 0, 0],
                "success": [True, True, True],
                "y_orig": [0, 0, 0],
                "target_class": [1, 1, 1],
                "x_orig_0": [0.0, 0.0, 0.0],
                "x_cf_0": [0.1, 0.2, 0.3],
            }
        )
    )

    assert list(df["method_label"]) == [
        "certcf_input_kmedoids",
        "certcf_latent_kmedoids",
        "DiCE",
    ]


def test_load_result_set_attaches_dataset_and_status(monkeypatch):
    demo = _demo_df()

    def fake_read_parquet(path):
        del path
        return demo.copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(Path, "exists", lambda self: True)

    df, status = load_result_set({"adult": Path("adult.parquet"), "compas": Path("compas.parquet")})

    assert set(df["dataset"].unique()) == {"adult", "compas"}
    assert status["available"].all()


def test_normalize_benchmark_df_accepts_method_label_fn():
    df = normalize_benchmark_df(
        _demo_df(),
        method_label_fn=lambda frame: frame["method"].map({"certcf": "CertCF top-3", "dice": "DiCE"}),
    )
    assert set(df["method_label"]) == {"CertCF top-3", "DiCE"}


def test_prepare_benchmark_df_filters_and_orders():
    df = normalize_benchmark_df(
        pd.DataFrame(
            {
                "dataset": ["compas", "adult", "adult"],
                "method": ["dice", "certcf", "dice"],
                "query_idx": [0, 0, 1],
                "success": [True, True, False],
                "y_orig": [0, 0, 1],
                "target_class": [1, 1, 0],
                "x_orig_0": [0.0, 0.0, 1.0],
                "x_cf_0": [0.1, 0.2, 0.3],
            }
        )
    )
    prepared = prepare_benchmark_df(
        df,
        methods=["CertCF"],
        datasets=["adult"],
        method_order=["CertCF"],
        dataset_order=["adult", "compas"],
        success_only_rows=True,
    )
    assert list(prepared["dataset"].astype(str)) == ["adult"]
    assert list(prepared["method_label"].astype(str)) == ["CertCF"]


def test_prepare_benchmark_df_sorts_by_method_order_without_dataset_order():
    df = normalize_benchmark_df(
        pd.DataFrame(
            {
                "method": ["dice", "certcf", "face"],
                "query_idx": [0, 0, 0],
                "success": [True, True, True],
                "y_orig": [0, 0, 0],
                "target_class": [1, 1, 1],
                "x_orig_0": [0.0, 0.0, 0.0],
                "x_cf_0": [0.1, 0.2, 0.3],
            }
        )
    )

    prepared = prepare_benchmark_df(
        df,
        method_order=["CertCF", "FACE", "DiCE"],
    )

    assert list(prepared["method_label"].astype(str)) == ["CertCF", "FACE", "DiCE"]


def test_resolve_existing_path_accepts_notebook_relative_result_path(monkeypatch):
    path = _resolve_existing_path(Path("../results/benchmark_adult.parquet"))
    assert path is None or path.name == "benchmark_adult.parquet"


def test_method_order_sorts_by_validity_desc():
    df = normalize_benchmark_df(_demo_df())
    assert method_order(df) == ["CertCF", "DiCE"]


def test_shared_success_subset_keeps_only_universal_success_tasks():
    df = normalize_benchmark_df(_demo_df())
    shared = shared_success_subset(df, by=("query_idx",), method_col="method_label")

    assert set(shared["query_idx"].unique()) == {0}
    assert shared["success"].all()
    assert set(shared["method_label"].unique()) == {"CertCF", "DiCE"}


def test_shared_success_subset_is_dataset_aware_when_requested():
    df = normalize_benchmark_df(
        pd.DataFrame(
            {
                "dataset": ["adult", "adult", "adult", "adult", "compas", "compas"],
                "method": ["certcf", "dice", "certcf", "dice", "certcf", "dice"],
                "query_idx": [0, 0, 1, 1, 0, 0],
                "success": [True, True, True, False, True, True],
                "space": ["raw"] * 6,
                "build_time_s": [1.0] * 6,
                "runtime_s": [0.1] * 6,
                "l2_distance": [0.5, 0.7, 0.4, np.nan, 0.3, 0.2],
                "l1_distance": [1.0, 1.2, 0.9, np.nan, 0.8, 0.7],
                "mad_l1_distance": [0.6, 0.8, 0.5, np.nan, 0.4, 0.3],
                "l0_sparsity": [0.2, 0.3, 0.1, np.nan, 0.2, 0.1],
                "redundancy": [0.1, 0.2, 0.05, np.nan, 0.1, 0.1],
                "y_orig": [0, 0, 1, 1, 0, 0],
                "target_class": [1, 1, 0, 0, 1, 1],
                "x_orig_0": [0.0, 0.0, 1.0, 1.0, 0.5, 0.5],
                "x_cf_0": [0.4, 0.6, 0.7, np.nan, 0.6, 0.7],
            }
        )
    )

    shared = shared_success_subset(df, by=("dataset", "query_idx"), method_col="method_label")

    kept_tasks = {
        (row.dataset, int(row.query_idx))
        for row in shared[["dataset", "query_idx"]].drop_duplicates().itertuples(index=False)
    }
    assert kept_tasks == {("adult", 0), ("compas", 0)}


def test_shared_success_retention_summary_reports_task_counts():
    df = normalize_benchmark_df(
        pd.DataFrame(
            {
                "dataset": ["adult", "adult", "adult", "adult", "compas", "compas"],
                "method": ["certcf", "dice", "certcf", "dice", "certcf", "dice"],
                "query_idx": [0, 0, 1, 1, 0, 0],
                "success": [True, True, True, False, True, True],
                "space": ["raw"] * 6,
                "build_time_s": [1.0] * 6,
                "runtime_s": [0.1] * 6,
                "l2_distance": [0.5, 0.7, 0.4, np.nan, 0.3, 0.2],
                "l1_distance": [1.0, 1.2, 0.9, np.nan, 0.8, 0.7],
                "mad_l1_distance": [0.6, 0.8, 0.5, np.nan, 0.4, 0.3],
                "l0_sparsity": [0.2, 0.3, 0.1, np.nan, 0.2, 0.1],
                "redundancy": [0.1, 0.2, 0.05, np.nan, 0.1, 0.1],
                "y_orig": [0, 0, 1, 1, 0, 0],
                "target_class": [1, 1, 0, 0, 1, 1],
                "x_orig_0": [0.0, 0.0, 1.0, 1.0, 0.5, 0.5],
                "x_cf_0": [0.4, 0.6, 0.7, np.nan, 0.6, 0.7],
            }
        )
    )

    summary = shared_success_retention_summary(
        df,
        by=("dataset", "query_idx"),
        group_cols=("dataset",),
        method_col="method_label",
    )

    assert int(summary.loc["adult", "n_total_tasks"]) == 2
    assert int(summary.loc["adult", "n_shared_tasks"]) == 1
    assert float(summary.loc["adult", "retention_pct"]) == 50.0
    assert int(summary.loc["compas", "n_total_tasks"]) == 1
    assert int(summary.loc["compas", "n_shared_tasks"]) == 1


def test_annotate_common_success_subset_marks_inside_and_outside():
    df = normalize_benchmark_df(_demo_df())
    annotated = annotate_common_success_subset(df, by=("query_idx",), method_col="method_label")
    subset_by_query = annotated.groupby("query_idx")["subset_group"].first().to_dict()
    assert subset_by_query[0] == "Common-success subset"
    assert subset_by_query[1] == "Outside common subset"


def test_annotate_common_success_subset_never_marks_failed_rows_as_common():
    df = normalize_benchmark_df(_demo_df())
    annotated = annotate_common_success_subset(df, by=("query_idx",), method_col="method_label")

    failed_row = annotated.loc[
        (annotated["query_idx"] == 1) & (annotated["method_label"] == "DiCE")
    ].iloc[0]

    assert bool(failed_row["success"]) is False
    assert failed_row["subset_group"] == "Outside common subset"


def test_validity_summary_returns_expected_counts():
    df = normalize_benchmark_df(_demo_df())
    summary = validity_summary(df)

    assert float(summary.loc["CertCF", "validity_pct"]) == 100.0
    assert float(summary.loc["DiCE", "validity_pct"]) == 50.0
    assert int(summary.loc["CertCF", "n_total"]) == 2


def test_method_comparison_summary_combines_validity_proximity_and_runtime():
    df = normalize_benchmark_df(
        pd.DataFrame(
            {
                "dataset": ["adult", "adult", "adult", "adult"],
                "method": ["certcf", "certcf", "dice", "dice"],
                "query_idx": [0, 1, 0, 1],
                "success": [True, True, True, False],
                "runtime_s": [0.1, 0.2, 0.3, 0.4],
                "l1_distance": [1.0, 2.0, 3.0, np.nan],
                "l2_distance": [0.5, 0.8, 1.5, np.nan],
                "y_orig": [0, 1, 0, 1],
                "target_class": [1, 0, 1, 0],
                "x_orig_0": [0.0, 1.0, 0.0, 1.0],
                "x_cf_0": [0.2, 0.8, 0.3, np.nan],
            }
        )
    )
    summary = method_comparison_summary(df, order=["CertCF", "DiCE"])
    assert float(summary.loc[0, "valid_pct"]) == 100.0
    assert float(summary.loc[1, "valid_pct"]) == 50.0
    assert float(summary.loc[0, "l1_mean"]) == 1.5
    assert float(summary.loc[1, "runtime_median_s"]) == 0.35


def test_matched_success_summary_returns_common_counts_and_rows():
    df = normalize_benchmark_df(
        pd.DataFrame(
            {
                "dataset": ["adult", "adult", "adult", "adult"],
                "method": ["certcf", "dice", "certcf", "dice"],
                "query_idx": [0, 0, 1, 1],
                "success": [True, True, True, False],
                "runtime_s": [0.1, 0.2, 0.1, 0.3],
                "l1_distance": [1.0, 1.2, 0.9, np.nan],
                "l2_distance": [0.5, 0.7, 0.4, np.nan],
                "y_orig": [0, 0, 1, 1],
                "target_class": [1, 1, 0, 0],
                "x_orig_0": [0.0, 0.0, 1.0, 1.0],
                "x_cf_0": [0.4, 0.6, 0.7, np.nan],
            }
        )
    )
    counts, matched = matched_success_summary(df, order=["CertCF", "DiCE"])
    assert int(counts.loc[0, "n_common_queries"]) == 1
    assert set(matched["method_label"].astype(str)) == {"CertCF", "DiCE"}
    assert set(matched["n_common_queries"]) == {1}


def test_matched_success_summary_ignores_failed_rows_even_if_task_is_common():
    df = normalize_benchmark_df(
        pd.DataFrame(
            {
                "dataset": ["adult", "adult", "adult", "adult", "adult"],
                "method": ["certcf", "certcf", "dice", "dice", "dice"],
                "query_idx": [0, 1, 0, 1, 1],
                "success": [True, True, True, True, False],
                "runtime_s": [0.1, 0.2, 0.3, 0.4, 9.9],
                "l1_distance": [1.0, 0.9, 1.2, 1.1, 99.0],
                "l2_distance": [0.5, 0.4, 0.7, 0.6, 99.0],
                "y_orig": [0, 1, 0, 1, 1],
                "target_class": [1, 0, 1, 0, 0],
                "x_orig_0": [0.0, 1.0, 0.0, 1.0, 1.0],
                "x_cf_0": [0.4, 0.7, 0.6, 0.8, np.nan],
            }
        )
    )

    counts, matched = matched_success_summary(df, order=["CertCF", "DiCE"])

    assert int(counts.loc[0, "n_common_queries"]) == 2
    dice_row = matched.loc[matched["method_label"].astype(str) == "DiCE"].iloc[0]
    assert np.isclose(float(dice_row["l1_mean"]), 1.15)
    assert np.isclose(float(dice_row["runtime_mean_s"]), 0.35)


def test_validity_proximity_curve_returns_auc_and_percent_curve():
    eps, curve, auc = validity_proximity_curve(np.array([0.2, 0.6, np.inf], dtype=float))

    assert eps.ndim == 1
    assert curve.ndim == 1
    assert len(eps) == len(curve)
    assert 0.0 <= auc <= 1.0
    assert curve[-1] <= 100.0


def test_curve_endpoint_table_uses_largest_epsilon_per_group():
    curve_df = pd.DataFrame(
        {
            "method_label": ["CertCF", "CertCF", "DiCE", "DiCE"],
            "dataset": ["adult", "adult", "adult", "adult"],
            "epsilon": [0.1, 0.2, 0.1, 0.2],
            "success_pct": [80.0, 60.0, 75.0, 55.0],
        }
    )

    table = curve_endpoint_table(curve_df, order=["CertCF", "DiCE"])

    assert float(table.loc["CertCF", "adult"]) == 60.0
    assert float(table.loc["DiCE", "adult"]) == 55.0


def test_manifoldness_summary_aggregates_available_metrics():
    df = pd.DataFrame(
        {
            "dataset": ["adult", "adult", "adult", "adult"],
            "method_label": ["CertCF", "CertCF", "DiCE", "DiCE"],
            "query_idx": [0, 1, 0, 1],
            "manifold_knn5_all": [0.2, 0.4, 0.6, 0.8],
            "manifold_knn5_target_pred": [0.1, 0.3, 0.5, 0.7],
            "manifold_neg_lof_all": [1.0, 1.5, 2.0, 2.5],
        }
    )
    summary = manifoldness_summary(df, order=["CertCF", "DiCE"])
    assert np.isclose(float(summary.loc[0, "knn5_all_mean"]), 0.3)
    assert np.isclose(float(summary.loc[1, "knn5_target_median"]), 0.6)
    assert np.isclose(float(summary.loc[0, "neg_lof_all_mean"]), 1.25)


def test_load_tabular_manifold_resources_defaults_data_dir_to_repo_data(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class FakeDataset:
        def load(self):
            return None

        def get_train(self):
            return np.array([[0.0], [1.0]], dtype=np.float32), np.array([0, 1], dtype=np.int64)

    class FakeDatasetRegistry:
        def create(self, name, **kwargs):
            captured["name"] = name
            captured["kwargs"] = kwargs
            return FakeDataset()

    fake_registries = {"dataset": FakeDatasetRegistry()}

    class FakeModel:
        def predict(self, x):
            return np.zeros(len(x), dtype=np.int64)

    config_path = tmp_path / "bench.yaml"
    config_path.write_text("dummy: true\n")

    monkeypatch.setattr(
        "notebooks.utils.manifoldness._resolve_existing_path",
        lambda path: config_path,
    )
    monkeypatch.setattr(
        "notebooks.utils.manifoldness.read_yaml",
        lambda path: {
            "datasets": [
                {
                    "name": "adult",
                    "model": {
                        "name": "tabular_classifier_ckpt",
                        "params": {
                            "checkpoint": "checkpoints/adult_classifier/best.ckpt",
                            "dataset_module": "adult",
                            "hidden_dims": [32, 8],
                            "dropout": 0.2,
                        },
                    },
                }
            ]
        },
    )
    monkeypatch.setattr(
        "notebooks.utils.manifoldness.create_default_registries",
        lambda: fake_registries,
    )
    monkeypatch.setattr(
        "notebooks.utils.manifoldness._build_torch_model_from_checkpoint",
        lambda **kwargs: FakeModel(),
    )

    resources = load_tabular_manifold_resources(config_path, datasets=["adult"], device="cpu")

    assert "adult" in resources
    assert captured["name"] == "adult"
    assert captured["kwargs"]["data_dir"] == str(ROOT / "data")


def test_sample_lp_ball_respects_radius_for_supported_norms():
    rng = np.random.default_rng(0)

    for norm in ("l1", "l2", "linf"):
        noise = sample_lp_ball(rng, n_samples=32, dim=5, radius=0.3, norm=norm)
        if norm == "l1":
            norms = np.linalg.norm(noise, ord=1, axis=1)
        elif norm == "l2":
            norms = np.linalg.norm(noise, ord=2, axis=1)
        else:
            norms = np.linalg.norm(noise, ord=np.inf, axis=1)
        assert np.all(norms <= 0.300001)


def test_evaluate_empirical_robustness_curves_returns_expected_schema():
    class ConstantModel:
        def predict(self, x):
            return np.zeros(len(x), dtype=np.int64)

    spec = get_tabular_dataset_spec("heloc")
    df = pd.DataFrame(
        {
            "dataset": ["heloc", "heloc"],
            "method_label": ["CertCF", "DiCE"],
            "success": [True, True],
            "target_class": [0, 0],
            **{
                f"x_cf_{idx}": [float(idx) / 10.0, float(idx + 1) / 10.0]
                for idx in range(spec.n_features)
            },
        }
    )

    curves = evaluate_empirical_robustness_curves(
        df,
        models_by_dataset={"heloc": ConstantModel()},
        eps_by_norm={"l1": [0.0], "l2": [0.1], "linf": [0.2]},
        n_samples=4,
        seed=0,
    )

    assert set(curves["norm"].unique()) == {"l1", "l2", "linf"}
    assert set(curves["method_label"].unique()) == {"CertCF", "DiCE"}
    assert {"dataset", "epsilon", "success_pct", "mean_preservation_pct", "n_queries", "n_samples"} <= set(curves.columns)


def test_evaluate_empirical_robustness_curves_falls_back_to_binary_target_from_source_class():
    class ConstantModel:
        def predict(self, x):
            return np.ones(len(x), dtype=np.int64)

    spec = get_tabular_dataset_spec("heloc")
    df = pd.DataFrame(
        {
            "dataset": ["heloc"],
            "method_label": ["CertCF"],
            "success": [True],
            "source_class": [0],
            **{
                f"x_cf_{idx}": [float(idx) / 10.0]
                for idx in range(spec.n_features)
            },
        }
    )

    curves = evaluate_empirical_robustness_curves(
        df,
        models_by_dataset={"heloc": ConstantModel()},
        eps_by_norm={"l2": [0.1]},
        n_samples=4,
        seed=0,
    )

    assert float(curves.loc[0, "success_pct"]) == 100.0


def test_ohe_block_valid_mask_accepts_strict_one_hot_blocks():
    spec = get_tabular_dataset_spec("compas")
    x_cf = np.zeros((1, spec.n_features), dtype=np.float32)
    x_cf[0, 0:4] = [35.0, 2.0, 1.0, -3.0]
    x_cf[0, 4:6] = [1.0, 0.0]
    x_cf[0, 6:12] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    x_cf[0, 12:14] = [0.0, 1.0]

    valid = _ohe_block_valid_mask(x_cf, dataset_name="compas")

    assert valid.shape == (1, 3)
    assert valid.all()


def test_constraint_quality_rows_flags_invalid_soft_and_multihot_blocks():
    spec = get_tabular_dataset_spec("compas")
    valid = np.zeros(spec.n_features, dtype=np.float32)
    invalid = np.zeros(spec.n_features, dtype=np.float32)

    valid[0:4] = [35.0, 2.0, 1.0, -3.0]
    valid[4:6] = [1.0, 0.0]
    valid[6:12] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    valid[12:14] = [0.0, 1.0]

    invalid[0:4] = [40.0, 1.0, 0.0, 5.0]
    invalid[4:6] = [0.5, 0.5]
    invalid[6:12] = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    invalid[12:14] = [0.2, 0.7]

    df = pd.DataFrame(
        {
            "dataset": ["compas", "compas"],
            "method_label": ["CertCF", "DiCE"],
            **{f"x_orig_{idx}": [valid[idx], valid[idx]] for idx in range(spec.n_features)},
            **{f"x_cf_{idx}": [valid[idx], invalid[idx]] for idx in range(spec.n_features)},
        }
    )

    rows = constraint_quality_rows(df)

    assert bool(rows.loc[0, "ohe_valid"]) is True
    assert bool(rows.loc[1, "ohe_valid"]) is False
    assert float(rows.loc[1, "invalid_block_fraction"]) == 1.0
    assert float(rows.loc[1, "ohe_snap_l1"]) > 0.0
    assert bool(rows.loc[0, "immutable_valid"]) is True
    assert bool(rows.loc[1, "immutable_valid"]) is False
    assert float(rows.loc[1, "immutable_change_fraction"]) == 1.0
    assert float(rows.loc[1, "immutable_l1"]) > 0.0


def test_constraint_quality_summary_excludes_numerical_only_datasets():
    heloc = pd.DataFrame(
        {
            "dataset": ["heloc"],
            "method_label": ["CertCF"],
            "x_cf_0": [0.1],
            "x_cf_1": [0.2],
        }
    )

    summary = constraint_quality_summary(heloc)
    tables = constraint_quality_tables(heloc)

    assert summary.empty
    assert all(table.empty for table in tables.values())


def test_constraint_quality_summary_and_tables_aggregate_by_method_and_dataset():
    spec = get_tabular_dataset_spec("adult")
    valid = np.zeros(spec.n_features, dtype=np.float32)
    soft = np.zeros(spec.n_features, dtype=np.float32)

    for idx, (start, end) in enumerate(spec.feature_slices):
        if spec.input_types[idx] == "numerical":
            valid[start:end] = [float(idx + 1)]
            soft[start:end] = [float(idx + 2)]
        else:
            valid[start:end] = 0.0
            valid[start] = 1.0
            soft[start:end] = 1.0 / (end - start)

    df = pd.DataFrame(
        {
            "dataset": ["adult", "adult", "adult", "adult"],
            "method_label": ["CertCF", "CertCF", "DiCE", "DiCE"],
            **{f"x_orig_{idx}": [valid[idx], valid[idx], valid[idx], valid[idx]] for idx in range(spec.n_features)},
            **{f"x_cf_{idx}": [valid[idx], valid[idx], valid[idx], soft[idx]] for idx in range(spec.n_features)},
        }
    )

    summary = constraint_quality_summary(df, order=["CertCF", "DiCE"])
    tables = constraint_quality_tables(df, order=["CertCF", "DiCE"])

    assert float(summary.loc["CertCF", "ohe_valid_pct"]) == 100.0
    assert float(summary.loc["DiCE", "ohe_valid_pct"]) == 50.0
    assert float(summary.loc["DiCE", "mean_invalid_block_pct"]) == 50.0
    assert float(summary.loc["DiCE", "mean_ohe_snap_l1"]) > 0.0
    assert float(summary.loc["CertCF", "immutable_valid_pct"]) == 100.0
    assert float(summary.loc["DiCE", "immutable_valid_pct"]) == 50.0
    assert float(summary.loc["DiCE", "mean_immutable_change_pct"]) == 50.0
    assert float(summary.loc["DiCE", "mean_immutable_l1"]) > 0.0
    assert list(tables["ohe_valid_pct"].index) == ["CertCF", "DiCE"]
    assert float(tables["ohe_valid_pct"].loc["CertCF", "adult"]) == 100.0
    assert float(tables["immutable_valid_pct"].loc["CertCF", "adult"]) == 100.0


def test_style_method_table_bolds_best_values_per_selected_column():
    table = pd.DataFrame(
        {
            "validity_pct": [100.0, 80.0],
            "mean_l2": [0.3, 0.5],
        },
        index=["CertCF", "DiCE"],
    )

    styler = style_method_table(
        table,
        fmt={"validity_pct": "{:.1f}%", "mean_l2": "{:.3f}"},
        maximize=["validity_pct"],
        minimize=["mean_l2"],
    )
    ctx = styler._compute().ctx

    assert ("font-weight", "bold") in ctx[(0, 0)]
    assert ("font-weight", "bold") in ctx[(0, 1)]
    assert (1, 0) not in ctx or ("font-weight", "bold") not in ctx[(1, 0)]


def test_method_palette_keeps_certcf_red():
    palette = method_palette(["DiCE", "CertCF", "FACE"])
    assert tuple(round(v, 3) for v in palette["CertCF"]) == (0.839, 0.153, 0.157)


def test_method_palette_normalizes_certcf_variants_to_red():
    palette = method_palette(["certcf", "CertCF", "cert cf"])
    expected = (0.839, 0.153, 0.157)
    assert tuple(round(v, 3) for v in palette["certcf"]) == expected
    assert tuple(round(v, 3) for v in palette["CertCF"]) == expected
    assert tuple(round(v, 3) for v in palette["cert cf"]) == expected


def test_method_palette_normalizes_cpp_aliases_to_red():
    palette = method_palette(["cpp", "CPP", "CertCF"])
    expected = (0.839, 0.153, 0.157)
    assert tuple(round(v, 3) for v in palette["cpp"]) == expected
    assert tuple(round(v, 3) for v in palette["CPP"]) == expected
    assert tuple(round(v, 3) for v in palette["CertCF"]) == expected


def test_plot_validity_distance_curves_keeps_certcf_red_with_variant_labels():
    curve_df = pd.DataFrame(
        {
            "method_label": ["cert cf", "cert cf", "DiCE", "DiCE"],
            "epsilon": [0.0, 1.0, 0.0, 1.0],
            "success_pct": [100.0, 80.0, 100.0, 60.0],
        }
    )
    palette = method_palette(["CertCF", "DiCE"])
    fig, ax = plot_validity_distance_curves(curve_df, palette=palette, title="Test")
    certcf_color = tuple(round(v, 3) for v in ax.lines[0].get_color()[:3])
    assert certcf_color == (0.839, 0.153, 0.157)
    plt.close(fig)


def test_plot_proximity_kdes_by_dataset_returns_one_figure_per_dataset():
    df = pd.DataFrame(
        {
            "dataset": ["adult", "adult", "adult", "adult"],
            "method_label": ["CertCF", "CertCF", "DiCE", "DiCE"],
            "l1_distance": [1.0, 1.2, 1.5, 1.7],
            "l2_distance": [0.5, 0.6, 0.8, 0.9],
        }
    )
    palette = method_palette(["CertCF", "DiCE"])
    figures = plot_proximity_kdes_by_dataset(
        df,
        dataset_order=["adult"],
        method_order=["CertCF", "DiCE"],
        palette=palette,
    )
    assert len(figures) == 1
    _, fig, axes = figures[0]
    assert len(axes) == 2
    plt.close(fig)


def test_plot_conditioned_proximity_kdes_returns_grid():
    df = pd.DataFrame(
        {
            "dataset": ["adult"] * 8,
            "method_label": ["CertCF"] * 4 + ["DiCE"] * 4,
            "subset_group": ["Common-success subset", "Common-success subset", "Outside common subset", "Outside common subset"] * 2,
            "l1_distance": [1.0, 1.1, 1.5, 1.6, 0.8, 0.9, 1.3, 1.4],
            "l2_distance": [0.5, 0.55, 0.9, 1.0, 0.4, 0.45, 0.7, 0.8],
        }
    )
    figures = plot_conditioned_proximity_kdes(
        df,
        dataset_order=["adult"],
        method_order=["CertCF", "DiCE"],
        subset_col="subset_group",
        subset_order=["Common-success subset", "Outside common subset"],
        subset_palette={
            "Common-success subset": "#1f77b4",
            "Outside common subset": "#d62728",
        },
    )
    assert len(figures) == 1
    _, fig, axes = figures[0]
    assert len(axes) == 2
    plt.close(fig)


def test_benchmark_notebooks_do_not_define_shared_helpers_inline():
    notebook_paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert notebook_paths, "Expected at least one active notebook in notebooks/."

    for path in notebook_paths:
        nb = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        assert "def validity_proximity_curve(" not in source
        assert "def shared_success_retention_summary(" not in source
        assert "def style_method_table(" not in source
