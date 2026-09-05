import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from notebooks.utils.verix_certcf import (
    FlatPostHocLiRPACertifier,
    attach_manifoldness_metrics,
    completion_table,
    evaluate_paper_empirical_robustness,
    method_summary,
    paired_query_table,
    paired_summary,
    summarize_empirical_robustness,
    summarize_quality_metrics,
)


def sample_frame():
    rows = []
    for query, verix_l1, certcf_l1 in [(1, 1.0, 2.0), (2, 2.0, 3.0)]:
        for method, l1, l0, runtime in [
            ("verix", verix_l1, 3, 0.4),
            ("certcf", certcf_l1, 2, 0.1),
        ]:
            rows.append(
                {
                    "architecture_id": "depth_01_width_016",
                    "depth": 1,
                    "width": 16,
                    "parameter_count": 562,
                    "query_index": query,
                    "method": method,
                    "paired_eligible": True,
                    "success": True,
                    "l1_distance": l1,
                    "l2_distance": l1 / 2,
                    "l0_changed": l0,
                    "runtime_seconds": runtime,
                    "offline_build_seconds": 10.0 if method == "certcf" else 0.0,
                    "target_margin": 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_completion_and_paired_summaries():
    frame = sample_frame()
    manifest = {
        "architectures": ["depth_01_width_016", "depth_05_width_256"],
        "query_indices": [1, 2],
    }
    completion = completion_table(frame, manifest)
    assert completion["complete"].tolist() == [True, True, False, False]

    summary = method_summary(frame)
    assert summary.loc[summary["method"] == "verix", "mean_l1"].iloc[0] == pytest.approx(1.5)
    paired = paired_query_table(frame)
    assert paired["certcf_over_verix_l1"].tolist() == pytest.approx([2.0, 1.5])
    aggregate = paired_summary(paired)
    assert aggregate.loc[0, "shared_successes"] == 2
    assert aggregate.loc[0, "mean_certcf_minus_verix_l0"] == pytest.approx(-1.0)


def test_notebook_has_required_comparison_sections():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks" / "VeriXCertCFSynthetic32.ipynb").read_text(
            encoding="utf-8"
        )
    )
    sources = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    ).lower()
    for required in (
        "benchmark completion",
        "method-level comparison",
        "paired query comparison",
        "distance and sparsity distributions",
        "query time and offline construction",
        "certcf_over_verix_l1",
        "on-manifoldness",
        "empirical robustness",
        "certified robustness",
        "interpretation and caveats",
    ):
        assert required in sources
    assert "relative proximity ratio" not in sources


def test_grid_notebook_has_scaling_and_timeout_sections():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (
            root
            / "notebooks"
            / "VeriXCertCFSynthetic32Grid.ipynb"
        ).read_text(encoding="utf-8")
    )
    sources = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    ).lower()
    for required in (
        "completion, timeout, and witness coverage",
        "complete per-architecture metric table",
        "shared-success comparison",
        "coverage heatmaps",
        "on-manifoldness",
        "empirical robustness",
        "mean certified $l_1$ radius",
        "mean certified $l_2$ radius",
        "mean certified $l_\\infty$ radius",
        "runtime and scaling",
        "failures and resource limits",
        "interpretation caveats",
    ):
        assert required in sources
    assert "relative-proximity ratio" in sources
    assert "relative proximity ratio" not in sources


class _ThresholdModel:
    def predict(self, values):
        values = np.asarray(values)
        return (values[:, 0] >= 0.0).astype(np.int64)


def _vector_frame():
    frame = sample_frame()
    frame["target_class"] = 1
    for idx in range(32):
        frame[f"x_orig_{idx}"] = 0.0
        frame[f"x_cf_{idx}"] = 1.0 if idx == 0 else 0.0
    return frame


def test_manifoldness_and_rpr_follow_paper_definitions():
    frame = _vector_frame()
    x_train = np.zeros((40, 32), dtype=np.float32)
    x_train[:20, 0] = np.linspace(-2.0, -1.0, 20)
    x_train[20:, 0] = np.linspace(0.5, 1.5, 20)
    y_train = np.repeat([0, 1], 20)

    result = attach_manifoldness_metrics(
        frame,
        x_train=x_train,
        y_train=y_train,
    )

    assert np.isfinite(result["log10_lof"]).all()
    assert np.isfinite(result["negative_lof_score"]).all()
    assert np.isfinite(result["isolation_forest_score"]).all()
    assert np.isfinite(result["relative_proximity_ratio"]).all()
    assert (result["natural_same_class_nn_l1"] > 0).all()
    assert (result["target_class_nn_l1"] >= 0).all()

    summary = summarize_quality_metrics(
        result.assign(
            certified_l1_radius=0.75,
            certified_l1_at_cap=False,
            certified_l2_radius=0.50,
            certified_l2_at_cap=False,
            certified_linf_radius=0.25,
            certified_linf_at_cap=False,
        )
    )
    assert set(summary["method"]) == {"verix", "certcf"}
    assert summary["mean_certified_linf_radius"].tolist() == pytest.approx(
        [0.25, 0.25]
    )
    assert summary["mean_certified_l2_radius"].tolist() == pytest.approx(
        [0.50, 0.50]
    )
    assert summary["mean_certified_l1_radius"].tolist() == pytest.approx(
        [0.75, 0.75]
    )


def test_empirical_robustness_uses_common_paper_grid():
    frame = _vector_frame()
    curves = evaluate_paper_empirical_robustness(
        frame,
        models_by_architecture={"depth_01_width_016": _ThresholdModel()},
        samples_per_sigma=10,
        seed=42,
    )

    assert sorted(curves["sigma"].unique().tolist()) == pytest.approx(
        [0.0, 0.01, 0.03, 0.05, 0.1]
    )
    assert len(curves) == 2 * 2 * 5
    assert curves["empirical_target_rate"].eq(1.0).all()
    summary = summarize_empirical_robustness(curves)
    assert summary["empirical_robustness_pct"].tolist() == pytest.approx(
        [100.0, 100.0]
    )


@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in scalar divide:RuntimeWarning"
)
def test_flat_posthoc_certifier_caps_a_robust_linear_point():
    model = torch.nn.Linear(32, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
        model.weight[0, 0] = -1.0
        model.weight[1, 0] = 1.0

    certifier = FlatPostHocLiRPACertifier(model, device="cpu")
    for norm in (1.0, 2.0, float("inf")):
        radii, capped = certifier.radii(
            np.array([[1.0] + [0.0] * 31], dtype=np.float32),
            np.array([1], dtype=np.int64),
            maximum=0.5,
            steps=4,
            norm=norm,
        )

        assert radii.tolist() == pytest.approx([0.5])
        assert capped.tolist() == [True]
