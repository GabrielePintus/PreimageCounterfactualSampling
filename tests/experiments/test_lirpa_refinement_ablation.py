import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.lirpa_refinement_ablation import (
    DEFAULT_CONFIG,
    LiRPARefinementAblationRunner,
    anchor_ball_candidate,
    anchor_pgd_candidate,
    config_fingerprint,
    project_feasible_intersection,
    project_l1_ball,
    project_simplex,
    validate_config,
)


def test_official_config_has_seven_unique_cases_and_three_methods(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["artifacts"]["output_dir"] = str(tmp_path / "run")
    validate_config(config)
    runner = LiRPARefinementAblationRunner(config)
    assert len(runner.cases) == 7
    assert runner.resolve_methods(None) == ["anchor_ball", "anchor_pgd", "certcf"]
    assert [case["depth"] for case in runner.cases if case["kind"] == "synthetic32"] == [1, 2, 3, 4, 5]


def test_cli_bootstrap_can_import_shared_benchmark_loader_from_any_cwd(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "lirpa_refinement_ablation.py"
    probe = (
        "import importlib, runpy; "
        f"runpy.run_path({str(script)!r}, run_name='_bootstrap_probe'); "
        "importlib.import_module('scripts.benchmark')"
    )
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_fingerprint_ignores_output_location_only():
    first = copy.deepcopy(DEFAULT_CONFIG)
    second = copy.deepcopy(DEFAULT_CONFIG)
    second["artifacts"]["output_dir"] = "somewhere/else"
    assert config_fingerprint(first) == config_fingerprint(second)
    second["data"]["eps_alpha"] = 0.1
    assert config_fingerprint(first) != config_fingerprint(second)


def test_prepare_is_deterministic_and_records_shared_geometry(monkeypatch, tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["experiment"]["require_cuda"] = False
    config["certcf"]["device"] = "cpu"
    config["experiment"]["cases"] = [
        {"id": "toy", "kind": "synthetic32", "depth": 1, "width": 128}
    ]
    config["data"]["queries_per_case"] = 2
    config["data"]["synthetic_support_total"] = 8
    config["data"]["anchors_per_predicted_class"] = 2
    config["artifacts"]["output_dir"] = str(tmp_path / "run")
    runner = LiRPARefinementAblationRunner(config)
    checkpoint = tmp_path / "model.ckpt"
    dataset_source = tmp_path / "dataset.npz"
    checkpoint.write_bytes(b"checkpoint")
    dataset_source.write_bytes(b"dataset")
    x_train = np.zeros((8, 32), dtype=np.float32)
    x_train[:4, 0] = -1.0
    x_train[4:, 0] = 1.0
    y_train = np.array([0] * 4 + [1] * 4)
    x_test = np.zeros((4, 32), dtype=np.float32)
    x_test[:, 0] = [-1.0, 1.0, -0.5, 0.5]
    y_test = np.array([0, 1, 0, 1])
    spec = __import__("dataset_specs", fromlist=["get_tabular_dataset_spec"]).get_tabular_dataset_spec(
        "network_complexity"
    )

    monkeypatch.setattr(runner, "_checkpoint", lambda case: checkpoint)
    monkeypatch.setattr(runner, "_dataset_source", lambda case: dataset_source)
    monkeypatch.setattr(
        runner,
        "_load_case",
        lambda case, device: (x_train, y_train, x_test, y_test, _ThresholdNet(), spec),
    )
    first = runner.prepare(force=True)
    with np.load(runner.paths.geometry("toy"), allow_pickle=False) as data:
        first_geometry = {key: data[key].copy() for key in data.files}
    second = runner.prepare(force=True)
    with np.load(runner.paths.geometry("toy"), allow_pickle=False) as data:
        second_geometry = {key: data[key].copy() for key in data.files}
    assert first["geometry_sha256"] == second["geometry_sha256"]
    assert first["dataset_sha256"]["toy"] == second["dataset_sha256"]["toy"]
    assert first_geometry.keys() == second_geometry.keys()
    for key in first_geometry:
        assert np.array_equal(first_geometry[key], second_geometry[key])


def test_invalid_pgd_schedule_and_timeout_are_rejected():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["anchor_pgd"]["lambda_schedule"] = [1.0, 0.1]
    with pytest.raises(ValueError, match="increasing"):
        validate_config(config)
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["runtime"]["timeout_seconds_per_query"] = 0
    with pytest.raises(ValueError, match="positive"):
        validate_config(config)


def test_l1_ball_projection_is_feasible_and_idempotent():
    center = np.array([1.0, -1.0, 0.5])
    point = np.array([4.0, 2.0, -3.0])
    projected = project_l1_ball(point, center, 1.5)
    assert np.linalg.norm(projected - center, 1) == pytest.approx(1.5)
    assert np.allclose(project_l1_ball(projected, center, 1.5), projected)


def test_simplex_and_intersection_preserve_ohe_fixed_and_ball():
    simplex = project_simplex(np.array([-1.0, 0.2, 2.0]))
    assert simplex.sum() == pytest.approx(1.0)
    assert np.all(simplex >= 0.0)
    anchor = np.array([0.0, 1.0, 0.0, 2.0], dtype=np.float32)
    query = np.array([0.4, 0.0, 1.0, 3.0], dtype=np.float32)
    projected = project_feasible_intersection(
        np.array([3.0, -1.0, 2.0, -4.0]),
        anchor=anchor,
        epsilon=2.0,
        ohe_slices=[(1, 3)],
        fixed_dims=np.array([3]),
        fixed_values=query,
        iterations=200,
    )
    assert np.linalg.norm(projected - anchor, 1) <= 2.0 + 1.0e-4
    assert projected[1:3].sum() == pytest.approx(1.0, abs=1.0e-5)
    assert projected[3] == pytest.approx(query[3], abs=1.0e-5)


def test_anchor_ball_projection_does_not_need_a_classifier():
    point, diagnostics = anchor_ball_candidate(
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([2.0, 0.0], dtype=np.float32),
        1.0,
    )
    assert point is not None
    assert np.linalg.norm(point - np.array([2.0, 0.0]), 1) <= 1.0 + 1.0e-5
    assert np.linalg.norm(point, 1) == pytest.approx(1.0, abs=1.0e-3)
    assert diagnostics["solver_calls"] == 1


def test_anchor_ball_direct_decode_is_exact_one_hot():
    query = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    anchor = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    point, diagnostics = anchor_ball_candidate(
        query,
        anchor,
        1.5,
        ohe_slices=[(1, 3)],
    )
    assert point is not None
    assert point[1:3].sum() == pytest.approx(1.0, abs=1.0e-5)
    assert np.count_nonzero(point[1:3] > 1.0e-5) == 1
    assert diagnostics["decode_mode"] in {"direct", "beam"}


class _ThresholdNet(torch.nn.Module):
    def forward(self, x):
        score = x[:, 0]
        return torch.stack([-score, score], dim=1)


def test_anchor_pgd_returns_target_valid_point_inside_initial_ball():
    config = copy.deepcopy(DEFAULT_CONFIG["anchor_pgd"])
    config.update(
        lambda_schedule=[1.0, 10.0],
        steps_per_lambda=30,
        projection_iterations=20,
    )
    query = np.array([-1.0], dtype=np.float32)
    anchor = np.array([1.0], dtype=np.float32)
    candidate, diagnostics = anchor_pgd_candidate(
        query,
        anchor,
        1.5,
        model=_ThresholdNet(),
        target=1,
        device="cpu",
        ball_start=np.array([-0.5], dtype=np.float32),
        config=config,
    )
    assert candidate is not None
    assert candidate[0] >= 0.0
    assert np.linalg.norm(candidate - anchor, 1) <= 1.5 + 1.0e-5
    assert diagnostics["model_evaluations"] > 0


def test_anchor_pgd_exactly_decodes_one_hot_blocks():
    config = copy.deepcopy(DEFAULT_CONFIG["anchor_pgd"])
    config.update(lambda_schedule=[1.0], steps_per_lambda=5, projection_iterations=30)
    query = np.array([-1.0, 1.0, 0.0], dtype=np.float32)
    anchor = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    candidate, _ = anchor_pgd_candidate(
        query,
        anchor,
        1.5,
        model=_ThresholdNet(),
        target=1,
        device="cpu",
        ohe_slices=[(1, 3)],
        config=config,
    )
    assert candidate is not None
    assert candidate[1:3].sum() == pytest.approx(1.0)
    assert np.count_nonzero(candidate[1:3] > 1.0e-6) == 1


def test_baseline_timeout_retains_target_anchor(monkeypatch, tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["experiment"]["cases"] = [
        {"id": "depth_01_width_128", "kind": "synthetic32", "depth": 1, "width": 128}
    ]
    config["artifacts"]["output_dir"] = str(tmp_path / "run")
    runner = LiRPARefinementAblationRunner(config)
    case = runner.cases[0]
    geometry = {
        "x_query": np.array([[-1.0]], dtype=np.float32),
        "target": np.array([1]),
        "x_anchor": np.array([[1.0]], dtype=np.float32),
        "y_anchor_pred": np.array([1]),
        "eps_initial": np.array([1.0], dtype=np.float32),
    }

    def interrupted(*args, **kwargs):
        raise TimeoutError("budget")

    monkeypatch.setattr(
        "experiments.lirpa_refinement_ablation.anchor_pgd_candidate", interrupted
    )
    candidate, diagnostics = runner._baseline_query(
        case, "anchor_pgd", 0, geometry, _ThresholdNet(), "cpu"
    )
    assert np.allclose(candidate, geometry["x_anchor"][0])
    assert diagnostics["timed_out_with_incumbent"] is True


def test_unknown_case_and_method_are_rejected(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["artifacts"]["output_dir"] = str(tmp_path / "run")
    runner = LiRPARefinementAblationRunner(config)
    with pytest.raises(ValueError, match="unknown cases"):
        runner.resolve_cases(["missing"])
    with pytest.raises(ValueError, match="unknown methods"):
        runner.resolve_methods(["missing"])


def test_notebook_covers_paired_quality_depth_and_robustness():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / "notebooks" / "LiRPARefinementAblation.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "shared_success_all" in source
    assert "shared_success_pgd_certcf" in source
    assert "mean_certified_l1_radius" in source
    assert "Synthetic32 depth trends" in source
    assert "Failure and resource diagnostics" in source
