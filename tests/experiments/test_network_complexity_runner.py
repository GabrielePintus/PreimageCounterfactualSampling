import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.network_complexity import (
    DEFAULT_CONFIG,
    AccuracyGateError,
    NetworkComplexityRunner,
    PhaseResourceMonitor,
    architecture_grid,
    architecture_id,
    sha256_file,
)


def tiny_config(tmp_path, *, widths=(4, 8), depths=(1, 2)):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg["experiment"].update(
        {
            "widths": list(widths),
            "depths": list(depths),
            "accuracy_gate": 0.0,
        }
    )
    cfg["dataset"].update(
        {
            "n_train": 400,
            "n_validation": 50,
            "n_test": 50,
            "n_queries": 20,
            "queries_per_class": 10,
        }
    )
    cfg["training"].update({"max_epochs": 1, "num_workers": 0})
    cfg["certcf"].update(
        {
            "train_pool_per_true_class": 30,
            "anchors_per_predicted_class": 2,
            "atlas_subsample_method": "random",
            "lirpa_batch_size": 4,
            "lirpa_batch_size_candidates": [4, 2, 1],
            "timeout_s_per_query": 1,
        }
    )
    cfg["artifacts"]["output_dir"] = str(tmp_path / "run")
    return cfg


def write_fake_training_artifact(runner, depth, width, accuracy=0.95):
    checkpoint = runner.paths.checkpoint(depth, width)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"{depth}-{width}".encode())
    metadata = {
        "architecture_id": architecture_id(depth, width),
        "test_accuracy": accuracy,
        "training_time_s": 0.1,
        "config_fingerprint": runner.fingerprint,
        "training_config_fingerprint": runner.training_fingerprint,
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    runner.paths.train_metadata(depth, width).write_text(json.dumps(metadata))


class FakeModel(torch.nn.Module):
    def forward(self, x):
        return torch.stack((-x[:, 0], x[:, 0]), dim=1)


class FakeMethod:
    def __init__(self, support):
        self._y_train = np.asarray(support)
        self.atlas = None

    def generate_batch(self, x, target_class, timeout_s_per_query):
        del timeout_s_per_query
        x_cf = np.asarray(x[0], dtype=np.float32).copy()
        x_cf[0] = 2.0 if int(target_class) == 1 else -2.0
        result = SimpleNamespace(
            x_cf=x_cf,
            success=True,
            distance=float(np.linalg.norm(x_cf - x[0], ord=1)),
            metadata={"fake": True},
        )
        return [result]


def test_grid_unique_paths_resume_hash_and_accuracy_gate(tmp_path):
    runner = NetworkComplexityRunner(tiny_config(tmp_path))
    assert len(architecture_grid(runner.config)) == 4
    assert len({runner.paths.checkpoint(*cell) for cell in runner.grid}) == 4
    for cell in runner.grid:
        write_fake_training_artifact(runner, *cell)
        assert runner._valid_training_artifact(*cell)
    runner.enforce_accuracy_gate()

    checkpoint = runner.paths.checkpoint(*runner.grid[0])
    checkpoint.write_bytes(b"changed")
    assert not runner._valid_training_artifact(*runner.grid[0])

    write_fake_training_artifact(runner, *runner.grid[0], accuracy=-1.0)
    with pytest.raises(AccuracyGateError, match="shared training configuration"):
        runner.enforce_accuracy_gate()


def test_benchmark_only_config_change_does_not_invalidate_training(tmp_path):
    cfg_top5 = tiny_config(tmp_path, widths=(4,), depths=(1,))
    cfg_top5["certcf"]["query_k_candidates"] = 5
    runner_top5 = NetworkComplexityRunner(cfg_top5)
    write_fake_training_artifact(runner_top5, 1, 4)

    cfg_top3 = tiny_config(tmp_path, widths=(4,), depths=(1,))
    cfg_top3["certcf"]["query_k_candidates"] = 3
    runner_top3 = NetworkComplexityRunner(cfg_top3)
    assert runner_top3.fingerprint != runner_top5.fingerprint
    assert runner_top3.benchmark_config_fingerprint != runner_top5.benchmark_config_fingerprint
    assert runner_top3.training_fingerprint == runner_top5.training_fingerprint
    assert runner_top3._valid_training_artifact(1, 4)


def test_phase_resource_metadata_cpu():
    with PhaseResourceMonitor("build", "cpu", interval_s=0.005) as monitor:
        _ = np.ones(10_000)
    assert monitor.metrics["build_wall_time_s"] >= 0
    assert monitor.metrics["build_rss_peak_bytes"] >= monitor.metrics["build_rss_baseline_bytes"]
    assert monitor.metrics["build_cuda_peak_allocated_bytes"] == 0
    assert monitor.metrics["build_cuda_peak_reserved_bytes"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_phase_resource_metadata_cuda():
    with PhaseResourceMonitor("query", "cuda", interval_s=0.005) as monitor:
        _ = torch.ones(1024, device="cuda")
    assert monitor.metrics["query_cuda_peak_allocated_bytes"] > 0
    assert monitor.metrics["query_cuda_peak_reserved_bytes"] > 0


def test_tiny_end_to_end_grid_combines_and_analyzes(monkeypatch, tmp_path):
    runner = NetworkComplexityRunner(tiny_config(tmp_path))
    runner.prepare()
    for cell in runner.grid:
        write_fake_training_artifact(runner, *cell)
    monkeypatch.setattr(runner, "_load_model", lambda depth, width, device: FakeModel().to(device))

    def fake_make(model, x_train, y_support, *, device, batch_size):
        del model, x_train, device
        return FakeMethod(y_support), {
            "build_wall_time_s": 0.01,
            "build_rss_baseline_bytes": 100,
            "build_rss_peak_bytes": 120,
            "build_rss_peak_delta_bytes": 20,
            "build_rss_sample_count": 2,
            "build_cuda_peak_allocated_bytes": 0,
            "build_cuda_peak_reserved_bytes": 0,
        }

    monkeypatch.setattr(runner, "_make_certcf", fake_make)
    benchmark = runner.benchmark()
    assert len(benchmark) == 4 * 20
    combined = runner.combine()
    assert len(combined) == 80
    assert combined["architecture_id"].nunique() == 4
    summary = runner.analyze()
    assert len(summary) == 4
    assert runner.paths.combined.exists()
    assert runner.paths.summary.exists()
    status = runner.status()
    assert status["training_complete"] == 4
    assert status["benchmark_complete"] == 4


def test_largest_pilot_reduces_one_common_lirpa_batch_on_memory_error(monkeypatch, tmp_path):
    runner = NetworkComplexityRunner(tiny_config(tmp_path, widths=(4,), depths=(1,)))
    runner.prepare()
    write_fake_training_artifact(runner, 1, 4)
    monkeypatch.setattr(runner, "_load_model", lambda depth, width, device: FakeModel().to(device))
    attempted = []

    def oom_then_succeed(model, x_train, y_support, *, device, batch_size):
        del model, x_train, device
        attempted.append(batch_size)
        if batch_size == 4:
            raise RuntimeError("CUDA out of memory")
        return FakeMethod(y_support), {
            "build_wall_time_s": 0.01,
            "build_rss_baseline_bytes": 100,
            "build_rss_peak_bytes": 120,
            "build_rss_peak_delta_bytes": 20,
            "build_rss_sample_count": 2,
            "build_cuda_peak_allocated_bytes": 0,
            "build_cuda_peak_reserved_bytes": 0,
        }

    monkeypatch.setattr(runner, "_make_certcf", oom_then_succeed)
    result = runner.benchmark(calibrate_largest=True)
    assert len(result) == 20
    assert attempted == [4, 2]
    assert runner.status()["effective_lirpa_batch_size"] == 2
    assert result["effective_lirpa_batch_size"].unique().tolist() == [2]


def test_all_finishes_training_before_any_benchmark(monkeypatch, tmp_path):
    runner = NetworkComplexityRunner(tiny_config(tmp_path))
    calls = []
    monkeypatch.setattr(
        runner,
        "prepare",
        lambda **kwargs: calls.append(("prepare", kwargs)),
    )
    monkeypatch.setattr(
        runner,
        "train",
        lambda **kwargs: calls.append(("train", kwargs)),
    )
    monkeypatch.setattr(
        runner,
        "benchmark",
        lambda **kwargs: calls.append(("benchmark", kwargs)),
    )
    expected = pd.DataFrame({"done": [True]})

    def fake_analyze(**kwargs):
        calls.append(("analyze", kwargs))
        return expected

    monkeypatch.setattr(runner, "analyze", fake_analyze)
    result = runner.all()
    assert result is expected
    assert [name for name, _ in calls] == ["prepare", "train", "benchmark", "analyze"]
    assert calls[1][1] == {"force": False, "enforce_gate": True}
    assert calls[2][1] == {
        "force": False,
        "calibrate_largest": True,
        "enforce_gate": True,
    }


def test_summary_calculations():
    frame = pd.DataFrame(
        {
            "architecture_id": ["a", "a"],
            "depth": [1, 1],
            "width": [16, 16],
            "parameter_count": [562, 562],
            "success": [True, False],
            "runtime_s": [0.1, 0.3],
            "query_time_median_s": [0.2, 0.2],
        }
    )
    summary = NetworkComplexityRunner.summarize_frame(frame)
    assert summary.loc[0, "validity"] == pytest.approx(0.5)
    assert summary.loc[0, "n_queries"] == 2
    assert summary.loc[0, "query_time_median_s"] == pytest.approx(0.2)
