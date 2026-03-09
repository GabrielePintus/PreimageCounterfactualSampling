from pathlib import Path

import numpy as np

from counterfactuals.core.registry import Registry
from counterfactuals.datasets.base_dataset import NumpyDataset
from counterfactuals.experiments.config import parse_experiment_config
from counterfactuals.experiments.runner import run_experiment
from counterfactuals.methods.wachter import WachterMethod
from counterfactuals.metrics.proximity import L2Proximity
from counterfactuals.metrics.sparsity import SparsityMetric
from counterfactuals.models.sklearn_model import build_sklearn_mlp
from counterfactuals.utils.config import read_yaml


def test_parse_config_schema():
    cfg = parse_experiment_config(
        {
            "method": "wachter",
            "model": "sklearn_mlp",
            "dataset": "adult",
            "metrics": ["proximity", "sparsity", "validity"],
            "seed": 123,
            "max_test_samples": 8,
        }
    )
    assert cfg.method.name == "wachter"
    assert len(cfg.metrics) == 3
    assert cfg.seed == 123


def test_read_yaml(tmp_path: Path):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("method: wachter\nmodel: sklearn_mlp\ndataset: adult\nmetrics: [proximity]\n", encoding="utf-8")
    loaded = read_yaml(str(config_path))
    assert loaded["method"] == "wachter"


def test_runner_smoke_with_custom_registries():
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(120, 4)).astype(np.float32)
    y_train = (x_train[:, 0] + x_train[:, 1] > 0.0).astype(np.int64)
    x_test = rng.normal(size=(20, 4)).astype(np.float32)
    y_test = (x_test[:, 0] + x_test[:, 1] > 0.0).astype(np.int64)

    registries = {
        "method": Registry("method"),
        "dataset": Registry("dataset"),
        "model": Registry("model"),
        "metric": Registry("metric"),
    }
    registries["method"].register("wachter", WachterMethod)
    registries["dataset"].register(
        "toy",
        lambda: NumpyDataset(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test),
    )
    registries["model"].register("sklearn_mlp", build_sklearn_mlp)
    registries["metric"].register("proximity", L2Proximity)
    registries["metric"].register("sparsity", SparsityMetric)

    cfg = parse_experiment_config(
        {
            "method": {"name": "wachter", "params": {"max_iter": 60}},
            "model": "sklearn_mlp",
            "dataset": "toy",
            "metrics": ["proximity", "sparsity", "validity"],
            "max_test_samples": 6,
            "seed": 7,
        }
    )

    summary = run_experiment(cfg=cfg, registries=registries)

    assert "success" in summary
    assert "validity" in summary
    assert 0.0 <= summary["success"] <= 1.0
