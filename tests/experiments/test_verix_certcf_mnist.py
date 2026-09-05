import json

import numpy as np
import pandas as pd
import pytest

from experiments.verix_certcf_mnist import (
    DEFAULT_CONFIG,
    VeriXCertCFMNISTRunner,
    balanced_class_indices,
    config_fingerprint,
    counterfactual_metrics,
    query_indices,
    validate_config,
)


def copy_config(tmp_path):
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config["artifacts"]["output_dir"] = str(tmp_path / "run")
    return config


def test_official_protocol_defaults_and_query_selection(tmp_path):
    config = copy_config(tmp_path)
    validate_config(config)

    assert query_indices(config).tolist() == list(range(100))
    assert query_indices(config, pilot=True).tolist() == [10]
    assert config["verix"]["epsilon"] == pytest.approx(0.05)
    assert config["verix"]["traversal"] == "reversal"
    assert config["verix"]["solve_with_milp"] is False
    assert config["certcf"]["query_k_candidates"] == 3
    assert config["certcf"]["atlas_subsample_method"] == "random"


def test_config_fingerprint_excludes_only_artifact_location(tmp_path):
    first = copy_config(tmp_path)
    second = copy_config(tmp_path)
    second["artifacts"]["output_dir"] = str(tmp_path / "elsewhere")
    assert config_fingerprint(first) == config_fingerprint(second)

    second["verix"]["epsilon"] = 0.1
    assert config_fingerprint(first) != config_fingerprint(second)


def test_config_rejects_kmedoids_and_nonpaper_traversal(tmp_path):
    config = copy_config(tmp_path)
    config["certcf"]["atlas_subsample_method"] = "kmedoids"
    with pytest.raises(ValueError, match="must not use k-medoids"):
        validate_config(config)

    config = copy_config(tmp_path)
    config["verix"]["traversal"] = "random"
    with pytest.raises(ValueError, match="reversal traversal"):
        validate_config(config)


def test_balanced_pool_has_exact_reproducible_count_per_class():
    labels = np.repeat(np.arange(10), 20)
    first = balanced_class_indices(labels, 7, seed=42)
    second = balanced_class_indices(labels, 7, seed=42)

    assert np.array_equal(first, second)
    assert len(first) == 70
    assert np.bincount(labels[first], minlength=10).tolist() == [7] * 10
    assert len(np.unique(first)) == len(first)


def test_counterfactual_metrics_require_target_and_pixel_feasibility():
    def logits(values):
        values = np.asarray(values)
        return np.stack((-values[:, 0], values[:, 0]), axis=1)

    query = np.zeros(784, dtype=np.float32)
    counterfactual = query.copy()
    counterfactual[0] = 0.5
    metrics = counterfactual_metrics(
        query,
        counterfactual,
        target=1,
        logits_function=logits,
    )
    assert metrics["success"]
    assert metrics["predicted_class"] == 1
    assert metrics["l1_distance"] == pytest.approx(0.5)
    assert metrics["l2_distance"] == pytest.approx(0.5)
    assert metrics["l0_changed"] == 1

    counterfactual[1] = 1.1
    metrics = counterfactual_metrics(
        query,
        counterfactual,
        target=1,
        logits_function=logits,
    )
    assert not metrics["success"]
    assert not metrics["pixel_feasible"]


def test_all_runs_all_verix_before_certcf_and_builds_once(monkeypatch, tmp_path):
    runner = VeriXCertCFMNISTRunner(copy_config(tmp_path))
    calls = []
    monkeypatch.setattr(
        runner, "prepare", lambda **kwargs: calls.append(("prepare", kwargs))
    )
    monkeypatch.setattr(
        runner,
        "run_verix",
        lambda indices=None, **kwargs: calls.append(
            ("verix", None if indices is None else list(indices), kwargs)
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_certcf",
        lambda indices=None, **kwargs: calls.append(("certcf", indices, kwargs)),
    )
    monkeypatch.setattr(
        runner, "robustness", lambda **kwargs: calls.append(("robustness", kwargs))
    )
    expected = pd.DataFrame({"done": [True]})
    monkeypatch.setattr(runner, "analyze", lambda: expected)

    result = runner.all(force=True)

    assert result is expected
    assert [call[0] for call in calls] == [
        "prepare",
        "verix",
        "verix",
        "certcf",
        "robustness",
    ]
    assert calls[1][1] == [10]
    assert calls[2][1] is None
    assert calls[3][1] is None
