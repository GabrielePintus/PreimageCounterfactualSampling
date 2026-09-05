import json

import numpy as np
import pytest

from experiments.verix_certcf_synthetic32 import (
    DEFAULT_CONFIG,
    Synthetic32Runner,
    _binary_target,
    architecture_cells,
    config_fingerprint,
    select_balanced_queries,
    validate_config,
)


def copy_config(tmp_path):
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config["artifacts"]["output_dir"] = str(tmp_path / "run")
    return config


def test_default_endpoint_protocol(tmp_path):
    config = copy_config(tmp_path)
    validate_config(config)
    assert architecture_cells(config) == [(1, 16), (5, 256)]
    assert config["queries"] == {"count": 10, "per_true_class": 5}
    assert config["verix"]["epsilon"] == pytest.approx(0.5)
    assert config["verix"]["traversal"] == "deletion_sensitivity"
    assert config["verix"]["solve_with_milp"] is False


def test_cartesian_architecture_grid(tmp_path):
    config = copy_config(tmp_path)
    config["experiment"]["architectures"] = None
    config["experiment"]["depths"] = [1, 2]
    config["experiment"]["widths"] = [16, 32, 64]

    validate_config(config)

    assert architecture_cells(config) == [
        (1, 16),
        (1, 32),
        (1, 64),
        (2, 16),
        (2, 32),
        (2, 64),
    ]


def test_balanced_queries_preserve_prepared_order():
    candidates = np.array([9, 4, 8, 3, 7, 2, 6, 1])
    labels = np.zeros(10, dtype=np.int64)
    labels[[4, 3, 2, 1]] = 1
    selected = select_balanced_queries(candidates, labels, per_true_class=2)
    assert selected.tolist() == [9, 4, 8, 3]
    assert np.bincount(labels[selected], minlength=2).tolist() == [2, 2]


def test_fingerprint_ignores_output_location_only(tmp_path):
    first = copy_config(tmp_path)
    second = copy_config(tmp_path)
    second["artifacts"]["output_dir"] = str(tmp_path / "other")
    assert config_fingerprint(first) == config_fingerprint(second)
    second["verix"]["epsilon"] = 0.1
    assert config_fingerprint(first) != config_fingerprint(second)


def test_invalid_query_balance_and_traversal_are_rejected(tmp_path):
    config = copy_config(tmp_path)
    config["queries"]["count"] = 9
    with pytest.raises(ValueError, match="2 \\* queries.per_true_class"):
        validate_config(config)

    config = copy_config(tmp_path)
    config["verix"]["traversal"] = "reversal"
    with pytest.raises(ValueError, match="deletion_sensitivity"):
        validate_config(config)

    config = copy_config(tmp_path)
    config["verix"]["query_timeout_seconds"] = 0
    with pytest.raises(ValueError, match="query_timeout_seconds"):
        validate_config(config)


def test_architecture_cli_resolution(tmp_path):
    runner = Synthetic32Runner(copy_config(tmp_path))
    assert runner.resolve_architectures(["depth_05_width_256"]) == [(5, 256)]
    with pytest.raises(ValueError, match="unknown configured architectures"):
        runner.resolve_architectures(["depth_03_width_064"])


def test_binary_target_uses_witness_when_present_and_opposite_as_fallback():
    assert _binary_target(0, 1) == (1, "verix_witness")
    assert _binary_target(1, 0) == (0, "verix_witness")
    assert _binary_target(0, None) == (1, "binary_opposite")
    assert _binary_target(1, None) == (0, "binary_opposite")
    with pytest.raises(RuntimeError, match="disagrees"):
        _binary_target(0, 0)
