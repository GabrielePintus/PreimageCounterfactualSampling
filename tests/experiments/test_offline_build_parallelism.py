from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.offline_build_parallelism import (
    compare_bounds_to_saved,
    validate_config,
)


def _config():
    return {
        "source_configs": {},
        "cases": {"small": {"kind": "network_complexity"}},
        "variants": {
            "serial": {"epsilon_parallelism": 1, "build_parallelism": 1}
        },
        "comparison": {
            "absolute_tolerance": 1.0e-6,
            "relative_tolerance": 1.0e-6,
            "chunk_elements": 2,
        },
        "artifacts": {"output_dir": "results/test"},
    }


def test_validate_config_rejects_nonpositive_parallelism():
    config = _config()
    config["variants"]["serial"]["build_parallelism"] = 0
    with pytest.raises(ValueError, match="build_parallelism"):
        validate_config(config)


def test_compare_bounds_to_saved_reports_exact_and_tolerant_matches(tmp_path):
    reference = tmp_path / "atlas"
    reference.mkdir()
    arrays = {
        "X": np.array([[1.0], [2.0]], dtype=np.float32),
        "eps": np.array([0.5, 0.25], dtype=np.float32),
    }
    np.savez_compressed(reference / "class_0.npz", **arrays)
    (reference / "manifest.json").write_text(
        json.dumps({"class_labels": [0], "files": {"0": "class_0.npz"}}),
        encoding="utf-8",
    )

    exact = compare_bounds_to_saved(
        {0: arrays}, reference, atol=1.0e-6, rtol=1.0e-6, chunk_elements=1
    )
    assert exact["all_exact"] is True
    assert exact["all_close"] is True

    perturbed = {0: {**arrays, "eps": arrays["eps"] + np.float32(5.0e-7)}}
    close = compare_bounds_to_saved(
        perturbed, reference, atol=1.0e-6, rtol=0.0, chunk_elements=1
    )
    assert close["all_exact"] is False
    assert close["all_close"] is True
    expected_difference = float(
        np.max(np.abs(perturbed[0]["eps"].astype(np.float64) - arrays["eps"]))
    )
    assert close["maximum_absolute_difference"] == expected_difference
