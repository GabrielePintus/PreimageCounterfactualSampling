from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from experiments.offline_build_parallelism import (
    OfflineBuildParallelismRunner,
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


def test_validate_config_rejects_nonpositive_lirpa_batch_size():
    config = _config()
    config["variants"]["serial"]["lirpa_batch_size"] = 0
    with pytest.raises(ValueError, match="lirpa_batch_size"):
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


def test_analyze_batch_size_sweep_aggregates_repetitions(tmp_path):
    config = _config()
    config["artifacts"]["output_dir"] = str(tmp_path)
    runner = OfflineBuildParallelismRunner(config, config_path=tmp_path / "config.yaml")
    rows = []
    for batch_size, times in ((1, (10.0, 12.0)), (4, (3.0, 5.0))):
        for repetition, build_time in enumerate(times):
            rows.append(
                {
                    "status": "complete",
                    "case_id": "small",
                    "case_kind": "network_complexity",
                    "batch_kind": "variable_epsilon",
                    "batch_size": batch_size,
                    "repetition": repetition,
                    "build_wall_time_s": build_time,
                    "lirpa_time_s": build_time - 1.0,
                    "epsilon_time_s": 1.0,
                    "build_rss_peak_delta_bytes": 100,
                    "build_cuda_peak_allocated_bytes": 200,
                    "atlas_all_close": True,
                    "atlas_decisions_exact": True,
                    "atlas_decisions_close": True,
                    "atlas_max_absolute_difference": 0.0,
                    "cnn_radius_bucket_fallback_count": 0,
                }
            )
    pd.DataFrame(rows).to_parquet(runner.paths.batch_size_sweep, index=False)

    summary = runner.analyze_batch_size_sweep().set_index("batch_size")

    assert summary.loc[1, "build_mean_s"] == pytest.approx(11.0)
    assert summary.loc[4, "build_mean_s"] == pytest.approx(4.0)
    assert summary.loc[4, "speedup_vs_batch1"] == pytest.approx(2.75)
    assert bool(summary.loc[4, "all_decisions_exact"]) is True
