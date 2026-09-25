from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from counterfactuals.benchmarks.paper_results import prepare


ROOT = Path(__file__).resolve().parents[1]


def _write_frame(path: Path, method: str, run_name: str, counts: dict[str, int]) -> None:
    rows = []
    for dataset, count in counts.items():
        for query_idx in range(count):
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "run_name": run_name,
                    "query_idx": query_idx,
                    "runtime_s": 0.1,
                    "success": True,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_consolidates_selected_runs_and_timing_override(tmp_path: Path) -> None:
    counts = {"one": 2, "two": 1}
    baseline = tmp_path / "baseline.parquet"
    override = tmp_path / "override.parquet"
    _write_frame(baseline, "method_a", "selected", counts)
    _write_frame(override, "method_a", "timed", {"one": 1, "two": 1})

    manifest = {
        "schema_version": 1,
        "expected_queries": counts,
        "results_output": str(tmp_path / "paper.parquet"),
        "timing_output": str(tmp_path / "timing.parquet"),
        "result_sources": [
            {
                "key": "baseline",
                "path": str(baseline),
                "sha256": _sha256(baseline),
                "selections": [
                    {
                        "method": "method_a",
                        "run_name": "selected",
                        "paper_label": "A",
                    }
                ],
            }
        ],
        "timing_overrides": [
            {
                "key": "timed",
                "path": str(override),
                "sha256": _sha256(override),
                "method": "method_a",
                "run_name": "timed",
                "paper_label": "A",
                "expected_queries_per_dataset": 1,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    result_path, timing_path = prepare(manifest_path)
    results = pd.read_parquet(result_path)
    timing = pd.read_parquet(timing_path)

    assert len(results) == 3
    assert set(results["paper_label"]) == {"A"}
    assert len(timing) == 2
    assert set(timing["timing_source"]) == {"timed"}


def test_prepare_rejects_checksum_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _write_frame(source, "method_a", "selected", {"one": 1})
    manifest = {
        "schema_version": 1,
        "expected_queries": {"one": 1},
        "results_output": str(tmp_path / "paper.parquet"),
        "timing_output": str(tmp_path / "timing.parquet"),
        "result_sources": [
            {
                "key": "source",
                "path": str(source),
                "sha256": "0" * 64,
                "selections": [
                    {
                        "method": "method_a",
                        "run_name": "selected",
                        "paper_label": "A",
                    }
                ],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        prepare(manifest_path)


def test_paper_protocol_is_explicit_and_complete() -> None:
    benchmark = yaml.safe_load(
        (ROOT / "configs" / "benchmarks" / "final_benchmark.yaml").read_text(
            encoding="utf-8"
        )
    )
    certcf = next(method for method in benchmark["methods"] if method["name"] == "certcf")
    assert certcf["params"]["k_per_class"] == 500
    assert certcf["params"]["atlas_subsample_method"] == "random"
    assert certcf["params"]["eps_alpha"] == pytest.approx(0.20)
    assert certcf["params"]["sparsity_lambda"] == pytest.approx(1.0)
    assert certcf["params"]["query_k_candidates"] == 5

    manifest = yaml.safe_load(
        (ROOT / "configs" / "paper" / "paper_results.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert sum(manifest["expected_queries"].values()) == 4773
    labels = {
        selection["paper_label"]
        for source in manifest["result_sources"]
        for selection in source["selections"]
    }
    assert labels == {"CertCF", "NN", "GS", "DiCE", "FACE"}
