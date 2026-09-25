#!/usr/bin/env python3
"""Validate raw artifacts and build the canonical paper result tables.

The raw experiments were intentionally run in separate jobs because FACE,
DiCE, and the optimized CertCF timing run have different resource profiles.
This script makes that provenance explicit and consolidates the selected rows
used by the paper.  It never modifies the raw artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "configs" / "paper" / "paper_results.yaml"


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if manifest.get("schema_version") != 1:
        raise ValueError("paper-result manifest must use schema_version: 1")
    return manifest


def _verify_hash(path: Path, expected: str | None, *, skip: bool) -> None:
    if skip or not expected:
        return
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"Checksum mismatch for {path}: expected {expected}, got {actual}"
        )


def _select_run(
    frame: pd.DataFrame,
    *,
    method: str,
    run_name: str,
    source_key: str,
    paper_label: str,
) -> pd.DataFrame:
    required = {"dataset", "method", "run_name", "query_idx", "runtime_s"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source_key} is missing required columns: {missing}")
    selected = frame[
        frame["method"].astype(str).eq(method)
        & frame["run_name"].astype(str).eq(run_name)
    ].copy()
    if selected.empty:
        available = (
            frame[["method", "run_name"]]
            .drop_duplicates()
            .sort_values(["method", "run_name"])
        )
        raise ValueError(
            f"No rows for method={method!r}, run_name={run_name!r} in "
            f"{source_key}. Available runs:\n{available.to_string(index=False)}"
        )
    selected["paper_label"] = paper_label
    selected["paper_source"] = source_key
    return selected


def _validate_counts(
    frame: pd.DataFrame,
    expected: dict[str, int],
    *,
    label: str,
) -> None:
    counts = frame.groupby("dataset", observed=True).size().to_dict()
    normalized = {str(key): int(value) for key, value in counts.items()}
    if normalized != expected:
        raise ValueError(
            f"Unexpected query counts for {label}: expected {expected}, "
            f"got {normalized}"
        )
    if frame.duplicated(["dataset", "query_idx"]).any():
        raise ValueError(f"Duplicate (dataset, query_idx) rows in {label}")


def prepare(
    manifest_path: Path,
    *,
    output: Path | None = None,
    timing_output: Path | None = None,
    skip_checksums: bool = False,
) -> tuple[Path, Path]:
    manifest = _load_manifest(manifest_path)
    expected = {
        str(dataset): int(count)
        for dataset, count in manifest["expected_queries"].items()
    }

    result_frames: list[pd.DataFrame] = []
    for source in manifest["result_sources"]:
        source_path = _resolve(source["path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing paper artifact: {source_path}")
        _verify_hash(source_path, source.get("sha256"), skip=skip_checksums)
        raw = pd.read_parquet(source_path)
        for selection in source["selections"]:
            selected = _select_run(
                raw,
                method=str(selection["method"]),
                run_name=str(selection["run_name"]),
                source_key=str(source["key"]),
                paper_label=str(selection["paper_label"]),
            )
            _validate_counts(selected, expected, label=str(selection["paper_label"]))
            result_frames.append(selected)

    results = pd.concat(result_frames, ignore_index=True, sort=False)
    expected_labels = [
        str(selection["paper_label"])
        for source in manifest["result_sources"]
        for selection in source["selections"]
    ]
    if results["paper_label"].drop_duplicates().tolist() != expected_labels:
        raise ValueError("Consolidated method order does not match the manifest")

    output_path = output or _resolve(manifest["results_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output_path, index=False)

    timing = results.copy()
    timing["timing_source"] = timing["paper_source"]
    for override in manifest.get("timing_overrides", []):
        source_path = _resolve(override["path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing timing artifact: {source_path}")
        _verify_hash(source_path, override.get("sha256"), skip=skip_checksums)
        raw = pd.read_parquet(source_path)
        selected = _select_run(
            raw,
            method=str(override["method"]),
            run_name=str(override["run_name"]),
            source_key=str(override["key"]),
            paper_label=str(override["paper_label"]),
        )
        expected_override = override.get("expected_queries")
        if expected_override == "paper":
            override_counts = expected
        else:
            per_dataset = int(override["expected_queries_per_dataset"])
            override_counts = {dataset: per_dataset for dataset in expected}
        _validate_counts(
            selected,
            override_counts,
            label=f"{override['paper_label']} timing override",
        )
        selected["timing_source"] = str(override["key"])
        timing = timing[~timing["paper_label"].eq(str(override["paper_label"]))]
        timing = pd.concat([timing, selected], ignore_index=True, sort=False)

    timing_path = timing_output or _resolve(manifest["timing_output"])
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing.to_parquet(timing_path, index=False)

    print(
        f"Wrote {len(results):,} result rows to {_display_path(output_path)}"
    )
    print(
        f"Wrote {len(timing):,} timing rows to {_display_path(timing_path)}"
    )
    return output_path, timing_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timing-output", type=Path)
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Validate schema and counts but do not compare artifact hashes.",
    )
    args = parser.parse_args()
    prepare(
        args.manifest.resolve(),
        output=args.output.resolve() if args.output else None,
        timing_output=args.timing_output.resolve() if args.timing_output else None,
        skip_checksums=args.skip_checksums,
    )


if __name__ == "__main__":
    main()
