"""Loading and normalization helpers for benchmark parquet analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .style import DEFAULT_METHOD_LABELS

_HELPERS_DIR = Path(__file__).resolve().parent
_NOTEBOOKS_DIR = _HELPERS_DIR.parent
_REPO_ROOT = _NOTEBOOKS_DIR.parent


def _resolve_existing_path(path: str | Path) -> Path | None:
    """Resolve a result path across common notebook launch locations."""
    requested = Path(path)
    candidates: list[Path] = []

    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.extend(
            [
                requested,
                Path.cwd() / requested,
                _NOTEBOOKS_DIR / requested,
                _REPO_ROOT / requested,
            ]
        )

    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.exists():
            return normalized

    return None


def normalize_benchmark_df(
    df: pd.DataFrame,
    *,
    dataset: str | None = None,
    method_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Attach standard display columns without mutating the raw schema."""
    labels = dict(DEFAULT_METHOD_LABELS)
    if method_labels:
        labels.update(method_labels)

    out = df.copy()
    if dataset is not None and "dataset" not in out.columns:
        out["dataset"] = dataset
    out["method_label"] = out["method"].map(labels).fillna(out["method"])

    if "source_class" not in out.columns and "y_orig" in out.columns:
        out["source_class"] = out["y_orig"]
    if "source_class" in out.columns:
        out["source_class"] = out["source_class"].astype("Int64")
    if "target_class" in out.columns:
        out["target_class"] = out["target_class"].astype("Int64")
    if "y_true" in out.columns:
        out["y_true"] = out["y_true"].astype("Int64")

    return out


def load_result(
    path: str | Path,
    *,
    dataset: str | None = None,
    method_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load one parquet file and normalize notebook-facing columns."""
    result_path = _resolve_existing_path(path)
    if result_path is None:
        raise FileNotFoundError(f"Missing benchmark file: {path}")
    df = pd.read_parquet(result_path)
    return normalize_benchmark_df(df, dataset=dataset, method_labels=method_labels)


def load_result_set(
    mapping: dict[str, str | Path],
    *,
    method_labels: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load multiple parquet files and attach dataset names."""
    frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []

    for dataset_name, path_like in mapping.items():
        resolved = _resolve_existing_path(path_like)
        exists = resolved is not None
        status_rows.append(
            {
                "dataset": dataset_name,
                "requested_path": str(path_like),
                "resolved_path": (str(resolved) if resolved is not None else None),
                "available": exists,
            }
        )
        if exists:
            frames.append(load_result(resolved, dataset=dataset_name, method_labels=method_labels))

    status_df = pd.DataFrame(status_rows)
    if not frames:
        raise RuntimeError("No benchmark parquet files were found.")

    return pd.concat(frames, ignore_index=True), status_df


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return sorted raw feature columns for factual and counterfactual arrays."""
    x_orig_cols = sorted(
        [col for col in df.columns if col.startswith("x_orig_")],
        key=lambda col: int(col.split("_")[-1]),
    )
    x_cf_cols = sorted(
        [col for col in df.columns if col.startswith("x_cf_")],
        key=lambda col: int(col.split("_")[-1]),
    )
    return x_orig_cols, x_cf_cols
