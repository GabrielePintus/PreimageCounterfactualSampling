#!/usr/bin/env python3
"""Refresh an incremental benchmark databank family.

This script treats per-method parquet files as the source of truth and rebuilds
derived artifacts for navigation:

- combined.parquet
- summary.json
- index.html

Expected family layout:

    results/benchmarks/<family_name>/
      manifest.json
      methods/
        <method_or_run_name>.parquet
        ...
      combined.parquet
      summary.json
      index.html

The refresh command validates each method parquet against the manifest task set
before including it in the combined parquet.

Optional convenience:
    If manifest.json does not exist yet, pass --init-from-parquet together with
    the required metadata flags to bootstrap the manifest from an existing
    parquet task set. This is useful for importing historical benchmark runs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_MANIFEST_KEYS = {
    "family_name",
    "dataset",
    "config_path",
    "seed",
    "n_queries",
    "task_definition",
    "query_indices",
    "y_orig",
    "target_class",
}

REQUIRED_RESULT_COLUMNS = {
    "method",
    "query_idx",
    "y_orig",
    "success",
    "runtime_s",
    "build_time_s",
    "l1_distance",
}


@dataclass
class FileValidationResult:
    path: Path
    status: str
    methods: list[str]
    n_rows: int
    message: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def _task_frame_from_manifest(manifest: dict[str, Any]) -> pd.DataFrame:
    missing = sorted(REQUIRED_MANIFEST_KEYS.difference(manifest))
    if missing:
        raise ValueError(f"Manifest {manifest!r} is missing required keys: {missing}")

    query_indices = manifest["query_indices"]
    y_orig = manifest["y_orig"]
    target_class = manifest["target_class"]
    lengths = {len(query_indices), len(y_orig), len(target_class)}
    if len(lengths) != 1:
        raise ValueError("Manifest task arrays query_indices, y_orig, and target_class must have the same length.")

    task_df = pd.DataFrame(
        {
            "query_idx": pd.Series(query_indices, dtype="int64"),
            "y_orig": pd.Series(y_orig, dtype="int64"),
            "target_class": pd.Series(target_class, dtype="int64"),
        }
    )
    if int(manifest["n_queries"]) != len(task_df):
        raise ValueError(
            f"Manifest n_queries={manifest['n_queries']} does not match task rows={len(task_df)}."
        )
    return task_df


def _normalize_task_frame(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["query_idx", "y_orig", "target_class"]]
        .astype({"query_idx": "int64", "y_orig": "int64", "target_class": "int64"})
        .sort_values(["query_idx", "target_class", "y_orig"], kind="stable")
        .reset_index(drop=True)
    )


def _infer_target_class_if_missing(df: pd.DataFrame, manifest_tasks: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "target_class" in out.columns:
        return out

    manifest_unique = _normalize_task_frame(manifest_tasks).drop_duplicates(["query_idx", "y_orig"])
    if len(manifest_unique) != len(manifest_tasks):
        raise ValueError(
            "Result parquet is missing target_class, but the manifest task set is not binary/inferable."
        )
    inferred = 1 - out["y_orig"].astype("int64")
    out["target_class"] = inferred
    return out


def _group_task_frame(df: pd.DataFrame, method_name: str) -> pd.DataFrame:
    group_column = "run_name" if "run_name" in df.columns else "method"
    return _normalize_task_frame(df[df[group_column] == method_name])


def _feature_dim_from_df(df: pd.DataFrame) -> int | None:
    cols = [c for c in df.columns if c.startswith("x_orig_")]
    return len(cols) if cols else None


def _validate_method_file(
    path: Path,
    manifest: dict[str, Any],
    manifest_tasks: pd.DataFrame,
) -> tuple[FileValidationResult, pd.DataFrame | None]:
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return FileValidationResult(path=path, status="rejected", methods=[], n_rows=0, message=f"read_error: {exc}"), None

    missing_cols = sorted(REQUIRED_RESULT_COLUMNS.difference(df.columns))
    if missing_cols:
        return (
            FileValidationResult(
                path=path,
                status="rejected",
                methods=[],
                n_rows=len(df),
                message=f"missing required columns: {missing_cols}",
            ),
            None,
        )

    try:
        df = _infer_target_class_if_missing(df, manifest_tasks)
    except Exception as exc:
        return FileValidationResult(path=path, status="rejected", methods=[], n_rows=len(df), message=str(exc)), None

    if "dataset" in df.columns:
        datasets = sorted(set(df["dataset"].dropna().astype(str)))
        if datasets and datasets != [str(manifest["dataset"])]:
            return (
                FileValidationResult(
                    path=path,
                    status="rejected",
                    methods=sorted(
                        df[("run_name" if "run_name" in df.columns else "method")]
                        .astype(str)
                        .unique()
                        .tolist()
                    ),
                    n_rows=len(df),
                    message=f"dataset mismatch: parquet has {datasets}, manifest expects {manifest['dataset']!r}",
                ),
                None,
            )

    feature_dim = manifest.get("feature_dim")
    if feature_dim is not None:
        df_feature_dim = _feature_dim_from_df(df)
        if df_feature_dim is not None and int(df_feature_dim) != int(feature_dim):
            return (
                FileValidationResult(
                    path=path,
                    status="rejected",
                    methods=sorted(
                        df[("run_name" if "run_name" in df.columns else "method")]
                        .astype(str)
                        .unique()
                        .tolist()
                    ),
                    n_rows=len(df),
                    message=f"feature_dim mismatch: parquet has {df_feature_dim}, manifest expects {feature_dim}",
                ),
                None,
            )

    normalized_manifest = _normalize_task_frame(manifest_tasks)
    method_column = "run_name" if "run_name" in df.columns else "method"
    methods = sorted(df[method_column].astype(str).unique().tolist())
    for method_name in methods:
        task_df = _group_task_frame(df, method_name)
        if len(task_df) != len(normalized_manifest):
            return (
                FileValidationResult(
                    path=path,
                    status="rejected",
                    methods=methods,
                    n_rows=len(df),
                    message=(
                        f"task count mismatch for method {method_name!r}: "
                        f"{len(task_df)} rows vs manifest {len(normalized_manifest)}"
                    ),
                ),
                None,
            )
        if not task_df.equals(normalized_manifest):
            return (
                FileValidationResult(
                    path=path,
                    status="rejected",
                    methods=methods,
                    n_rows=len(df),
                    message=f"task-set mismatch for method {method_name!r}",
                ),
                None,
            )

    accepted = df.copy()
    accepted["dataset"] = str(manifest["dataset"])
    accepted["family_name"] = str(manifest["family_name"])
    accepted["source_file"] = path.name
    return (
        FileValidationResult(
            path=path,
            status="accepted",
            methods=methods,
            n_rows=len(accepted),
            message=None,
        ),
        accepted,
    )


def _summary_rows(combined: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_column = "run_name" if "run_name" in combined.columns else "method"
    for run_name, group in combined.groupby(group_column, sort=True):
        success_mask = group["success"].astype(bool)
        success_group = group[success_mask]
        build_values = group["build_time_s"].dropna()
        rows.append(
            {
                "method": str(group["method"].iloc[0]) if "method" in group.columns else str(run_name),
                "run_name": str(run_name),
                "n_rows": int(len(group)),
                "n_queries": int(group["query_idx"].nunique()),
                "n_success": int(success_mask.sum()),
                "success_rate": float(success_mask.mean()),
                "mean_l1_distance": float(success_group["l1_distance"].mean()) if len(success_group) else None,
                "mean_runtime_s": float(group["runtime_s"].mean()),
                "build_time_s": float(build_values.iloc[0]) if len(build_values) else None,
                "source_files": sorted(group["source_file"].dropna().astype(str).unique().tolist()),
            }
        )
    return rows


def _render_html(summary: dict[str, Any]) -> str:
    family = summary["family"]
    files = summary["files"]
    methods = summary["methods"]
    refreshed_at = escape(str(summary["refreshed_at"]))

    def _fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return escape(str(value))

    files_rows = "\n".join(
        (
            "<tr>"
            f"<td><a href=\"{escape(item['relative_path'])}\">{escape(item['relative_path'])}</a></td>"
            f"<td>{escape(item['status'])}</td>"
            f"<td>{', '.join(escape(m) for m in item.get('methods', []))}</td>"
            f"<td>{item.get('n_rows', '')}</td>"
            f"<td>{escape(item.get('message') or '')}</td>"
            "</tr>"
        )
        for item in files
    )

    method_rows = "\n".join(
        (
            "<tr>"
            f"<td>{escape(str(row['run_name']))}</td>"
            f"<td>{row['n_rows']}</td>"
            f"<td>{row['n_queries']}</td>"
            f"<td>{row['n_success']}</td>"
            f"<td>{_fmt(row['success_rate'])}</td>"
            f"<td>{_fmt(row['mean_l1_distance'])}</td>"
            f"<td>{_fmt(row['mean_runtime_s'])}</td>"
            f"<td>{_fmt(row['build_time_s'])}</td>"
            f"<td>{', '.join(escape(v) for v in row.get('source_files', []))}</td>"
            "</tr>"
        )
        for row in methods
    )

    family_items = "\n".join(
        f"<li><strong>{escape(str(key))}:</strong> {escape(str(value))}</li>"
        for key, value in family.items()
        if key not in {"query_indices", "y_orig", "target_class"}
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Benchmark Databank - {escape(str(family['family_name']))}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
    h1, h2 {{ margin-bottom: 0.4rem; }}
    p.meta {{ color: #555; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.55rem 0.7rem; text-align: left; }}
    th {{ background: #f6f6f6; cursor: pointer; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    code {{ background: #f3f3f3; padding: 0.1rem 0.25rem; border-radius: 4px; }}
    ul {{ line-height: 1.5; }}
    .links a {{ margin-right: 1rem; }}
  </style>
</head>
<body>
  <h1>Benchmark Databank: {escape(str(family['family_name']))}</h1>
  <p class="meta">Last refreshed: {refreshed_at}</p>

  <div class="links">
    <a href="combined.parquet">combined.parquet</a>
    <a href="summary.json">summary.json</a>
    <a href="manifest.json">manifest.json</a>
  </div>

  <h2>Family metadata</h2>
  <ul>
    {family_items}
  </ul>

  <h2>Method summary</h2>
  <table id="method-summary">
    <thead>
      <tr>
        <th>run_name</th>
        <th>n_rows</th>
        <th>n_queries</th>
        <th>n_success</th>
        <th>success_rate</th>
        <th>mean_l1_distance</th>
        <th>mean_runtime_s</th>
        <th>build_time_s</th>
        <th>source_files</th>
      </tr>
    </thead>
    <tbody>
      {method_rows}
    </tbody>
  </table>

  <h2>Files</h2>
  <table id="file-status">
    <thead>
      <tr>
        <th>file</th>
        <th>status</th>
        <th>methods</th>
        <th>n_rows</th>
        <th>message</th>
      </tr>
    </thead>
    <tbody>
      {files_rows}
    </tbody>
  </table>

  <script>
    for (const table of document.querySelectorAll('table')) {{
      for (const th of table.tHead.rows[0].cells) {{
        th.addEventListener('click', () => {{
          const tbody = table.tBodies[0];
          const rows = Array.from(tbody.rows);
          const idx = th.cellIndex;
          const asc = th.dataset.order !== 'asc';
          rows.sort((a, b) => {{
            const av = a.cells[idx].innerText.trim();
            const bv = b.cells[idx].innerText.trim();
            const an = Number(av);
            const bn = Number(bv);
            if (!Number.isNaN(an) && !Number.isNaN(bn)) {{
              return asc ? an - bn : bn - an;
            }}
            return asc ? av.localeCompare(bv) : bv.localeCompare(av);
          }});
          tbody.replaceChildren(...rows);
          th.dataset.order = asc ? 'asc' : 'desc';
        }});
      }}
    }}
  </script>
</body>
</html>
"""


def _manifest_bootstrap_from_parquet(
    family_dir: Path,
    parquet_path: Path,
    *,
    dataset: str,
    config_path: str,
    seed: int,
    task_definition: str,
    notes: str | None,
) -> dict[str, Any]:
    df = pd.read_parquet(parquet_path)
    missing = sorted({"query_idx", "y_orig"}.difference(df.columns))
    if missing:
        raise ValueError(f"Cannot bootstrap manifest from {parquet_path}: missing columns {missing}")

    if "target_class" not in df.columns:
        inferred_target = 1 - df["y_orig"].astype("int64")
        df = df.copy()
        df["target_class"] = inferred_target

    methods = sorted(df["method"].astype(str).unique().tolist()) if "method" in df.columns else []
    if methods:
        first_method = methods[0]
        first_tasks = _group_task_frame(df, first_method)
        for method_name in methods[1:]:
            if not _group_task_frame(df, method_name).equals(first_tasks):
                raise ValueError(
                    f"Cannot bootstrap manifest from {parquet_path}: methods disagree on the task set."
                )
        task_df = first_tasks
    else:
        task_df = _normalize_task_frame(df)

    manifest = {
        "family_name": family_dir.name,
        "dataset": dataset,
        "config_path": config_path,
        "seed": int(seed),
        "n_queries": int(len(task_df)),
        "task_definition": task_definition,
        "query_indices": task_df["query_idx"].astype(int).tolist(),
        "y_orig": task_df["y_orig"].astype(int).tolist(),
        "target_class": task_df["target_class"].astype(int).tolist(),
        "created_at": _now_iso(),
        "notes": notes or "",
        "feature_dim": _feature_dim_from_df(df),
        "source": {"bootstrap_parquet": str(parquet_path)},
    }
    return manifest


def _ensure_manifest(args: argparse.Namespace, family_dir: Path) -> dict[str, Any]:
    manifest_path = family_dir / "manifest.json"
    if manifest_path.exists():
        return _read_json(manifest_path)

    if not args.init_from_parquet:
        raise FileNotFoundError(
            f"Missing manifest: {manifest_path}. "
            "Create manifest.json first, or rerun with --init-from-parquet and the required metadata flags."
        )

    required = {
        "dataset": args.dataset,
        "config_path": args.config_path,
        "seed": args.seed,
        "task_definition": args.task_definition,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(
            f"Bootstrapping manifest requires values for: {missing}. "
            "Pass --dataset, --config-path, --seed, and --task-definition."
        )

    manifest = _manifest_bootstrap_from_parquet(
        family_dir,
        args.init_from_parquet,
        dataset=args.dataset,
        config_path=args.config_path,
        seed=args.seed,
        task_definition=args.task_definition,
        notes=args.notes,
    )
    _write_json(manifest_path, manifest)
    print(f"[INFO] Bootstrapped manifest to {manifest_path}")
    return manifest


def refresh_family(family_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    family_dir.mkdir(parents=True, exist_ok=True)
    methods_dir = family_dir / "methods"
    methods_dir.mkdir(parents=True, exist_ok=True)

    manifest = _ensure_manifest(args, family_dir)
    manifest_tasks = _task_frame_from_manifest(manifest)

    accepted_frames: list[pd.DataFrame] = []
    file_results: list[FileValidationResult] = []
    seen_methods: dict[str, str] = {}

    for parquet_path in sorted(methods_dir.glob("*.parquet")):
        file_result, df = _validate_method_file(parquet_path, manifest, manifest_tasks)
        if df is not None:
            duplicated = [m for m in file_result.methods if m in seen_methods]
            if duplicated:
                dup_msg = ", ".join(
                    f"{m} (already seen in {seen_methods[m]})" for m in duplicated
                )
                raise SystemExit(
                    f"Duplicate method labels across parquet files are not allowed: {dup_msg}."
                )
            for method_name in file_result.methods:
                seen_methods[method_name] = parquet_path.name
            accepted_frames.append(df)
        file_results.append(file_result)

    accepted = [r for r in file_results if r.status == "accepted"]
    rejected = [r for r in file_results if r.status == "rejected"]

    if not accepted_frames:
        raise SystemExit(
            "No valid method parquet files were accepted for this family. "
            "Check manifest compatibility and methods/*.parquet contents."
        )

    combined = pd.concat(accepted_frames, ignore_index=True)
    combined = combined.sort_values(["method", "query_idx", "target_class"], kind="stable").reset_index(drop=True)
    combined_path = family_dir / "combined.parquet"
    combined.to_parquet(combined_path, index=False, compression="gzip")

    summary = {
        "family": {
            key: value
            for key, value in manifest.items()
            if key not in {"query_indices", "y_orig", "target_class"}
        },
        "refreshed_at": _now_iso(),
        "artifacts": {
            "manifest": "manifest.json",
            "combined_parquet": "combined.parquet",
            "summary_json": "summary.json",
            "index_html": "index.html",
        },
        "files": [
            {
                "relative_path": f"methods/{res.path.name}",
                "status": res.status,
                "methods": res.methods,
                "n_rows": res.n_rows,
                "message": res.message,
            }
            for res in file_results
        ],
        "methods": _summary_rows(combined),
        "counts": {
            "accepted_files": len(accepted),
            "rejected_files": len(rejected),
            "accepted_rows": int(len(combined)),
            "accepted_methods": len(seen_methods),
        },
    }

    summary_path = family_dir / "summary.json"
    _write_json(summary_path, summary)

    html_path = family_dir / "index.html"
    html_path.write_text(_render_html(summary))

    print(f"[INFO] Refreshed family: {family_dir}")
    print(f"[INFO] Accepted files: {len(accepted)} | Rejected files: {len(rejected)}")
    print(f"[INFO] Wrote combined parquet: {combined_path}")
    print(f"[INFO] Wrote summary json:    {summary_path}")
    print(f"[INFO] Wrote html index:      {html_path}")
    if rejected:
        print("\n[REJECTED FILES]")
        for item in rejected:
            print(f"- {item.path.name}: {item.message}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh an incremental benchmark databank family.")
    parser.add_argument("--family", required=True, help="Path to the benchmark family directory.")
    parser.add_argument(
        "--init-from-parquet",
        default=None,
        help="Optional parquet path used to bootstrap manifest.json if it does not already exist.",
    )
    parser.add_argument("--dataset", default=None, help="Dataset name used only when bootstrapping a manifest.")
    parser.add_argument("--config-path", default=None, help="Benchmark config path used only when bootstrapping a manifest.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used only when bootstrapping a manifest.")
    parser.add_argument(
        "--task-definition",
        default=None,
        help="Short task-set label used only when bootstrapping a manifest (for example: final_benchmark).",
    )
    parser.add_argument("--notes", default=None, help="Optional notes used only when bootstrapping a manifest.")
    args = parser.parse_args()

    family_dir = Path(args.family)
    if args.init_from_parquet is not None:
        args.init_from_parquet = Path(args.init_from_parquet)
    refresh_family(family_dir, args)


if __name__ == "__main__":
    main()
