"""Typed data structures for benchmark results.

Hierarchy
---------
BenchmarkResult
    x_queries  — shared query set, stored once
    y_orig     — shared source-class labels used by the benchmark task
    y_true     — optional ground-truth labels, stored once
    method_results: list[MethodResult]
        params        — native dict (no JSON)
        build_time_s  — once per run, not repeated per query
        query_results: list[QueryResult]
            x_cf      — None on failure
            metrics   — pre-computed at benchmark time

Serialization
-------------
BenchmarkResult.save(path)  — pickle to .bmk file
BenchmarkResult.load(path)  — unpickle

Notebook compatibility
----------------------
BenchmarkResult.to_dataframe()  — flatten to the same column schema used by
                                   the analysis notebooks (x_orig_*, x_cf_*, etc.)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class QueryResult:
    """Outcome of one counterfactual generation call."""

    query_idx: int                    # index into the original test set
    x_cf: Optional[np.ndarray]       # None on failure; shape (d,)
    y_cf: Optional[int]              # predicted class of x_cf; None on failure
    success: bool
    runtime_s: float
    error: Optional[str]             # None on success
    l2_distance: float
    l1_distance: float
    l0_sparsity: float
    mad_l1_distance: float
    redundancy: float
    target_class: Optional[int] = None


@dataclass
class MethodResult:
    """All results for one method run (one entry in the grid)."""

    method: str                       # registry key, e.g. "face"
    run_name: str                     # e.g. "face_knn_n_neighbors=5_k_per_class=200"
    params: Dict[str, Any]            # native Python dict — no JSON serialization
    build_time_s: float
    space: str                        # "raw" or "gen"
    query_results: List[QueryResult] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Top-level container for a complete benchmark run."""

    dataset: str
    seed: int
    x_queries: np.ndarray             # (n_queries, d) — stored once
    y_orig: np.ndarray                # (n_queries,) — stored once
    y_true: Optional[np.ndarray] = None
    method_results: List[MethodResult] = field(default_factory=list)

    # ---------------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize to a .bmk file using pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkResult":
        """Deserialize from a .bmk file."""
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected BenchmarkResult, got {type(obj)}")
        return obj

    # ---------------------------------------------------------------------------
    # Notebook compatibility
    # ---------------------------------------------------------------------------

    def to_dataframe(self, include_features: bool = True) -> pd.DataFrame:
        """Flatten to a DataFrame matching the legacy parquet column schema.

        Columns produced (identical to the old row-builder output):
          method, query_idx, space, y_orig, source_class, y_cf, target_class, success,
          build_time_s, runtime_s, params (JSON string), error,
          l2_distance, l1_distance, l0_sparsity, mad_l1_distance, redundancy
          [x_orig_0 .. x_orig_{d-1}, x_cf_0 .. x_cf_{d-1}]  (if include_features)
        """
        n_features = self.x_queries.shape[1] if self.x_queries.ndim == 2 else 0
        rows: List[Dict[str, Any]] = []

        for mr in self.method_results:
            params_json = json.dumps(mr.params)
            for pos, qr in enumerate(mr.query_results):
                x_orig = self.x_queries[pos]
                y_orig_val = int(self.y_orig[pos])

                row: Dict[str, Any] = {
                    "method": mr.run_name,
                    "query_idx": int(qr.query_idx),
                    "space": mr.space,
                    "y_orig": y_orig_val,
                    "source_class": y_orig_val,
                    "y_cf": int(qr.y_cf) if qr.y_cf is not None else float("nan"),
                    "target_class": int(qr.target_class) if qr.target_class is not None else float("nan"),
                    "success": bool(qr.success),
                    "build_time_s": float(mr.build_time_s),
                    "runtime_s": float(qr.runtime_s),
                    "params": params_json,
                    "error": qr.error,
                    "l2_distance": float(qr.l2_distance),
                    "l1_distance": float(qr.l1_distance),
                    "l0_sparsity": float(qr.l0_sparsity),
                    "mad_l1_distance": float(qr.mad_l1_distance),
                    "redundancy": float(qr.redundancy),
                }
                if self.y_true is not None:
                    row["y_true"] = int(self.y_true[pos])

                if include_features:
                    for k in range(n_features):
                        row[f"x_orig_{k}"] = float(x_orig[k])
                    if qr.x_cf is not None:
                        for k in range(n_features):
                            row[f"x_cf_{k}"] = float(qr.x_cf[k])
                    else:
                        for k in range(n_features):
                            row[f"x_cf_{k}"] = float("nan")

                rows.append(row)

        return pd.DataFrame(rows)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------

    def summary(self) -> None:
        """Print a terminal summary table (one row per method run)."""
        try:
            from tabulate import tabulate
        except ImportError:
            print("[WARNING] tabulate not installed; skipping summary table.")
            return

        rows = []
        for mr in self.method_results:
            qrs = mr.query_results
            n_total = len(qrs)
            ok = [qr for qr in qrs if qr.success]
            n_ok = len(ok)
            validity = 100.0 * n_ok / n_total if n_total > 0 else float("nan")
            l2 = float(np.mean([qr.l2_distance for qr in ok])) if ok else float("nan")
            l1 = float(np.mean([qr.l1_distance for qr in ok])) if ok else float("nan")
            sp = 100.0 * float(np.mean([qr.l0_sparsity for qr in ok])) if ok else float("nan")
            query_t = float(np.mean([qr.runtime_s for qr in qrs])) if qrs else float("nan")
            rows.append([
                mr.run_name,
                f"{validity:.1f}%",
                f"{l2:.3f}",
                f"{l1:.3f}",
                f"{sp:.1f}%",
                f"{mr.build_time_s:.2f}",
                f"{query_t:.3f}",
                n_total - n_ok,
            ])

        print("\nBENCHMARK SUMMARY")
        print(tabulate(
            rows,
            headers=["method", "validity%", "l2_mean", "l1_mean", "sparsity%",
                     "build_s", "query_s", "n_failed"],
            tablefmt="simple",
        ))

    # ---------------------------------------------------------------------------
    # Legacy parquet loader
    # ---------------------------------------------------------------------------

    @classmethod
    def load_legacy_parquet(cls, path: str | Path) -> "BenchmarkResult":
        """Load an old-style flat parquet file into a BenchmarkResult.

        The parquet must contain the columns produced by the old row-builder
        functions: method, query_idx, space, y_orig, y_cf, success,
        build_time_s, runtime_s, params (JSON), error, l2_distance, etc.,
        plus x_orig_* and x_cf_* feature columns.
        """
        df = pd.read_parquet(Path(path))

        # Detect feature columns.
        orig_cols = sorted(
            [c for c in df.columns if c.startswith("x_orig_")],
            key=lambda c: int(c.split("_")[-1]),
        )
        cf_cols = sorted(
            [c for c in df.columns if c.startswith("x_cf_")],
            key=lambda c: int(c.split("_")[-1]),
        )

        # Recover shared query set from the first method group.
        # All methods share the same query set so any group works.
        first_method = df["method"].iloc[0]
        first_grp = df[df["method"] == first_method].sort_values("query_idx")
        x_queries = first_grp[orig_cols].to_numpy(dtype=np.float32)
        y_orig = first_grp["y_orig"].to_numpy(dtype=np.int64)
        y_true = first_grp["y_true"].to_numpy(dtype=np.int64) if "y_true" in first_grp.columns else None
        query_idx_order = first_grp["query_idx"].tolist()

        # Build an index from query_idx → position in x_queries.
        idx_to_pos = {q: pos for pos, q in enumerate(query_idx_order)}

        method_results: List[MethodResult] = []
        for run_name, grp in df.groupby("method", sort=False):
            grp = grp.sort_values("query_idx")
            build_time_s = float(grp["build_time_s"].iloc[0]) if "build_time_s" in grp.columns else 0.0
            space = str(grp["space"].iloc[0]) if "space" in grp.columns else "raw"
            params_raw = grp["params"].iloc[0] if "params" in grp.columns else "{}"
            try:
                params = json.loads(params_raw) if isinstance(params_raw, str) else {}
            except (json.JSONDecodeError, TypeError):
                params = {}

            query_results: List[QueryResult] = []
            for _, row in grp.iterrows():
                q_idx = int(row["query_idx"])
                success = bool(row["success"])
                x_cf_arr = None
                if success and cf_cols:
                    cf_vals = row[cf_cols].to_numpy(dtype=np.float32)
                    if not np.any(np.isnan(cf_vals)):
                        x_cf_arr = cf_vals

                query_results.append(QueryResult(
                    query_idx=q_idx,
                    x_cf=x_cf_arr,
                    y_cf=int(row["y_cf"]) if success and not pd.isna(row.get("y_cf")) else None,
                    success=success,
                    runtime_s=float(row.get("runtime_s", 0.0)),
                    error=row.get("error") if not success else None,
                    l2_distance=float(row.get("l2_distance", float("nan"))),
                    l1_distance=float(row.get("l1_distance", float("nan"))),
                    l0_sparsity=float(row.get("l0_sparsity", float("nan"))),
                    mad_l1_distance=float(row.get("mad_l1_distance", float("nan"))),
                    redundancy=float(row.get("redundancy", float("nan"))),
                    target_class=(
                        int(row["target_class"])
                        if "target_class" in row and not pd.isna(row.get("target_class"))
                        else None
                    ),
                ))

            # Sort query_results by position in x_queries so to_dataframe() aligns correctly.
            query_results.sort(key=lambda qr: idx_to_pos.get(qr.query_idx, 0))

            method_results.append(MethodResult(
                method=str(run_name),
                run_name=str(run_name),
                params=params,
                build_time_s=build_time_s,
                space=space,
                query_results=query_results,
            ))

        return cls(
            dataset="unknown",
            seed=0,
            x_queries=x_queries,
            y_orig=y_orig,
            y_true=y_true,
            method_results=method_results,
        )
