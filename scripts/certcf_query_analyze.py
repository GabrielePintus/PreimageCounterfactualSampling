#!/usr/bin/env python3
"""Analyze CertCF query benchmark outputs and print optimization-focused diagnostics.

Usage:
    python scripts/certcf_query_analyze.py \
      --per-query results/certcf_query_benchmark_adult.parquet \
      --summary results/certcf_query_benchmark_adult_summary.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CertCF query benchmark artifacts")
    parser.add_argument("--per-query", required=True, help="Per-query parquet path")
    parser.add_argument("--summary", required=True, help="Summary parquet path")
    args = parser.parse_args()

    per_query_path = Path(args.per_query)
    summary_path = Path(args.summary)

    perq = pd.read_parquet(per_query_path)
    summary = pd.read_parquet(summary_path)

    print("=== SUMMARY ===")
    print(_fmt(summary))

    print("\n=== DIFFICULTY BREAKDOWN ===")
    diff = (
        perq.groupby(["variant", "difficulty"]).agg(
            n=("query_idx", "count"),
            time_p50=("total_time_ms", lambda s: float(np.quantile(s, 0.5))),
            time_p95=("total_time_ms", lambda s: float(np.quantile(s, 0.95))),
            n_qp_mean=("n_qp_solved", "mean"),
            validity=("validity", "mean"),
        )
        .reset_index()
    )
    print(_fmt(diff))

    print("\n=== PHASE TIMING (NON-OVERLAPPING) ===")
    phase_cols = [
        c
        for c in [
            "search_time_ms",
            "projection_time_ms",
            "query_loop_time_ms",
            "total_time_ms",
        ]
        if c in perq.columns
    ]
    phase = perq[["variant"] + phase_cols].groupby("variant").mean().reset_index()
    if "search_time_ms" in phase.columns and "projection_time_ms" in phase.columns:
        phase["search_share_%"] = 100.0 * phase["search_time_ms"] / (phase["search_time_ms"] + phase["projection_time_ms"] + 1e-12)
        phase["projection_share_%"] = 100.0 * phase["projection_time_ms"] / (phase["search_time_ms"] + phase["projection_time_ms"] + 1e-12)
    print(_fmt(phase))

    print("\n=== PRUNING STATS ===")
    prune_cols = [
        c
        for c in [
            "n_candidates_considered",
            "n_candidates_total",
            "n_candidates_pruned_by_bound",
            "n_nodes_popped",
            "n_nodes_pruned",
            "n_leaves_visited",
            "max_queue_size",
            "best_lower_bound_at_termination",
        ]
        if c in perq.columns
    ]
    if prune_cols:
        prune = perq[["variant"] + prune_cols].groupby("variant").mean().reset_index()
        if {"n_candidates_total", "n_candidates_considered"}.issubset(set(prune.columns)):
            prune["candidate_prune_rate_%"] = (
                100.0 * (prune["n_candidates_total"] - prune["n_candidates_considered"]) / (prune["n_candidates_total"] + 1e-12)
            )
        print(_fmt(prune))


if __name__ == "__main__":
    main()
