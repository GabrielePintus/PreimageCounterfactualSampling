# CPP Query Benchmark Suite

## Purpose

This benchmark is focused on **online CPP query-time performance** and pruning effectiveness.
It is designed for ablation studies before implementing search/pruning optimizations.

## Entrypoint

Run:

```bash
python scripts/cpp_query_benchmark.py --config configs/cpp_query_benchmark_adult.yaml
```

## What It Logs

Per-query artifacts include:
- `total_time_ms`
- `search_time_ms`
- `projection_time_ms`
- `n_qp_solved`
- method-specific pruning stats from atlas profiling:
  - BVH: node pops/prunes, leaves visited, queue peak
  - sorted: candidates considered/pruned, best lower bound at stop
- quality fields (`success`, `validity`, `distance_eval_l2`)
- difficulty bucket (`easy` / `medium` / `hard`) from nearest opposite-class train distance

Outputs:
- Per-query parquet
- Summary parquet
- Markdown report table

## Ablation Workflow

1. Start with baseline variants (`bvh`, `sorted`, `knn`).
2. Add one optimization at a time (single variable change).
3. Compare:
   - p50 / p95 / p99 latency
   - median and tail `n_qp_solved`
   - prune rates (`n_candidates_pruned_by_bound`, node prunes)
   - quality guardrails (`validity`, distance)

## Notes

- Keep seed and query set fixed while comparing variants.
- Include warmup runs when adding heavy new kernels or solver settings.
- If using `kmedoids` subsampling, ensure `scikit-learn-extra` is installed.
