# CertCF Speedup Ideas

Working note for future CertCF optimization work, based on the current Adult
benchmark behavior in `configs/benchmarks/test_certcf.yaml`.

## Current Observation

For the current Adult `certcf` setup:
- `query_method: nearest_anchor`
- `norm: 2`
- `query_k_candidates: 1` or `3`
- `k_per_class: 5000`
- `device: cuda`

thread-based query parallelism did not produce useful throughput gains. Query
time still appears dominated by projection and OHE decode, not by outer-loop
Python orchestration.

## Main Speedup Directions

1. Make decode cheaper.
   The biggest likely win is reducing work in `_polytope_aware_decode(...)` in
   `src/certcf/atlas.py`. In particular, avoid or budget the exact fallback
   path when heuristic OHE snap fails.

2. Make `nearest_anchor` truly top-k.
   The current `query_k_candidates` behavior tries the top-k anchors first, but
   then scans additional anchors on fallback. A strict top-k mode would make
   the runtime meaning of the knob much clearer and could substantially reduce
   query cost.

3. Expose real CVXPY solver controls.
   The current L2 projection path rebuilds and solves a fresh CVXPY problem in
   `_project_cvxpy(...)`. Add configurable solver choice, iteration caps, and
   looser tolerances for benchmark-oriented fast modes.

4. Reuse CVXPY problem structure when possible.
   A deeper but potentially important optimization is to avoid rebuilding the
   full optimization problem from scratch for every anchor/decode subproblem.
   Reusing parameterized problem structure could reduce canonicalization and
   solver setup overhead.

5. Reduce exact decode subsolves.
   Even without a full approximation mode, cap or shrink the exact
   enumeration/branch-and-bound work in decode. Examples: node budgets, solver
   call budgets, or stronger pruning before child solves.

## Practical Priority Order

If the goal is to speed up CertCF soon with limited code churn, the current
recommended order is:

1. cheaper decode mode
2. strict top-k nearest-anchor mode
3. CVXPY tuning controls
4. exact-decode budgets
5. CVXPY problem reuse / deeper refactor

## Why `query_k_candidates` Alone Did Not Help Much

Lowering `query_k_candidates` from `3` to `1` did not materially reduce Adult
query time because the current `nearest_anchor` implementation can still fall
back to scanning additional anchors if the first anchor does not yield a
feasible certified result. Also, one anchor can still be expensive on its own
due to CVXPY projection and exact OHE decode fallback.
