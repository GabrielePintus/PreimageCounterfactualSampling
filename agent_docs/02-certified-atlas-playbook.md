# Certified Atlas Playbook

## What This Module Does

`src/preimage_sampling/` implements the CPP pipeline:
- certify local regions via LiRPA bounds
- represent class preimages as unions of certified polytopes
- query nearest certified counterfactual by convex projection
- accelerate search with BVH pruning

## Primary API

From `preimage_sampling` package:

```python
from preimage_sampling import CertifiedAtlas
from preimage_sampling import ConstantEpsStrategy, NearestOppositeClassClearanceStrategy
```

Main calls:
- `CertifiedAtlas(...).build(...)`
- `atlas.find_counterfactual(...)`
- `atlas.find_counterfactual_batch(...)`
- `atlas.verify_counterfactual(...)`

## Build Phase (Offline)

`build(...)` computes and stores:
- LiRPA lower/upper linear bounds per class/sample
- per-class BVH index over certified regions
- optional 2D polygon unions for visualization (`build_unions=True`)

Important parameters:
- `eps` or `eps_strategy` (mutually exclusive)
- `norm` in `{1, 2, inf}`
- `max_samples_per_class` and `batch_size` for scalability

Epsilon strategies:
- `ConstantEpsStrategy(eps)` for uniform radius
- `NearestOppositeClassClearanceStrategy(alpha)` for per-sample adaptive radius

## Query Phase (Online)

`find_counterfactual(...)` supports:
- `method='bvh'` (recommended exact branch-and-bound over BVH)
- `method='knn'` heuristic over nearest anchors

Robust counterfactual options:
- `delta > 0` activates polytope erosion
- `robust_norm` can differ from certification norm

Feature control:
- `fixed_dims` keeps specified dimensions unchanged during projection

## Solver Dispatch

Projection backend is norm-aware:
- L1/L2: CVXPY path (`CLARABEL` solver)
- L-infinity: SLSQP path (all-linear constraints)

If CVXPY is unavailable, code falls back where possible, but robust behavior for L1/L2 depends on CVXPY support.

## Module Responsibilities

- `certification/lirpa.py`: LiRPA orchestration and batch bound computation.
- `certification/wrapping.py`: one-vs-all wrapped model for certification.
- `geometry/polytopes.py`: ball/box constraints, 2D polygon helpers.
- `geometry/operations.py`: polygon unions and overlap refinement.
- `indexing/bvh.py`: BVH tree + branch-and-bound query.
- `sampling/sampler.py`: lower-level sampler API (legacy-oriented).
- `atlas.py`: high-level orchestrator intended for most users.

## Agent Editing Guardrails

- Preserve `CertifiedAtlas` method signatures unless explicitly requested.
- Keep 2D and CNN pathways both functional (`cnn` flag and flattening logic).
- Do not break assumptions that dataset is TensorDataset-like with `.tensors`.
- Avoid changing numerical tolerances globally without benchmarking impact on feasibility/success.

## Minimal Runtime Check

After edits in `src/preimage_sampling/`, run a tiny end-to-end check:
1. Build a small atlas on synthetic/spiral data.
2. Run one `find_counterfactual` call.
3. Confirm result object fields (`success`, `distance`, `n_qp_solved`) behave as expected.
