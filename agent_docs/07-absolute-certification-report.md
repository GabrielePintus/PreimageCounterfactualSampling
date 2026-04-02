# Report for Gemini Pro: Why CertCF Validity Is <100% and How to Achieve Absolute Certification

## 1) Problem Statement

In our benchmark, `certcf` (the benchmark-facing name for our CertCFAtlas-based method) reports validity below 100%:

- `nearest_neighbor`: 100.0%
- `certcf`: 96.0% (8 failures out of 200)

This is surprising because CertCF should produce certified counterfactuals.

We need to explain the discrepancy and redesign the pipeline to obtain **absolutely certified counterfactuals in the final evaluation space**.

---

## 2) What “validity” currently means in our benchmark

In `scripts/benchmark.py`, validity is computed as:

- per-query `success = (y_cf == target_class)`
- summary `validity% = 100 * sum(success) / n_total`

So validity is **post-hoc class flip rate on the final evaluated counterfactual**, not “QP solved” and not “method returned an output”.

Relevant logic:

- per-query success assignment in generation loop: `scripts/benchmark.py` (around `cf_success = (y_cf == target_class)`)
- summary aggregation: `scripts/benchmark.py` (groupby on `success`)

---

## 3) What we observed in the saved benchmark rows

From `results/benchmark_adult_test.parquet`:

- `certcf` rows: 200
- `success=True`: 192
- `success=False`: 8
- all 8 failures have `error = None`
- all 8 failures have `y_cf = y_orig` (no class flip)

Therefore failures are **not** from timeout or exceptions; they are outputs that do not flip class under final evaluation.

---

## 4) Root cause: certification space ≠ evaluation space

Our CertCF pipeline currently has a space mismatch:

1. Certified projection is solved in a continuous space with linear constraints.
2. For tabular categorical features (OHE), the solution can be fractional in categorical blocks.
3. A decoder / argmax snapping step maps continuous outputs to discrete OHE vectors.
4. Validity is then re-evaluated on the snapped/decoded point.

This decode/snap can move points across the decision boundary.

So the method can be certified in optimization space but not guaranteed valid in the final discrete evaluation space.

In short: **we are certifying one object and evaluating another**.

---

## 5) Why this violates “absolute certification”

For absolute certification, we need a guarantee on the exact returned counterfactual `x_cf_eval`:

1. `x_cf_eval` is in the valid data manifold/constraint set (including strict OHE validity), and
2. classifier predicts target class on `x_cf_eval`, and
3. (optional robustness) for perturbations in chosen norm, target class remains stable.

Current pipeline only guarantees properties for a continuous pre-decoding point `z_cf`; the final evaluated point may differ.

---

## 6) Target guarantee we want

We want the following guarantee contract:

> Returned `x_cf` is the exact point checked by certification and by benchmark validity, with no non-certified post-processing step.

Equivalent requirement:

- remove decode/snap mismatch, or
- certify the full composed mapping (continuous solution -> decode/snap -> classifier), which is hard/non-smooth for argmax.

---

## 7) Proposed technical path to absolute certified CFs

### A) Unify optimization and evaluation spaces (recommended)

Operate directly in final raw feature space and avoid post-hoc decode/snap.

For categorical OHE blocks, enforce exact one-hot semantics during optimization:

- either exact mixed-integer constraints (binary OHE variables + network constraints), or
- structured search over categorical assignments, each solved with certified continuous projection over numerical subspace.

Given current codebase and APIs, the second option is more practical.

### B) Certified branch over categorical assignments

For each query and target class:

1. Choose candidate categorical assignments (all combinations or pruned top-K).
2. Convert assignment into fixed OHE dimensions.
3. Call atlas projection with `fixed_dims` so categorical dims are exact and unchanged by post-processing.
4. Solve certified projection for numerical dims only.
5. Return best feasible candidate.

This removes decode/snap drift and makes the returned point equal to the certified point.

### C) Add a strict final certificate check before accepting output

Before returning success, verify on the same deterministic model used for certification:

- target margin `m(x_cf) = f_t(x_cf) - max_{c!=t} f_c(x_cf)` is non-negative with tolerance, and
- optional robustness margin if delta-robust CF is requested.

If check fails, mark failure and continue search.

### D) Benchmark alignment

Ensure benchmark validity uses the same model and same point used by certification.

No extra decode/snap after method output.

---

## 8) Concrete code-level changes requested

Please propose and (if possible) provide a patch plan for:

1. `scripts/benchmark.py`
   - for `certcf`, remove post-hoc decode/snap path from validity evaluation (or gate it behind a legacy mode)
   - add explicit columns distinguishing:
     - `solver_success`
     - `cert_check_success`
     - `eval_success`

2. `src/counterfactuals/methods/certcf.py`
   - extend `generate(...)` to support strict categorical fixing strategy and strict acceptance criteria
   - include metadata fields for certificate diagnostics

3. `src/certcf/atlas.py`
   - add/query helper for categorical assignment constrained search
   - ensure returned `x_cf` is already evaluation-ready (no extra snapping required)
   - add optional strict final target-margin check routine

4. configs
   - add options for:
     - `strict_certification: true`
     - categorical search strategy (`all`, `topk`, `beam`)
     - strict tolerance settings (`margin_tol`, feasibility tolerances)

---

## 9) Acceptance criteria (must pass)

For strict mode:

1. **No post-processing drift**
   - benchmark evaluates exactly the returned `x_cf`
2. **Certification/evaluation consistency**
   - if `success=True`, then `eval_success=True` by construction
3. **Deterministic validity**
   - repeated runs with same seed produce identical success outcomes
4. **Diagnostic transparency**
   - every failure has explicit reason (`infeasible`, `margin_check_failed`, `timeout`, etc.)
5. **Empirical target**
   - if strict guarantees hold, observed validity should be 100% on solvable queries; unsolved queries must be explicit, not silent class-mismatch outputs

---

## 10) Questions for Gemini Pro

Please answer with:

1. A rigorous guarantee statement we can claim after the redesign.
2. The best practical algorithm in this codebase for strict OHE validity + certified class flip.
3. Tradeoff analysis (exact MILP vs constrained categorical search + certified projection).
4. A minimal incremental patch sequence (small PRs) to get to strict mode safely.
5. Potential numerical pitfalls/tolerances that can create false certificate passes/fails.
6. Suggested unit/integration tests to prove end-to-end certification consistency.

---

## 11) Summary in one sentence

Our current 96% validity is caused by a certification/evaluation mismatch (continuous certified point vs decoded/snapped evaluated point); absolute certification requires guaranteeing validity on the exact final returned point, with no non-certified post-processing.
