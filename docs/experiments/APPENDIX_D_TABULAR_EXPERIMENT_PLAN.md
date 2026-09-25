# Appendix D: seven-dataset ablation protocol

## Goal

Make every experiment in Appendix D refer to the same seven tabular datasets:
Adult, COMPAS, German Credit, Give Me Some Credit, HELOC, Lending Club, and
Wisconsin Breast Cancer. Results must remain available per dataset before any
macro-average is computed.

## Common protocol

- Seed: 42.
- Classifier checkpoints and preprocessing: exactly those of the main benchmark.
- Radius and projection norm: `L1` unless the norm itself is ablated.
- Radius multiplier: `alpha = 0.20` unless `alpha` is ablated.
- Training reference: random sample of at most 10,000 total training points.
- Atlas: random sample of 500 anchors per predicted class unless a smaller,
  explicitly controlled atlas is required.
- Query retrieval: `k = 5` unless `k` is ablated.
- Immutable features: `sex` for Adult and COMPAS, as in their dataset specs.
- Categorical features: the existing one-hot decoding and feasibility checks.
- Aggregation: compute every metric within each dataset, then macro-average the
  seven dataset-level values so that dataset size does not determine its weight.

## Existing experiments that do not need to be rerun

The following already use all seven tabular datasets and can be retained:

1. `alpha` sensitivity;
2. sparsity weight `lambda`;
3. `L1` versus `L2` projection objective.

Their existing artifacts remain the source of the corresponding Appendix D
figures and table.

## New experiment 1: adaptive shrinkage

Compare atlas construction with and without adaptive shrinkage for
`alpha = 0.01, 0.05, ..., 0.95, 0.99` on every dataset. Use 500 random anchors
per predicted class.

Save one row per region with:

- initial and final radius;
- final-to-initial radius ratio;
- center-certification indicator and slack;
- number of shrink steps.

Report per-dataset curves. For a compact cross-dataset result, macro-average
dimensionless quantities such as center-certification rate, radius-retention
ratio, and shrink frequency. Raw radii should not be the sole aggregate because
the encoded dimensionality differs across datasets.

Entrypoint: `scripts/tabular_atlas_ablation.py shrinkage`.

## New experiment 2: LiRPA backend

Compare CROWN, optimized CROWN, and alpha-CROWN on each dataset using the same
300 prediction-balanced random anchors and up to 200 fixed test queries. Repeat
each build three times.

Save:

- build time and resource measurements;
- per-region radius, center certification, and shrinkage diagnostics;
- the fraction of 1,024 uniform samples from the final `L1` ball retained by
  the certified halfspaces;
- per-query validity, `L1`, `L2`, `L0`, and query time.

The retained fraction is comparable across dimensions. Raw log-volume is kept
per dataset but is not macro-averaged because the datasets have different
encoded dimensions.

Entrypoint: `scripts/tabular_atlas_ablation.py backend`.

## New experiment 3: certification mechanism

Compare Anchor-Ball, Anchor-PGD, and CertCF on every dataset. Each method uses
the same support data, 1,000 prediction-balanced random anchors, initial radii,
top-5 retrieval, immutable-feature constraints, and categorical feasibility
rules. Use up to 500 test queries per dataset; smaller test sets are used in
full. PGD retains the existing five penalty weights, 200 steps per weight,
three initializations, and a 120-second timeout per query.

Report success on all queries. Compute proximity, sparsity, plausibility, and
empirical retention on shared-success queries, first per dataset and then by
macro-average. Empirical retention uses the main benchmark's Cartesian grid
`{0, 0.01, 0.03, 0.05, 0.10} x {0, 0.01, 0.03, 0.05, 0.10}` for numerical
noise and categorical-block flip probability, with ten samples per cell.

Entrypoint: `scripts/lirpa_refinement_ablation.py` with
`configs/experiments/lirpa_refinement_ablation_tabular.yaml`.

## New experiment 4: top-k retrieval

For each dataset, compare strict nearest-anchor prefixes `k = 1, ..., 7` with
exact exhaustive search over the same certified atlas. Use up to 200 fixed test
queries per dataset; German Credit and Wisconsin Breast Cancer use their full
smaller test sets. The comparison respects immutable features and categorical
decoding.

Save and report:

- strict and exhaustive success;
- empirical miss probability and one-sided 95% Clopper-Pearson upper bound;
- unconditional and miss-conditional absolute `L1` gap;
- relative gap, to support comparison across datasets;
- minimum `k` recovering the exhaustive result;
- projection counts and runtime.

Optimality probabilities use only queries for which exhaustive search returns a
counterfactual; success rates retain all queries. Gap summaries use
shared-success queries. The unconditional gap includes exact recoveries (zero
gap), whereas the conditional gap includes only misses.

Entrypoint: `scripts/tabular_topk_ablation.py`.

## Execution order

1. Run the lightweight `prepare` stages and pilots.
2. Run shrinkage, which requires only atlas builds.
3. Run the backend comparison.
4. Run top-k, saving after every exhaustive query.
5. Run the certification-mechanism experiment last because multi-restart PGD is
   the dominant cost and supports per-query resumption.
6. Aggregate each family only after all per-dataset artifacts have been checked.
7. Update Appendix D from the per-dataset and macro summaries; move the
   synthetic-only description out of Appendix D after it is no longer used there.

## Expected full-run sizes

- Shrinkage: `7 datasets x 2 variants x 21 alpha values = 294` builds.
- Backend: `7 datasets x 3 backends x 3 repetitions = 63` builds, with up to
  200 queries per build.
- Certification mechanism: up to 500 queries per dataset and three methods.
- Top-k: up to 200 queries per dataset, seven prefixes per query, plus one
  exhaustive search per query.

The backend, top-k, and certification-mechanism runners keep separate artifacts
per dataset or query so interrupted experiments can resume without discarding
completed work.
