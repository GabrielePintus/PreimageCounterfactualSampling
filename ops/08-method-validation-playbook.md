# Method Validation Playbook

## Goal

This document defines how we validate counterfactual methods in this repository.
It is meant to keep benchmarking and notebook analysis consistent across runs,
methods, and future paper iterations.

The key principle is simple:

- use the benchmark pipeline to generate reproducible result parquets
- evaluate methods with a small set of motivated metrics
- visualize those metrics with a small set of motivated plots

This avoids ad hoc notebook analyses that are hard to compare across time.

## Notebook Helper Contract

Validation notebooks should prefer the shared helper layer in `notebooks/utils/`
over large inline blocks of pandas or seaborn code.

Expected pattern:

- load results through `load_result(...)` or `load_result_set(...)`
- normalize and filter through `prepare_benchmark_df(...)`
- build tables through summary helpers
- build standard visualizations through shared plot helpers

Inline notebook code is still acceptable for:

- notebook-specific exploratory checks
- one-off debug visualizations
- short glue code that composes existing helpers

If logic is reused across notebooks, move it into `notebooks/utils/` before
copying it again.

---

## How To Benchmark

### 1. Use the benchmark entrypoint

The official benchmark entrypoint is:

```bash
python scripts/benchmark.py --config configs/benchmarks/<config>.yaml
```

Examples:

```bash
python scripts/benchmark.py --config configs/benchmarks/benchmark_smoke_all.yaml
```

```bash
python scripts/benchmark.py \
  --config configs/benchmarks/benchmark_meeting_all_200q.yaml \
  --datasets adult compas german_credit
```

### 2. Keep the task set fixed across methods

When comparing methods, they must be evaluated on the same query set.

This means:
- same dataset split
- same benchmark seed
- same query indices
- same target policy

If methods are not run on the same query set, proximity and validity comparisons
are much less meaningful.

### 3. Keep method comparisons prediction-aligned

The benchmark pipeline now uses model-predicted training labels during method
fit. This is the intended evaluation semantics for the repository benchmark.

### 4. Prefer benchmark families for iterative experiments

When adding methods incrementally, use the benchmark databank layout:

```text
results/benchmarks/<family_name>/
  manifest.json
  methods/
    <run_name>.parquet
  combined.parquet
  summary.json
  index.html
```

This is the safest way to ensure:
- the task set is locked
- methods are validated against the same manifest
- combined analysis remains reproducible

### 5. Compare both overall and matched-success subsets

Every serious method comparison should include two views:

- overall benchmark results
- matched-success comparison on the intersection of query points where all
  compared methods succeeded

Purpose:
- overall results capture real-world method reliability
- matched-success results isolate conditional quality when methods all succeed

Both views matter, and neither should replace the other.

---

## Metrics We Care About

### 1. Validity / Success Rate

Definition:
- fraction of benchmark tasks with `success == True`

Purpose:
- this is the primary feasibility metric
- a method with low validity cannot be considered strong even if its distances
  are excellent on the few queries it solves

Why it matters:
- counterfactual quality is irrelevant if the method frequently fails
- this is the first metric to inspect before proximity

### 2. Method Success vs Target Reached

Definition:
- `method_success`: whether the method itself reported success
- `target_reached`: whether the returned point actually reaches the requested
  target class

Purpose:
- separates internal method failure from external benchmark failure
- helps diagnose whether a method fails because it returns no candidate, or
  because it returns a point that is not actually valid

Why it matters:
- critical for debugging and honest failure analysis

### 3. L2 Proximity

Definition:
- Euclidean distance between factual and counterfactual in raw feature space

Purpose:
- captures overall geometric closeness
- useful as a smooth, global notion of change magnitude

Why it matters:
- standard in the literature
- often more stable than sparse/discrete metrics

### 4. L1 Proximity

Definition:
- Manhattan distance between factual and counterfactual in raw feature space

Purpose:
- captures total amount of feature movement
- often more interpretable than L2 in tabular settings

Why it matters:
- tends to reflect “how much changed” more directly than L2
- especially useful when a method spreads small changes across many features

### 5. MAD-normalized L1

Definition:
- L1 distance weighted by median absolute deviation per feature

Purpose:
- prevents high-variance features from dominating the distance unfairly
- gives a scale-aware proximity measure

Why it matters:
- raw L1 can be distorted by heterogeneous feature scales
- useful for cross-feature fairness in tabular data

### 6. Sparsity (`L0`)

Definition:
- fraction or count of changed features

Purpose:
- captures how many features were modified, not just by how much

Why it matters:
- many counterfactuals are easier to understand when fewer features change
- complements L1/L2, since a low-distance CF may still touch many dimensions

### 7. Redundancy

Definition:
- fraction of changed features that can be individually reverted without losing
  the target prediction

Purpose:
- measures unnecessary feature changes

Why it matters:
- a good counterfactual should not contain many irrelevant edits
- helps distinguish minimal explanations from bloated ones

### 8. Runtime

Definition:
- per-query generation time

Purpose:
- measures practical usability at inference time

Why it matters:
- a method can be excellent in quality but unusable in practice if query time
  is too high
- especially important for CertCF, where build time and query time are separate

### 9. Build Time

Definition:
- one-time fit/build cost before query generation

Purpose:
- measures offline scalability

Why it matters:
- especially important for CertCF atlas construction
- should always be analyzed separately from query time

### 10. Failure Breakdown

Definition:
- grouped causes of failed queries: timeout, fit/build error, no valid CF, etc.

Purpose:
- turns a single failure rate into actionable debugging information

Why it matters:
- tells us what to improve next
- critical during method development

### 11. Constraint Quality

Definition:
- OHE validity
- invalid categorical block fraction
- distance to snapped valid OHE representation
- immutable-feature violations

Purpose:
- checks whether returned CFs satisfy structural data constraints

Why it matters:
- a method may be “valid” with respect to the classifier while still producing
  malformed categorical or immutable-feature edits

### 12. Empirical Robustness

Definition:
- how often small random perturbations around the returned CF preserve the
  target prediction

Purpose:
- measures local stability of counterfactuals

Why it matters:
- boundary-hugging CFs are fragile
- robustness matters if we want explanations that are not one-bit artifacts

### 13. Manifoldness / Plausibility

Definition:
- kNN distance to training data
- class-conditional kNN distance
- optional LOF-based outlier score

Purpose:
- measures whether a counterfactual lies in a realistic region of the data
  distribution

Why it matters:
- this is the best operational proxy we have for plausibility / on-manifoldness
- especially important when low-distance CFs still look unrealistic

Important note:
- this is not the same thing as interpretability
- it should be reported as plausibility / manifoldness, not conflated with
  human interpretability

---

## Plots And Visualizations We Care About

### 1. Validity Bar Plot

Plot:
- per-method validity percentage

Purpose:
- first overview of which methods are actually reliable

Why we care:
- this is the quickest way to detect whether a method is even competitive

### 2. Proximity KDE Plots

Plot:
- KDE distributions of `L1` and `L2` by method, usually per dataset

Purpose:
- compare the full distribution, not just the mean

Why we care:
- means can hide tails, multimodality, and outliers
- if two methods have nearly identical distributions, tiny mean differences are
  probably not meaningful

### 3. Boxplots For Proximity / Runtime

Plot:
- boxplots of `L1`, `L2`, or runtime by method

Purpose:
- compact view of medians, spread, and outliers

Why we care:
- easier than KDE when distributions overlap heavily
- useful for quick sanity checks

### 4. Validity-Distance Curves

Plot:
- success rate as a function of a distance threshold

Purpose:
- combines validity and proximity into one view

Why we care:
- tells us how quickly a method accumulates successful low-distance CFs
- often more informative than comparing only mean distances

### 5. Runtime Bar / Log-Runtime Plot

Plot:
- mean query runtime on linear and log scales

Purpose:
- compare practical efficiency across methods

Why we care:
- log scale is essential when methods differ by orders of magnitude

### 6. Runtime Distribution Plot

Plot:
- runtime boxplot or KDE by method

Purpose:
- inspect tail latency, not only the mean

Why we care:
- important for methods with unstable or heavy-tailed query time

### 7. Failure Breakdown Stacked Bar

Plot:
- stacked bar of failure categories by method

Purpose:
- show *why* methods fail

Why we care:
- helps prioritize engineering/debugging work
- separates “method is slow” from “method returns invalid CFs”

### 8. Matched-Success Comparison Tables

Visualization:
- table computed only on the intersection of successful queries across methods

Purpose:
- compare quality on the same subset of tasks

Why we care:
- removes confounding from different success sets
- should be used alongside, not instead of, overall results

### 9. Feature-Change Heatmap

Plot:
- normalized mean absolute change per feature and method

Purpose:
- show *where* each method tends to act

Why we care:
- helps identify whether a method concentrates on a few meaningful features or
  spreads change broadly across the input

### 10. Constraint Quality Tables / Heatmaps

Visualization:
- OHE validity rate
- invalid categorical block fraction
- immutable-feature violation rate

Purpose:
- verify structural correctness of CFs on tabular datasets

Why we care:
- classifier validity alone is not enough for trustworthy evaluation

### 11. Empirical Robustness Curves

Plot:
- robustness success percentage as epsilon increases, for different norms

Purpose:
- inspect local stability around counterfactuals

Why we care:
- reveals whether a method produces brittle solutions near the decision boundary

### 12. Manifoldness Summary Tables

Visualization:
- per-method table of:
  - kNN distance to all train points
  - kNN distance to target-class train points
  - optional LOF score

Purpose:
- quantify plausibility / on-manifoldness

Why we care:
- useful complement to proximity
- low-distance CFs are not necessarily realistic CFs

---

## Recommended Analysis Order

When building a notebook, the default order should be:

1. Benchmark overview
2. Validity summary
3. Runtime summary
4. Proximity summary
5. Proximity distributions
6. Matched-success comparison
7. Failure analysis
8. Constraint quality
9. Empirical robustness
10. Manifoldness / plausibility

This order keeps the analysis honest:
- first ask whether the method works
- then ask how close it is
- then ask whether it is stable and plausible

---

## Practical Rule For New Notebooks

A new benchmark-analysis notebook should ideally answer these questions:

1. Does the method succeed often enough to matter?
2. When it succeeds, is it actually closer than the alternatives?
3. Is any observed gain real across the distribution, or only in the mean?
4. Are the returned CFs structurally valid?
5. Are they locally robust?
6. Are they plausible / on-manifold?
7. Is the runtime/build-time cost acceptable?

If a notebook does not help answer at least some of these, it is probably not a
good validation notebook.
