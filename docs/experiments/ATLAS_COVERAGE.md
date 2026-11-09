# Held-out atlas coverage

This experiment measures the fraction of held-out inputs predicted as class
`t` that belong to the union of the saved certified regions for `t`. It does
not measure geometric preimage volume, ground-truth classification recall,
counterfactual-generation success, or proximity to the atlas.

## Synthetic-32 pilot

The pilot reuses the artifacts identified by
`configs/experiments/topk_heuristic_ablation.yaml`:

- `results/network_complexity/data/dataset.npz`: standardized inputs, with
  10,000 training, 1,000 validation and 1,000 test examples;
- the saved `depth_03_width_128` classifier;
- `results/topk_heuristic_ablation/atlas/`: 500 certified regions per class.

No model training, dataset generation, certification, or source-artifact
updates are performed. All 1,000 test examples are evaluated, grouped by
model prediction, not by ground-truth label. Erosion is disabled. The membership
tolerance is `1e-6`, consistent with the production membership check.

For a class with `N` regions, each of ten seeded random permutations supplies
nested prefixes of lengths `ceil(f*N)` for `f = .10, .25, .50, .75, 1.0`.
The complete saved atlas is the 100% endpoint; it is not an atlas constructed
from all training examples. Existing region geometry stays fixed across prefixes.
The plot shows mean coverage and the min–max range across permutations, not
a confidence interval for population coverage. At 100%, every ordering has
exactly the same coverage. This does not capture atlas-construction variability.

## Run

From the repository root:

```bash
MPLCONFIGDIR=/tmp/certcf-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/atlas_coverage.py \
  --config configs/experiments/atlas_coverage_synthetic32.yaml
```

The first run checks the source dataset, checkpoint and atlas hashes, evaluates
the model once on the test set, and caches boolean membership matrices. Later
runs reuse these matrices when their hashes and computation fingerprint match;
changing analysis fractions, permutation count or seed does not invalidate them.
`--force` recomputes coverage only; it never rebuilds the source atlas.

Outputs under `results/atlas_coverage/synthetic32/`:

- `membership.npz`: test indices, labels, predictions and one matrix per class.
  Each matrix has all test rows in original `x_test` order and one column per
  saved atlas region. `balls_*` are enclosing-ball diagnostic matrices, not
  certified membership. The analysis selects rows by predicted class.
- `membership_metadata.json`: provenance, source hashes, split checks, accuracy
  and class-level diagnostics.
- `orderings.npz`: exact saved region permutations per class.
- `coverage_by_ordering.csv`: covered counts and fractions for each prefix.
- `coverage_summary.csv`: per-class mean, standard deviation and range.
- `coverage_full.csv`: exact full-atlas covered counts and denominators.
- `coverage_curve.png` and `.pdf`: class-separated coverage curves.
- `analysis_metadata.json`: configuration and interpretation of variability.

## Checks

- Source artifact fingerprints must match the existing completed build.
- Test rows must not exactly overlap training, validation or atlas anchors.
- Every saved anchor must occur in the training data and pass both scalar
  production membership and vectorized membership in its own region.
- One hundred sampled test/region pairs per class are compared against the
  scalar production membership check.
- No certified membership may contradict the model's predicted class.
- Prefix coverage must be nondecreasing and all full-atlas endpoints must match.

Unit tests cover positive/negative membership, L1/L2/L-infinity geometry,
classification margin, input bounds, empty regions, empty evaluation classes,
zero hits, reproducibility and nested-prefix counts:

```bash
MPLCONFIGDIR=/tmp/certcf-mpl .venv/bin/pytest tests/experiments/test_atlas_coverage.py -q
```

## Pilot result on 9 November 2026

| Predicted class | Saved regions | Held-out points | Covered points | Coverage |
| --- | ---: | ---: | ---: | ---: |
| 0 | 500 | 500 | 0 | 0% |
| 1 | 500 | 500 | 0 | 0% |

All subset coverages are consequently zero. Test accuracy is 98%. None of the
test points is inside a final enclosing ball of its predicted class, so the
zero coverage already follows from the radius restrictions, before applying
the affine certification constraints. The median final L1 radii are approximately
1.364 for class 0 and 1.272 for class 1, in standardized 32D input coordinates.
All 1,000 anchors passed membership checks, and no exact split overlaps were found.

This is zero observed held-out coverage for this fixed atlas and test sample,
not a proof of zero population coverage. It also does not imply failure to
generate counterfactuals: projection can reach an atlas region even when the
original query or another independently sampled point lies outside every region.
No parameters were tuned to change this result.

## Initial-ball alpha diagnostic

The separate diagnostic sweeps alpha from 0.05 to 1.00 in steps of 0.05,
without evaluating the classifier again or running certification. Anchor
locations and the reference set remain fixed. In this saved top-k experiment,
the epsilon strategy uses the 1,000 selected anchors as its reference set, not
the entire 10,000-point training pool. Recomputing nearest-opposite L1
clearances from this set reproduces the stored initial radii at alpha=0.2.

For each held-out point predicted as class t, its first-entry scale is
`min_i distance_L1(x, anchor_i) / clearance_i` over class-t anchors. This
need not be the nearest anchor: a farther anchor may have a larger clearance.
The output records both exact first-entry scales and scales allowing the
configured absolute membership tolerance of `1e-6`.

```bash
MPLCONFIGDIR=/tmp/certcf-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/atlas_initial_ball_sweep.py \
  --config configs/experiments/atlas_initial_ball_sweep_synthetic32.yaml
```

The outputs are in `results/atlas_coverage/synthetic32/initial_ball_alpha_sweep/`:
`initial_ball_coverage.csv`, `entry_alpha_by_point.csv`, `distances.npz`,
`metadata.json`, and the PNG/PDF `initial_ball_coverage` plots. The script
validates the saved coverage cache against the source artifacts and reuses its
predictions. It does not overwrite the original coverage results or the atlas.

| Alpha | Class 0 initial-ball coverage | Class 1 initial-ball coverage |
| ---: | ---: | ---: |
| 0.20 | 0.0% | 0.0% |
| 0.50 | 0.0% | 0.0% |
| 0.60 | 0.0% | 0.8% |
| 0.70 | 9.0% | 15.6% |
| 0.80 | 49.4% | 52.2% |
| 0.90 | 85.6% | 84.8% |
| 1.00 | 98.2% | 97.8% |

Each class has 500 held-out points. The first point enters at approximately
alpha=0.60636 for class 0 and alpha=0.54749 for class 1; median first-entry
scales are approximately 0.80112 and 0.79397 respectively. These thresholds
ignore the negligible membership tolerance. Even the initial balls at alpha=0.2
cover no held-out points, independently of any subsequent shrinkage.

These are **uncertified geometric coverage** values, not results for newly
constructed certified atlases. Within the same chosen initial balls, subsequent
shrinking and clipping can only reduce coverage. This experiment does not
establish how much coverage a certified atlas at a larger alpha would retain.

## Certified alpha comparison

The follow-up builds new certified atlases for alpha=0.8 and alpha=1.0 with
the same saved 1,000 anchors, classifier, reference set, margin, norm, LiRPA
backend and adaptive-shrinkage settings. The original alpha=0.2 atlas is loaded,
not rebuilt. All variants are saved separately from the original source.

```bash
MPLCONFIGDIR=/tmp/certcf-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/atlas_certified_alpha_sweep.py \
  --config configs/experiments/atlas_certified_alpha_sweep_synthetic32.yaml
```

Results are under `results/atlas_coverage/synthetic32/certified_alpha_sweep/`.
`coverage_stages.csv` and the matching PNG/PDF compare coverage of the initial
balls, final balls and certified regions. Each `alpha_*` directory contains
the corresponding serialized atlas, membership matrices and a completion
record with input/output hashes. A repeat invocation reuses completed variants
only after validating those hashes. Source-atlas hashes are also checked after
the experiment to ensure the original files were not modified.

### Results on 9 November 2026

| Alpha | Class | Initial coverage | Final-ball coverage | Certified coverage | Median final radius | Median halvings |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 0 | 0.0% | 0.0% | 0.0% | 1.364 | 2 |
| 0.2 | 1 | 0.0% | 0.0% | 0.0% | 1.272 | 2 |
| 0.8 | 0 | 49.4% | 0.0% | 0.0% | 1.364 | 4 |
| 0.8 | 1 | 52.2% | 0.0% | 0.0% | 1.272 | 4 |
| 1.0 | 0 | 98.2% | 0.0% | 0.0% | 1.529 | 4 |
| 1.0 | 1 | 97.8% | 0.0% | 0.0% | 1.520 | 4 |

All variants retain all 500 anchors per class, and every retained region passes
both vectorized and scalar anchor-membership checks. Every anchor undergoes
shrinkage. All held-out coverage initially available at the larger alphas is
lost upon shrinking the enclosing balls, before the final affine constraints
are applied in the coverage calculation.

For alpha=0.8, all final radii are exactly equal to their alpha=0.2 counterparts,
not just equal in median, and each anchor undergoes exactly two extra halvings.
The initial radius is four times larger, so halving twice returns to the same
radius sequence. Alpha=1.0 changes that sequence, but the resulting final balls
still contain no held-out points of their corresponding predicted class.

This establishes a limitation of the tested construction and shrinkage policy
on this fixed 32D problem, not the impossibility of larger sound regions. The
experiment does not distinguish intrinsic decision-boundary geometry from
LiRPA conservatism or quantify the best achievable certified coverage.

## Relaxing anchor inclusion on continuous inputs

The isolated `atlas_nonempty_policy` experiment replaces anchor inclusion with
nonemptiness as the retention criterion: recompute LiRPA after halving only
when the current certified region is empty, and retain the first nonempty
region. All other settings and the original model/reference/anchor arrays
remain fixed. The standard CertCF implementation is unchanged.

For this unconstrained binary L1 problem there is exactly one affine target
constraint per anchor. A region is nonempty precisely when
`a @ center + b - gamma + eps * max(abs(a)) >= 0`. The maximizer is available
explicitly at an L1-ball vertex; if the anchor already satisfies the constraint,
it serves as the witness instead. No surrogate bound or anchor test replaces
this feasibility condition. Numerical witnesses are independently checked with
production membership and the classifier. The condition establishes existence,
not positive region volume. Multiclass, categorical and additional input-bound
constraints are explicitly outside this experiment's supported scope.

```bash
MPLCONFIGDIR=/tmp/certcf-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/atlas_nonempty_policy.py \
  --config configs/experiments/atlas_nonempty_synthetic32.yaml
```

Outputs reside in `results/atlas_coverage/synthetic32/nonempty_policy/`:
`nonempty_coverage.csv`, `policy_comparison.csv`, and per-alpha atlas,
membership, witness, attempted-radius trace and provenance files. Standard
atlases are not overwritten. The altered anchor invariant has not been audited
for use by the online search/decoding code; these artifacts are for the offline
coverage experiment, not a replacement of the production atlas.

### Results on 9 November 2026

| Alpha | Class | Standard median radius | Nonempty-policy median radius | Retained regions excluding anchor | Certified held-out coverage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 0 | 1.364 | 2.375 | 181 / 500 | 0% |
| 0.2 | 1 | 1.272 | 2.284 | 222 / 500 | 0% |
| 0.8 | 0 | 1.364 | 2.375 | 181 / 500 | 0% |
| 0.8 | 1 | 1.272 | 2.284 | 222 / 500 | 0% |
| 1.0 | 0 | 1.529 | 1.668 | 177 / 500 | 0% |
| 1.0 | 1 | 1.520 | 1.581 | 136 / 500 | 0% |

Every variant retains all 1,000 regions and all witnesses are target-valid.
All 1,000 initial certified polytopes are empty for each tested alpha, even
without the anchor-inclusion requirement. At alpha=0.2, 700 become nonempty
after one halving, 999 cumulatively after two, and all 1,000 after three.
Alpha=0.8 follows the same pattern after two additional halvings; alpha=1.0
accepts 282 after three, 989 cumulatively after four, and 1,000 after five.
No held-out points enter even the final enclosing balls under the relaxed
policy, so certified coverage is still zero.

Relaxing anchor inclusion thus changes the accepted domains and retains
hundreds of regions that the standard criterion would shrink further, but
does not fix held-out coverage on this configuration. Larger enclosing radii
do not by themselves establish larger certified volumes: the halfspaces are
recomputed and may carve out regions away from the anchor. Initial emptiness
describes the current affine under-approximations, not the true target-class
preimage, which still contains target-valid points.

## PREMAP global-box pilot

On 9 November 2026, the optional original-PREMAP adapter was run on the same
32D dataset and saved depth-3/width-128 classifier (37,506 parameters; test
accuracy 98%). There was no retraining, dataset regeneration, or modification
of the existing atlas. Its 1,000 test points are again grouped by model
prediction, with 500 in each class. The logit margin remains `1e-4`.

This is a **different domain policy**, not a controlled backend-only replacement
of CertCF: one global axis-aligned box is defined by coordinate minima and maxima
over all 10,000 training points and reused for each target class. No test inputs
are used to choose or expand it. The box contains 498/500 held-out points in
each class (99.6%). There is no alpha or nearest-opposite epsilon rule here,
no adaptive shrinkage, and no anchor-inclusion requirement.

Configuration: `configs/experiments/atlas_premap_synthetic32.yaml`. Each class
receives a separate PREMAP request, with either 1 leaf (no splitting) or a cap
of 32 leaves, 1,000 samples for the coverage heuristic, target heuristic coverage
0.9, seed 42, and a 120-second timeout. The unmodified upstream volume-surrogate
sampler also uses 1,000 points internally with its own hard-coded seed. CPU
execution uses one OpenMP/MKL/OpenBLAS thread. Timing includes worker startup.

| Leaf budget | Class | Build time (s) | Returned leaves | Nonempty certified regions | Held-out coverage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0.67 | 1 | 0 | 0 / 500 |
| 1 | 1 | 0.72 | 1 | 0 | 0 / 500 |
| 32 | 0 | 37.92 | 32 | 0 | 0 / 500 |
| 32 | 1 | 37.73 | 32 | 0 | 0 / 500 |

All four runs stopped at the leaf budget, not the wall-clock timeout. Both
tolerance-`1e-6` membership and strict zero-tolerance membership give zero hits.
The internal uniform-sampling coverage is also zero. Ten fixed randomly selected
opposite-class queries per target were projected onto each atlas; all failed,
as expected for empty certified unions. This is an integration diagnostic, not
a full counterfactual benchmark. Unlike the original CertCF atlas, which has
nonempty regions but zero held-out coverage, these global-box PREMAP outputs
contain no feasible region at all under the tested budgets.

Emptiness was checked by linear programming and independently by maximizing
each affine lower-bound margin over its box. With one binary margin constraint
per leaf, its maximum is `a_positive @ upper + a_negative @ lower + b`. After
32 leaves, even the largest maximum across leaves is approximately -461.65 for
class 0 and -356.60 for class 1; it must be nonnegative to retain any point.
The float32 sigmoid of both values is exactly zero. The optimizer logs report
no bound improvement and longest-edge splitting fallback. This is consistent
with very loose relaxations and numerical saturation of the volume surrogate
on these broad boxes; it is not a proof that PREMAP cannot succeed with a
different initialization, budget, or optimization setup. No optimizer settings
were silently changed to avoid the negative result.

As a separate positive control, the first existing anchor of each class was
placed in a small box with coordinate halfwidth 0.05. PREMAP returned a nonempty
certified region containing that anchor for both classes, in approximately
0.67 seconds each. Independent witness evaluations gave target margins 4.32 and
10.88. These tiny boxes cover no held-out points; they check that the integration
can certify regions for this exact 32D model, not that generalization improved.

Outputs in `results/atlas_coverage/synthetic32/premap_global_box/`:

- `metadata.json`: source/config/code hashes, domain, model and baseline details.
- `coverage.csv`: all four cases, budgets, timings, feasibility and coverage.
- `regions_*/class_*/`: numeric atlas, membership matrices, LP witnesses,
  query outcomes, worker-log excerpt and hashed completion record.
- `diagnostics.json`: exact affine maxima, float32 saturation checks and controls.
- `positive_control/class_*/`: the two local-control atlases.

Use `scripts/atlas_premap.py` for the pilot and add `--diagnostics` for the
separate checks; both accept `--premap-python` and `--premap-repository`.
The main pilot reuses completed cases only when input/config/code and artifact
hashes match. Diagnostics are small independent reruns. Neither run updates
the original dataset, classifier or atlas. These are exploratory results on
one trained model and seed, not evidence of universal PREMAP performance.

## Matched local-box counterfactual metrics

`scripts/premap_metrics.py` evaluates counterfactual quality, not just coverage.
The pilot selects 20 anchors per predicted class uniformly without replacement
from the saved 500-per-class support, with seed 42. All three variants use the
same 40 source anchors and original trained model:

1. `certcf_l1`: the original certified L1 regions for those anchors, reused
   exactly, with their original adaptive shrinkage.
2. `box_crown`: explicit local boxes, certified with plain backward CROWN in
   the external PREMAP environment; no slope optimization or splitting.
3. `box_premap`: the same initial boxes with PREMAP volume-oriented optimization
   and up to four leaves per anchor; no shrinkage or anchor-inclusion condition.

The initial box halfwidth is alpha times nearest-opposite-anchor **Linf** distance,
computed using all original 1,000 anchors. This is different from the original
L1 clearance rule. Endpoints are explicitly rounded to float32 and recorded by
the backend. The initial alpha is 0.2; alpha=0.1 is a separate follow-up based
on the first pilot's offline feasibility, not selection on test-quality metrics.
Each class/backend receives a 180-second worker timeout. If it expires before
all 20 anchors are processed, the runner refuses to compare unequal processed
anchor sets. Empty leaves are identified using a linear program and omitted
from projection, without changing any surviving certificate.

For each query, all methods select the same five nearest *source anchors* in
L1 distance. Every retained leaf for those anchors is projected, and the same
selection score chooses the result. If no candidate is found, the remaining
source anchors are examined one at a time. This avoids changing the effective
number of source anchors when one backend subdivides a box into multiple leaves.
Online projection uses L1 and two reweighted-L1 refinement iterations, lambda=1,
eta=0.001, and the paper's CLARABEL settings. Erosion is disabled. Timing is
serial and includes grouping/subset setup, but excludes model validation and
metric computation; it is not directly comparable to the paper's parallel timing.

If each variant has usable regions for both classes, a 40-query pilot is followed
by all 1,000 test queries, using the same 40-anchor atlases throughout. This is
not yet a replication of the original 1,000-anchor configuration. Results are
reported both on each method's successful queries and on the exact intersection
of queries solved by all three methods; validity always retains all attempted
queries in its denominator.

Metric definitions are adapted from `notebooks/DatasetMetrics.ipynb`:

- L1 distance and count of changed coordinates above `1e-6`.
- Mean novelty-mode `log10(LOF)` with 20 neighbors fitted on all training inputs.
- Empirical target retention under Gaussian sigma in `[0, .01, .03, .05, .1]`,
  with ten perturbations per cell. Since there are no categorical features, the
  paper's five categorical-flip grid entries are five noise replicates. Common
  query-specific noise is used across methods, unlike the notebook's independent
  method seeds; this makes the comparison paired without changing noise laws.
- Depth-3 decision-tree query-to-CF regression, minimum leaf size 5, fixed 70/30
  split, seed 0; test L1 error normalized by the mean query-to-target nearest
  training-reference L1 distance over all 1,000 queries. A tree score is omitted
  if fewer than 20 successful examples are available.
- RPR: nearest same-class training distance normalized by mean leave-one-out
  same-class training distance, using training labels as in the paper notebook.

Outputs include `build_summary.json`, combined atlases, hashed class-build
checkpoints, one JSON per query, `metrics_per_query.parquet`, and
`metrics_summary.csv`/`.json`. Source artifacts are never overwritten.

### Local-box feasibility checks

The first run, retained in `results/premap_metrics/synthetic32_local_pilot/`,
used alpha=0.2. Plain CROWN returned 40 empty leaves (1.66 seconds total), and
PREMAP returned 160 empty leaves (294.14 seconds total). Neither class timed
out. The runner stopped before query evaluation because neither box variant
had a usable certified region. These are empty inner approximations, not a
claim that the underlying boxes contain no target-class inputs.

The separate alpha=0.1 follow-up is stored in
`results/premap_metrics/synthetic32_local_alpha010/`. Plain CROWN retains 34
nonempty regions from 34 of the 40 anchors, taking 1.65 seconds. PREMAP retains
79 nonempty leaves from 35 anchors (99 leaves before feasibility filtering),
taking 106.37 seconds. The original L1 baseline retains all 40 selected regions;
its offline time is not remeasured because these regions are reused. All three
unions still cover zero of the 1,000 held-out inputs. Nevertheless, all variants
successfully answer the initial 40 counterfactual queries. Containment of held-out
inputs and the ability to project a query onto a nonempty target region are
different properties; zero observed containment does not imply zero query success.

### Completed 1,000-query comparison (alpha=0.1 boxes)

All 1,000 queries succeed for every method, so the all-success and shared-success
populations coincide. Every returned candidate passes an independent classifier
check and membership check against its method's certified atlas. No source
dataset, trained model, or original atlas artifact was changed.

| Metric | Original CertCF L1 | Box CROWN control | Box PREMAP |
| --- | ---: | ---: | ---: |
| Validity (%) | 100.00 | 100.00 | 100.00 |
| Mean L1 proximity (lower better) | 27.6162 | 25.1884 | 24.8144 |
| Mean changed coordinates (lower better) | 22.457 | 26.679 | 26.450 |
| Empirical robustness (%) | 100.00 | 100.00 | 100.00 |
| Mean log10 LOF (lower better) | 0.003588 | -0.000208 | -0.000171 |
| Interpretability proxy (lower better) | 0.88789 | 0.83916 | 0.84734 |
| RPR (reference scale at 1) | 0.11075 | 0.30407 | 0.30349 |
| Serial query time (s) | 0.22320 | 0.18313 | 0.43179 |
| Nonempty certified regions | 40 | 34 | 79 |
| Offline build time (s) | Not remeasured | 1.652 | 106.372 |
| Held-out containment (%) | 0.00 | 0.00 | 0.00 |

PREMAP improves mean L1 proximity by about 10.1% over the original L1-region
baseline, but changes about four more coordinates and takes about 1.93 times
as long online. Comparing **the same initial boxes**, PREMAP improves mean L1
proximity by only about 1.5% over plain CROWN and reduces changed coordinates
from 26.679 to 26.450, while taking about 2.36 times as long online and 64 times
as long offline. The largest observed proximity change therefore occurs when
switching the original L1 regions to this local-box construction; it cannot be
attributed to PREMAP refinement alone. This comparison also changes shrinkage
policy and the bound-library version relative to the original baseline.

The LOF values are all close to zero on the logarithmic scale (negative values
are possible when novelty LOF is below one). They do not establish a meaningful
plausibility advantage of PREMAP over box CROWN. The tree proxy is slightly
worse for PREMAP than box CROWN. Empirical robustness is saturated under this
perturbation grid; these observations are not an erosion-based radius certificate
or evidence of a robustness improvement. RPR increases relative to the original
baseline but remains below the reference scale of one for both box methods.

This is a one-seed, one-model pilot with only 20 anchors per class, not a result
over the seven tabular datasets or the original 1,000-anchor atlas. No statistical
significance or general superiority claim is made. The pilot supports keeping
the modular backend for further study, but not replacing the default backend.
The next controlled extensions would vary the region budget and anchor count
while retaining the matched-box CROWN control and separating build/query costs.

## Wisconsin Breast Cancer: real-tabular backend pilot

`configs/experiments/premap_metrics_wbc.yaml` runs the same numerical-only
comparison on the saved Wisconsin Breast Cancer benchmark artifacts. The source
is `configs/experiments/lirpa_refinement_ablation_tabular.yaml`, with 457 training
inputs, 30 standardized numerical features, the existing 64/32-hidden-unit ReLU
classifier, and all 56 prepared test queries. The preparation manifest, model,
raw data and saved atlas are validated before use; loaded train/test rows are
also checked against the saved support/query arrays. Nothing is retrained.

The pilot selects 20 anchors per predicted class (seed 42) from the existing
457-anchor atlas. All three variants use those same 40 source anchors, top-five
anchor-group retrieval, L1 projection, and the same sparsity settings. The
box halfwidth uses alpha=0.1 times nearest-opposite-anchor Linf distance over
all 457 saved anchors. PREMAP uses the same four-leaf budget, 1,000 samples,
0.9 sampling coverage threshold, and 180-second per-class timeout as the
synthetic follow-up. There was no parameter search on Wisconsin test metrics.
The source L1 regions retain their original alpha=0.2 initialization and
adaptive shrinkage. Certification margin is 1e-4 and erosion is disabled.

| Metric | Original CertCF L1 | Box CROWN control | Box PREMAP |
| --- | ---: | ---: | ---: |
| Validity (%) | 100.00 | 100.00 | 100.00 |
| Mean L1 proximity (lower better) | 19.81233 | 19.13961 | 19.13488 |
| Mean changed coordinates (lower better) | 19.50000 | 26.46429 | 26.46429 |
| Empirical robustness (%) | 99.76429 | 99.34286 | 99.26429 |
| Mean log10 LOF (lower better) | 0.030450 | 0.018348 | 0.018407 |
| Interpretability proxy (lower better) | 0.538409 | 0.455748 | 0.456071 |
| RPR (reference scale at 1) | 0.222685 | 0.271366 | 0.271369 |
| Serial query time (s) | 0.206405 | 0.201076 | 0.313987 |
| Nonempty certified regions | 40 | 40 | 46 |
| Offline build time (s) | Not remeasured | 1.56324 | 4.32071 |
| Held-out containment (%) | 0.00 | 0.00 | 0.00 |

All 56 queries succeed in all variants, so the shared-success comparison is
identical. All returned points pass independent target-class and atlas-membership
checks. Both box methods retain regions for all 40 anchors, without timeouts or
empty leaves. PREMAP produces 23 leaves per class versus CROWN's 20 per class.
The original source files pass the post-run hash check.

Compared with the original L1 regions, PREMAP reduces mean L1 distance by about
3.4%, but changes about seven more coordinates and has slightly lower empirical
robustness. Against CROWN on identical initial boxes, PREMAP's distance reduction
is only about 0.025%, with identical mean changed-coordinate count; its online
time is about 1.56 times higher and offline time about 2.76 times higher. The
other quality metrics are very similar between the box methods. This pilot
does not support a material quality improvement from PREMAP refinement on this
configuration. The tree proxy uses only 56 points and one split, so small
differences should not be overinterpreted.

This can support an **atlas-construction ablation**, not a claim that PREMAP is
a competing end-to-end counterfactual generator. Report the original L1 method
and the matched-box CROWN control separately: original-versus-PREMAP changes
domain shape, initialization/shrinkage policy, and library version, whereas
box-CROWN-versus-PREMAP isolates volume optimization plus splitting within the
same external bound library. The pilot is one seed and a reduced atlas, not a
replacement for the full main benchmark. Categorical and immutable-feature
datasets are explicitly rejected by this runner pending a matched decoding and
constraint protocol.

Outputs are in `results/premap_metrics/wisconsin_breast_cancer_alpha010/`, with
the same build, per-query, provenance, and metric artifacts as the synthetic run.
The numerical tabular loader is now shared by the experiment runner. Because
code hashes are part of cache identity, rerunning a historical synthetic config
after this extension requires a fresh output directory; its existing results
remain intact and are never silently reused under changed code.

## PREMAP adapter audit and historical-result status (2026-11-09)

All PREMAP results above use the original PREMAP input-splitting implementation,
not PREMAP2. They were produced before sampling protocol `input-coverage-v2`.
The audit found that the initial coverage hook did not reset its RNG as the
original sampler does. This is now corrected; pairwise coverage already reset
its seed. The corrected sampler also matches upstream's argmax tie convention
and half-sample count for very small coverage budgets.

Consequently, the initial estimate, stopping decision, and resulting leaves can
change. The historical numerical tables are retained as records of those runs,
not claimed as reruns of the corrected adapter. Before using this ablation in
the paper, rerun the matched-box CROWN and PREMAP variants in fresh output
directories and regenerate their downstream metrics. Existing code-hash checks
prevent silently resuming old pilot results under the revised implementation.

The original optimizer independently uses 1,000 samples and a fixed seed of 1;
the configurable sample count controls coverage/split estimates only. The
reported pilots used 1,000 coverage samples too, but changing the public seed
does not independently resample that optimization objective. New atlas metadata
records both sampling roles explicitly. See the certification README for the
differential-test scope and the sampling contract.


## Completed corrected-protocol reruns (2026-11-09)

Both pilots were rerun from scratch with code commit `19f49a7` and sampling
protocol `input-coverage-v2`, after the full suite passed (435 passed, 1 skipped,
including real upstream differential tests). This completes the rerun requested
in the audit above. The original pilot directories were not overwritten.

Configurations:

- `configs/experiments/premap_metrics_synthetic32_alpha010_coverage_v2.yaml`
- `configs/experiments/premap_metrics_wbc_coverage_v2.yaml`

All experimental settings match their historical configs: seed 42, 20 anchors
per class, box alpha 0.1, four-leaf budget, 1,000 coverage samples, coverage
threshold 0.9, and identical retrieval/projection/metric options. Only the
sampling-protocol implementation, the explicit `coverage_samples` alias, and
the output directories changed. The external optimizer remains at its original
1,000 samples and fixed seed 1. Runs were sequential, not concurrent, to avoid
cross-run contention in latency measurements.

### Checks against the historical pilots

For both datasets and all three methods, explicit comparisons confirmed:

- Identical source-run identifiers, source hashes, and selected anchor indices.
- Identical final ordered region geometry, including affine coefficients, boxes,
  norm balls where present, and retrieval representatives.
- Identical targets, success flags, and counterfactual coordinates for every
  query (1,000 on synthetic-32 and 56 on Wisconsin per method).
- Identical non-timing aggregate metrics in both all-success and shared-success
  populations; all queries succeeded, so these populations coincide.
- No PREMAP/CROWN build timeout; both classes processed all 20 requested anchors.
- Source model/data/atlas hashes still matched after evaluation.

Thus the sampling correction does **not** change final regions or quality
results for these two configurations. This is an observed result of these runs,
not a claim that initial sampling or seed choices never affect the algorithm.
Wall-clock measurements were repeated and differ slightly, as expected.

### Quality results under the corrected protocol

Validity is 100% for every row. Robustness below is empirical (erosion disabled),
not a positive-radius certified-robustness experiment.

| Dataset | Method | L1 proximity | Changed coordinates | Emp. robustness (%) | log10 LOF | Interp. proxy | RPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthetic-32 | CertCF L1 | 27.616196 | 22.457000 | 100.000000 | 0.003588 | 0.887891 | 0.110748 |
| Synthetic-32 | Box CROWN | 25.188413 | 26.679000 | 100.000000 | -0.000208 | 0.839157 | 0.304069 |
| Synthetic-32 | Box PREMAP | 24.814378 | 26.450000 | 100.000000 | -0.000171 | 0.847342 | 0.303494 |
| Wisconsin | CertCF L1 | 19.812327 | 19.500000 | 99.764286 | 0.030450 | 0.538409 | 0.222685 |
| Wisconsin | Box CROWN | 19.139611 | 26.464286 | 99.342857 | 0.018348 | 0.455748 | 0.271366 |
| Wisconsin | Box PREMAP | 19.134876 | 26.464286 | 99.264286 | 0.018407 | 0.456071 | 0.271369 |

### Remeasured computational costs and coverage

| Dataset | Method | Nonempty regions | Offline build (s) | Mean serial query (s) | Held-out containment (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Synthetic-32 | CertCF L1 | 40 | Not remeasured | 0.224318 | 0.00 |
| Synthetic-32 | Box CROWN | 34 | 1.665452 | 0.183086 | 0.00 |
| Synthetic-32 | Box PREMAP | 79 | 106.251239 | 0.435490 | 0.00 |
| Wisconsin | CertCF L1 | 40 | Not remeasured | 0.205189 | 0.00 |
| Wisconsin | Box CROWN | 40 | 1.552942 | 0.199337 | 0.00 |
| Wisconsin | Box PREMAP | 46 | 4.320760 | 0.310721 | 0.00 |

Original L1 atlases were reused, not retrained or rebuilt. All query measurements
include the same serial group-selection and subset-atlas setup overhead as the
historical pilot; they are not directly comparable to the paper's parallel
benchmark timings. Zero containment was observed for each class separately.

The interpretation is unchanged: against matched-box CROWN, original PREMAP
reduces mean L1 distance by about 1.49% on synthetic-32 and 0.025% on Wisconsin,
with higher build and query costs. Against CertCF's L1 regions, the box methods
improve proximity but change more coordinates. These reduced-atlas, single-seed
pilots do not establish a general ranking or say anything about PREMAP2.

### Outputs and provenance

Fresh output directories:

- `results/premap_metrics/synthetic32_local_alpha010_coverage_v2/`
- `results/premap_metrics/wisconsin_breast_cancer_alpha010_coverage_v2/`

Each contains the full build artifacts, per-query JSON files, metric summaries,
and `validation_against_legacy.json` recording the explicit comparison checks
and old/new run fingerprints. Generated result directories remain local and
git-ignored; this documentation and the runnable configs are versioned.
