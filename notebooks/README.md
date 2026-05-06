# Notebooks

Interactive notebooks for training evaluation, benchmark analysis, and CertCF diagnostics.

## Current Layout

- `notebooks/`: active notebooks only
- `notebooks/utils/`: shared plotting and dataframe helpers for benchmark analysis

## Active Notebooks

- `6.12 - CertCF boundary-biased latent anchor diagnostics.ipynb`
  Prediction-aligned boundary-biased anchor notebook used during the latent-anchor investigation.
- `6.13 - CertCF proximity distribution histograms.ipynb`
  Distribution-focused proximity analysis from saved benchmark results.
- `6.14 - CertCF random anchor k-sweep analysis.ipynb`
  Analysis notebook for the random/boundary-random anchor budget sweeps.
- `6.15 - Method comparison KDE analysis.ipynb`
  Main current comparison notebook for CertCF, NN, GS, and DiCE, including KDE plots, shared-success summaries, and manifoldness diagnostics.
- `6.16 - CertCF 2D Gaussian mixture coverage diagnostics.ipynb`
  Visual diagnostic notebook built around a hard 2D Gaussian mixture so we can inspect decision regions, atlas coverage, and query behavior directly.
- `6.17 - CertCF 2D Adult PCA coverage diagnostics.ipynb`
  Visual diagnostic notebook that projects Adult into a 2D PCA plane so we can inspect CertCF coverage on a real tabular dataset in a controlled setting.
- `6.18 - CertCF Adult validity and proximity diagnostics.ipynb`
  Lightweight Adult notebook that runs CertCF in the normal preprocessed tabular space and prints aggregate validity and proximity metrics across multiple `eps_alpha` values.
- `6.19 - CertCF merged polytope certification diagnostics.ipynb`
  2D diagnostic notebook that wraps nearby certified polytopes into a convex hull, runs a fresh LiRPA certification pass on the wrapper, and compares area/query-distance proxies against the original local union.
- `6.20 - CertCF Adult box merge compression diagnostics.ipynb`
  Adult full-space diagnostic notebook that replaces small local groups of anchor regions with freshly certified axis-aligned boxes and reports atlas compression plus a nearest-target-box query proxy.
- `6.21 - CertCF Adult simplex merge compression diagnostics.ipynb`
  Adult full-space diagnostic notebook that replaces small local groups of anchor regions with freshly certified convex hulls of anchor centers and reports whether high-dimensional convex-polytope compression becomes certifiable.
- `6.22 - CertCF Adult Minkowski capsule diagnostics.ipynb`
  Adult full-space diagnostic notebook that tests full-dimensional certified merging via `Conv(anchors) + B_1(eps)`, using exact first-layer capsule bounds and interval propagation through the rest of the MLP.
- `6.23 - CertCF Adult full-region hull merge diagnostics.ipynb`
  Adult full-space diagnostic notebook that tests true region-containing merges by building the exact lifted convex hull of small groups of original certified `L1` regions and certifying the merged set with LiRPA affine bounds plus a final LP.
- `6.24 - FACE hyperparameter heuristic diagnostics.ipynb`
  Tabular benchmark notebook that loads the datasets and checkpoints used by `benchmark_full.yaml`, computes FACE graph and candidate-pool diagnostics, and proposes a per-dataset heuristic for choosing `epsilon` and `tp`.
- `6.25 - Benchmark result tables and manifoldness plots.ipynb`
  Final benchmark analysis notebook scaffold that loads the main L1 benchmark and separate FACE result files into one combined dataframe for validity-distance plots and manifoldness tables.
- `6.26 - Counterfactual surrogate probe diagnostics.ipynb`
  Auxiliary benchmark notebook that trains simple surrogates on generated counterfactuals and evaluates how well they recover real test-label structure and original-model decision structure across datasets and methods.
- `6.28 - CertCF single-dataset probe.ipynb`
  Lightweight CertCF notebook that runs one non-Adult dataset through the benchmark path and immediately inspects validity, proximity, query metadata, failures, and feature-level changes.
- `6.30 - CertCF 2D intuition figure.ipynb`
  Minimal synthetic 2D notebook for building a CertCF atlas with alpha 0.15 and producing a clean intuition figure with certified regions and counterfactual moves.

## Usage Notes

- Run cells top-to-bottom in a fresh kernel.
- Some notebooks expect benchmark outputs under `../results/` when launched from the `notebooks/` directory.
- For k-medoids experiments, install `scikit-learn-extra`.

## Benchmark Notebook Helpers

- Reusable benchmark-analysis helpers live in `notebooks/utils/`.
- Keep benchmark notebooks thin: configure paths/options, load results through helpers, then compose summaries and plots.
- Shared-success filtering is useful for fair per-query method comparisons, but keep an eye on retention counts so hard queries do not disappear silently.
- Add shared plots or dataframe wrangling to `notebooks/utils/` before duplicating code across notebooks.
- Treat `method_label` as a normalized display field produced by the helper layer.
- Prefer `run_name`-aware comparisons whenever multiple variants of the same implementation appear in one result file.

## Recommended Notebook Skeleton

```python
from notebooks.utils import (
    load_result,
    prepare_benchmark_df,
    method_comparison_summary,
    plot_proximity_kdes_by_dataset,
    setup_notebook_style,
)

setup_notebook_style()

df = load_result(RESULT_PATH)
comparison_df = prepare_benchmark_df(df, dataset_order=DATASET_ORDER, method_order=METHOD_ORDER)
success_df = prepare_benchmark_df(
    df,
    dataset_order=DATASET_ORDER,
    method_order=METHOD_ORDER,
    success_only_rows=True,
)

display(method_comparison_summary(comparison_df, order=METHOD_ORDER))
plot_proximity_kdes_by_dataset(
    success_df,
    dataset_order=DATASET_ORDER,
    method_order=METHOD_ORDER,
    palette=PALETTE,
)
```

Use inline notebook code only for experiment-specific glue or one-off diagnostics that are not yet reusable.
