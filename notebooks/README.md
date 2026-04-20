# Notebooks

Interactive notebooks for training evaluation, benchmark analysis, and CertCF diagnostics.

## Current Layout

- `notebooks/`: active notebooks only
- `archive/notebooks/`: historical notebooks kept for reference
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

## Archived Notebooks

The following notebooks were moved to `archive/notebooks/` to reduce clutter in the active workspace:

- early MNIST / spiral / preimage exploration (`1.x`, `2`, `3.x`)
- older benchmark analysis notebooks (`6.0`, `6.2`–`6.10`)

They are still available in git history and in the archive directory, but they are no longer treated as part of the active research surface.

## Usage Notes

- Run cells top-to-bottom in a fresh kernel.
- Some notebooks expect benchmark outputs under `../results/` when launched from the `notebooks/` directory.
- Some older archived notebooks may expect checkpoints under `../checkpoints/`.
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
