# Notebooks

Interactive notebooks for training evaluation and counterfactual generation demos.

## Naming Convention

- `1.x`: model evaluation notebooks (MNIST classifier/autoencoder).
- `2`: preimage approximation and sampling walkthrough.
- `3.x`: spiral/MNIST sampling experiments.
- `6.x`: benchmark analysis notebooks.

## Usage Notes

- Run cells top-to-bottom in a fresh kernel.
- Some notebooks expect checkpoints under `../checkpoints/`.
- For k-medoids cells, install `scikit-learn-extra`.

## Benchmark Analysis

- `6.2 - Multi-dataset benchmark analysis.ipynb`: multi-dataset tabular benchmark analysis on the shared-success subset.
- `6.3 - MNIST benchmark analysis.ipynb`: MNIST-specific benchmark analysis on the shared-success subset.
- `6.4 - CertCF atlas subsampling diagnostics.ipynb`: manual diagnostics notebook for choosing `k_per_class` before larger CertCF benchmark runs.
- `6.5 - CertCF atlas anchor selection diagnostics.ipynb`: manual diagnostics notebook for comparing atlas anchor-selection rules at fixed budget.
- `6.6 - CertCF adult atlas visualization.ipynb`: Adult-specific 2D visualization notebook for the `k`-medoids atlas, including PCA/t-SNE projections and sampled continuous certified polytope points.
- `6.7 - CertCF latent anchor selection diagnostics.ipynb`: Adult-only notebook comparing input-space and penultimate-space `k`-medoids anchor selection under the same CertCF atlas/query pipeline.
- `6.8 - CertCF density-bias evaluation diagnostics.ipynb`: Adult-only notebook comparing input/latent/density-stratified anchor selectors and weighted coverage diagnostics to probe density bias in CertCF atlas evaluation.
- `6.9 - CertCF proximity metric diagnostics.ipynb`: Adult-only notebook isolating the effect of density-aware proximity summaries while keeping the atlas fixed to input-space `k`-medoids.
- `6.10 - CertCF input vs latent benchmark analysis.ipynb`: benchmark-results notebook for the latest input-vs-latent CertCF comparison, with dataset-level summaries, delta tables, and per-query scatter views from the combined parquet.
- `6.11 - CertCF boundary-biased latent anchor diagnostics.ipynb`: Adult-only exploratory notebook comparing input `k`-medoids, latent `k`-medoids, and a notebook-only boundary-weighted latent `k`-medoids heuristic before codebase integration.

## Benchmark Notebook Helpers

- Reusable benchmark-analysis helpers now live in `notebooks/utils/`.
- Keep benchmark notebooks thin: configure paths/options, load results through helpers, then compose summaries and plots.
- Benchmark notebooks now filter to the shared-success task set by default, so all comparisons use the same per-dataset query subset across methods.
- Use the retention summary at the top of each notebook to see how many tasks remain after the fair shared-success restriction.
- `6.2` also includes diagnostic OHE constraint-quality checks and immutable-feature change checks where dataset metadata is available; treat these as debugging/ablation views rather than headline benchmark metrics.
- Add shared benchmark plots or dataframe wrangling to `notebooks/utils/` before duplicating code across notebooks.
