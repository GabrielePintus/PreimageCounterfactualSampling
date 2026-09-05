# Notebooks

Paper-facing notebooks for CertCF.

- `NetworkComplexityScaling.ipynb` validates and analyzes the 25-cell ReLU MLP
  scaling grid, including tables, heatmaps, resource scaling, marginal trends,
  and explicit failure reporting.

## Current Layout

- `Results.ipynb`
  Main paper-results notebook. It loads the final benchmark parquet files and computes the comparison tables and plots used for the main experimental results.
- `Appendix.ipynb`
  Compact appendix notebook. It collects the paper appendix tables and plots for hyperparameters, robustness, computational cost, CertCF ablations, and LiRPA backend diagnostics.
- `DatasetMetrics.ipynb`
  Per-dataset appendix metrics notebook. It recomputes the non-aggregated tables requested for the appendix, with one row per dataset and method.
- `MNISTMetrics.ipynb`
  Dedicated analysis for the MNIST LeNet-5 CertCF run. It reports multiclass
  task coverage, image-domain quality, empirical robustness, full-training-set
  plausibility/privacy diagnostics, CertCF timings, and qualitative examples.
- `NormAblation.ipynb`
  Reviewer-facing paired analysis of the L1-vs-L2 CertCF query-objective
  ablation. It reads per-dataset progress safely while the benchmark runs and
  emits final tables only after all seven matched comparisons are complete.
- `VeriXCertCFSynthetic32.ipynb`
  Progress-aware paired comparison of native VERIX witnesses and CertCF
  counterfactuals on the standardized 32D synthetic dataset. It reports
  validity, proximity, sparsity, online runtime, offline atlas construction,
  and per-query CertCF/VERIX distance ratios.
- `VeriXCertCFSynthetic32Grid.ipynb`
  Width-by-depth scaling analysis for the 25 paired synthetic-32 runs. It
  reports VERIX timeouts and witness coverage alongside proximity, sparsity,
  on-manifoldness, empirical and certified robustness, online time, and CertCF
  atlas construction time.
- `CIFARResNetScaling.ipynb`
  Result-only analysis of independent ResNet20/32/56 runs, covering offline
  scaling, LiRPA epsilon contraction, image quality, robustness, online time,
  qualitative examples, and resource-limit reporting.
- `LiRPARefinementAblation.ipynb`
  Paired analysis of Anchor-Ball, Anchor-PGD, and CertCF on identical Eq. (1)
  neighborhoods, including success, proximity, feasibility, manifoldness,
  empirical/certified robustness, runtime, failure modes, and depth trends.
- `utils/`
  Shared plotting and dataframe helpers for benchmark analysis notebooks.

## Usage Notes

- Run active notebooks top-to-bottom from the repository root or from `notebooks/`.
- Active notebooks expect benchmark outputs under `results/`.
- Expensive appendix recomputations are disabled by default. Flip the explicit `RECOMPUTE_*` flags only when you want to rerun those diagnostics.
- Notebook outputs are stripped in git to keep diffs readable. Re-run cells locally to regenerate displays.

## Maintenance

- Keep paper-facing notebooks thin and result-file driven.
- Add reusable plotting or dataframe logic to `notebooks/utils/` instead of duplicating it across notebooks.
- Avoid committing large generated notebook outputs, figures, or benchmark artifacts.
