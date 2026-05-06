# Notebooks

Paper-facing notebooks for CertCF.

## Current Layout

- `Results.ipynb`
  Main paper-results notebook. It loads the final benchmark parquet files and computes the comparison tables and plots used for the main experimental results.
- `Appendix.ipynb`
  Compact appendix notebook. It collects the paper appendix tables and plots for hyperparameters, robustness, computational cost, CertCF ablations, and LiRPA backend diagnostics.
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
