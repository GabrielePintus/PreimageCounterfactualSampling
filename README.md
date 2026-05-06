# CertCF: Certified Counterfactuals for Neural Networks

This repository contains the research code for **CertCF**, a counterfactual explanation method that connects counterfactual generation with neural network verification.

CertCF builds certified polytopic under-approximations of a target class preimage using LiRPA/CROWN bounds. At query time, counterfactual search becomes projection onto certified regions rather than an unconstrained pointwise search. Returned counterfactuals are accepted only when they satisfy the certified target-class constraints, giving validity by construction.

The current paper experiments focus on binary tabular datasets with one-hot encoded categorical features, actionability constraints, adaptive epsilon shrinkage, and a group-aware sparsity objective.

## Method At A Glance

```text
trained classifier + support data
        |
        v
LiRPA/CROWN bounds around target-class support points
        |
        v
certified atlas: union of target-class certified polytopes
        |
        v
nearest-anchor projection with tabular/actionability constraints
        |
        v
certified counterfactual
```

CertCF supports:

- certified target-class validity through LiRPA preimage under-approximations;
- L1/L2/Linf atlas and distance norms, with the paper benchmark using L1;
- one-hot simplex constraints and post-projection categorical decoding;
- immutable and directional feature constraints for actionability;
- adaptive epsilon shrinkage when a local certificate is too loose;
- optional group-aware reweighted-L1 sparsity selection;
- optional certified robustness through polytope erosion.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/certcf/` | Core certified atlas, LiRPA integration, geometry, projection, and query logic. |
| `src/counterfactuals/` | Modular benchmark framework and method wrappers for CertCF, NN, Growing Spheres, DiCE, and FACE. |
| `src/training/` and `src/models/` | Lightning training modules and tabular classifier architectures. |
| `src/dataset_specs/` | Tabular feature metadata, one-hot slices, and feature groups. |
| `configs/training/` | Classifier training configs for the seven tabular datasets. |
| `configs/benchmarks/final_benchmark.yaml` | Final paper benchmark configuration. |
| `scripts/` | Executable training and benchmark entrypoints. |
| `notebooks/Results.ipynb` | Main paper result tables and plots. |
| `notebooks/Appendix.ipynb` | Appendix tables, ablations, and diagnostics. |
| `tests/counterfactuals/` | Unit tests for the benchmark framework and CertCF integration. |

For more detailed module-level notes, see [src/README.md](src/README.md), [configs/README.md](configs/README.md), [scripts/README.md](scripts/README.md), and [notebooks/README.md](notebooks/README.md).

## Installation

```bash
git clone https://github.com/gabrielepintus/PreimageCounterfactualSampling.git
cd PreimageCounterfactualSampling
pip install -e .
```

For notebooks and development tools:

```bash
pip install -e .[dev]
```

The project depends on PyTorch, auto_LiRPA, CVXPY/CLARABEL, Lightning, scikit-learn, pandas, NumPy, SciPy, and related scientific Python packages declared in [setup.py](setup.py).

Generated data, checkpoints, results, and notebook outputs are intentionally kept out of git. The benchmark configs expect trained classifier checkpoints under `checkpoints/<dataset>_classifier/best.ckpt`.

## Train Classifiers

Training uses LightningCLI configs under `configs/training/`.

```bash
# Train one tabular classifier
python scripts/train_classifier.py fit --config configs/training/compas_classifier.yaml

# Quick smoke run with fewer epochs
python scripts/train_classifier.py fit \
  --config configs/training/compas_classifier.yaml \
  --trainer.max_epochs=2

# Test a trained checkpoint
python scripts/train_classifier.py test \
  --config configs/training/compas_classifier.yaml \
  --ckpt_path checkpoints/compas_classifier/best.ckpt
```

Available training configs cover:

- Adult
- COMPAS
- German Credit
- Give Me Some Credit
- HELOC
- Lending Club
- Wisconsin Breast Cancer

See [configs/training/README.md](configs/training/README.md) for details.

## Run The Final Benchmark

The final benchmark evaluates CertCF against Nearest Neighbor, Growing Spheres, DiCE, and FACE on the seven tabular datasets.

```bash
python scripts/benchmark.py --config configs/benchmarks/final_benchmark.yaml
```

Useful variants:

```bash
# Run only selected datasets
python scripts/benchmark.py \
  --config configs/benchmarks/final_benchmark.yaml \
  --datasets compas heloc

# Run only selected methods
python scripts/benchmark.py \
  --config configs/benchmarks/final_benchmark.yaml \
  --methods certcf face

# Resume an interrupted benchmark if the parquet already exists
python scripts/benchmark.py --config configs/benchmarks/final_benchmark.yaml

# Force recomputation from scratch
python scripts/benchmark.py \
  --config configs/benchmarks/final_benchmark.yaml \
  --force
```

The configured output is:

```text
results/final_benchmark.parquet
```

For multi-dataset runs, the benchmark also writes one per-dataset parquet next to the combined output. See [configs/benchmarks/README.md](configs/benchmarks/README.md) for method settings, result semantics, resume behavior, and FACE graph-radius values.

## Paper Analysis Notebooks

The cleaned repository keeps two paper-facing notebooks:

| Notebook | Purpose |
| --- | --- |
| `notebooks/Results.ipynb` | Loads benchmark parquet files and computes the main result plots/tables. |
| `notebooks/Appendix.ipynb` | Collects appendix hyperparameters, robustness heatmaps, computational cost tables, CertCF ablations, and LiRPA backend diagnostics. |

Run notebooks from either the repository root or the `notebooks/` directory. They expect local benchmark outputs under `results/`; expensive recomputations are disabled by explicit flags.

## Final Benchmark Configuration

The paper benchmark uses:

| Method | Main settings |
| --- | --- |
| CertCF | L1 distance, CROWN/backward LiRPA, `eps_alpha=0.20`, adaptive shrinkage, nearest-anchor query with 5 anchors, group-aware reweighted-L1 sparsity with `sparsity_lambda=1.0`. |
| Nearest Neighbor | L1 distance to target-class training points. |
| Growing Spheres | L1 distance, `max_radius=50.0`, `radius_step=0.25`, `n_in_layer=1000`. |
| DiCE | Single counterfactual, no diversity term, proximity weight 1, batch size 1. |
| FACE | L2 graph distance, kNN density estimator, `tp=0.5`, dataset-specific graph radius. |

Each dataset is capped at up to `10000` training points per class and up to `1000` benchmark queries.

## Tests

```bash
pytest tests/counterfactuals -q
```

The tests cover dataset specs, metrics, method wrappers, CertCF projection/configuration behavior, and benchmark registry wiring.

## Citation

If you use this code, please cite the repository and the paper once available.

```bibtex
@software{certcf2026,
  author = {Anonymous},
  title = {Certified Counterfactuals for Neural Networks},
  year = {2026},
}
```

## Acknowledgments

CertCF builds on:

- [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) for neural network bound propagation;
- [CVXPY](https://www.cvxpy.org/) and [CLARABEL](https://github.com/oxfordcontrol/Clarabel.jl) for convex projection problems;
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for classifier training;
- [DiCE](https://github.com/interpretml/DiCE), FACE, Growing Spheres, and nearest-neighbor baselines for comparative evaluation.
