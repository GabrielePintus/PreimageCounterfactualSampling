# Source Code

This directory contains the importable Python package code. Executable entrypoints live one level up in `scripts/`.

## Package Map

- `certcf/`: CertCF atlas construction, LiRPA preimage approximation, polytope projection, decoding, and visualization helpers.
- `counterfactuals/`: modular counterfactual benchmark framework, method wrappers, datasets, metrics, preprocessing, and notebook utilities.
- `dataset_specs/`: shared tabular dataset metadata, including feature names, OHE slices, and actionability metadata used by benchmarks.
- `models/`: neural-network architectures used by the classifiers.
- `training/`: Lightning modules and tabular data modules used to train benchmark classifiers.

## Entrypoints

Use these scripts from the repository root:

```bash
python scripts/train_classifier.py fit --config configs/training/adult_classifier.yaml
python scripts/benchmark.py --config configs/benchmarks/final_benchmark.yaml
```

## Import Example

```python
from certcf import CertCFAtlas
from counterfactuals.benchmarks import create_default_registries
```
