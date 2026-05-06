# training

Lightning modules and data modules used to train benchmark classifiers.

## Contents

- `lit_classifier.py`: Lightning module for classifier training.
- `datamodules/`: dataset-specific data modules for the tabular benchmark datasets.
- `lit_autoencoder.py`: legacy autoencoder Lightning module retained for older experiments.

Active training configs live in `configs/training/` and currently target the tabular classifiers used by `configs/benchmarks/final_benchmark.yaml`.

## Example

```python
from training.lit_classifier import LitClassifier
```
