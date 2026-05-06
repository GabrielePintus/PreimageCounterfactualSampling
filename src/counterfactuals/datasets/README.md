# datasets

Dataset adapters used by the benchmark runner.

## Contents

- `base_dataset.py`: in-memory dataset abstractions.
- `loaders.py`: concrete loaders for Adult, COMPAS, German Credit, Give Me Some Credit, HELOC, Lending Club, Wisconsin Breast Cancer, and legacy MNIST support.

These adapters expose already-split NumPy arrays to the method-agnostic benchmark stack. Dataset-specific feature metadata is centralized in `dataset_specs/`.

## Example

```python
from counterfactuals.datasets.loaders import AdultDataset

dataset = AdultDataset(data_dir="data/")
dataset.load()
x_train, y_train = dataset.get_train()
```
