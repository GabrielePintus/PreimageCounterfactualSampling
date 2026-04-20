# datasets

Dataset adapters used by the benchmark runner.

## Contents
- `base_dataset.py`: in-memory dataset abstractions.
- `loaders.py`: concrete loaders (for example, Adult and Wisconsin Breast Cancer adapters).

## Example
```python
from counterfactuals.datasets.loaders import AdultDataset

dataset = AdultDataset(data_dir="data/")
dataset.load()
x_train, y_train = dataset.get_train()
```
