# src

This directory contains the Python source code for training models, building certified preimage approximations, and running modular counterfactual experiments.

## Main packages
- `training/`: Lightning modules and data modules.
- `models/`: neural network architectures.
- `preimage_sampling/`: certified atlas and geometric counterfactual tooling.
- `counterfactuals/`: modular benchmarking framework for multiple CF methods.

## Example
```python
from preimage_sampling import CertifiedAtlas
from counterfactuals.experiments.runner import run_from_config_path
```
