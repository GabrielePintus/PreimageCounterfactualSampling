# src

This directory contains the Python source code for training models, building certified preimage approximations, and running modular counterfactual benchmarks.

## Main packages
- `training/`: Lightning modules and data modules.
- `models/`: neural network architectures.
- `certcf/`: certified atlas and geometric counterfactual tooling.
- `counterfactuals/`: modular benchmarking framework for multiple CF methods.

## Example
```python
from certcf import CertCFAtlas
from counterfactuals.benchmarks import create_default_registries
```
