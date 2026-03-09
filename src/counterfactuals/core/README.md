# core

Core contracts used across the modular counterfactual framework.

## Contents
- `interfaces.py`: dataset/model/metric interfaces.
- `base_classes.py`: base method class and standard result objects.
- `registry.py`: plugin registry for methods, models, datasets, and metrics.

## Example
```python
from counterfactuals.core.registry import Registry

method_registry = Registry("method")
```
