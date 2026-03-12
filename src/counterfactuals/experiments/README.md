# experiments

Config parsing and experiment orchestration for reproducible benchmark runs.

## Contents
- `config.py`: typed config objects and YAML parser.
- `runner.py`: registry-based execution loop and metric aggregation.

## Example
```python
from counterfactuals.experiments.runner import run_from_config_path

summary = run_from_config_path("configs/counterfactual_experiment.yaml")
```

## Shared preprocessing

`run_experiment` supports an optional `preprocessing` block in the YAML config.
When enabled (e.g. `name: pca`), methods operate in transformed space while all
metrics are computed after inverse-transform in the original feature space.
