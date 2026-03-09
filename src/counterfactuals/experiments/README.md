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
