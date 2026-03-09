# utils

Utility helpers for deterministic runs, logging, and config loading.

## Contents
- `seed.py`: centralized seeding utility.
- `logging.py`: standard logger creation.
- `config.py`: YAML reading helper.

## Example
```python
from counterfactuals.utils.seed import seed_everything

seed_everything(42)
```
