# utils

Utility helpers for deterministic runs, logging, config loading, and prototype selection.

## Contents

- `seed.py`: centralized seeding utility.
- `logging.py`: standard logger creation.
- `config.py`: YAML reading helper.
- `clustering.py`: prototype-selection utilities used by benchmark methods.

## Example

```python
from counterfactuals.utils.seed import seed_everything

seed_everything(42)
```
