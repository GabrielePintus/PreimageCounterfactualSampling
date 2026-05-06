# models

Model wrappers exposing a common prediction API for method-agnostic experiments.

## Contents

- `base_model.py`: shared wrapper behavior.
- `sklearn_model.py`: sklearn estimator adapter and simple MLP builder.
- `torch_model.py`: torch module adapter, including flat-to-tensor handling for image models.

## Example

```python
from counterfactuals.models.sklearn_model import build_sklearn_mlp

model = build_sklearn_mlp(random_seed=42)
```
