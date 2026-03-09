# methods

Counterfactual generation methods behind a shared interface.

## Contents
- `wachter.py`: Wachter-style search.
- `dice.py`: DiCE-style diverse candidate generation.
- `growing_spheres.py`: shell expansion baseline.
- `my_method.py`: adapter for `CertifiedAtlas`.

## Example
```python
from counterfactuals.methods.wachter import WachterMethod

method = WachterMethod(max_iter=500, random_seed=42)
```
