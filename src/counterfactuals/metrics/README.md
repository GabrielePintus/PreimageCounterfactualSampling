# metrics

Independent metric implementations for evaluating counterfactual quality.

## Contents
- `proximity.py`: L1/L2 distance metrics.
- `validity.py`: checks if prediction reached target class.
- `sparsity.py`: fraction of changed features.
- `plausibility.py`: k-NN manifold proximity.

## Example
```python
from counterfactuals.metrics.proximity import L2Proximity

score = L2Proximity().evaluate(x_orig, x_cf)
```
