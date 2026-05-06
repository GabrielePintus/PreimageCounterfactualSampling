# metrics

Independent metric implementations for evaluating counterfactual quality.

## Contents

- `proximity.py`: L1, L2, and MAD-weighted L1 proximity metrics.
- `validity.py`: checks whether a prediction reaches the target class.
- `sparsity.py`: fraction of changed encoded features.
- `redundancy.py`: redundant-change diagnostics.
- `plausibility.py`: k-NN manifold proximity.

## Example

```python
from counterfactuals.metrics.proximity import L1Proximity

score = L1Proximity().evaluate(x_orig, x_cf)
```
