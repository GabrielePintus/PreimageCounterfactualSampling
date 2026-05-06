# certification

LiRPA-related wrappers and bound-computation utilities.

## Contents

- `wrapping.py`: model wrappers used before handing networks to auto_LiRPA.
- `lirpa.py`: CROWN/optimized-CROWN/alpha-CROWN orchestration and `PreimageApproximation`.
- `bounds.py`: small helpers for extracting lower/upper bound values.

The final benchmark uses the standard backward CROWN path through `lirpa_method: backward`.

## Example

```python
from certcf.certification import PreimageApproximation
```
