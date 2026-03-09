# sampling

Sampling and projection routines used to search certified regions for counterfactuals.

## Contents
- `sampler.py`: `CounterfactualSampler` and atlas-level projection logic.

## Example
```python
from preimage_sampling.sampling import CounterfactualSampler

sampler = CounterfactualSampler(solver="ECOS")
```
