# sampling

Lower-level certified-polytope projection utilities.

## Contents

- `sampler.py`: standalone `CounterfactualSampler` and `PolytopeAtlas` helpers.

The benchmark-facing CertCF path uses `certcf.atlas.CertCFAtlas`, which contains the current OHE decoding, adaptive epsilon, and sparsity-aware selection logic. This subpackage remains useful for direct geometric experiments.

## Example

```python
from certcf.sampling import CounterfactualSampler

sampler = CounterfactualSampler(solver="CLARABEL", distance_norm=1)
```
