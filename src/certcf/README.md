# CertCF Core

Core implementation of CertCF: certified preimage construction, certified-polytope search, and counterfactual projection.

## Contents

- `atlas.py`: high-level `CertCFAtlas` API used by the benchmark wrapper.
- `eps_strategies.py`: support-radius policies, including nearest-opposite-class clearance.
- `certification/`: auto_LiRPA/CROWN bound computation and preimage approximation helpers.
- `geometry/`: halfspace, Lp-ball, polygon, and projection utilities.
- `indexing/`: BVH spatial index implementation retained for exhaustive/search variants.
- `sampling/`: lower-level projection sampler utilities.
- `visualization/`: plotting helpers for 2D certification diagnostics.

## Example

```python
from certcf import CertCFAtlas

atlas = CertCFAtlas(model=model, dataset=dataset, device="cpu", norm=1)
atlas.build()
result = atlas.find_counterfactual(x_query, target_class=1)
```
