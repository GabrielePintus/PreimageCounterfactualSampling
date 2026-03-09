# preimage_sampling

Main library for Certified Polyhedral Projection (CPP) and certified counterfactual generation.

## Contents
- `atlas.py`: high-level `CertifiedAtlas` API.
- `eps_strategies.py`: epsilon selection policies.
- subpackages for certification, geometry, indexing, and sampling.

## Example
```python
from preimage_sampling import CertifiedAtlas

atlas = CertifiedAtlas(model=model, dataset=dataset, device="cpu")
```
