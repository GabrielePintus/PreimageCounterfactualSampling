# certcf

Main library for CertCF and certified counterfactual generation.

## Contents
- `atlas.py`: high-level `CertCFAtlas` API.
- `eps_strategies.py`: epsilon selection policies.
- subpackages for certification, geometry, indexing, and sampling.

## Example
```python
from certcf import CertCFAtlas

atlas = CertCFAtlas(model=model, dataset=dataset, device="cpu")
```
