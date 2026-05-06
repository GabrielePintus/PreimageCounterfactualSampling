# indexing

Spatial indexing utilities for candidate certified-polytope retrieval.

## Contents

- `bvh.py`: Bounding Volume Hierarchy (BVH) implementation over polytope bounding boxes.

The final benchmark uses nearest-anchor search, but BVH remains available for exhaustive and lower-bound-pruned search modes.

## Example

```python
from certcf.indexing import BVHIndex
```
