# models

Neural network architectures used by Lightning training modules and certification workflows.

## Contents
- `classifiers.py`: tabular, MNIST, and simple classifiers.
- `ae.py` and `ae_channels.py`: autoencoder variants.

## Example
```python
from models.classifiers import TabularClassifier

model = TabularClassifier(input_types=["numerical"], cardinalities=[])
```
