# models

Neural network architectures used by Lightning training modules and certification workflows.

## Contents

- `classifiers.py`: tabular classifier used by the final benchmark, plus legacy MNIST and simple MLP classifiers.
- `ae.py` and `ae_channels.py`: legacy autoencoder variants retained for older experiments.

## Example

```python
from models.classifiers import TabularClassifier

model = TabularClassifier(
    input_types=["numerical", "categorical"],
    cardinalities=[3],
    hidden_dims=[64, 32],
    num_classes=2,
    dropout=0.2,
)
```
