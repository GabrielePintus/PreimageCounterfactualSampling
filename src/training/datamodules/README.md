# datamodules

Dataset-specific LightningDataModule implementations.

## Contents
- `mnist.py`: MNIST pipeline.
- `spiral.py`: synthetic spiral pipeline.
- `adult.py`: Adult tabular pipeline.

## Example
```python
from training.datamodules.adult import AdultDataModule

dm = AdultDataModule(data_dir="data/")
dm.setup()
```
