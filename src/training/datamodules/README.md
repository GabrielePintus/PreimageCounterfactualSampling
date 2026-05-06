# datamodules

Dataset-specific LightningDataModule implementations.

## Contents

Tabular benchmark data modules:

- `adult.py`: Adult.
- `compas.py`: COMPAS.
- `german_credit.py`: German Credit.
- `give_me_some_credit.py`: Give Me Some Credit.
- `heloc.py`: HELOC.
- `lending_club.py`: Lending Club.
- `wisconsin_breast_cancer.py`: Wisconsin Breast Cancer.

Legacy/experimental data modules:

- `mnist.py`: MNIST image pipeline.
- `spiral.py`: synthetic spiral pipeline.

Shared tabular feature metadata lives in `dataset_specs/`; these data modules own data loading and fitted preprocessing.

## Example

```python
from training.datamodules.adult import AdultDataModule

dm = AdultDataModule(filepath="data/Adult/raw.parquet")
dm.setup()
```
