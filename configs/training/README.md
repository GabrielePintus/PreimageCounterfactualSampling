# Training Configuration

This directory contains the LightningCLI YAML files used to train the tabular classifiers evaluated by the final benchmark.

Run a training job with:

```bash
python scripts/train_classifier.py fit --config configs/training/<config>.yaml
```

For example:

```bash
python scripts/train_classifier.py fit --config configs/training/adult_classifier.yaml
python scripts/train_classifier.py fit --config configs/training/compas_classifier.yaml
```

## Available Configs

| Config | Dataset | Checkpoint directory |
| --- | --- | --- |
| `adult_classifier.yaml` | Adult | `checkpoints/adult_classifier/` |
| `compas_classifier.yaml` | COMPAS | `checkpoints/compas_classifier/` |
| `german_credit_classifier.yaml` | German Credit | `checkpoints/german_credit_classifier/` |
| `give_me_some_credit_classifier.yaml` | Give Me Some Credit | `checkpoints/give_me_some_credit_classifier/` |
| `heloc_classifier.yaml` | HELOC | `checkpoints/heloc_classifier/` |
| `lending_club_classifier.yaml` | Lending Club | `checkpoints/lending_club_classifier/` |
| `wisconsin_breast_cancer_classifier.yaml` | Wisconsin Breast Cancer | `checkpoints/wisconsin_breast_cancer_classifier/` |

## Shared Setup

All configs train `training.lit_classifier.LitClassifier` around a tabular neural classifier with:

- hidden dimensions `[64, 32]`
- dropout `0.2`
- AdamW optimization through the Lightning module
- cosine learning-rate decay from `5e-3` to `1e-6`
- validation-loss checkpointing with `save_top_k: 1`
- deterministic best-checkpoint path `checkpoints/<dataset>_classifier/best.ckpt`
- no `last.ckpt` tracking

The final benchmark loads the resulting checkpoints from `checkpoints/<dataset>_classifier/best.ckpt`.

## Overrides

LightningCLI arguments can override YAML values at runtime. For quick checks, reduce the number of epochs:

```bash
python scripts/train_classifier.py fit \
  --config configs/training/compas_classifier.yaml \
  --trainer.max_epochs=2
```

To test a trained checkpoint:

```bash
python scripts/train_classifier.py test \
  --config configs/training/compas_classifier.yaml \
  --ckpt_path checkpoints/compas_classifier/best.ckpt
```
