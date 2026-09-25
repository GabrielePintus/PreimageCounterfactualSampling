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
| `mnist_lenet5_classifier.yaml` | MNIST | `checkpoints/mnist_lenet5_classifier/` |

## Shared Setup

All configs train `training.lit_classifier.LitClassifier` around a tabular neural classifier with:

- hidden dimensions `[64, 32]`
- dropout `0.2`
- AdamW optimization through the Lightning module
- cosine learning-rate decay from `5e-3` to `1e-6`
- validation-loss checkpointing with `save_top_k: 1`
- deterministic best-checkpoint path `checkpoints/<dataset>_classifier/best.ckpt`
- no `last.ckpt` tracking
- local CSV logging under `lightning_logs/` by default

The final benchmark loads the resulting checkpoints from `checkpoints/<dataset>_classifier/best.ckpt`.

The default configurations are intentionally offline and require no tracking
account. To use Weights & Biases, install `.[tracking]` and pass a Lightning
logger override explicitly; this is not required for paper reproduction.

The MNIST config is a separate reproducibility experiment. It trains the
ReLU LeNet-5-style classifier for 10 epochs without a W&B logger and writes
`checkpoints/mnist_lenet5_classifier/best.ckpt`.

LeNet-5 uses the classifier wrapper's per-step cosine learning-rate schedule.
Set its start and end values under `model.init_args`:

```yaml
model:
  init_args:
    initial_lr: 1e-3
    final_lr: 1e-6
```

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
