# Training Playbook

## Entrypoint

Training and testing run through LightningCLI:

```bash
python train.py fit --config <config.yaml>
python train.py test --config <config.yaml> --ckpt_path <checkpoint.ckpt>
```

`train.py` pre-imports modules so YAML `class_path` strings resolve at runtime.

## Supported Configs

- `configs/mnist_classifier.yaml`
- `configs/mnist_ae.yaml`
- `configs/spiral_classifier.yaml`
- `configs/adult_classifier.yaml`

Each config defines `trainer`, `model`, and `data` sections.

## Model Wrappers

- `training.lit_classifier.LitClassifier`
  - Cross-entropy training
  - AdamW optimizer
  - Linear warmup + cosine schedule
- `training.lit_autoencoder.LitAutoencoder`
  - VAE-style reconstruction + KL loss
  - AdamW optimizer
  - Linear warmup + cosine schedule

## Datamodules

- `training.datamodules.mnist.MNISTDataModule`
  - downloads MNIST
  - optional train augmentation
  - split into train/val/test
- `training.datamodules.spiral.SpiralDataModule`
  - generates synthetic spiral data via `make_spiral`
- `training.datamodules.adult.AdultDataModule`
  - fetches OpenML adult dataset
  - categorical encoding + numerical scaling

## Common Operations

### Quick smoke run

```bash
python train.py fit --config configs/spiral_classifier.yaml --trainer.max_epochs=1
```

### Override hyperparameters at runtime

```bash
python train.py fit \
  --config configs/mnist_classifier.yaml \
  --model.init_args.initial_lr=0.005 \
  --trainer.max_epochs=5
```

### Evaluate from a saved checkpoint

```bash
python train.py test \
  --config configs/mnist_classifier.yaml \
  --ckpt_path checkpoints/classifier/last.ckpt
```

## Known Footguns

- Keep YAML `class_path` values aligned with importable module paths under `src/`.
- Config files currently include repeated `logger` keys in some YAMLs; when editing, ensure final YAML shape is intentional.
- Changes to model signatures must stay compatible with existing config `init_args`.
- For Adult tabular experiments, preserve feature order and cardinalities used by `TabularClassifier`.

## Minimal Validation Matrix For Agent Edits

If touching training code, run at least:
1. One classifier smoke fit (`spiral` preferred for speed).
2. One autoencoder smoke fit (`mnist_ae` with low epochs).
3. One test invocation from a checkpoint path if checkpoint handling was changed.
