#!/usr/bin/env python3
"""
LightningCLI entrypoint for training NN models.

Usage
-----
Train MNIST classifier:
    python scripts/train_classifier.py fit --config configs/training/mnist_classifier.yaml

Train convolutional VAE:
    python scripts/train_classifier.py fit --config configs/training/mnist_ae.yaml

Override parameters on the fly:
    python scripts/train_classifier.py fit --config configs/training/mnist_classifier.yaml --trainer.max_epochs=100

Test (after training):
    python scripts/train_classifier.py test --config configs/training/mnist_classifier.yaml \
        --ckpt_path "data/Trained Models/mnist_classifier.ckpt"
"""

import sys
from pathlib import Path

from lightning.pytorch.cli import LightningCLI

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Register all modules so LightningCLI can resolve class_path strings from YAML
import training.lit_classifier      # noqa: F401
import training.lit_autoencoder     # noqa: F401
import training.datamodules.mnist   # noqa: F401
import training.datamodules.spiral  # noqa: F401
import training.datamodules.adult   # noqa: F401
import training.datamodules.compas  # noqa: F401
import training.datamodules.german_credit  # noqa: F401
import training.datamodules.heloc                # noqa: F401
import training.datamodules.give_me_some_credit  # noqa: F401
import training.datamodules.lending_club         # noqa: F401
import training.datamodules.wisconsin_breast_cancer  # noqa: F401
import models.classifiers           # noqa: F401
import models.ae                    # noqa: F401
import models.ae_channels           # noqa: F401


def main():
    LightningCLI(
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
