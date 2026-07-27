from types import SimpleNamespace

import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.classifiers import LeNet5Classifier
from training.lit_classifier import LitClassifier


def test_lenet5_uses_configured_cosine_lr_endpoints():
    initial_lr = 1.0e-3
    final_lr = 1.0e-6
    total_steps = 4
    module = LitClassifier(
        model=LeNet5Classifier(),
        initial_lr=initial_lr,
        final_lr=final_lr,
    )
    module._trainer = SimpleNamespace(estimated_stepping_batches=total_steps)

    optimizer_config = module.configure_optimizers()
    optimizer = optimizer_config["optimizer"]
    scheduler_config = optimizer_config["lr_scheduler"]
    scheduler = scheduler_config["scheduler"]

    assert isinstance(optimizer, AdamW)
    assert isinstance(scheduler, CosineAnnealingLR)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(initial_lr)
    assert scheduler.eta_min == pytest.approx(final_lr)
    assert scheduler.T_max == total_steps
    assert scheduler_config["interval"] == "step"

    for _ in range(total_steps):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(final_lr)
