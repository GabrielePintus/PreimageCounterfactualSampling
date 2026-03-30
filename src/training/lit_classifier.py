"""LightningModule wrapper for classification models."""

import torch
import torch.nn as nn
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class LitClassifier(L.LightningModule):
    """
    LightningModule wrapping any PyTorch classifier.

    Handles training and validation with CrossEntropyLoss, AdamW optimizer,
    linear LR warmup followed by CosineAnnealingLR.

    LR schedule: ~0 → (linear, warmup_steps) → initial_lr → (cosine) → final_lr

    Parameters
    ----------
    model : nn.Module
        The classifier to train (e.g. MNISTClassifier, SimpleClassifier).
    initial_lr : float
        Peak learning rate reached after warmup; starting point of cosine decay.
    weight_decay : float
        L2 regularization coefficient for AdamW.
    warmup_steps : int
        Number of optimizer steps for linear LR warmup.
    final_lr : float
        Minimum learning rate at the end of cosine annealing.
    """

    def __init__(
        self,
        model: nn.Module,
        initial_lr: float = 1e-2,
        weight_decay: float = 1e-4,
        warmup_steps: int = 500,
        final_lr: float = 1e-6,
        class_weights: list[float] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
            if weight_tensor.ndim != 1:
                raise ValueError("class_weights must be a 1D sequence of per-class weights")
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        return {"loss": loss, "acc": acc}, loss

    def training_step(self, batch, batch_idx):
        prediction, loss = self.shared_step(batch, batch_idx)
        self.log_dict({f"train/{k}": v for k, v in prediction.items()}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        prediction, loss = self.shared_step(batch, batch_idx)
        self.log_dict({f"val/{k}": v for k, v in prediction.items()}, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        prediction, loss = self.shared_step(batch, batch_idx)
        self.log_dict({f"test/{k}": v for k, v in prediction.items()}, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.initial_lr,
            weight_decay=self.hparams.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup = LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=self.hparams.warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - self.hparams.warmup_steps,
            eta_min=self.hparams.final_lr,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.hparams.warmup_steps],
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
