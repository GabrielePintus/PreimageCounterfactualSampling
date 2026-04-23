"""LightningModule wrapper for classification models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class LitClassifier(L.LightningModule):
    """
    LightningModule wrapping any PyTorch classifier.

    Handles training and validation with CrossEntropyLoss, AdamW optimizer,
    and CosineAnnealingLR.

    Parameters
    ----------
    model : nn.Module
        The classifier to train (e.g. MNISTClassifier, SimpleClassifier).
    initial_lr : float
        Peak learning rate reached after warmup; starting point of cosine decay.
    weight_decay : float
        L2 regularization coefficient for AdamW.
    final_lr : float
        Minimum learning rate at the end of cosine annealing.
    """

    def __init__(
        self,
        model: nn.Module,
        initial_lr: float = 5e-3,
        weight_decay: float = 1e-4,
        final_lr: float = 1e-6,
        class_weights: list[float] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self._class_weights_loaded = False
        weight_tensor = self._build_weight_tensor(class_weights)
        self.register_buffer(
            "class_weight_tensor",
            weight_tensor if weight_tensor is not None else torch.empty(0, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _build_weight_tensor(class_weights: list[float] | None) -> torch.Tensor | None:
        if class_weights is None:
            return None
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        if weight_tensor.ndim != 1:
            raise ValueError("class_weights must be a 1D sequence of per-class weights")
        return weight_tensor

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self._class_weights_loaded:
            return
        if self.hparams.class_weights is not None:
            self._class_weights_loaded = True
            return

        datamodule = getattr(self.trainer, "datamodule", None)
        class_weights = getattr(datamodule, "class_weights", None)
        if class_weights is None:
            return

        weight_tensor = self._build_weight_tensor(class_weights)
        self.class_weight_tensor = weight_tensor
        self._class_weights_loaded = True

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        weight = self.class_weight_tensor if self.class_weight_tensor.numel() else None
        loss = F.cross_entropy(logits, y, weight=weight)
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
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps),
            eta_min=self.hparams.final_lr,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": cosine, "interval": "step"}}
