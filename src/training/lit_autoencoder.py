"""LightningModule wrapper for the convolutional VAE."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class LitAutoencoder(L.LightningModule):
    """
    LightningModule wrapping ConvAutoencoder (VAE).

    Uses the combined VAE loss: MSE reconstruction + beta * KL divergence.

    Parameters
    ----------
    model : nn.Module
        ConvAutoencoder instance (must return mu, logvar, reconstruction).
    initial_lr : float
        Peak learning rate reached after warmup; starting point of cosine decay.
    weight_decay : float
        L2 regularization coefficient for AdamW.
    beta : float
        Weight for the KL divergence term (beta-VAE coefficient).
    warmup_steps : int
        Number of optimizer steps for linear LR warmup.
    final_lr : float
        Minimum learning rate at the end of cosine annealing.
    """

    def __init__(
        self,
        model: nn.Module,
        initial_lr: float = 5e-3,
        weight_decay: float = 1e-4,
        beta: float = 1.0,
        warmup_steps: int = 1000,
        final_lr: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.mse = nn.MSELoss(reduction="mean")
        self.kl = lambda mu, logvar: -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def vae_loss(self, x, mu, logvar, recon):
        recon_loss = self.mse(recon, x)
        kl_loss = self.kl(mu, logvar)
        total = recon_loss + self.hparams.beta * kl_loss
        return {"loss": total, "recon_loss": recon_loss, "kl_loss": kl_loss}

    def forward(self, x):
        return self.model(x)
    

    def shared_step(self, batch, batch_idx):
        x, _ = batch
        mu, logvar, recon = self(x)
        loss = self.vae_loss(x, mu, logvar, recon)
        return {"mu": mu, "logvar": logvar, "recon": recon}, loss

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
