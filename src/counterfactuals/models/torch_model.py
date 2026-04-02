"""PyTorch model wrapper implementing the common model interface."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base_model import BaseModelWrapper


class FlatToTensorModel(nn.Module):
    """Wrap a module so it can consume flattened inputs."""

    def __init__(self, model: nn.Module, input_shape: tuple[int, ...]):
        super().__init__()
        self.model = model
        self.input_shape = tuple(int(v) for v in input_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.view(x.shape[0], *self.input_shape)
        return self.model(x)


class TorchModelWrapper(BaseModelWrapper):
    """Adapter around ``torch.nn.Module`` for model-agnostic methods."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        input_shape: tuple[int, ...] | None = None,
    ):
        super().__init__(n_classes=None)
        if input_shape is not None:
            model = FlatToTensorModel(model=model, input_shape=input_shape)
        self.model = model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_2d = self._ensure_2d(x)
        x_tensor = torch.from_numpy(x_2d).to(self.device)
        logits = self.model(x_tensor)
        probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy().astype(np.float32)
