"""Model wrappers for a unified prediction interface."""

from .base_model import BaseModelWrapper
from .sklearn_model import SklearnModelWrapper
from .torch_model import TorchModelWrapper

__all__ = ["BaseModelWrapper", "SklearnModelWrapper", "TorchModelWrapper"]
