"""Neural network model architectures."""

from .classifiers import LeNet5Classifier, MNISTClassifier, SimpleClassifier
from .ae import ConvAutoencoder
from .ae_channels import ConvAutoencoder as ConvAutoencoderChannels

__all__ = [
    "SimpleClassifier",
    "ConvAutoencoderChannels",
    "ConvAutoencoder",
    "MNISTClassifier",
    "LeNet5Classifier",
]
