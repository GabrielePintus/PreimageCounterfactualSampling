"""Pinned CIFAR ResNets used by the CertCF scaling experiment.

Architecture adapted from chenyaofo/pytorch-cifar-models commit
786c16252c0fc58ee9adac063f8337cc4a7a497a (BSD-3-Clause). The original
implementation is itself derived from torchvision's ResNet.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


MODEL_SPECS = {
    "resnet20": {
        "blocks": 3,
        "url": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt",
        "sha256": "4118986f0df73003d572b0e397f0ac7b3f60af1f31aff3d2da164536e36f6ec8",
    },
    "resnet32": {
        "blocks": 5,
        "url": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet32-ef93fc4d.pt",
        "sha256": "ef93fc4d3ea83c08d75bfd6fc31af56047cfa241608c9ba3a1f6dc11cca20dd9",
    },
    "resnet56": {
        "blocks": 9,
        "url": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt",
        "sha256": "187c023aee0c9cf3093a682d9447a538cbaf5489f7ae78a7df4edf2246ce380b",
    },
}


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class CifarResNet(nn.Module):
    def __init__(self, blocks_per_stage: int, num_classes: int = 10):
        super().__init__()
        self.inplanes = 16
        self.conv1 = _conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(16, blocks_per_stage)
        self.layer2 = self._make_layer(32, blocks_per_stage, stride=2)
        self.layer3 = self._make_layer(64, blocks_per_stage, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers.extend(BasicBlock(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


class NormalizeCifar10(nn.Module):
    """Keep experiment inputs in raw pixel space while matching training preprocessing."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # No leading singleton is stored: auto-LiRPA otherwise interprets it
        # as a batch dimension on the constant subtraction/division operands.
        self.register_buffer("mean", torch.tensor([0.4914, 0.4822, 0.4465])[:, None, None])
        self.register_buffer("std", torch.tensor([0.2023, 0.1994, 0.2010])[:, None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


def build_cifar10_resnet(name: str, checkpoint: str | Path | None = None) -> nn.Module:
    if name not in MODEL_SPECS:
        raise ValueError(f"Unknown CIFAR ResNet {name!r}; expected one of {sorted(MODEL_SPECS)}")
    model = CifarResNet(int(MODEL_SPECS[name]["blocks"]))
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    return NormalizeCifar10(model).eval()


def count_parameters_and_relu_activations(model: nn.Module) -> tuple[int, int]:
    """Count trainable parameters and executed scalar ReLU outputs for one image."""
    count = 0
    hooks = []

    def hook(_module, _inputs, output):
        nonlocal count
        count += int(output[0].numel())

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(torch.zeros(1, 3, 32, 32))
    finally:
        for item in hooks:
            item.remove()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return int(parameters), int(count)
