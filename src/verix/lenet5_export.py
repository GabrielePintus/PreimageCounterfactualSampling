"""Function-preserving LeNet-5 export for the VERIX Marabou backend."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn


class FixedAveragePoolConv2d(nn.Conv2d):
    """Exact ``AvgPool2d(2, 2)`` expressed as an ordinary convolution.

    Marabou's ONNX parser supports standard convolutions but not AveragePool or
    grouped/depthwise convolutions. A dense channel-diagonal kernel therefore
    preserves the function while remaining parser-compatible.
    """

    def __init__(self, channels: int) -> None:
        channels = int(channels)
        if channels <= 0:
            raise ValueError("channels must be positive")
        super().__init__(
            in_channels=channels,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        with torch.no_grad():
            self.weight.zero_()
            for channel in range(channels):
                self.weight[channel, channel, :, :] = 0.25
        self.weight.requires_grad_(False)


class MarabouCompatibleLeNet5(nn.Module):
    """Functionally equivalent LeNet-5 with parser-compatible pooling."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        if not hasattr(source, "features") or not hasattr(source, "classifier"):
            raise TypeError("source must expose LeNet-5 features and classifier modules")

        features = copy.deepcopy(source.features)
        current_channels: int | None = None
        replacements = 0
        for index, layer in enumerate(features):
            if isinstance(layer, nn.Conv2d):
                current_channels = int(layer.out_channels)
            elif isinstance(layer, nn.AvgPool2d):
                if (
                    layer.kernel_size not in {2, (2, 2)}
                    or layer.stride not in {2, (2, 2)}
                    or layer.padding not in {0, (0, 0)}
                    or bool(layer.ceil_mode)
                    or not bool(layer.count_include_pad)
                ):
                    raise ValueError(
                        "only AvgPool2d(kernel=2, stride=2, padding=0) is supported"
                    )
                if current_channels is None:
                    raise ValueError("could not infer channels before AvgPool2d")
                features[index] = FixedAveragePoolConv2d(current_channels)
                replacements += 1

        if replacements != 2:
            raise ValueError(
                f"expected exactly two LeNet-5 average pools, found {replacements}"
            )
        self.features = features
        self.classifier = copy.deepcopy(source.classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.classifier(torch.flatten(h, 1))


def load_lenet5_checkpoint(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    num_classes: int = 10,
) -> nn.Module:
    """Load the repository's Lightning LeNet-5 checkpoint without CLI metadata."""

    from models.classifiers import LeNet5Classifier

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing LeNet-5 checkpoint: {checkpoint_path}")
    payload = torch.load(
        checkpoint_path,
        map_location=torch.device(device),
        weights_only=False,
    )
    state_dict = payload.get("state_dict", payload)
    prefix = "model."
    model_state = {
        (key[len(prefix):] if str(key).startswith(prefix) else str(key)): value
        for key, value in state_dict.items()
    }
    model = LeNet5Classifier(num_classes=int(num_classes))
    model.load_state_dict(model_state, strict=True)
    return model.eval().to(device)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_lenet5_onnx(
    model: nn.Module,
    output_path: str | Path,
    *,
    opset_version: int = 13,
) -> Path:
    """Export logits with a dynamic batch dimension and stable node names."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = copy.deepcopy(model).eval().cpu()
    example = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        example,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=int(opset_version),
        do_constant_folding=True,
        dynamo=False,
    )
    return path


def onnx_operator_types(path: str | Path) -> tuple[str, ...]:
    """Return sorted unique operation types from an ONNX graph."""

    import onnx

    graph = onnx.load(str(path)).graph
    return tuple(sorted({str(node.op_type) for node in graph.node}))


def validate_marabou_parse(path: str | Path) -> dict[str, int]:
    """Parse the exported network and report flattened input/output sizes."""

    from maraboupy import Marabou

    network = Marabou.read_onnx(filename=str(path), outputNames=None)
    return {
        "input_variables": int(network.inputVars[0].size),
        "output_variables": int(network.outputVars[0].size),
    }


@torch.no_grad()
def validate_lenet5_equivalence(
    original: nn.Module,
    rewritten: nn.Module,
    onnx_path: str | Path,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: str | torch.device = "cpu",
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
) -> dict[str, Any]:
    """Compare original, rewritten, and ONNX logits over supplied batches."""

    import onnxruntime as ort

    original = original.eval().to(device)
    rewritten = rewritten.eval().to(device)
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name

    n_samples = 0
    original_correct = 0
    rewritten_correct = 0
    onnx_correct = 0
    rewritten_max_abs = 0.0
    onnx_max_abs = 0.0
    rewritten_argmax_equal = True
    onnx_argmax_equal = True
    allclose_rewritten = True
    allclose_onnx = True

    for images, labels in batches:
        images_device = images.to(device=device, dtype=torch.float32)
        labels_np = labels.detach().cpu().numpy().reshape(-1)
        original_logits = original(images_device).detach().cpu().numpy()
        rewritten_logits = rewritten(images_device).detach().cpu().numpy()
        onnx_logits = np.asarray(
            session.run(
                None,
                {input_name: images.detach().cpu().numpy().astype(np.float32)},
            )[0]
        )

        rewritten_max_abs = max(
            rewritten_max_abs,
            float(np.max(np.abs(original_logits - rewritten_logits))),
        )
        onnx_max_abs = max(
            onnx_max_abs,
            float(np.max(np.abs(original_logits - onnx_logits))),
        )
        allclose_rewritten &= bool(
            np.allclose(original_logits, rewritten_logits, atol=atol, rtol=rtol)
        )
        allclose_onnx &= bool(
            np.allclose(original_logits, onnx_logits, atol=atol, rtol=rtol)
        )

        original_pred = original_logits.argmax(axis=1)
        rewritten_pred = rewritten_logits.argmax(axis=1)
        onnx_pred = onnx_logits.argmax(axis=1)
        rewritten_argmax_equal &= bool(np.array_equal(original_pred, rewritten_pred))
        onnx_argmax_equal &= bool(np.array_equal(original_pred, onnx_pred))
        original_correct += int(np.sum(original_pred == labels_np))
        rewritten_correct += int(np.sum(rewritten_pred == labels_np))
        onnx_correct += int(np.sum(onnx_pred == labels_np))
        n_samples += int(len(labels_np))

    if n_samples == 0:
        raise ValueError("equivalence validation requires at least one sample")

    report: dict[str, Any] = {
        "n_samples": n_samples,
        "atol": float(atol),
        "rtol": float(rtol),
        "rewritten_max_abs_logit_error": rewritten_max_abs,
        "onnx_max_abs_logit_error": onnx_max_abs,
        "rewritten_allclose": allclose_rewritten,
        "onnx_allclose": allclose_onnx,
        "rewritten_argmax_identical": rewritten_argmax_equal,
        "onnx_argmax_identical": onnx_argmax_equal,
        "original_accuracy": original_correct / n_samples,
        "rewritten_accuracy": rewritten_correct / n_samples,
        "onnx_accuracy": onnx_correct / n_samples,
    }
    report["passed"] = bool(
        allclose_rewritten
        and allclose_onnx
        and rewritten_argmax_equal
        and onnx_argmax_equal
        and original_correct == rewritten_correct == onnx_correct
    )
    return report
