import numpy as np
import pytest
import torch

from models.classifiers import LeNet5Classifier
from verix.lenet5_export import (
    FixedAveragePoolConv2d,
    MarabouCompatibleLeNet5,
    export_lenet5_onnx,
    onnx_operator_types,
)


def test_fixed_convolution_exactly_matches_average_pooling():
    x = torch.randn(3, 6, 28, 28)
    expected = torch.nn.AvgPool2d(2, 2)(x)
    actual = FixedAveragePoolConv2d(6)(x)

    assert torch.equal(expected, actual)


def test_rewritten_lenet5_preserves_logits():
    torch.manual_seed(7)
    original = LeNet5Classifier().eval()
    rewritten = MarabouCompatibleLeNet5(original).eval()
    x = torch.rand(8, 1, 28, 28)

    assert torch.equal(original(x), rewritten(x))
    assert not any(isinstance(layer, torch.nn.AvgPool2d) for layer in rewritten.modules())


def test_onnx_export_has_no_average_pool_and_preserves_predictions(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    torch.manual_seed(11)
    original = LeNet5Classifier().eval()
    rewritten = MarabouCompatibleLeNet5(original).eval()
    path = export_lenet5_onnx(rewritten, tmp_path / "lenet5.onnx")

    operators = onnx_operator_types(path)
    assert "AveragePool" not in operators
    assert "Conv" in operators

    x = np.random.default_rng(3).random((4, 1, 28, 28), dtype=np.float32)
    expected = rewritten(torch.from_numpy(x)).detach().numpy()
    session = ort.InferenceSession(str(path))
    actual = session.run(None, {session.get_inputs()[0].name: x})[0]

    assert np.allclose(expected, actual, atol=1.0e-5, rtol=1.0e-5)
    assert np.array_equal(expected.argmax(axis=1), actual.argmax(axis=1))
