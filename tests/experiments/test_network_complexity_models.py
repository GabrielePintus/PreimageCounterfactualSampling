import pytest
import torch

from experiments.network_complexity import parameter_count
from models.classifiers import (
    TabularClassifier,
    infer_tabular_classifier_dims_from_state_dict,
)


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_tabular_classifier_supports_all_grid_depths(depth):
    hidden_dims = [16] * depth
    model = TabularClassifier(
        input_types=["numerical"] * 32,
        cardinalities=[],
        hidden_dims=hidden_dims,
        num_classes=2,
        dropout=0.2,
    )
    output = model(torch.zeros(7, 32))
    assert output.shape == (7, 2)
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count(
        32, hidden_dims, 2
    )
    inferred_hidden, inferred_classes = infer_tabular_classifier_dims_from_state_dict(
        model.state_dict()
    )
    assert inferred_hidden == hidden_dims
    assert inferred_classes == 2


def test_historical_two_layer_state_dict_layout_remains_compatible():
    model = TabularClassifier(
        input_types=["numerical"] * 32,
        cardinalities=[],
        hidden_dims=[64, 32],
        num_classes=2,
        dropout=0.2,
    )
    model(torch.zeros(1, 32))
    assert list(model.state_dict()) == [
        "net.1.weight",
        "net.1.bias",
        "net.4.weight",
        "net.4.bias",
        "net.6.weight",
        "net.6.bias",
    ]
    hidden, classes = infer_tabular_classifier_dims_from_state_dict(
        {f"model.{key}": value for key, value in model.state_dict().items()}
    )
    assert hidden == [64, 32]
    assert classes == 2


def test_dropout_is_applied_after_every_hidden_relu():
    model = TabularClassifier(
        input_types=["numerical"],
        cardinalities=[],
        hidden_dims=[1],
        num_classes=2,
        dropout=1.0,
    )
    model(torch.ones(1, 1))  # materialize LazyLinear
    with torch.no_grad():
        model.net[1].weight.fill_(1.0)
        model.net[1].bias.fill_(1.0)
        model.net[3].weight.fill_(1.0)
        model.net[3].bias.copy_(torch.tensor([3.0, 4.0]))
    model.train()
    torch.testing.assert_close(model(torch.ones(1, 1)), torch.tensor([[3.0, 4.0]]))


@pytest.mark.parametrize("hidden_dims", [[], [0], [-1, 8]])
def test_tabular_classifier_rejects_invalid_hidden_dims(hidden_dims):
    with pytest.raises(ValueError, match="hidden_dims"):
        TabularClassifier(["numerical"], [], hidden_dims=hidden_dims)
