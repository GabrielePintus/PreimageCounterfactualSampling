import numpy as np
import pytest
import torch

from certcf import CertifiedRegion, RegionGeometry
from experiments.atlas_premap import explicit_graph, region_feasibility, training_box
from models.classifiers import TabularClassifier


def test_training_box_contains_training_data_and_preserves_float32():
    x = np.array([[-1.2, 0], [3, 2]], dtype=np.float32)
    g = training_box(x)
    assert g.contains(x, tolerance=0).all()
    np.testing.assert_array_equal(g.lower, x.min(0).astype(float))
    assert not g.contains([100, 100])
    with pytest.raises(ValueError):
        training_box([[0, 1], [1, 1]])


def test_explicit_graph_preserves_loaded_lazy_classifier():
    original = TabularClassifier(["numerical"] * 2, [], hidden_dims=(4, 4), num_classes=2, dropout=.1).eval()
    example = torch.randn(8, 2)
    original(example)
    restored = TabularClassifier(["numerical"] * 2, [], hidden_dims=(4, 4), num_classes=2, dropout=.1).eval()
    restored.load_state_dict(original.state_dict())
    graph = explicit_graph(restored, example)
    torch.testing.assert_close(graph(example), original(example), rtol=0, atol=0)
    assert all(type(layer) in (torch.nn.Linear, torch.nn.ReLU) for layer in graph)
    assert graph[0].in_features == 2


def test_region_feasibility_does_not_count_empty_leaf_as_certificate():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[-1., 0], [1., 0]]))
        model[0].bias.zero_()
    empty = CertifiedRegion("empty", 1, RegionGeometry([[1, 0]], [-2], [-1, -1], [1, 1]), "box")
    valid = CertifiedRegion("valid", 1, RegionGeometry([[1, 0]], [-.2], [-1, -1], [1, 1]), "box")
    assert not region_feasibility(empty, model, .001)["feasible"]
    assert region_feasibility(valid, model, .001)["feasible"]


def test_affine_box_maximum_detects_empty_binary_halfspace():
    from experiments.atlas_premap_diagnostics import affine_box_maxima
    geometry = RegionGeometry([[2, -3]], [-10], [-1, -1], [1, 1])
    np.testing.assert_array_equal(affine_box_maxima(geometry), [-5])
    assert not geometry.contains(np.array([[1, -1], [-1, 1]])).any()
