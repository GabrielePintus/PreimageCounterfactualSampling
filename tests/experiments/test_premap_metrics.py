import numpy as np
import pytest
import torch

from certcf import CertifiedRegion, RegionGeometry
from experiments.premap_metrics import local_domains, nonempty_regions, query_grouped, metric_rows, METHODS
from experiments.atlas_premap import explicit_graph


def test_explicit_graph_preserves_numerical_sequential_with_lazy_layer():
    model = torch.nn.Sequential(torch.nn.LazyLinear(4), torch.nn.ReLU(),
                                torch.nn.Dropout(.2), torch.nn.Linear(4, 2)).eval()
    example = torch.ones((3, 2))
    model(example)
    graph = explicit_graph(model, example)
    torch.testing.assert_close(graph(example), model(example), rtol=0, atol=0)
    assert graph[0].in_features == 2
    assert len(graph) == 3


def test_explicit_graph_rejects_unsupported_layer():
    with pytest.raises(TypeError, match="Unsupported layer"):
        explicit_graph(torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Sigmoid()), torch.ones(1, 2))


def test_local_clearance_uses_full_reference_set():
    x = np.array([[0., 0.], [10., 0.], [1., 0.]], dtype=np.float32)
    y = np.array([0, 1, 1])
    domains = local_domains(x, y, np.array([0, 1]), .2)
    assert domains[0].geometry.upper[0] == pytest.approx(.2)
    assert domains[0].domain_id == '0'


def region(key, source, lo, hi):
    return CertifiedRegion(key, 1, RegionGeometry(np.empty((0, 2)), [], lo, hi), source)


def test_grouped_retrieval_evaluates_all_leaves_and_fallback():
    anchors = np.array([[1., 0], [3., 0]])
    labels = np.array([1, 1])
    groups = {"0": (region("a", "0", [2, 0], [2, 0]), region("b", "0", [.5, 0], [.5, 0])),
              "1": (region("c", "1", [3, 0], [3, 0]),)}
    options = {"k_anchors": 1, "distance_norm": 1}
    cf = query_grouped(groups, anchors, labels, np.array([0, 1]), np.zeros(2), 1, options)
    np.testing.assert_allclose(cf, [.5, 0], atol=1e-6)
    groups["0"] = ()
    cf = query_grouped(groups, anchors, labels, np.array([0, 1]), np.zeros(2), 1, options)
    np.testing.assert_allclose(cf, [3, 0], atol=1e-6)


def test_box_filter_preserves_geometry():
    valid = region("valid", "a", [0, 0], [1, 1])
    invalid = CertifiedRegion("empty", 1, RegionGeometry([[1, 0]], [-2], [0, 0], [1, 1]), "b")
    assert nonempty_regions((valid, invalid)) == (valid,)


def test_metrics_keep_failure_denominators_and_pair_queries():
    x = np.array([[-1., 0], [1., 0], [-2., 0], [2., 0]], dtype=np.float32)
    train = np.random.default_rng(7).normal(size=(60, 2)).astype(np.float32)
    y = (train[:, 0] >= 0).astype(int)
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[-1., 0], [1., 0]]))
        model.bias.zero_()
    records = []
    for method in METHODS:
        for i in range(4):
            success = method != 'box_crown' or i != 0
            records.append(dict(method=method, query_id=i, target=int(x[i, 0] < 0), success=success,
                                seconds=.01, x_cf=[-float(x[i, 0]), 0] if success else None))
    cfg = dict(lof_neighbors=5, l0_tolerance=1e-6, sigma_grid=[0], robustness_repeats=1,
               perturbations_per_cell=2, random_state=0, tree_test_size=.3, tree_depth=3, tree_min_samples_leaf=5)
    frame, summaries = metric_rows(records, x, train, y, y, model, cfg)
    for row in summaries:
        assert row['n_queries'] == 4
        assert row['validity_pct'] == (75 if row['method'] == 'box_crown' else 100)
        if row['population'] == 'shared_successes':
            assert row['n_quality'] == 3
        assert row['robustness_pct'] == 100
        assert row['l0'] == 1
        assert row['interpretability_proxy'] is None
    assert len(frame) == 12
