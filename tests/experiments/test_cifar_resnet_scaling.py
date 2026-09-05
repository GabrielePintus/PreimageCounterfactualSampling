import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

from certcf import CertCFAtlas, NearestOppositeClassClearanceStrategy
from certcf.certification.lirpa import PreimageApproximation
from experiments.cifar_resnet_scaling import (
    DEFAULT_CONFIG,
    CifarResNetScalingRunner,
    config_fingerprint,
    validate_config,
)
from models.cifar_resnet import (
    MODEL_SPECS,
    build_cifar10_resnet,
    count_parameters_and_relu_activations,
)


@pytest.mark.parametrize(
    ("network", "parameters", "relus"),
    [
        ("resnet20", 272_474, 188_416),
        ("resnet32", 466_906, 303_104),
        ("resnet56", 855_770, 532_480),
    ],
)
def test_cifar_resnet_sizes_are_measured_dynamically(network, parameters, relus):
    model = build_cifar10_resnet(network)
    assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 10)
    assert count_parameters_and_relu_activations(model) == (parameters, relus)


def test_full_support_clearance_differs_from_anchor_only_clearance():
    anchors = np.array([[0.0], [10.0]], dtype=np.float32)
    labels = np.array([0, 1])
    strategy = NearestOppositeClassClearanceStrategy(alpha=0.2, chunk_size=1)
    anchor_only = strategy.compute_eps(anchors, labels, norm=1)
    strategy.set_reference(
        np.array([[0.0], [2.0], [10.0]], dtype=np.float32),
        np.array([0, 1, 1]),
    )
    full_support = strategy.compute_eps(anchors, labels, norm=1)
    np.testing.assert_allclose(anchor_only, [2.0, 2.0])
    np.testing.assert_allclose(full_support, [0.4, 2.0])


def test_flat_rgb_cnn_shape_is_explicit_not_mnist_specific():
    dataset = TensorDataset(torch.zeros(2, 3 * 32 * 32), torch.tensor([0, 1]))
    preimage = PreimageApproximation(
        torch.nn.Identity(), dataset, torch.device("cpu"), cnn=True, model_input_shape=(3, 32, 32)
    )
    assert preimage._reshape_cnn_batch(torch.zeros(2, 3 * 32 * 32)).shape == (2, 3, 32, 32)


def test_numeric_atlas_bounds_round_trip_without_pickle(tmp_path):
    dataset = TensorDataset(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]]), torch.tensor([0, 1])
    )
    model = torch.nn.Linear(2, 2)
    atlas = CertCFAtlas(model, dataset, "cpu", norm=1)
    atlas.bounds = {}
    for label, center in ((0, [0.0, 0.0]), (1, [1.0, 1.0])):
        atlas.bounds[label] = {
            "X": np.asarray([center], dtype=np.float32),
            "eps": np.asarray([0.1], dtype=np.float64),
            "eps_initial": np.asarray([0.2], dtype=np.float64),
            "lA": np.zeros((1, 1, 2), dtype=np.float32),
            "lbias": np.ones((1, 1), dtype=np.float32),
            "uA": np.zeros((1, 1, 2), dtype=np.float32),
            "ubias": np.ones((1, 1), dtype=np.float32),
        }
    atlas.save_bounds(tmp_path)
    restored = CertCFAtlas(model, dataset, "cpu", norm=1).load_bounds(tmp_path)
    for label in (0, 1):
        np.testing.assert_array_equal(restored.bounds[label]["X"], atlas.bounds[label]["X"])
        assert restored.bvh_indices[label].n_polytopes == 1


def test_reused_lirpa_graph_matches_fresh_graph():
    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2)
    ).eval()
    dataset = TensorDataset(
        torch.tensor([[0.1, 0.2], [0.8, 0.9]], dtype=torch.float32),
        torch.tensor([0, 1]),
    )
    fresh = PreimageApproximation(
        model, dataset, torch.device("cpu"), reuse_lirpa_graph=False
    ).compute_all_bounds(eps=0.01, norm=1)
    reused = PreimageApproximation(
        model, dataset, torch.device("cpu"), reuse_lirpa_graph=True
    ).compute_all_bounds(eps=0.01, norm=1)
    for label in (0, 1):
        np.testing.assert_allclose(reused[label]["lA"], fresh[label]["lA"], atol=1e-7)
        np.testing.assert_allclose(reused[label]["lbias"], fresh[label]["lbias"], atol=1e-7)


def test_completed_lirpa_classes_can_be_resumed(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2)).eval()
    dataset = TensorDataset(
        torch.tensor([[0.1, 0.2], [0.8, 0.9]], dtype=torch.float32),
        torch.tensor([0, 1]),
    )
    completed = {}
    first = PreimageApproximation(model, dataset, torch.device("cpu")).compute_all_bounds(
        eps=0.01,
        norm=1,
        class_completed_callback=lambda label, values: completed.update({label: values}),
    )
    assert set(completed) == {0, 1}

    def fail_if_recomputed(*_args, **_kwargs):
        raise AssertionError("completed classes must not be recomputed")

    monkeypatch.setattr("certcf.certification.lirpa.run_lirpa", fail_if_recomputed)
    resumed = PreimageApproximation(model, dataset, torch.device("cpu")).compute_all_bounds(
        eps=0.01, norm=1, precomputed_bounds=completed
    )
    for label in (0, 1):
        np.testing.assert_array_equal(resumed[label]["lA"], first[label]["lA"])


def test_official_protocol_uses_all_10k_support_points_l1_and_top3():
    validate_config(DEFAULT_CONFIG)
    assert DEFAULT_CONFIG["dataset"]["support_per_true_class"] == 1000
    assert DEFAULT_CONFIG["dataset"]["queries_per_true_class"] == 10
    assert DEFAULT_CONFIG["certcf"]["use_all_support_points"] is True
    assert "anchors_per_predicted_class" not in DEFAULT_CONFIG["certcf"]
    assert "anchors_per_predicted_class" not in DEFAULT_CONFIG["pilot"]
    assert DEFAULT_CONFIG["certcf"]["query_k_candidates"] == 3
    assert DEFAULT_CONFIG["certcf"]["norm"] == 1
    assert DEFAULT_CONFIG["certcf"]["distance_norm"] == 1
    invalid = deepcopy(DEFAULT_CONFIG)
    invalid["certcf"]["query_k_candidates"] = 5
    with pytest.raises(ValueError, match="top-k"):
        validate_config(invalid)
    invalid = deepcopy(DEFAULT_CONFIG)
    invalid["certcf"]["use_all_support_points"] = False
    with pytest.raises(ValueError, match="every selected support point"):
        validate_config(invalid)


def test_runner_disables_second_anchor_subsampling(tmp_path):
    config = deepcopy(DEFAULT_CONFIG)
    config["artifacts"]["output_dir"] = str(tmp_path)
    runner = CifarResNetScalingRunner(config)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10))

    method = runner._method(model)

    assert method.k_per_class is None


def test_full_size_pilot_artifacts_are_reused_for_official_run(tmp_path):
    config = deepcopy(DEFAULT_CONFIG)
    config["artifacts"]["output_dir"] = str(tmp_path)
    runner = CifarResNetScalingRunner(config)
    network = "resnet20"

    pilot_atlas = runner.paths.atlas(network, pilot=True)
    pilot_atlas.mkdir(parents=True)
    (pilot_atlas / "class_0.npz").write_bytes(b"bounds")
    (pilot_atlas / "manifest.json").write_text(
        json.dumps({"files": {"0": "class_0.npz"}}), encoding="utf-8"
    )
    runner.paths.build(network, pilot=True).write_text(
        json.dumps(
            {
                "status": "complete",
                "pilot": True,
                "config_fingerprint": runner.fingerprint,
                "checkpoint_sha256": MODEL_SPECS[network]["sha256"],
            }
        ),
        encoding="utf-8",
    )
    pilot_query = runner.paths.query_dir(network, pilot=True)
    pilot_query.mkdir(parents=True)
    (pilot_query / "query_000.json").write_text(
        json.dumps({"config_fingerprint": runner.fingerprint, "pilot": True}),
        encoding="utf-8",
    )
    np.savez(pilot_query / "query_000.npz", x=np.zeros(1), x_cf=np.ones(1))

    assert runner._promote_pilot_artifacts(network)
    assert runner._valid_build(network)
    assert (runner.paths.atlas(network) / "class_0.npz").read_bytes() == b"bounds"
    promoted_build = json.loads(runner.paths.build(network).read_text(encoding="utf-8"))
    assert promoted_build["pilot"] is False
    assert promoted_build["reused_from_pilot"] is True
    promoted_query = json.loads(
        (runner.paths.query_dir(network) / "query_000.json").read_text(encoding="utf-8")
    )
    assert promoted_query["pilot"] is False
    assert promoted_query["reused_from_pilot"] is True
    assert (runner.paths.query_dir(network) / "query_000.npz").exists()


def test_balanced_sampling_and_shared_nearest_targets_are_deterministic():
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(10), 4)
    first = CifarResNetScalingRunner._balanced_indices(labels, 2, rng)
    second = CifarResNetScalingRunner._balanced_indices(
        labels, 2, np.random.default_rng(42)
    )
    np.testing.assert_array_equal(first, second)
    assert np.bincount(labels[first], minlength=10).tolist() == [2] * 10

    x_support = np.array([[[[0.0]]], [[[0.2]]], [[[0.9]]]], dtype=np.float32)
    y_support = np.array([0, 1, 2])
    targets = CifarResNetScalingRunner._nearest_other_targets(
        np.array([[[[0.1]]]], dtype=np.float32),
        np.array([0]),
        x_support,
        y_support,
    )
    np.testing.assert_array_equal(targets, [1])


def test_strict_aggregation_requires_every_network(tmp_path):
    config = deepcopy(DEFAULT_CONFIG)
    config["artifacts"]["output_dir"] = str(tmp_path)
    runner = CifarResNetScalingRunner(config)
    network = "resnet20"
    runner.paths.atlas(network).mkdir(parents=True)
    (runner.paths.atlas(network) / "manifest.json").write_text("{}", encoding="utf-8")
    runner.paths.build(network).write_text(
        json.dumps(
            {
                "status": "complete",
                "config_fingerprint": config_fingerprint(config),
                "checkpoint_sha256": MODEL_SPECS[network]["sha256"],
            }
        ),
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {
            "network": [network] * 100,
            "config_fingerprint": [runner.fingerprint] * 100,
            "success": [True] * 100,
        }
    )
    runner.paths.queries(network).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(runner.paths.queries(network), index=False)
    with pytest.raises(RuntimeError, match="Missing or stale"):
        runner.aggregate()
    assert len(runner.aggregate(allow_partial=True)) == 100


def test_cifar_scaling_notebook_contains_required_sections():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks" / "CIFARResNetScaling.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"]).lower()
    for phrase in (
        "complete per-network results",
        "offline scaling",
        "lirpa coarseness",
        "counterfactual quality",
        "matched qualitative examples",
        "failures and resource limits",
        "300",
    ):
        assert phrase in source
