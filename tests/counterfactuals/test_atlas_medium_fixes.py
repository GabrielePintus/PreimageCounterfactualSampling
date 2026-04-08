from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from certcf.atlas import CertCFAtlas
from certcf.certification.lirpa import PreimageApproximation
from counterfactuals.methods.certcf import CertCF


class LinearRawLabelModel(torch.nn.Module):
    def forward(self, x):
        if x.ndim > 2:
            x = x.view(x.shape[0], -1)
        score = x[:, 1] - x[:, 0]
        return torch.stack([-score, score], dim=1)


class TinyTabularTorchModel:
    def __init__(self):
        self.model = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 2),
        )
        with torch.no_grad():
            self.model[1].weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                    ],
                    dtype=torch.float32,
                )
            )
            self.model[1].bias.zero_()
            self.model[3].weight.copy_(
                torch.tensor(
                    [
                        [1.0, -1.0, 0.0],
                        [-1.0, 1.0, 0.0],
                    ],
                    dtype=torch.float32,
                )
            )
            self.model[3].bias.zero_()
        self.device = "cpu"


class _DummyBVH:
    def __init__(self, x_cf: np.ndarray):
        self.n_polytopes = 1
        self.tree_depth = 1
        self._x_cf = np.asarray(x_cf, dtype=np.float64)

    def query_sorted_lower_bounds(self, x_query, eps_array, project_fn, distance_norm, stats_out):
        stats_out["n_candidates_considered"] = 1
        stats_out["n_candidates_total"] = 1
        stats_out["n_candidates_pruned_by_bound"] = 0
        stats_out["best_lower_bound_at_termination"] = 0.0
        return self._x_cf.copy(), 0.0, 0, 0

    def query_nearest(self, x_query, project_fn, distance_norm, stats_out):
        stats_out["n_nodes_popped"] = 1
        stats_out["n_nodes_pruned"] = 0
        stats_out["n_leaves_visited"] = 1
        stats_out["max_queue_size"] = 1
        stats_out["n_candidates_considered"] = 1
        return self._x_cf.copy(), 0.0, 0, 0


def _manual_atlas(
    *,
    centers_by_label: dict[int, np.ndarray] | None = None,
    eps: float = 1.0,
    ohe_slices=None,
):
    atlas = CertCFAtlas.__new__(CertCFAtlas)
    atlas.model = LinearRawLabelModel()
    atlas.device = torch.device("cpu")
    atlas.cnn = False
    atlas.norm = 2
    atlas.distance_norm = 2
    atlas.ohe_slices = ohe_slices
    atlas.solver_maxiter = 100
    atlas.class_labels = [2, 5]
    atlas.label_to_index = {2: 0, 5: 1}
    atlas.n_classes = 2
    atlas.model_input_shape = (3,)
    atlas._class_unions = None

    if centers_by_label is None:
        centers_by_label = {
            2: np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
            5: np.array([[0.0, 1.0, 0.0]], dtype=np.float64),
        }

    bounds = {}
    for label, centers in centers_by_label.items():
        centers = np.asarray(centers, dtype=np.float64)
        n_anchors, dim = centers.shape
        bounds[label] = {
            "lA": np.zeros((n_anchors, 1, dim), dtype=np.float64),
            "lbias": np.ones((n_anchors, 1), dtype=np.float64),
            "uA": np.zeros((n_anchors, 1, dim), dtype=np.float64),
            "ubias": np.ones((n_anchors, 1), dtype=np.float64),
            "X": centers,
            "eps": np.full(n_anchors, eps, dtype=np.float64),
        }
    atlas.bounds = bounds
    atlas.bvh_indices = {
        label: _DummyBVH(centers[0]) for label, centers in centers_by_label.items()
    }
    return atlas


def test_certcf_method_forwards_delta_and_robust_norm():
    calls = []

    class FakeAtlas:
        def find_counterfactual(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                x_cf=np.array([0.0, 1.0], dtype=np.float32),
                success=True,
                distance=1.25,
                profiling={"ok": True},
            )

    method = CertCF(model=object(), delta=0.3, robust_norm="inf", random_seed=7)
    method._is_fitted = True
    method.atlas = FakeAtlas()

    result = method.generate(np.array([0.2, 0.8], dtype=np.float32), target_class=5)

    assert result.success is True
    assert calls[0]["target_class"] == 5
    assert np.isclose(calls[0]["delta"], 0.3)
    assert calls[0]["robust_norm"] == np.inf


def test_certcf_rejects_invalid_subsample_space():
    with pytest.raises(ValueError, match="subsample_space must be one of"):
        CertCF(model=TinyTabularTorchModel(), subsample_space="bad-space")


def test_certcf_fit_uses_input_space_for_subsampling(monkeypatch):
    seen_spaces = []

    def fake_selector(X, k, method, random_state):
        del k, method, random_state
        seen_spaces.append(np.array(X, copy=True))
        return np.array([0], dtype=np.int64)

    monkeypatch.setattr("counterfactuals.utils.clustering.select_prototype_indices", fake_selector)
    monkeypatch.setattr(CertCF, "_fit", lambda self: None)

    x_train = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1], dtype=np.int64)
    method = CertCF(model=TinyTabularTorchModel(), k_per_class=1, subsample_method="kmedoids", subsample_space="input")

    method.fit(x_train=x_train, y_train=y_train)

    assert len(seen_spaces) == 2
    assert all(space.shape[1] == x_train.shape[1] for space in seen_spaces)
    assert np.allclose(method._x_train, np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    assert np.array_equal(method._y_train, np.array([0, 1], dtype=np.int64))


def test_certcf_fit_uses_penultimate_latent_space_for_subsampling(monkeypatch):
    seen_spaces = []

    def fake_selector(X, k, method, random_state):
        del k, method, random_state
        seen_spaces.append(np.array(X, copy=True))
        return np.array([0], dtype=np.int64)

    monkeypatch.setattr("counterfactuals.utils.clustering.select_prototype_indices", fake_selector)
    monkeypatch.setattr(CertCF, "_fit", lambda self: None)

    x_train = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1], dtype=np.int64)
    method = CertCF(model=TinyTabularTorchModel(), k_per_class=1, subsample_method="kmedoids", subsample_space="latent")

    method.fit(x_train=x_train, y_train=y_train)

    assert len(seen_spaces) == 2
    assert all(space.shape[1] == 3 for space in seen_spaces)
    assert seen_spaces[0].shape != x_train[y_train == 0].shape
    assert seen_spaces[1].shape != x_train[y_train == 1].shape
    assert np.allclose(method._x_train, np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    assert np.array_equal(method._y_train, np.array([0, 1], dtype=np.int64))


def test_certcf_fit_rejects_latent_subsampling_for_unsupported_models(monkeypatch):
    class UnsupportedTorchModel:
        def __init__(self):
            self.model = torch.nn.Linear(2, 2)
            self.device = "cpu"

    monkeypatch.setattr(CertCF, "_fit", lambda self: None)

    method = CertCF(model=UnsupportedTorchModel(), k_per_class=1, subsample_space="latent")

    with pytest.raises(ValueError, match="requires the wrapped model to be an nn.Sequential"):
        method.fit(
            x_train=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            y_train=np.array([0, 1], dtype=np.int64),
        )


def test_l2_ray_step_stays_inside_ball():
    ref_offset = np.array([0.9, 0.0], dtype=np.float64)
    direction = np.array([0.2, 0.0], dtype=np.float64)

    step = CertCFAtlas._largest_l2_ray_step(ref_offset, direction, ball_eps=1.0)
    candidate = ref_offset + step * direction

    assert np.isclose(step, 0.5, atol=1e-9)
    assert np.linalg.norm(candidate, ord=2) <= 1.0 + 1e-9


def test_projection_initial_guess_preserves_fixed_dims_and_ohe_assignments():
    atlas = _manual_atlas(ohe_slices=[(0, 2)])
    x_query = np.array([0.7, 0.3, 2.5], dtype=np.float64)
    center = np.array([0.2, 0.8, 0.0], dtype=np.float64)
    guess = atlas._projection_initial_guess(
        x_query,
        A_full=np.zeros((1, 3), dtype=np.float64),
        b_full=np.ones(1, dtype=np.float64),
        center=center,
        box_eps=10.0,
        ball_eps=10.0,
        fixed_dims=np.array([2], dtype=np.int64),
        ohe_slices=[(0, 2)],
        fixed_ohe_assignments={0: 1},
    )

    assert np.allclose(guess[:2], np.array([0.0, 1.0]))
    assert np.isclose(guess[2], x_query[2])


def test_verify_counterfactual_reports_prediction_ohe_and_polytope_diagnostics():
    atlas = _manual_atlas(ohe_slices=[(0, 2)])
    x_cf = np.array([0.4, 0.6, 0.0], dtype=np.float32)

    report = atlas.verify_counterfactual(x_cf, target_class=5)

    assert report["predicted"] == 5
    assert report["prediction_valid"] is True
    assert report["ohe_valid"] is False
    assert report["in_certified_polytope"] is True
    assert report["robustly_certified"] is True
    assert report["overall_valid"] is False
    assert report["valid"] is False


def test_verify_counterfactual_can_fail_robust_certification():
    atlas = _manual_atlas(ohe_slices=None, eps=1.0)
    x_cf = np.array([0.25, 0.75, 0.0], dtype=np.float32)

    report = atlas.verify_counterfactual(x_cf, target_class=5, delta=0.8, robust_norm=2)

    assert report["prediction_valid"] is True
    assert report["in_certified_polytope"] is True
    assert report["robustly_certified"] is False
    assert report["overall_valid"] is False


def test_verify_counterfactual_anchor_idx_does_not_silently_scan_other_anchors():
    atlas = _manual_atlas(
        centers_by_label={
            2: np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
            5: np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        },
        eps=0.2,
        ohe_slices=None,
    )
    x_cf = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    report = atlas.verify_counterfactual(x_cf, target_class=5, anchor_idx=1)

    assert report["prediction_valid"] is True
    assert report["in_certified_polytope"] is False
    assert report["verified_anchor_idx"] is None
    assert report["overall_valid"] is False


def test_preimage_approximation_preserves_raw_label_keys(monkeypatch):
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(
            [[0.0, 1.0], [0.1, 0.9], [1.0, 0.0], [0.9, 0.1]],
            dtype=torch.float32,
        ),
        torch.tensor([2, 2, 5, 5], dtype=torch.int64),
    )
    seen_label_indices = []

    def fake_run_lirpa(model, label, X, n_classes, device, eps, norm, dtype, lirpa_method):
        seen_label_indices.append(int(label))
        n_samples, dim = X.shape
        zeros = np.zeros((n_samples, n_classes - 1, dim), dtype=np.float32)
        ones = np.ones((n_samples, n_classes - 1), dtype=np.float32)
        return zeros, ones, zeros, ones

    monkeypatch.setattr("certcf.certification.lirpa.run_lirpa", fake_run_lirpa)

    preimage = PreimageApproximation(
        model=torch.nn.Linear(2, 2),
        dataset=dataset,
        device=torch.device("cpu"),
        cnn=False,
    )
    bounds = preimage.compute_all_bounds(eps=0.2, norm=2)

    assert preimage.class_labels == [2, 5]
    assert set(bounds.keys()) == {2, 5}
    assert seen_label_indices == [0, 1]


def test_preimage_approximation_forwards_lirpa_method(monkeypatch):
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(
            [[0.0, 1.0], [0.1, 0.9], [1.0, 0.0], [0.9, 0.1]],
            dtype=torch.float32,
        ),
        torch.tensor([2, 2, 5, 5], dtype=torch.int64),
    )
    seen_methods = []

    def fake_run_lirpa(model, label, X, n_classes, device, eps, norm, dtype, lirpa_method):
        seen_methods.append(lirpa_method)
        n_samples, dim = X.shape
        zeros = np.zeros((n_samples, n_classes - 1, dim), dtype=np.float32)
        ones = np.ones((n_samples, n_classes - 1), dtype=np.float32)
        return zeros, ones, zeros, ones

    monkeypatch.setattr("certcf.certification.lirpa.run_lirpa", fake_run_lirpa)

    preimage = PreimageApproximation(
        model=torch.nn.Linear(2, 2),
        dataset=dataset,
        device=torch.device("cpu"),
        cnn=False,
    )
    preimage.compute_all_bounds(eps=0.2, norm=2, lirpa_method="alpha-crown")

    assert seen_methods == ["alpha-crown", "alpha-crown"]


def test_certcf_method_forwards_lirpa_method_to_atlas(monkeypatch):
    init_calls = []

    class FakeAtlas:
        def __init__(self, *args, **kwargs):
            init_calls.append(kwargs)

        def build(self):
            return self

    monkeypatch.setattr("counterfactuals.methods.certcf.CertCFAtlas", FakeAtlas)

    class TinyTorchModel:
        def __init__(self):
            self.model = torch.nn.Sequential(torch.nn.Linear(2, 2))
            self.device = "cpu"

    method = CertCF(model=TinyTorchModel(), lirpa_method="alpha-crown", random_seed=7)
    method.fit(
        x_train=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        y_train=np.array([0, 1], dtype=np.int64),
    )

    assert init_calls
    assert init_calls[0]["lirpa_method"] == "alpha-crown"


def test_find_counterfactual_supports_non_contiguous_target_labels():
    atlas = _manual_atlas()

    result = atlas.find_counterfactual(
        x_query=np.array([0.2, 0.8, 0.0], dtype=np.float32),
        target_class=5,
        method="sorted",
    )

    assert result.success is True
    assert result.target_class == 5
    assert np.allclose(result.x_cf, np.array([0.0, 1.0, 0.0], dtype=np.float64))


def test_find_counterfactual_rejects_unknown_target_label():
    atlas = _manual_atlas()

    with pytest.raises(ValueError, match="Unknown target_class 7"):
        atlas.find_counterfactual(
            x_query=np.array([0.2, 0.8, 0.0], dtype=np.float32),
            target_class=7,
            method="sorted",
        )
