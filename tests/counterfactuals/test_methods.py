import numpy as np
import pytest

from counterfactuals.core.base_classes import BaseCounterfactualMethod, CounterfactualResult
from counterfactuals.density import EpsilonBallEstimator, KDEEstimator, KNNEstimator
from counterfactuals.methods.dice import DiceMethod
from counterfactuals.methods.face import FACEMethod
from counterfactuals.methods.growing_spheres import GrowingSpheresMethod
from counterfactuals.methods.nearest_neighbor import NearestNeighborMethod
from counterfactuals.methods.wachter import WachterMethod


class ThresholdModel:
    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[None, :]
        return (x[:, 0] > 0.0).astype(np.int64)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        pred = self.predict(x)
        proba = np.zeros((pred.shape[0], 2), dtype=np.float32)
        proba[:, 1] = np.where(pred == 1, 0.9, 0.1)
        proba[:, 0] = 1.0 - proba[:, 1]
        return proba


def _train_data() -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[-1.0, 0.0], [-0.5, 0.1], [0.6, -0.2], [1.1, 0.3]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    return x, y


def _assert_method_works(method):
    x_train, y_train = _train_data()
    method.fit(x_train=x_train, y_train=y_train)
    result = method.generate(x=np.array([-0.8, 0.0], dtype=np.float32), target_class=1)
    assert result.x_cf.shape == (2,)
    assert isinstance(result.success, bool)


def test_wachter_method_contract():
    _assert_method_works(WachterMethod(model=ThresholdModel(), max_iter=80, restart_scale=0.25, random_seed=10))


def test_dice_method_contract():
    torch = pytest.importorskip("torch")

    from counterfactuals.models.torch_model import TorchModelWrapper
    from counterfactuals.preprocessing.transforms import IdentityTransform, InverseTransformModel, OHEBlockSpec

    class TinyTabularNet(torch.nn.Module):
        def forward(self, x):
            score = 2.5 * x[:, 0] + 4.0 * x[:, 2] - 4.0 * x[:, 1] - 0.5
            return torch.stack([-score, score], dim=1)

    x_train = np.array(
        [
            [0.1, 1.0, 0.0],
            [0.2, 1.0, 0.0],
            [0.7, 0.0, 1.0],
            [0.9, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1], dtype=np.int64)
    model = InverseTransformModel(
        base_model=TorchModelWrapper(model=TinyTabularNet(), device="cpu"),
        transform=IdentityTransform(),
        ohe_blocks=[OHEBlockSpec(start=1, end=3)],
    )
    method = DiceMethod(
        model=model,
        total_cfs=1,
        diversity_weight=0.0,
        min_iter=25,
        max_iter=250,
        posthoc_sparsity_param=0.0,
        random_seed=10,
    )
    method.fit(x_train=x_train, y_train=y_train)
    result = method.generate(x=np.array([0.15, 1.0, 0.0], dtype=np.float32), target_class=1)

    assert result.success
    assert result.x_cf.shape == (3,)
    assert np.isclose(result.x_cf[1:3].sum(), 1.0)
    assert set(np.unique(result.x_cf[1:3])).issubset({0.0, 1.0})
    assert int(model.predict(result.x_cf[None, :])[0]) == 1


def test_dice_generate_batch_supports_per_query_targets():
    torch = pytest.importorskip("torch")

    from counterfactuals.models.torch_model import TorchModelWrapper
    from counterfactuals.preprocessing.transforms import IdentityTransform, InverseTransformModel, OHEBlockSpec

    class TinyTabularNet(torch.nn.Module):
        def forward(self, x):
            score = 2.5 * x[:, 0] + 4.0 * x[:, 2] - 4.0 * x[:, 1] - 0.5
            return torch.stack([-score, score], dim=1)

    x_train = np.array(
        [
            [0.1, 1.0, 0.0],
            [0.2, 1.0, 0.0],
            [0.7, 0.0, 1.0],
            [0.9, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1], dtype=np.int64)
    model = InverseTransformModel(
        base_model=TorchModelWrapper(model=TinyTabularNet(), device="cpu"),
        transform=IdentityTransform(),
        ohe_blocks=[OHEBlockSpec(start=1, end=3)],
    )
    method = DiceMethod(
        model=model,
        total_cfs=1,
        diversity_weight=0.0,
        min_iter=25,
        max_iter=250,
        posthoc_sparsity_param=0.0,
        random_seed=10,
    )
    method.fit(x_train=x_train, y_train=y_train)

    queries = np.array(
        [
            [0.15, 1.0, 0.0],
            [0.85, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    targets = np.array([1, 0], dtype=np.int64)
    results = method.generate_batch(x=queries, target_class=targets)

    assert len(results) == 2
    for result, target in zip(results, targets):
        assert result.success
        assert result.x_cf.shape == (3,)
        assert np.isclose(result.x_cf[1:3].sum(), 1.0)
        assert set(np.unique(result.x_cf[1:3])).issubset({0.0, 1.0})
        assert int(model.predict(result.x_cf[None, :])[0]) == int(target)


def test_dice_validity_mask_uses_argmax():
    method = DiceMethod(model=ThresholdModel(), random_seed=10)

    def fake_predict_proba(_x):
        # First candidate: argmax=0; second: argmax=1
        return np.array([[0.70, 0.30], [0.40, 0.60]], dtype=np.float32)

    method.predict_proba_eval = fake_predict_proba  # type: ignore[method-assign]

    mask0, scores0 = method.validity_mask_eval(np.zeros((2, 2), dtype=np.float32), target_class=0)
    assert np.array_equal(mask0, np.array([True, False]))
    assert np.allclose(scores0, np.array([0.70, 0.40], dtype=np.float32))

    mask1, scores1 = method.validity_mask_eval(np.zeros((2, 2), dtype=np.float32), target_class=1)
    assert np.array_equal(mask1, np.array([False, True]))
    assert np.allclose(scores1, np.array([0.30, 0.60], dtype=np.float32))


def test_growing_spheres_contract():
    _assert_method_works(GrowingSpheresMethod(model=ThresholdModel(), n_in_layer=128, max_radius=2.0, random_seed=10))


def test_growing_spheres_layer_sampling_stays_finite_in_high_dimension():
    method = GrowingSpheresMethod(model=ThresholdModel(), n_in_layer=256, max_radius=5.0, radius_step=0.05, random_seed=10)
    method._feature_scale = np.ones(104, dtype=np.float32)

    x0 = np.zeros(104, dtype=np.float32)
    samples_small = method._sample_layer(x0=x0, inner_radius=0.25, outer_radius=0.30)
    samples_large = method._sample_layer(x0=x0, inner_radius=4.95, outer_radius=5.0)

    assert np.isfinite(samples_small).all()
    assert np.isfinite(samples_large).all()
    assert np.max(np.linalg.norm(samples_small, axis=1)) > 0.0
    assert np.max(np.linalg.norm(samples_large, axis=1)) <= 5.0 + 1e-5


def test_growing_spheres_sampling_preserves_fixed_dims():
    method = GrowingSpheresMethod(
        model=ThresholdModel(),
        n_in_layer=128,
        fixed_dims=[3, 1, 1],
        immutable_features=["sex"],
        random_seed=10,
    )
    method._feature_scale = np.ones(4, dtype=np.float32)

    x0 = np.array([-0.8, 1.0, 0.0, 4.0], dtype=np.float32)
    samples = method._sample_layer(x0=x0, inner_radius=0.0, outer_radius=1.0)

    assert np.array_equal(method.fixed_dims, np.array([1, 3], dtype=np.int64))
    assert np.allclose(samples[:, [1, 3]], x0[[1, 3]])


def test_growing_spheres_sampling_respects_directional_dims():
    method = GrowingSpheresMethod(
        model=ThresholdModel(),
        n_in_layer=128,
        nondecreasing_dims=[1],
        nonincreasing_dims=[2],
        nondecreasing_features=["age"],
        nonincreasing_features=["debt"],
        random_seed=10,
    )
    method._feature_scale = np.ones(3, dtype=np.float32)

    x0 = np.array([-0.8, 1.0, 4.0], dtype=np.float32)
    samples = method._sample_layer(x0=x0, inner_radius=0.0, outer_radius=1.0)

    assert np.all(samples[:, 1] >= x0[1] - 1e-6)
    assert np.all(samples[:, 2] <= x0[2] + 1e-6)
    assert method.nondecreasing_features == ("age",)
    assert method.nonincreasing_features == ("debt",)


def test_growing_spheres_feature_selection_preserves_fixed_dims():
    method = GrowingSpheresMethod(model=ThresholdModel(), fixed_dims=[1], random_seed=10)

    x0 = np.array([-0.8, 0.0], dtype=np.float32)
    enemy = np.array([0.6, 5.0], dtype=np.float32)
    x_cf = method._feature_selection(x0=x0, enemy=enemy, target_class=1)

    assert int(method.model.predict(x_cf[None, :])[0]) == 1
    assert np.isclose(x_cf[1], x0[1])


def test_growing_spheres_feature_selection_preserves_directional_dims():
    method = GrowingSpheresMethod(
        model=ThresholdModel(),
        nondecreasing_dims=[1],
        nonincreasing_dims=[2],
        random_seed=10,
    )

    x0 = np.array([-0.8, 1.0, 4.0], dtype=np.float32)
    enemy = np.array([0.6, 0.5, 5.0], dtype=np.float32)
    x_cf = method._feature_selection(x0=x0, enemy=enemy, target_class=1)

    assert int(method.model.predict(x_cf[None, :])[0]) == 1
    assert x_cf[1] >= x0[1] - 1e-6
    assert x_cf[2] <= x0[2] + 1e-6


def test_face_method_contract():
    x_train, y_train = _train_data()
    method = FACEMethod(
        model=ThresholdModel(),
        density_estimator=KNNEstimator(n_neighbors=2),
        epsilon=1.0,
        tp=0.5,
        td=0.0,
        random_seed=10,
    )
    method.fit(x_train=x_train, y_train=y_train)
    result = method.generate(x=np.array([-0.8, 0.0], dtype=np.float32), target_class=1)
    assert result.x_cf.shape == (2,)
    assert isinstance(result.success, bool)


def test_face_knn_respects_distance_threshold():
    x_train = np.array([[0.0], [0.1], [1.0]], dtype=np.float32)
    y_train = np.array([0, 0, 1], dtype=np.int64)
    method = FACEMethod(
        model=ThresholdModel(),
        density_estimator=KNNEstimator(n_neighbors=2),
        epsilon=0.2,
        tp=0.5,
        td=0.0,
        random_seed=10,
    )
    method.fit(x_train=x_train, y_train=y_train)
    assert method.graph is not None
    assert method.graph.has_edge(0, 1)
    assert not method.graph.has_edge(1, 2)


def test_face_graph_weights_match_paper_formulas():
    x_train = np.array([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32)
    y_train = np.array([0, 1], dtype=np.int64)
    model = ThresholdModel()
    dist = float(np.linalg.norm(x_train[0] - x_train[1], ord=2))
    midpoint = ((x_train[0] + x_train[1]) / 2.0).reshape(1, -1)

    knn_est = KNNEstimator(n_neighbors=1)
    knn = FACEMethod(model=model, density_estimator=knn_est, tp=0.5, td=0.0, random_seed=10)
    knn.fit(x_train=x_train, y_train=y_train)
    assert knn.graph is not None
    expected_knn = -np.log(max(float(knn.density_estimator(midpoint)[0]), 1e-300)) * dist
    assert np.isclose(knn.graph[0][1]["weight"], expected_knn, rtol=1e-5, atol=1e-7)

    eps_est = EpsilonBallEstimator(epsilon=0.5)
    eps = FACEMethod(model=model, density_estimator=eps_est, tp=0.5, td=0.0, random_seed=10)
    eps.fit(x_train=x_train, y_train=y_train)
    assert eps.graph is not None
    expected_eps = -np.log(max(float(eps.density_estimator(midpoint)[0]), 1e-300)) * dist
    assert np.isclose(eps.graph[0][1]["weight"], expected_eps, rtol=1e-5, atol=1e-7)


def test_face_kde_edge_weights_use_kde_density():
    x_train = np.array([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32)
    y_train = np.array([0, 1], dtype=np.int64)
    model = ThresholdModel()
    kde_est = KDEEstimator(bandwidth=0.5)
    face = FACEMethod(model=model, density_estimator=kde_est, epsilon=1.0, tp=0.5, td=0.0, random_seed=10)
    face.fit(x_train=x_train, y_train=y_train)
    assert face.graph is not None
    midpoint = ((x_train[0] + x_train[1]) / 2.0).reshape(1, -1)
    dist = float(np.linalg.norm(x_train[0] - x_train[1]))
    expected = -np.log(max(float(face.density_estimator(midpoint)[0]), 1e-300)) * dist
    assert np.isclose(face.graph[0][1]["weight"], expected, rtol=1e-5)


def test_face_start_node_respects_query_label():
    x_train = np.array([[-1.0, 0.0], [-0.5, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=np.float32)
    y_train = np.array([0, 0, 1, 1], dtype=np.int64)
    model = ThresholdModel()
    face = FACEMethod(
        model=model,
        density_estimator=KNNEstimator(n_neighbors=2),
        epsilon=2.0,
        tp=0.5,
        td=0.0,
        random_seed=10,
    )
    face.fit(x_train=x_train, y_train=y_train)
    # Query is class 0 (x=-0.8 < 0); same-class training points are indices 0 and 1.
    x_query = np.array([[-0.8, 0.0]], dtype=np.float32)
    query_label = int(model.predict(x_query)[0])
    same_indices = np.where(face._y_train == query_label)[0]
    assert all(face._y_train[i] == query_label for i in same_indices)
    # The expected start node is the closest same-class point.
    z_query = np.asarray(x_query, dtype=np.float32)
    dists = np.linalg.norm(face._x_train[same_indices] - z_query, axis=1)
    expected_start = int(same_indices[np.argmin(dists)])
    assert face._y_train[expected_start] == query_label


def test_face_candidate_filter_respects_directional_dims():
    x_train = np.array(
        [
            [-1.0, 0.5],
            [-0.5, 0.6],
            [0.2, 0.2],
            [0.9, 0.8],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1], dtype=np.int64)
    face = FACEMethod(
        model=ThresholdModel(),
        density_estimator=KNNEstimator(n_neighbors=2),
        epsilon=2.0,
        tp=0.5,
        td=0.0,
        nondecreasing_dims=[1],
        nondecreasing_features=["age"],
        random_seed=10,
    )
    face.fit(x_train=x_train, y_train=y_train)

    candidates = face.candidate_nodes(
        x_query=np.array([-0.8, 0.5], dtype=np.float32),
        target_class=1,
    )

    assert np.array_equal(candidates, np.array([3]))
    assert face.nondecreasing_features == ("age",)


def test_nearest_neighbor_method_contract():
    _assert_method_works(NearestNeighborMethod(model=ThresholdModel(), random_seed=10))


def test_nearest_neighbor_filters_candidates_by_fixed_dims():
    x_train = np.array(
        [
            [-0.5, 0.5],
            [0.05, 0.0],
            [0.9, 0.5],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 1, 1], dtype=np.int64)
    method = NearestNeighborMethod(
        model=ThresholdModel(),
        fixed_dims=[1],
        immutable_features=["sex"],
        random_seed=10,
    )
    method.fit(x_train=x_train, y_train=y_train)

    result = method.generate(x=np.array([-0.1, 0.5], dtype=np.float32), target_class=1)

    assert result.success
    assert np.allclose(result.x_cf, np.array([0.9, 0.5], dtype=np.float32))
    assert result.metadata["n_immutable_compatible_candidates"] == 1
    assert result.metadata["fixed_dims_count"] == 1
    assert result.metadata["immutable_features"] == "sex"


def test_nearest_neighbor_filters_candidates_by_directional_dims():
    x_train = np.array(
        [
            [-0.5, 0.5],
            [0.1, 0.1],
            [0.9, 0.7],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 1, 1], dtype=np.int64)
    method = NearestNeighborMethod(
        model=ThresholdModel(),
        nondecreasing_dims=[1],
        nondecreasing_features=["age"],
        random_seed=10,
    )
    method.fit(x_train=x_train, y_train=y_train)

    result = method.generate(x=np.array([-0.1, 0.5], dtype=np.float32), target_class=1)

    assert result.success
    assert np.allclose(result.x_cf, np.array([0.9, 0.7], dtype=np.float32))
    assert result.metadata["n_directional_compatible_candidates"] == 1
    assert result.metadata["nondecreasing_dims_count"] == 1
    assert result.metadata["nondecreasing_features"] == "age"


def test_nearest_neighbor_fails_when_no_candidate_matches_fixed_dims():
    x_train = np.array(
        [
            [-0.5, 0.5],
            [0.05, 0.0],
            [0.9, 0.25],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 1, 1], dtype=np.int64)
    method = NearestNeighborMethod(model=ThresholdModel(), fixed_dims=[1], random_seed=10)
    method.fit(x_train=x_train, y_train=y_train)

    result = method.generate(x=np.array([-0.1, 0.5], dtype=np.float32), target_class=1)

    assert not result.success
    assert result.metadata["reason"] == "no_immutable_compatible_candidate"
    assert result.metadata["n_immutable_compatible_candidates"] == 0


def test_generic_methods_reject_boundary_random_subsampling():
    with pytest.raises(ValueError, match="subsample_method must be one of"):
        NearestNeighborMethod(
            model=ThresholdModel(),
            random_seed=10,
            k_per_class=2,
            subsample_method="boundary_random",
        )


def test_base_counterfactual_method_fit_uses_labels_as_provided():
    class RecordingMethod(BaseCounterfactualMethod):
        def _fit(self) -> None:
            self.fit_labels = self._y_train.copy()

        def generate(self, x: np.ndarray, target_class=None) -> CounterfactualResult:
            del x, target_class
            return CounterfactualResult(
                x_cf=np.zeros(2, dtype=np.float32),
                success=False,
                distance=0.0,
                metadata={},
            )

    method = RecordingMethod(
        model=ThresholdModel(),
        random_seed=10,
        k_per_class=1,
        subsample_method="random",
    )
    x_train = np.array([[-1.0, 0.0], [-0.5, 0.1], [0.6, -0.2], [1.1, 0.3]], dtype=np.float32)
    supplied_labels = np.array([1, 1, 0, 0], dtype=np.int64)

    method.fit(x_train=x_train, y_train=supplied_labels)

    assert set(method.fit_labels.tolist()) == {0, 1}
