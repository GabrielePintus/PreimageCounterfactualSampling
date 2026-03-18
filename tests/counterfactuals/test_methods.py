import numpy as np
import pytest

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


def test_dice_binary_validity_threshold_for_target_zero_uses_positive_class_probability():
    method = DiceMethod(model=ThresholdModel(), random_seed=10)

    def fake_predict_proba(_x):
        return np.array([[0.70, 0.30], [0.76, 0.24]], dtype=np.float32)

    method._predict_proba_eval = fake_predict_proba  # type: ignore[method-assign]
    mask, scores = method._validity_mask_eval(np.zeros((2, 2), dtype=np.float32), target_class=0)

    assert np.array_equal(mask, np.array([False, True]))
    assert np.allclose(scores, np.array([0.30, 0.24], dtype=np.float32))


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


def test_nearest_neighbor_method_contract():
    _assert_method_works(NearestNeighborMethod(model=ThresholdModel(), random_seed=10))
