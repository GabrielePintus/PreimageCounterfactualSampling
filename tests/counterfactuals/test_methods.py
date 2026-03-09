import numpy as np

from counterfactuals.core.base_classes import CounterfactualExample
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
    model = ThresholdModel()
    method.fit(x_train=x_train, y_train=y_train, model=model)
    result = method.generate(CounterfactualExample(x=np.array([-0.8, 0.0], dtype=np.float32), target_class=1), model=model)
    assert result.x_cf.shape == (2,)
    assert isinstance(result.success, bool)


def test_wachter_method_contract():
    _assert_method_works(WachterMethod(max_iter=80, step_scale=0.25, random_seed=10))


def test_dice_method_contract():
    _assert_method_works(DiceMethod(num_candidates=128, random_seed=10))


def test_growing_spheres_contract():
    _assert_method_works(GrowingSpheresMethod(n_in_layer=128, max_radius=2.0, random_seed=10))


def test_face_method_contract():
    x_train, y_train = _train_data()
    model = ThresholdModel()
    method = FACEMethod(graph_mode="knn", n_neighbors=2, tp=0.5, td=0.0, random_seed=10)
    method.fit(x_train=x_train, y_train=y_train, model=model)
    result = method.generate(
        CounterfactualExample(x=np.array([-0.8, 0.0], dtype=np.float32), target_class=1),
        model=model,
    )
    assert result.x_cf.shape == (2,)
    assert isinstance(result.success, bool)


def test_nearest_neighbor_method_contract():
    _assert_method_works(NearestNeighborMethod(random_seed=10))
