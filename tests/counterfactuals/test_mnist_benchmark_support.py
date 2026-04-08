from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from counterfactuals.benchmarks.results import BenchmarkResult, MethodResult, QueryResult
from counterfactuals.datasets.loaders import MNISTDataset
from counterfactuals.methods.certcf import _strip_dropout_modules
from counterfactuals.models.torch_model import TorchModelWrapper


def _load_benchmark_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DummyMulticlassModel:
    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        return x[:, 0].astype(np.int64)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        pred = self.predict(x)
        proba = np.full((pred.shape[0], 10), 0.01, dtype=np.float32)
        proba[np.arange(pred.shape[0]), pred] = 0.91
        return proba


def test_build_query_tasks_expands_all_other_classes():
    benchmark = _load_benchmark_module()
    x_test = np.zeros((20, 4), dtype=np.float32)
    y_test = np.repeat(np.arange(10, dtype=np.int64), 2)
    x_test[:, 0] = y_test

    query_idx, y_true, y_orig, target_classes = benchmark._build_query_tasks(
        dataset_name="mnist",
        x_test=x_test,
        y_test=y_test,
        model=DummyMulticlassModel(),
        sampling_cfg={"balanced_per_class": 1, "target_policy": "all_other_classes"},
        rng=np.random.default_rng(0),
    )

    assert len(query_idx) == 90
    assert len(y_true) == 90
    assert len(y_orig) == 90
    assert len(target_classes) == 90
    assert set(np.unique(y_true)) == set(range(10))
    for source, target in zip(y_orig, target_classes):
        assert source != target


def test_torch_model_wrapper_reshapes_flat_inputs():
    torch = pytest.importorskip("torch")

    class ShapeNet(torch.nn.Module):
        def forward(self, x):
            assert x.shape[1:] == (1, 2, 2)
            score = x.view(x.shape[0], -1).sum(dim=1)
            return torch.stack([-score, score], dim=1)

    wrapper = TorchModelWrapper(model=ShapeNet(), device="cpu", input_shape=(1, 2, 2))
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    proba = wrapper.predict_proba(x)

    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_mnist_dataset_flattens_images(monkeypatch):
    torch = pytest.importorskip("torch")

    class FakeTensorTrainDataset:
        def __init__(self):
            self.tensors = (
                torch.arange(2 * 28 * 28, dtype=torch.float32).view(2, 1, 28, 28) / 255.0,
                torch.tensor([1, 2], dtype=torch.long),
            )

    class FakeVisionTestDataset:
        def __init__(self):
            self.data = torch.arange(2 * 28 * 28, dtype=torch.uint8).view(2, 28, 28)
            self.targets = torch.tensor([3, 4], dtype=torch.long)

    class FakeMNISTDataModule:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def setup(self, stage=None):
            del stage
            self.train_ds = FakeTensorTrainDataset()
            self.test_ds = FakeVisionTestDataset()

    fake_module = types.ModuleType("training.datamodules.mnist")
    fake_module.MNISTDataModule = FakeMNISTDataModule
    monkeypatch.setitem(sys.modules, "training.datamodules.mnist", fake_module)

    ds = MNISTDataset(data_dir="data/")
    ds.load()
    x_train, y_train = ds.get_train()
    x_test, y_test = ds.get_test()

    assert x_train.shape == (2, 784)
    assert x_test.shape == (2, 784)
    assert y_train.tolist() == [1, 2]
    assert y_test.tolist() == [3, 4]
    assert float(x_test.max()) <= 1.0


def test_benchmark_result_dataframe_includes_multiclass_columns():
    result = BenchmarkResult(
        dataset="mnist",
        seed=7,
        x_queries=np.array([[0.0, 1.0]], dtype=np.float32),
        y_orig=np.array([3], dtype=np.int64),
        y_true=np.array([3], dtype=np.int64),
        method_results=[
            MethodResult(
                method="nn",
                run_name="nn",
                params={},
                build_time_s=0.0,
                space="gen",
                query_results=[
                    QueryResult(
                        query_idx=11,
                        x_cf=np.array([1.0, 0.0], dtype=np.float32),
                        y_cf=8,
                        success=True,
                        runtime_s=0.1,
                        error=None,
                        l2_distance=1.0,
                        l1_distance=2.0,
                        l0_sparsity=0.5,
                        mad_l1_distance=2.0,
                        redundancy=0.0,
                        target_class=8,
                    )
                ],
            )
        ],
    )

    df = result.to_dataframe()
    assert {"source_class", "target_class", "y_true"}.issubset(df.columns)
    assert int(df.loc[0, "source_class"]) == 3
    assert int(df.loc[0, "target_class"]) == 8
    assert int(df.loc[0, "y_true"]) == 3


def test_strip_dropout_modules_preserves_custom_forward():
    torch = pytest.importorskip("torch")

    class TinyCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(1, 2, kernel_size=1),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5),
            )
            self.head = torch.nn.Linear(2 * 2 * 2, 3)

        def forward(self, x):
            h = self.encoder(x)
            h = h.view(h.shape[0], -1)
            return self.head(h)

    model = TinyCNN().eval()
    clean = _strip_dropout_modules(model).eval()
    x = torch.randn(4, 1, 2, 2)
    y = clean(x)

    assert y.shape == (4, 3)
    assert not any(isinstance(m, torch.nn.Dropout) for m in clean.modules())


def test_build_certcf_method_forwards_atlas_subsample_space(monkeypatch):
    torch = pytest.importorskip("torch")
    benchmark = _load_benchmark_module()
    init_calls = []

    class FakeLitModel:
        def __init__(self):
            self.net = torch.nn.Sequential(
                torch.nn.Linear(2, 3),
                torch.nn.ReLU(),
                torch.nn.Linear(3, 2),
            )

        def eval(self):
            return self

        def to(self, _device):
            return self

    class FakeCertCF:
        def __init__(self, **kwargs):
            init_calls.append(kwargs)

        def fit(self, x_train, y_train):
            self.x_train = x_train
            self.y_train = y_train

    monkeypatch.setattr(benchmark, "_load_lit_checkpoint_resilient", lambda *args, **kwargs: SimpleNamespace(model=FakeLitModel()))
    monkeypatch.setattr("counterfactuals.methods.certcf.CertCF", FakeCertCF)

    benchmark._build_certcf_method(
        params={
            "checkpoint": "checkpoints/adult_classifier/best.ckpt",
            "device": "cpu",
            "atlas_subsample_method": "kmedoids",
            "atlas_subsample_space": "latent",
            "k_per_class": 10,
        },
        model_params={
            "dataset_module": "adult",
            "hidden_dims": [32, 8],
            "dropout": 0.2,
        },
        dataset_name="adult",
        x_train=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        y_train=np.array([0, 1], dtype=np.int64),
        x_queries=np.array([[0.5, 0.5]], dtype=np.float32),
        seed=7,
    )

    assert init_calls
    assert init_calls[0]["subsample_space"] == "latent"
