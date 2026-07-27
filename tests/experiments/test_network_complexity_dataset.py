import json

import numpy as np

from counterfactuals.datasets.loaders import NetworkComplexityDataset
from training.datamodules.network_complexity import (
    NetworkComplexityDataModule,
    prepare_network_complexity_dataset,
)


def test_deterministic_stratified_train_scaled_dataset_and_fixed_queries(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    kwargs = dict(
        n_train=800,
        n_validation=100,
        n_test=100,
        n_features=32,
        seed=42,
        n_queries=40,
        queries_per_class=20,
    )
    metadata_first = prepare_network_complexity_dataset(first, **kwargs)
    metadata_second = prepare_network_complexity_dataset(second, **kwargs)
    assert metadata_first["fingerprint"] == metadata_second["fingerprint"]
    assert metadata_first["split_sizes"] == {"train": 800, "validation": 100, "test": 100}
    assert metadata_first["split_class_counts"]["train"] == [400, 400]
    assert metadata_first["split_class_counts"]["validation"] == [50, 50]
    assert metadata_first["split_class_counts"]["test"] == [50, 50]
    assert metadata_first["split_class_counts"]["queries"] == [20, 20]
    assert metadata_first["rows_generated"] == 1_000
    assert metadata_first["retained_rows"] == 1_000

    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        for key in (
            "x_train",
            "y_train",
            "x_validation",
            "y_validation",
            "x_test",
            "y_test",
            "query_indices",
        ):
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_allclose(a["x_train"].mean(axis=0), 0.0, atol=2e-6)
        np.testing.assert_allclose(a["x_train"].std(axis=0), 1.0, atol=2e-6)
        expected_test = (a["x_test_raw"] - a["scaler_mean"]) / a["scaler_scale"]
        np.testing.assert_allclose(a["x_test"], expected_test, rtol=2e-6, atol=2e-6)
        parsed = json.loads(str(a["metadata_json"].item()))
        assert parsed["scaler_fit_split"] == "train"


def test_lightning_and_benchmark_load_exact_same_cache(tmp_path):
    cache = tmp_path / "dataset.npz"
    prepare_network_complexity_dataset(
        cache,
        n_train=400,
        n_validation=50,
        n_test=50,
        n_features=32,
        n_queries=20,
        queries_per_class=10,
    )
    dm = NetworkComplexityDataModule(
        cache_path=str(cache),
        n_train=400,
        n_validation=50,
        n_test=50,
        n_features=32,
        n_queries=20,
        queries_per_class=10,
        num_workers=0,
    )
    dm.setup()
    benchmark_dataset = NetworkComplexityDataset(cache_path=str(cache))
    benchmark_dataset.load()
    x_train, y_train = benchmark_dataset.get_train()
    x_test, y_test = benchmark_dataset.get_test()
    np.testing.assert_array_equal(dm.train_ds.tensors[0].numpy(), x_train)
    np.testing.assert_array_equal(dm.train_ds.tensors[1].numpy(), y_train)
    np.testing.assert_array_equal(dm.test_ds.tensors[0].numpy(), x_test)
    np.testing.assert_array_equal(dm.test_ds.tensors[1].numpy(), y_test)
    np.testing.assert_array_equal(dm.query_indices, benchmark_dataset.query_indices)
