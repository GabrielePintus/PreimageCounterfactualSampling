from __future__ import annotations

import importlib
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

from counterfactuals.datasets.loaders import (
    AdultDataset,
    CompasDataset,
    GermanCreditDataset,
    GiveMeSomeCreditDataset,
    HELOCDataset,
    LendingClubDataset,
    WisconsinBreastCancerDataset,
)
from counterfactuals.preprocessing import snap_ohe_blocks
from dataset_specs import OHEBlockSpec, TABULAR_DATASET_SPECS, get_tabular_dataset_spec


DATASET_MODULES = {
    "adult": "training.datamodules.adult",
    "compas": "training.datamodules.compas",
    "german_credit": "training.datamodules.german_credit",
    "lending_club": "training.datamodules.lending_club",
    "heloc": "training.datamodules.heloc",
    "give_me_some_credit": "training.datamodules.give_me_some_credit",
    "wisconsin_breast_cancer": "training.datamodules.wisconsin_breast_cancer",
}

DATASET_ADAPTERS = {
    "adult": AdultDataset,
    "compas": CompasDataset,
    "german_credit": GermanCreditDataset,
    "lending_club": LendingClubDataset,
    "heloc": HELOCDataset,
    "give_me_some_credit": GiveMeSomeCreditDataset,
    "wisconsin_breast_cancer": WisconsinBreastCancerDataset,
}


def _load_benchmark_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dataset_name", sorted(TABULAR_DATASET_SPECS))
def test_tabular_spec_derived_fields(dataset_name: str):
    spec = get_tabular_dataset_spec(dataset_name)

    assert len(spec.input_types) == len(spec.feature_names)
    assert len(spec.feature_slices) == len(spec.feature_names)
    assert len(spec.ohe_feature_types) == spec.n_features
    assert spec.feature_slices[0][0] == 0
    assert all(
        spec.feature_slices[idx][1] == spec.feature_slices[idx + 1][0]
        for idx in range(len(spec.feature_slices) - 1)
    )
    assert tuple(end - start for start, end in spec.categorical_slices) == spec.cardinalities
    assert spec.ohe_blocks == tuple(
        OHEBlockSpec(start=start, end=end) for start, end in spec.categorical_slices
    )


@pytest.mark.parametrize("dataset_name", sorted(DATASET_MODULES))
def test_datamodule_constants_match_shared_spec(dataset_name: str):
    pytest.importorskip("lightning")
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")

    module = importlib.import_module(DATASET_MODULES[dataset_name])
    spec = get_tabular_dataset_spec(dataset_name)

    assert tuple(module.INPUT_TYPES) == spec.input_types
    assert tuple(module.CARDINALITIES) == spec.cardinalities
    assert tuple(module.OHE_FEATURE_TYPES) == spec.ohe_feature_types
    assert int(module.N_FEATURES) == spec.n_features


@pytest.mark.parametrize("dataset_name", sorted(DATASET_ADAPTERS))
def test_tabular_dataset_adapters_expose_shared_spec(dataset_name: str):
    assert DATASET_ADAPTERS[dataset_name].spec is get_tabular_dataset_spec(dataset_name)


def test_snap_ohe_blocks_with_shared_spec_preserves_shape_and_validity():
    spec = get_tabular_dataset_spec("compas")
    x = np.linspace(0.1, 0.9, spec.n_features, dtype=np.float32)
    snapped = snap_ohe_blocks(x, spec.ohe_blocks)

    assert snapped.shape == x.shape

    cat_indices: set[int] = set()
    for block in spec.ohe_blocks:
        cat_indices.update(range(block.start, block.end))
        values = snapped[block.start:block.end]
        assert np.all(np.isin(values, [0.0, 1.0]))
        assert np.isclose(values.sum(), 1.0)

    for idx in range(spec.n_features):
        if idx not in cat_indices:
            assert np.isclose(snapped[idx], x[idx])


def test_benchmark_uses_shared_specs_for_tabular_metadata():
    benchmark = _load_benchmark_module()
    spec = benchmark._get_tabular_spec("adult")

    assert spec is get_tabular_dataset_spec("adult")
    assert list(spec.cardinalities) == [7, 16, 7, 14, 6, 5, 2, 41]
    assert list(spec.categorical_slices) == [
        (1, 8),
        (9, 25),
        (26, 33),
        (33, 47),
        (47, 53),
        (53, 58),
        (58, 60),
        (63, 104),
    ]
    assert tuple(spec.ohe_blocks) == tuple(
        OHEBlockSpec(start=start, end=end) for start, end in spec.categorical_slices
    )
    assert benchmark._get_tabular_spec("mnist") is None


def test_benchmark_resolves_immutable_feature_dims_from_shared_spec():
    benchmark = _load_benchmark_module()

    dims, features = benchmark._resolve_immutable_feature_dims(
        dataset_name="adult",
        immutable_features=["sex"],
        preprocessing_name="identity",
    )

    assert features == ["sex"]
    assert np.array_equal(dims, np.array([58, 59], dtype=np.int64))


def test_benchmark_rejects_unknown_immutable_feature():
    benchmark = _load_benchmark_module()

    with pytest.raises(ValueError, match="Unknown immutable feature"):
        benchmark._resolve_immutable_feature_dims(
            dataset_name="adult",
            immutable_features=["not_a_feature"],
            preprocessing_name="identity",
        )


def test_benchmark_rejects_immutable_features_with_non_identity_preprocessing():
    benchmark = _load_benchmark_module()

    with pytest.raises(ValueError, match="identity preprocessing"):
        benchmark._resolve_immutable_feature_dims(
            dataset_name="adult",
            immutable_features=["sex"],
            preprocessing_name="pca",
        )


def test_benchmark_resolves_directional_feature_dims_from_shared_spec():
    benchmark = _load_benchmark_module()

    inc_dims, dec_dims, inc_features, dec_features = benchmark._resolve_directional_feature_dims(
        dataset_name="compas",
        nondecreasing_features=["age"],
        nonincreasing_features=[],
        preprocessing_name="identity",
    )

    assert inc_features == ["age"]
    assert dec_features == []
    assert np.array_equal(inc_dims, np.array([0], dtype=np.int64))
    assert dec_dims is None


def test_benchmark_rejects_categorical_directional_feature():
    benchmark = _load_benchmark_module()

    with pytest.raises(ValueError, match="Categorical feature"):
        benchmark._resolve_directional_feature_dims(
            dataset_name="compas",
            nondecreasing_features=["sex"],
            nonincreasing_features=[],
            preprocessing_name="identity",
        )


def test_benchmark_rejects_unknown_directional_feature():
    benchmark = _load_benchmark_module()

    with pytest.raises(ValueError, match="Unknown directional feature"):
        benchmark._resolve_directional_feature_dims(
            dataset_name="compas",
            nondecreasing_features=["not_a_feature"],
            nonincreasing_features=[],
            preprocessing_name="identity",
        )


def test_benchmark_rejects_directional_features_with_non_identity_preprocessing():
    benchmark = _load_benchmark_module()

    with pytest.raises(ValueError, match="identity preprocessing"):
        benchmark._resolve_directional_feature_dims(
            dataset_name="compas",
            nondecreasing_features=["age"],
            nonincreasing_features=[],
            preprocessing_name="pca",
        )


def test_benchmark_keeps_directional_feature_lists_as_atomic_grid_params():
    benchmark = _load_benchmark_module()

    expanded = benchmark._expand_grid([
        {
            "name": "nearest_neighbor",
            "run_name": "nn",
            "params": {"norm": [1, 2], "nondecreasing_features": ["age"]},
        }
    ])

    assert len(expanded) == 2
    assert [entry["params"]["norm"] for entry in expanded] == [1, 2]
    assert all(entry["params"]["nondecreasing_features"] == ["age"] for entry in expanded)


def test_benchmark_attaches_immutable_constraints_to_nn_and_gs():
    benchmark = _load_benchmark_module()

    nn_params = benchmark._attach_immutable_feature_constraints(
        method_name="nearest_neighbor",
        method_params={"norm": 1, "immutable_features": ["sex"]},
        dataset_name="adult",
        preprocessing_name="identity",
    )
    gs_params = benchmark._attach_immutable_feature_constraints(
        method_name="growing_spheres",
        method_params={"immutable_features": ["sex"]},
        dataset_name="adult",
        preprocessing_name="identity",
    )

    assert nn_params["norm"] == 1
    assert nn_params["immutable_features"] == ["sex"]
    assert np.array_equal(nn_params["fixed_dims"], np.array([58, 59], dtype=np.int64))
    assert np.array_equal(gs_params["fixed_dims"], np.array([58, 59], dtype=np.int64))


def test_benchmark_attaches_directional_constraints_to_supported_methods():
    benchmark = _load_benchmark_module()

    params = benchmark._attach_feature_constraints(
        method_name="face",
        method_params={
            "immutable_features": ["sex"],
            "nondecreasing_features": ["age"],
        },
        dataset_name="compas",
        preprocessing_name="identity",
    )

    assert params["immutable_features"] == ["sex"]
    assert np.array_equal(params["fixed_dims"], np.array([4, 5], dtype=np.int64))
    assert params["nondecreasing_features"] == ["age"]
    assert np.array_equal(params["nondecreasing_dims"], np.array([0], dtype=np.int64))
    assert params["nonincreasing_dims"] is None


def test_benchmark_rejects_immutable_constraints_for_unsupported_method():
    benchmark = _load_benchmark_module()

    with pytest.raises(ValueError, match="not supported for method 'dice'"):
        benchmark._attach_immutable_feature_constraints(
            method_name="dice",
            method_params={"immutable_features": ["sex"]},
            dataset_name="adult",
            preprocessing_name="identity",
        )


def test_no_counterfactual_code_imports_datamodule_constants():
    root = Path(__file__).resolve().parents[2]
    forbidden = re.compile(
        r"from\s+training\.datamodules\.[\w_]+\s+import\s+.*\b(INPUT_TYPES|CARDINALITIES|OHE_FEATURE_TYPES)\b"
    )

    checked_paths = list((root / "src" / "counterfactuals").rglob("*.py"))
    checked_paths.append(root / "scripts" / "benchmark.py")
    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        assert forbidden.search(text) is None, f"Datamodule constants import leaked into {path}"


def test_no_dataset_specific_ohe_branching_remains_in_benchmark():
    benchmark_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark.py"
    text = benchmark_path.read_text(encoding="utf-8")

    assert "adult_ohe_blocks" not in text
    assert "compas_ohe_blocks" not in text
    assert "german_credit_ohe_blocks" not in text
    assert "lending_club_ohe_blocks" not in text
    assert "spec.ohe_blocks" in text
    assert "spec.categorical_slices" in text
