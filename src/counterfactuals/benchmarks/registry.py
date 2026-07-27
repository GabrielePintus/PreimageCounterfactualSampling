"""Shared registry setup for benchmark scripts and analysis tooling."""

from __future__ import annotations

from typing import Dict

from counterfactuals.core.registry import Registry
from counterfactuals.datasets.loaders import (
    AdultDataset,
    CompasDataset,
    GermanCreditDataset,
    GiveMeSomeCreditDataset,
    HELOCDataset,
    LendingClubDataset,
    MNISTDataset,
    NetworkComplexityDataset,
    WisconsinBreastCancerDataset,
)
from counterfactuals.methods.certcf import CertCF
from counterfactuals.methods.dice import DiceMethod
from counterfactuals.methods.face import FACEMethod
from counterfactuals.methods.growing_spheres import GrowingSpheresMethod
from counterfactuals.methods.nearest_neighbor import NearestNeighborMethod
from counterfactuals.methods.wachter import WachterMethod
from counterfactuals.metrics.proximity import L2Proximity
from counterfactuals.metrics.sparsity import SparsityMetric
from counterfactuals.models.sklearn_model import SklearnModelWrapper, build_sklearn_mlp
from counterfactuals.models.torch_model import TorchModelWrapper
from counterfactuals.preprocessing import IdentityTransform, PCATransform


def create_default_registries() -> Dict[str, Registry]:
    """Create the default registries used by benchmark scripts."""
    method_registry: Registry = Registry("method")
    dataset_registry: Registry = Registry("dataset")
    model_registry: Registry = Registry("model")
    metric_registry: Registry = Registry("metric")
    preprocessing_registry: Registry = Registry("preprocessing")

    method_registry.register("wachter", WachterMethod)
    method_registry.register("dice", DiceMethod)
    method_registry.register("face", FACEMethod)
    method_registry.register("growing_spheres", GrowingSpheresMethod)
    method_registry.register("nearest_neighbor", NearestNeighborMethod)
    method_registry.register("certcf", CertCF)

    dataset_registry.register("adult", AdultDataset)
    dataset_registry.register("compas", CompasDataset)
    dataset_registry.register("german_credit", GermanCreditDataset)
    dataset_registry.register("heloc", HELOCDataset)
    dataset_registry.register("give_me_some_credit", GiveMeSomeCreditDataset)
    dataset_registry.register("lending_club", LendingClubDataset)
    dataset_registry.register("wisconsin_breast_cancer", WisconsinBreastCancerDataset)
    dataset_registry.register("mnist", MNISTDataset)
    dataset_registry.register("network_complexity", NetworkComplexityDataset)

    model_registry.register("sklearn_model", SklearnModelWrapper)
    model_registry.register("sklearn_mlp", build_sklearn_mlp)
    model_registry.register("torch_model", TorchModelWrapper)

    metric_registry.register("proximity", L2Proximity)
    metric_registry.register("sparsity", SparsityMetric)

    preprocessing_registry.register("identity", IdentityTransform)
    preprocessing_registry.register("pca", PCATransform)

    return {
        "method": method_registry,
        "dataset": dataset_registry,
        "model": model_registry,
        "metric": metric_registry,
        "preprocessing": preprocessing_registry,
    }
