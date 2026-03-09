"""Config-driven runner for counterfactual benchmarking experiments."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np

from counterfactuals.core.base_classes import CounterfactualExample
from counterfactuals.core.interfaces import MetricInterface
from counterfactuals.core.registry import Registry
from counterfactuals.datasets.loaders import AdultDataset
from counterfactuals.experiments.config import ExperimentConfig, parse_experiment_config
from counterfactuals.methods.dice import DiceMethod
from counterfactuals.methods.face import FACEMethod
from counterfactuals.methods.growing_spheres import GrowingSpheresMethod
from counterfactuals.methods.nearest_neighbor import NearestNeighborMethod
from counterfactuals.methods.my_method import CertifiedAtlasMethod
from counterfactuals.methods.wachter import WachterMethod
from counterfactuals.metrics.plausibility import KNNPlausibility
from counterfactuals.metrics.proximity import L2Proximity
from counterfactuals.metrics.sparsity import SparsityMetric
from counterfactuals.metrics.validity import ValidityMetric
from counterfactuals.models.sklearn_model import SklearnModelWrapper, build_sklearn_mlp
from counterfactuals.models.torch_model import TorchModelWrapper
from counterfactuals.utils.config import read_yaml
from counterfactuals.utils.logging import get_logger
from counterfactuals.utils.seed import seed_everything

LOGGER = get_logger(__name__)


def create_default_registries() -> Dict[str, Registry]:
    """Create registries for methods, datasets, models, and metrics."""
    method_registry: Registry = Registry("method")
    dataset_registry: Registry = Registry("dataset")
    model_registry: Registry = Registry("model")
    metric_registry: Registry = Registry("metric")

    method_registry.register("wachter", WachterMethod)
    method_registry.register("dice", DiceMethod)
    method_registry.register("face", FACEMethod)
    method_registry.register("growing_spheres", GrowingSpheresMethod)
    method_registry.register("nearest_neighbor", NearestNeighborMethod)
    method_registry.register("my_method", CertifiedAtlasMethod)

    dataset_registry.register("adult", AdultDataset)

    model_registry.register("sklearn_model", SklearnModelWrapper)
    model_registry.register("sklearn_mlp", build_sklearn_mlp)
    model_registry.register("torch_model", TorchModelWrapper)

    metric_registry.register("proximity", L2Proximity)
    metric_registry.register("sparsity", SparsityMetric)

    return {
        "method": method_registry,
        "dataset": dataset_registry,
        "model": model_registry,
        "metric": metric_registry,
    }


def run_experiment(cfg: ExperimentConfig, registries: Optional[Dict[str, Registry]] = None) -> Dict[str, float]:
    """Run one benchmark and return averaged metrics."""
    registries = registries or create_default_registries()
    seed_everything(cfg.seed)

    dataset = registries["dataset"].create(cfg.dataset.name, **cfg.dataset.params)
    dataset.load()
    x_train, y_train = dataset.get_train()
    x_test, y_test = dataset.get_test()
    del y_test

    model = registries["model"].create(cfg.model.name, **cfg.model.params)
    _fit_model_if_needed(model=model, x_train=x_train, y_train=y_train)

    method = registries["method"].create(cfg.method.name, random_seed=cfg.seed, **cfg.method.params)
    method.fit(x_train=x_train, y_train=y_train, model=model)

    metrics = _build_metrics(cfg=cfg, registries=registries, model=model, x_train=x_train)
    n_samples = min(cfg.max_test_samples, len(x_test))

    rows: List[Dict[str, float]] = []
    for i in range(n_samples):
        x_orig = x_test[i]
        target_class = _default_target_class(x_orig=x_orig, model=model)
        result = method.generate(
            example=CounterfactualExample(x=x_orig, target_class=target_class),
            model=model,
        )
        row: Dict[str, float] = {
            "success": 1.0 if result.success else 0.0,
            "distance": float(result.distance),
        }
        context = {"target_class": target_class}
        for metric in metrics:
            row[metric.name] = float(metric.evaluate(x_orig=x_orig, x_cf=result.x_cf, context=context))
        rows.append(row)

    summary = _aggregate(rows)
    LOGGER.info("Experiment summary: %s", summary)
    return summary


def run_from_config_path(path: str) -> Dict[str, float]:
    """Load YAML config, parse schema, and execute the benchmark."""
    raw = read_yaml(path)
    cfg = parse_experiment_config(raw)
    LOGGER.info("Running with config: %s", asdict(cfg))
    return run_experiment(cfg)


def _build_metrics(
    cfg: ExperimentConfig,
    registries: Dict[str, Registry],
    model,
    x_train: np.ndarray,
) -> List[MetricInterface]:
    built: List[MetricInterface] = []
    for metric_cfg in cfg.metrics:
        if metric_cfg.name == "validity":
            built.append(ValidityMetric(model=model))
        elif metric_cfg.name == "plausibility":
            built.append(KNNPlausibility(x_train=x_train, **metric_cfg.params))
        else:
            built.append(registries["metric"].create(metric_cfg.name, **metric_cfg.params))
    return built


def _default_target_class(x_orig: np.ndarray, model) -> int:
    pred = int(model.predict(x_orig)[0])
    probs = model.predict_proba(x_orig)
    if probs.shape[1] != 2:
        raise ValueError("Multi-class tasks require explicit target_class in this runner")
    return 1 - pred


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = rows[0].keys()
    summary: Dict[str, float] = {}
    for key in keys:
        values = np.asarray([r[key] for r in rows], dtype=np.float32)
        summary[key] = float(np.mean(values))
    return summary


def _fit_model_if_needed(model, x_train: np.ndarray, y_train: np.ndarray) -> None:
    """Fit wrapped estimators when they expose sklearn-style fit method."""
    estimator = getattr(model, "estimator", None)
    if estimator is not None and hasattr(estimator, "fit"):
        estimator.fit(x_train, y_train)
