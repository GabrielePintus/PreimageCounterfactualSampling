"""Typed experiment configuration objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ComponentConfig:
    """Named component plus keyword arguments."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Root configuration for one benchmark run."""

    method: ComponentConfig
    model: ComponentConfig
    dataset: ComponentConfig
    metrics: List[ComponentConfig]
    seed: int = 42
    max_test_samples: int = 64


def parse_experiment_config(raw: Dict[str, Any]) -> ExperimentConfig:
    """Parse normalized YAML dict into a typed config object."""

    def parse_component(key: str) -> ComponentConfig:
        value = raw.get(key)
        if isinstance(value, str):
            return ComponentConfig(name=value, params={})
        if isinstance(value, dict) and "name" in value:
            return ComponentConfig(name=str(value["name"]), params=dict(value.get("params", {})))
        raise ValueError(f"Invalid '{key}' config. Use either a string or mapping with 'name'.")

    metrics_raw = raw.get("metrics", [])
    metrics: List[ComponentConfig] = []
    for m in metrics_raw:
        if isinstance(m, str):
            metrics.append(ComponentConfig(name=m, params={}))
        elif isinstance(m, dict) and "name" in m:
            metrics.append(ComponentConfig(name=str(m["name"]), params=dict(m.get("params", {}))))
        else:
            raise ValueError("Invalid metric entry. Use string or {'name': ..., 'params': ...}.")

    return ExperimentConfig(
        method=parse_component("method"),
        model=parse_component("model"),
        dataset=parse_component("dataset"),
        metrics=metrics,
        seed=int(raw.get("seed", 42)),
        max_test_samples=int(raw.get("max_test_samples", 64)),
    )
