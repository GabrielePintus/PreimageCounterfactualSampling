"""Configuration file helpers for experiment execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def read_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file into a Python dictionary."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at config root, got {type(data).__name__}")
    return data
