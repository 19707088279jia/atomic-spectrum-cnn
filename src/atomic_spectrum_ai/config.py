"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a YAML configuration file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    required_sections = {"project", "data", "model", "training", "output"}
    missing = sorted(required_sections.difference(config))
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")

    classes = config["data"].get("classes")
    if not isinstance(classes, list) or len(classes) < 2:
        raise ValueError("data.classes must contain at least two class names")
    if len(classes) != len(set(classes)):
        raise ValueError("data.classes contains duplicate class names")

    return config
