"""Load and validate the DeepCrack experiment configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML and reject configurations that would invalidate the protocol."""
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    required = {"dataset", "model", "training", "runtime"}
    missing = required.difference(config or {})
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")
    dataset = config["dataset"]
    if dataset.get("name") != "deepcrack":
        raise ValueError("This reference pipeline supports dataset.name: deepcrack")
    if not 0 < float(dataset.get("validation_fraction", 0)) < 1:
        raise ValueError("dataset.validation_fraction must be between 0 and 1")
    if int(config["model"].get("num_classes", 0)) != 2:
        raise ValueError("DeepCrack binary segmentation requires model.num_classes: 2")
    if int(dataset.get("image_size", 0)) < 32:
        raise ValueError("dataset.image_size must be at least 32")
    return config
