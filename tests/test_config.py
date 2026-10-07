from pathlib import Path

import pytest

from src.config import load_config


def test_valid_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "dataset:\n  name: deepcrack\n  image_size: 128\n  validation_fraction: 0.2\n"
        "model:\n  num_classes: 2\ntraining: {}\nruntime: {}\n",
        encoding="utf-8",
    )
    assert load_config(path)["dataset"]["name"] == "deepcrack"


def test_invalid_validation_fraction_fails(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "dataset:\n  name: deepcrack\n  image_size: 128\n  validation_fraction: 1.0\n"
        "model:\n  num_classes: 2\ntraining: {}\nruntime: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="validation_fraction"):
        load_config(path)
