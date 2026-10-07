"""Validate a manually obtained DeepCrack dataset without downloading unknown files."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import PROJECT_ROOT
from .data_loader import discover_records


def validate_deepcrack(root: str | Path) -> None:
    root = Path(root)
    print("DeepCrack must be obtained under its own non-commercial research terms.")
    print("Expected directories: train_img/, train_lab/, test_img/, test_lab/.")
    train = discover_records(root, "train")
    test = discover_records(root, "test")
    print(f"Validated {len(train)} labelled train pair(s) and {len(test)} labelled official test pair(s).")
    print("The pipeline derives validation only from train; test remains held out until evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT / "data"))
    args = parser.parse_args()
    validate_deepcrack(args.root)
