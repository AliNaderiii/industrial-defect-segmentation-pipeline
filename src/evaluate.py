"""Evaluate a trained DeepCrack checkpoint exactly once on the held-out test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .config import PROJECT_ROOT, load_config
from .data_loader import get_dataloaders
from .metrics import SegmentationMeter
from .models import get_model


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = {"model_state_dict", "model", "dataset", "split_counts"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Unsupported checkpoint. Train it using the current pipeline first.")
    model_spec = payload["model"]
    model = get_model(
        model_spec["name"],
        num_classes=int(model_spec["num_classes"]),
        encoder=model_spec["encoder"],
        pretrained=False,
    )
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval(), payload


@torch.no_grad()
def evaluate_test(checkpoint_path: str | Path, config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Evaluate only DeepCrack's untouched official test folder."""
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(checkpoint_path, device)
    model_spec = payload["model"]
    if model_spec["num_classes"] != config["model"]["num_classes"]:
        raise ValueError("Checkpoint and config disagree about class count")
    loaders, manifest = get_dataloaders(
        data_root=PROJECT_ROOT / config["dataset"]["root"],
        batch_size=int(config["training"]["batch_size"]),
        image_size=int(payload["dataset"]["image_size"]),
        validation_fraction=float(config["dataset"]["validation_fraction"]),
        split_seed=int(config["dataset"]["split_seed"]),
        num_workers=int(config["runtime"]["num_workers"]),
    )
    meter = SegmentationMeter()
    for images, masks in tqdm(loaders["test"], desc="held-out test"):
        meter.update(model(images.to(device)), masks.to(device))
    results = meter.compute()
    results.update(
        {
            "split": "official_test",
            "checkpoint": str(checkpoint_path),
            "protocol": payload["dataset"]["protocol"],
            "split_counts": payload["split_counts"],
            "current_manifest_counts": {key: len(manifest[f"{key}_ids"]) for key in ("train", "validation", "test")},
        }
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    metrics = evaluate_test(args.checkpoint, args.config)
    output = PROJECT_ROOT / "reports" / "test_metrics.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
