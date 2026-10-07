"""Train DeepCrack models without using the official test set for selection."""
from __future__ import annotations

import argparse
import json
import platform
import random
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .config import PROJECT_ROOT, load_config
from .data_loader import get_dataloaders, write_split_manifest
from .metrics import SegmentationMeter
from .models import CombinedLoss, count_parameters, get_model


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    meter = SegmentationMeter()
    total_loss, total_items = 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, masks in tqdm(loader, desc="train" if training else "validate", leave=False):
            images, masks = images.to(device), masks.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().item()) * images.shape[0]
            total_items += images.shape[0]
            meter.update(logits, masks)
    if total_items == 0:
        raise RuntimeError("Encountered an empty data loader")
    return total_loss / total_items, meter.compute()


def checkpoint_payload(
    model: torch.nn.Module,
    config: dict[str, Any],
    split_manifest: dict[str, Any],
    epoch: int,
    validation_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "model_state_dict": model.state_dict(),
        "model": deepcopy(config["model"]),
        "dataset": {
            "name": "DeepCrack",
            "image_size": config["dataset"]["image_size"],
            "validation_fraction": config["dataset"]["validation_fraction"],
            "split_seed": config["dataset"]["split_seed"],
            "protocol": "train/validation split derived only from official train; official test held out",
        },
        "epoch": epoch,
        "validation_metrics": validation_metrics,
        "split_counts": {
            "train": len(split_manifest["train_ids"]),
            "validation": len(split_manifest["validation_ids"]),
            "test": len(split_manifest["test_ids"]),
        },
    }


def train(config: dict[str, Any]) -> dict[str, Any]:
    """Train on train, select on validation, and never inspect test metrics."""
    dataset, model_config, training, runtime = (
        config["dataset"],
        config["model"],
        config["training"],
        config["runtime"],
    )
    set_seed(int(runtime["seed"]), bool(runtime["deterministic"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    loaders, manifest = get_dataloaders(
        data_root=PROJECT_ROOT / dataset["root"],
        batch_size=int(training["batch_size"]),
        image_size=int(dataset["image_size"]),
        validation_fraction=float(dataset["validation_fraction"]),
        split_seed=int(dataset["split_seed"]),
        num_workers=int(runtime["num_workers"]),
    )
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    write_split_manifest(manifest, checkpoint_dir / "split_manifest.json")

    model = get_model(
        model_config["name"],
        num_classes=int(model_config["num_classes"]),
        encoder=model_config["encoder"],
        pretrained=bool(model_config["pretrained"]),
    ).to(device)
    print(f"Model: {model_config['name']}/{model_config['encoder']} ({count_parameters(model):.2f}M parameters)")
    criterion = CombinedLoss(
        dice_weight=float(training["loss_dice_weight"]),
        ce_weight=float(training["loss_ce_weight"]),
        foreground_class_weight=float(training["foreground_class_weight"]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=float(training["scheduler_factor"]), patience=int(training["scheduler_patience"])
    )

    history: list[dict[str, Any]] = []
    best_foreground_iou, epochs_without_improvement = float("-inf"), 0
    for epoch in range(1, int(training["epochs"]) + 1):
        train_loss, train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        validation_loss, validation_metrics = run_epoch(model, loaders["validation"], criterion, device)
        selection_metric = validation_metrics["foreground_iou"]
        if selection_metric is None:
            raise RuntimeError("Validation set contains no crack pixels; choose a different split seed or inspect labels")
        scheduler.step(selection_metric)
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d} | train loss={train_loss:.4f}, crack IoU={train_metrics['foreground_iou']} "
            f"| validation loss={validation_loss:.4f}, crack IoU={selection_metric}, crack Dice={validation_metrics['foreground_dice']}"
        )
        if selection_metric > best_foreground_iou:
            best_foreground_iou = selection_metric
            epochs_without_improvement = 0
            torch.save(checkpoint_payload(model, config, manifest, epoch, validation_metrics), checkpoint_dir / "best.pt")
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(training["early_stopping_patience"]):
            print("Early stopping: validation foreground IoU did not improve.")
            break

    torch.save(checkpoint_payload(model, config, manifest, history[-1]["epoch"], history[-1]["validation"]), checkpoint_dir / "last.pt")
    summary = {
        "protocol": "test split was not loaded for model selection or scheduler decisions",
        "best_validation_foreground_iou": best_foreground_iou,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device)},
        "history": history,
    }
    (checkpoint_dir / "training_history.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    result = train(load_config(args.config))
    print(json.dumps({"best_validation_foreground_iou": result["best_validation_foreground_iou"]}, indent=2))
