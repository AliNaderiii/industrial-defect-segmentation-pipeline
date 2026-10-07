"""Generate a small dashboard only from artifacts created by a real experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import PROJECT_ROOT


def build_dashboard(history_path: str | Path, test_metrics_path: str | Path, output_path: str | Path) -> None:
    """Create a reproducible training/test dashboard; no values are hard-coded."""
    history = json.loads(Path(history_path).read_text(encoding="utf-8"))
    test_metrics = json.loads(Path(test_metrics_path).read_text(encoding="utf-8"))
    epochs = [entry["epoch"] for entry in history["history"]]
    train_loss = [entry["train_loss"] for entry in history["history"]]
    validation_loss = [entry["validation_loss"] for entry in history["history"]]
    train_iou = [entry["train"]["foreground_iou"] for entry in history["history"]]
    validation_iou = [entry["validation"]["foreground_iou"] for entry in history["history"]]
    cm = np.asarray(test_metrics["confusion_matrix"])

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(epochs, train_loss, label="train", linewidth=2)
    axes[0].plot(epochs, validation_loss, label="validation", linewidth=2)
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Combined loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, train_iou, label="train crack IoU", linewidth=2)
    axes[1].plot(epochs, validation_iou, label="validation crack IoU", linewidth=2)
    axes[1].set(title="Foreground IoU", xlabel="Epoch", ylabel="IoU")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    image = axes[2].imshow(cm, cmap="Blues")
    figure.colorbar(image, ax=axes[2], fraction=0.046)
    axes[2].set(title="Held-out test confusion matrix", xlabel="Predicted", ylabel="Ground truth", xticks=[0, 1], yticks=[0, 1])
    axes[2].set_xticklabels(["background", "crack"])
    axes[2].set_yticklabels(["background", "crack"])
    for row in range(2):
        for column in range(2):
            axes[2].text(column, row, str(cm[row, column]), ha="center", va="center")
    figure.suptitle(
        f"DeepCrack experiment | held-out crack IoU: {test_metrics['foreground_iou']} | "
        f"held-out crack Dice: {test_metrics['foreground_dice']}"
    )
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", default=str(PROJECT_ROOT / "checkpoints" / "training_history.json"))
    parser.add_argument("--test-metrics", default=str(PROJECT_ROOT / "reports" / "test_metrics.json"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reports" / "experiment_dashboard.png"))
    args = parser.parse_args()
    build_dashboard(args.history, args.test_metrics, args.output)
