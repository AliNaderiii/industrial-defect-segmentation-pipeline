"""Dataset-level metrics for binary crack segmentation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def _safe_mean(values: torch.Tensor) -> float:
    finite = values[~torch.isnan(values)]
    return float(finite.mean().item()) if finite.numel() else 0.0


def _maybe_float(value: torch.Tensor) -> float | None:
    return None if torch.isnan(value) else round(float(value.item()), 6)


@dataclass
class SegmentationMeter:
    """Accumulate a global pixel confusion matrix for a binary segmentation run.

    Rows represent ground-truth labels; columns represent predicted labels.
    Metrics are derived after all batches have been seen, preventing the
    batch-size bias of averaging per-batch mIoU or Dice values.
    """

    num_classes: int = 2
    confusion_matrix: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if self.num_classes != 2:
            raise ValueError("This meter is intentionally defined for binary segmentation")
        self.confusion_matrix = torch.zeros((2, 2), dtype=torch.int64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        if logits.ndim != 4 or target.ndim != 3:
            raise ValueError("Expected logits [B,C,H,W] and targets [B,H,W]")
        if logits.shape[1] != self.num_classes or logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
            raise ValueError("Logit/target shapes are incompatible")
        prediction = logits.argmax(dim=1).detach().to("cpu", dtype=torch.int64)
        target = target.detach().to("cpu", dtype=torch.int64)
        if target.numel() and (target.min() < 0 or target.max() >= self.num_classes):
            raise ValueError("Masks must contain only 0 (background) and 1 (crack)")
        encoded = self.num_classes * target.reshape(-1) + prediction.reshape(-1)
        self.confusion_matrix += torch.bincount(encoded, minlength=4).reshape(2, 2)

    def compute(self) -> dict[str, Any]:
        cm = self.confusion_matrix.to(torch.float64)
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        support = cm.sum(dim=1)
        union = tp + fp + fn
        iou = torch.where(union > 0, tp / union, torch.nan)
        dice_denominator = 2 * tp + fp + fn
        dice = torch.where(dice_denominator > 0, 2 * tp / dice_denominator, torch.nan)
        precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.nan)
        recall = torch.where(support > 0, tp / support, torch.nan)
        total = cm.sum()
        accuracy = float((tp.sum() / total).item()) if total else 0.0
        return {
            "pixel_accuracy": accuracy,
            "mean_iou": _safe_mean(iou),
            "mean_dice": _safe_mean(dice),
            "foreground_iou": _maybe_float(iou[1]),
            "foreground_dice": _maybe_float(dice[1]),
            "foreground_precision": _maybe_float(precision[1]),
            "foreground_recall": _maybe_float(recall[1]),
            "per_class_iou": [_maybe_float(value) for value in iou],
            "per_class_dice": [_maybe_float(value) for value in dice],
            "support": [int(value) for value in support.tolist()],
            "confusion_matrix": self.confusion_matrix.tolist(),
        }
