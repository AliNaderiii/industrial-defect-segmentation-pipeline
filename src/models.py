"""Binary crack-segmentation models and an imbalance-aware objective."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as functional

SUPPORTED_MODELS = {"unet", "deeplabv3plus", "fpn"}


def get_model(
    model_name: str,
    num_classes: int,
    encoder: str,
    pretrained: bool = True,
    in_channels: int = 3,
) -> nn.Module:
    """Create a SMP model with raw logits, never an implicit activation."""
    name = model_name.lower().replace("+", "plus")
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {sorted(SUPPORTED_MODELS)}")
    common = {
        "encoder_name": encoder,
        "encoder_weights": "imagenet" if pretrained else None,
        "in_channels": in_channels,
        "classes": num_classes,
        "activation": None,
    }
    if name == "unet":
        return smp.Unet(**common)
    if name == "deeplabv3plus":
        return smp.DeepLabV3Plus(**common)
    return smp.FPN(**common)


class ForegroundDiceLoss(nn.Module):
    """Soft Dice loss calculated for the crack class only.

    Averaging Dice across background and crack can conceal poor crack recall on
    this highly imbalanced dataset. Optimising the foreground channel makes the
    objective match the task's material-risk target.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] != 2:
            raise ValueError("ForegroundDiceLoss expects two output channels")
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        foreground = (targets == 1).to(dtype=probabilities.dtype)
        intersection = (probabilities * foreground).sum(dim=(1, 2))
        denominator = probabilities.sum(dim=(1, 2)) + foreground.sum(dim=(1, 2))
        return 1 - ((2 * intersection + self.smooth) / (denominator + self.smooth)).mean()


class CombinedLoss(nn.Module):
    """Foreground Dice plus weighted cross-entropy for sparse crack masks."""

    def __init__(
        self,
        dice_weight: float = 0.7,
        ce_weight: float = 0.3,
        foreground_class_weight: float = 3.0,
    ) -> None:
        super().__init__()
        if dice_weight < 0 or ce_weight < 0 or dice_weight + ce_weight == 0:
            raise ValueError("Loss weights must be non-negative and not both zero")
        if foreground_class_weight <= 0:
            raise ValueError("foreground_class_weight must be positive")
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = ForegroundDiceLoss()
        self.register_buffer("class_weights", torch.tensor([1.0, foreground_class_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = functional.cross_entropy(logits, targets, weight=self.class_weights)
        return self.dice_weight * self.dice(logits, targets) + self.ce_weight * cross_entropy


def count_parameters(model: nn.Module) -> float:
    """Return parameter count in millions."""
    return sum(parameter.numel() for parameter in model.parameters()) / 1_000_000
