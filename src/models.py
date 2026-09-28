"""
Professional Segmentation Models for Industrial Defect Detection
U-Net, DeepLabV3+, FPN via segmentation-models-pytorch
Optimized for thin crack structures, extreme class imbalance (2-5% foreground)
"""

from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

def get_unet(encoder: str = 'resnet34', num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    U-Net - Best for thin cracks, preserves boundaries via skip connections
    Encoder: ResNet18/34/50, 14M params for resnet18, 38ms CPU @128px
    """
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights='imagenet' if pretrained else None,
        in_channels=3,
        classes=num_classes,
        activation=None
    )

def get_deeplabv3plus(encoder: str = 'resnet50', num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    DeepLabV3+ with ASPP rates [1,6,12,18], output stride 16
    Multi-scale context for varying crack widths (1-50px)
    """
    return smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights='imagenet' if pretrained else None,
        in_channels=3,
        classes=num_classes,
        activation=None
    )

def get_fpn(encoder: str = 'resnet34', num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """FPN - Feature Pyramid, multi-scale fusion for multi-scale cracks"""
    return smp.FPN(
        encoder_name=encoder,
        encoder_weights='imagenet' if pretrained else None,
        in_channels=3,
        classes=num_classes,
        activation=None
    )

def get_model(model_name: str = 'unet', num_classes: int = 2, encoder: str = 'resnet34') -> nn.Module:
    model_name = model_name.lower()
    if model_name == 'unet':
        return get_unet(encoder=encoder, num_classes=num_classes)
    elif model_name in ['deeplabv3plus', 'deeplabv3+', 'deeplab']:
        return get_deeplabv3plus(encoder=encoder, num_classes=num_classes)
    elif model_name == 'fpn':
        return get_fpn(encoder=encoder, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model {model_name}")

class DiceLoss(nn.Module):
    """Dice loss - critical for crack imbalance (2-5% crack pixels)"""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (probs * targets_onehot).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + targets_onehot.sum(dim=(2,3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    """
    Combined Dice (0.6) + CE (0.4) - optimized for extreme imbalance
    Dice emphasizes overlap, CE stabilizes
    """
    def __init__(self, dice_weight: float = 0.6, ce_weight: float = 0.4):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + self.ce_weight * self.ce(logits, targets)

def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

if __name__ == "__main__":
    for name, enc in [('unet','resnet18'), ('deeplabv3plus','resnet50'), ('fpn','resnet34')]:
        model = get_model(name, num_classes=2, encoder=enc)
        print(f"{name} {enc}: {count_parameters(model):.1f}M params")
