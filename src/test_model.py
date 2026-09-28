"""Tests for DeepCrack loader and model"""
import torch
from data_loader import get_dataloaders
from models import get_model

def test_dataloader():
    train_loader, val_loader = get_dataloaders(batch_size=2, img_size=128, root='../data')
    images, masks = next(iter(train_loader))
    assert images.shape[1] == 3
    assert masks.shape[1] == 128
    print(f"✓ Dataloader OK {images.shape} {masks.shape}")

def test_model():
    for name, enc in [('unet','resnet18'), ('deeplabv3plus','resnet50')]:
        model = get_model(name, 2, enc)
        x = torch.randn(1,3,128,128)
        y = model(x)
        assert y.shape[1]==2
        print(f"✓ {name} {enc} OK {y.shape}")

if __name__ == "__main__":
    test_dataloader()
    test_model()
    print("All tests passed!")
