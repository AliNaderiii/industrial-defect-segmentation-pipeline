"""
Real Industrial Defect Dataset Loader - DeepCrack 2019
537 real crack images with manual annotations, multi-scale, multi-scene
Reference: DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation, Neurocomputing 2019
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from pathlib import Path

class DeepCrackDataset(Dataset):
    """
    Real DeepCrack Dataset - 537 RGB images with binary crack masks
    Train: 300 images, Test: 237 images
    All images manually annotated for crack segmentation
    """
    def __init__(self, root, split='train', transform=None, img_size=256):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.img_size = img_size
        
        if split == 'train':
            self.img_dir = self.root / "train_img"
            self.mask_dir = self.root / "train_lab"
        else:
            self.img_dir = self.root / "test_img"
            self.mask_dir = self.root / "test_lab"
        
        self.images = sorted(list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png")))
        if len(self.images) == 0:
            # Try with .JPG extension
            self.images = sorted(list(self.img_dir.glob("*.JPG")))
        
        print(f"DeepCrack {split}: {len(self.images)} real images found at {self.img_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        # Mask has same name but in mask_dir
        mask_path = self.mask_dir / img_path.name
        # If mask not found with same ext, try png
        if not mask_path.exists():
            mask_path = self.mask_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            mask_path = self.mask_dir / (img_path.stem + ".jpg")
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        
        # Load mask - binary
        if mask_path.exists():
            mask = Image.open(mask_path).convert('L')
            mask_np = np.array(mask)
            # Binarize: >127 -> 1 (crack), else 0
            mask_np = (mask_np > 127).astype(np.uint8)
        else:
            # Fallback - shouldn't happen with real data
            mask_np = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
        
        if self.transform:
            transformed = self.transform(image=img_np, mask=mask_np)
            img_np = transformed['image']
            mask_np = transformed['mask']
        
        return img_np, mask_np.long()

def get_transforms(train=True, img_size=256):
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

def get_dataloaders(batch_size=8, img_size=256, root='./data', num_workers=2):
    train_transform = get_transforms(train=True, img_size=img_size)
    val_transform = get_transforms(train=False, img_size=img_size)
    
    train_dataset = DeepCrackDataset(root=root, split='train', transform=train_transform, img_size=img_size)
    val_dataset = DeepCrackDataset(root=root, split='test', transform=val_transform, img_size=img_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=4, img_size=256, root='../data')
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    for img, mask in train_loader:
        print(f"Image: {img.shape}, Mask: {mask.shape}, Unique: {torch.unique(mask)}, Crack pixels: {(mask==1).sum()}/{mask.numel()} = {(mask==1).sum()/mask.numel()*100:.2f}%")
        break
