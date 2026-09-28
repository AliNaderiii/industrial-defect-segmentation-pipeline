"""Download DeepCrack dataset"""
from pathlib import Path
import os

def download_deepcrack(root="./data"):
    """
    DeepCrack dataset - 537 real crack images
    Manual download from https://github.com/yhlleo/DeepCrack
    Place DeepCrack.zip in data/ and unzip, or use this script to guide
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    print(f"DeepCrack dataset should be in {root_path}")
    print("1. Download DeepCrack.zip from https://github.com/yhlleo/DeepCrack")
    print("2. Unzip to data/train_img, data/train_lab, data/test_img, data/test_lab")
    print("3. Expected: 300 train + 237 test = 537 images")
    
    # Check if exists
    train_img = root_path / "train_img"
    if train_img.exists():
        count = len(list(train_img.glob("*.jpg")) + list(train_img.glob("*.png")))
        print(f"Found {count} train images")
    else:
        print("Not found, please download manually")

if __name__ == "__main__":
    download_deepcrack()
