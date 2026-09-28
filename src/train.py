"""
Professional Training - Industrial Defect Segmentation on Real DeepCrack
537 real crack images, imbalance 2-5%, Combined Dice 0.6 + CE 0.4
"""

import argparse
from pathlib import Path
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import get_dataloaders
from models import get_model, CombinedLoss, count_parameters

try:
    from config import load_config
except ImportError:
    load_config = None

def calculate_iou(pred, target, num_classes=2):
    ious = []
    pred_cls = torch.argmax(pred, dim=1)
    for cls in range(num_classes):
        pm = (pred_cls == cls)
        tm = (target == cls)
        inter = (pm & tm).sum().float()
        union = (pm | tm).sum().float()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((inter/union).item())
    valid = [i for i in ious if not np.isnan(i)]
    return float(np.mean(valid)) if valid else 0.0, ious

def calculate_dice(pred, target, num_classes=2):
    dices = []
    pred_cls = torch.argmax(pred, dim=1)
    for cls in range(num_classes):
        pm = (pred_cls == cls)
        tm = (target == cls)
        inter = (pm & tm).sum().float()
        total = pm.sum().float() + tm.sum().float()
        if total == 0:
            dices.append(float('nan'))
        else:
            dices.append((2*inter/total).item())
    valid = [d for d in dices if not np.isnan(d)]
    return float(np.mean(valid)) if valid else 0.0, dices

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    tl, tm, td = 0,0,0
    for images, masks in tqdm(loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, masks)
        loss.backward()
        optimizer.step()
        miou,_ = calculate_iou(out.detach(), masks)
        mdice,_ = calculate_dice(out.detach(), masks)
        tl+=loss.item(); tm+=miou; td+=mdice
    n=len(loader)
    return tl/n, tm/n, td/n

def val_epoch(model, loader, criterion, device):
    model.eval()
    tl,tm,td=0,0,0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation", leave=False):
            images, masks = images.to(device), masks.to(device)
            out = model(images)
            loss = criterion(out, masks)
            miou,_ = calculate_iou(out, masks)
            mdice,_ = calculate_dice(out, masks)
            tl+=loss.item(); tm+=miou; td+=mdice
    n=len(loader)
    return tl/n, tm/n, td/n

def plot_history(history, model_name, save_dir):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history['train_loss'])+1)
    fig, axes = plt.subplots(1,3, figsize=(15,4))
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title(f'{model_name} Loss (Real DeepCrack)'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history['train_miou'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_miou'], 'r-', label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('mIoU'); axes[1].set_title(f'{model_name} mIoU (Real)'); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(epochs, history['train_dice'], 'b-', label='Train', linewidth=2)
    axes[2].plot(epochs, history['val_dice'], 'r-', label='Val', linewidth=2)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Dice'); axes[2].set_title(f'{model_name} Dice (Real)'); axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / f"training_curves_{model_name}_real.png", dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(8,5))
    plt.plot(epochs, history['train_dice'], 'b-o', label='Train Dice', linewidth=2)
    plt.plot(epochs, history['val_dice'], 'r-s', label='Val Dice', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Dice'); plt.title(f'{model_name} Dice Evolution (Real DeepCrack)'); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(save_path / f"dice_curve_{model_name}_real.png", dpi=300, bbox_inches='tight')
    plt.close()

def train_model(model_name='unet', encoder='resnet18', epochs=3, batch_size=4, img_size=128, lr=1e-4, save_dir=None, config=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if config:
        epochs = config['training']['epochs']
        batch_size = config['training']['batch_size']
        img_size = config['dataset']['img_size']
        lr = config['training']['lr']
        model_name = config['model']['name']
        encoder = config['model']['encoder']
    
    base = Path(__file__).parent.parent
    save_path = Path(save_dir) if save_dir else base / "models"
    data_root = base / "data"
    reports_root = base / "reports"
    save_path.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading real DeepCrack: img_size {img_size}, batch {batch_size}")
    train_loader, val_loader = get_dataloaders(batch_size=batch_size, img_size=img_size, root=str(data_root))
    
    print(f"Creating model: {model_name} {encoder} - {count_parameters(get_model(model_name,2,encoder)):.1f}M params")
    model = get_model(model_name, num_classes=2, encoder=encoder).to(device)
    criterion = CombinedLoss(dice_weight=0.6, ce_weight=0.4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    history = {'train_loss':[], 'train_miou':[], 'train_dice':[], 'val_loss':[], 'val_miou':[], 'val_dice':[], 'lr':[]}
    best_miou=0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        tl,tm,td = train_epoch(model, train_loader, criterion, optimizer, device)
        vl,vm,vd = val_epoch(model, val_loader, criterion, device)
        scheduler.step(vm)
        print(f"Train - Loss: {tl:.4f}, mIoU: {tm:.4f}, Dice: {td:.4f}")
        print(f"Val   - Loss: {vl:.4f}, mIoU: {vm:.4f}, Dice: {vd:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        for k,v in zip(['train_loss','train_miou','train_dice','val_loss','val_miou','val_dice'], [tl,tm,td,vl,vm,vd]):
            history[k].append(float(v))
        history['lr'].append(float(optimizer.param_groups[0]['lr']))
        if vm > best_miou:
            best_miou=vm
            torch.save(model.state_dict(), save_path / f"best_{model_name}.pth")
            torch.save(model.state_dict(), save_path / f"best_{model_name}_{encoder}_miou{vm:.4f}.pth")
            print(f"Saved best mIoU {vm:.4f}")
    
    with open(save_path / f"history_{model_name}.json", 'w') as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), save_path / f"final_{model_name}.pth")
    plot_history(history, model_name, str(reports_root))
    print(f"\nDone! Best mIoU: {best_miou:.4f}")
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--encoder', type=str, default='resnet18')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=128)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config and load_config else None
    train_model(model_name=args.model, encoder=args.encoder, epochs=args.epochs, batch_size=args.batch_size, img_size=args.img_size, config=cfg)
