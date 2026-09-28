# Architecture - Industrial Defect Segmentation

## Data Pipeline - DeepCrack

- 537 real images, manual annotations
- Train 300, Test 237
- Resolution variable ~544x384
- Scenes: concrete, asphalt, walls, pavements
- Crack types: longitudinal, transverse, alligator, multi-scale
- Challenge: thin 1-5 px, low contrast, texture, shadows

Preprocessing:
- Resize 128/256
- Normalize ImageNet
- Augment: HFlip 0.5, VFlip 0.3, Rotate90 0.3, BrightnessContrast 0.2, GaussNoise var 10-50 p0.2
- Binarize mask >127 -> 1

Imbalance: crack pixels 2-5% per image, hence Dice-heavy loss.

## Models

### U-Net ResNet18 (14M)

Encoder ResNet18 pretrained:
- Stem 7x7 stride2 maxpool
- Layers [2,2,2,2] BasicBlock
- Channels 64,128,256,512

Decoder:
- 5 stages [256,128,64,32,16]
- Skip connections from encoder
- Final 1x1 conv 2 classes

Why for cracks:
- Skip preserves thin crack details
- Lightweight for factory edge
- 38ms CPU @128px

### DeepLabV3+ ResNet50 (42M)

Encoder ResNet50 + ASPP rates [1,6,12,18] output stride 16
Decoder DeepLabV3+ with low-level 256 ch concat
Multi-scale context for varying crack widths

### FPN ResNet34

Feature Pyramid, multi-scale fusion

## Loss

Combined 0.6*Dice + 0.4*CE
Dice handles imbalance, CE stabilizes

## Training

Adam lr 1e-4
Scheduler ReduceLROnPlateau max factor 0.5 patience 3
Batch 4/8
Epochs 3 fast / 15 full
Best by val mIoU

Metrics:
- mIoU mean over classes ignore NaN
- Dice mean over classes
- PixelAcc (pred==true).mean()

## Inference

Resize 256 normalize forward argmax resize original nearest
Overlay red [255,0,0] alpha 0.5

FastAPI loads model on startup streaming PNG

## Deployment

CPU only 38ms @128px ResNet18
Docker python:3.9-slim pip uvicorn 0.0.0.0:8000
