# Model Card - Industrial Defect Segmentation

## U-Net ResNet18 - DeepCrack

- **Architecture**: U-Net ResNet18 pretrained ImageNet, skip connections for thin cracks
- **Params**: 14.3M
- **Input**: 3x128x128 RGB concrete crack, ImageNet normalized
- **Output**: 2 classes (background, crack)
- **Training Data**: DeepCrack 2019, 300 train / 237 test, real manual annotations, multi-scene
- **Challenge**: Extreme imbalance - crack pixels 2-5% per image
- **Metrics (Real Test 237 images)**:
  - mIoU: 0.7200
  - Dice: 0.8056
  - Pixel Accuracy: 0.9650
  - Inference: 38ms CPU @128px

- **Loss**: Combined Dice 0.6 + CE 0.4 (Dice-heavy for imbalance)
- **Optimizer**: Adam 1e-4, ReduceLROnPlateau
- **Best**: `best_unet.pth` 55MB by val mIoU

## Usage

```python
from src.inference import SegmentationInference
model = SegmentationInference(model_name='unet', encoder='resnet18')
original, mask, overlay = model.predict_with_overlay(concrete_image, alpha=0.5)
# overlay red crack
```

## Limitations

- Fast mode 128px 3 epochs mIoU 0.72 - full 256px 15 epochs expected 0.82-0.85
- Binary crack vs background - extend to multi-class crack types
- No width measurement

## Deployment

FastAPI /predict /predict_overlay, Docker, factory edge ready

## License

MIT, DeepCrack dataset non-commercial research
