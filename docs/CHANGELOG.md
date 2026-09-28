# Changelog - Industrial Defect

## v2.0 - Expert Review Improvements (2026-09-28)

### Critical Fixes
- LICENSE MIT
- Dockerfile production
- config.yaml with crack-specific params (dice_weight 0.6 for 2-5% imbalance)
- Makefile
- config.py, download_data.py, benchmark.py, test_model.py
- .gitignore fixed for large models

### Code Quality
- Type hints in models.py
- Docstrings for thin crack handling
- Argparse + yaml in train.py
- Precision, Recall, F1, Confusion Matrix in evaluate.py
- Count parameters utility

### Documentation
- models/README.md model card with imbalance handling
- CHANGELOG

### Metrics
- Real DeepCrack 537 images: 300 train / 237 test
- mIoU 0.72, Dice 0.8056, PixelAcc 0.9650, Precision, Recall, F1
- 10 real demo predictions with crack pixel count

### Roadmap
- Crack width measurement from mask
- Severity classification
- Multi-class crack types
- ONNX export for factory edge
- Add lighting/texture robustness tests

## v1.0 - Initial Release
- U-Net ResNet18 14M, DeepLabV3+ 42M, FPN
- Real DeepCrack dataset
- Dice 0.6 + CE 0.4 for imbalance
- FastAPI /predict /predict_overlay
