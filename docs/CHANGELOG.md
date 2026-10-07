# Changelog


## 2.1.0 — recorded DeepCrack CPU baseline

- Added a real, protocol-compliant DeepCrack experiment record: U-Net / pretrained ResNet-18, 128 × 128, seed 42, with validation derived only from official training data.
- Selected epoch 14 exclusively by validation crack IoU (`0.595122`), then evaluated the untouched 237-image official test split once.
- Recorded held-out crack IoU `0.665591`, Dice `0.799225`, precision `0.778466`, and recall `0.821121`.
- Versioned a dashboard and safe provenance artifacts while continuing to exclude the raw DeepCrack data and checkpoint weights.

## 2.0.0 — split integrity and reproducibility revision

### Scientific validity

- Reserved the official DeepCrack test set for final evaluation only.
- Added deterministic train/validation derivation from official training data.
- Persisted the split manifest and checkpoint protocol metadata.
- Replaced silent empty masks with strict one-to-one image/mask validation.
- Replaced per-batch averaged metrics with a global confusion-matrix metric accumulator.
- Made foreground crack IoU the checkpoint selection metric and added foreground Dice, precision, and recall.

### Engineering and presentation

- Added package-safe module entry points, test coverage, linting, CI, Docker fixes, and version-pinned CPU dependencies.
- Added a report generator that creates a dashboard only from current experiment artifacts.
- Made FastAPI checkpoint-safe and input-safe.
- Removed stale dashboards, thumbnails, notebook, demo images, old history, and unsupported accuracy/latency claims.
