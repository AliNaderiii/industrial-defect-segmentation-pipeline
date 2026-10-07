# Changelog

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
