# DeepCrack CPU baseline — recorded v2 experiment

> **Scope.** This is one protocol-compliant, CPU-only reference run of the current v2 pipeline. It is not a production-readiness claim, a safety certification, or a comparison with runs using different resolution, split, preprocessing, or hardware.

## Result at a glance

| Held-out official DeepCrack test metric | Value |
| --- | ---: |
| Foreground / crack IoU | **0.665591** |
| Foreground / crack Dice | **0.799225** |
| Foreground precision | 0.778466 |
| Foreground recall | 0.821121 |
| Pixel accuracy | 0.982151 |
| Mean IoU | 0.823542 |

The official test result above was generated once from the selected validation checkpoint. Pixel accuracy is included for completeness but is not the primary score because crack pixels are sparse.

## Protocol and provenance

| Item | Recorded value |
| --- | --- |
| Code commit used for training | `4aa8a120ea4f59357c82bbf172068448a239a795` |
| Versioned configuration | [`experiments/configs/deepcrack_cpu_baseline.yaml`](../../experiments/configs/deepcrack_cpu_baseline.yaml) |
| Model | U-Net with a ResNet-18 encoder; ImageNet initialization enabled |
| Input resolution | 128 × 128 |
| Training data | 240 images from the official DeepCrack `train_*` folders |
| Validation data | 60 images, deterministic 20% split from official training data only |
| Split seed | 42 |
| Official held-out test data | 237 images from the untouched `test_*` folders |
| Checkpoint selection metric | Validation foreground / crack IoU |
| Selected checkpoint | `best.pt`, epoch 14; validation foreground IoU 0.595122 |
| Runtime | CPU-only, 8 PyTorch threads; Python 3.11.9; torch 2.6.0+cpu |
| Training command wall time | 834.87 s (13 m 54.87 s; includes the one-time pretrained-weight download) |
| Official-test evaluation wall time | 22.63 s |

The training history records that the official test split was not loaded for model selection or scheduler decisions. The test set was used only after the configuration and the validation-selected checkpoint were fixed.

## Official held-out test metrics

The confusion matrix uses rows = ground truth and columns = prediction.

| Ground truth \ Prediction | Background | Crack |
| --- | ---: | ---: |
| Background | 3,675,756 | 39,256 |
| Crack | 30,051 | 137,945 |

| Class | IoU | Dice |
| --- | ---: | ---: |
| Background | 0.981494 | 0.990660 |
| Crack | 0.665591 | 0.799225 |

The test support contains 3,715,012 background pixels and 167,996 crack pixels. The higher crack recall than precision should be interpreted together with the false-positive count above; no single aggregate metric is sufficient for sparse-defect segmentation.

## Versioned evidence bundle

The dashboard and the JSON artifacts below are copies of this recorded run, not synthetic examples:

- [`assets/experiments/deepcrack_cpu_baseline_dashboard.png`](../../assets/experiments/deepcrack_cpu_baseline_dashboard.png) — generated from the recorded history and held-out test metrics;
- [`experiments/runs/deepcrack-cpu-baseline/test_metrics.json`](../../experiments/runs/deepcrack-cpu-baseline/test_metrics.json) — exact held-out output;
- [`experiments/runs/deepcrack-cpu-baseline/training_history.json`](../../experiments/runs/deepcrack-cpu-baseline/training_history.json) — per-epoch losses and split metrics;
- [`experiments/runs/deepcrack-cpu-baseline/split_manifest.json`](../../experiments/runs/deepcrack-cpu-baseline/split_manifest.json) — deterministic split provenance;
- [`experiments/runs/deepcrack-cpu-baseline/run_metadata.json`](../../experiments/runs/deepcrack-cpu-baseline/run_metadata.json) and [`artifact_manifest.json`](../../experiments/runs/deepcrack-cpu-baseline/artifact_manifest.json) — environment, command timing, and SHA-256 fingerprints.

Raw DeepCrack images/masks and model weights are deliberately not published. The artifact manifest fingerprints the local selected checkpoint without distributing it.

## Interpretation and limitations

- This result is one seed (`42`) at 128 × 128 resolution; it is not a confidence interval or a multi-seed benchmark.
- The DeepCrack source project's data terms limit use to non-commercial research and education. The dataset is excluded from this repository, and the MIT code license does not relicense it.
- The CPU wall-clock time is recorded for reproducibility context, not a portable latency or throughput claim.
- A model with these image-level segmentation scores still requires task-specific false-negative analysis, operating-condition validation, monitoring, and human review before any real inspection workflow.
