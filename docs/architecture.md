# Architecture and evaluation protocol

## Split integrity

The original DeepCrack folders are treated as source training data (`train_img`/`train_lab`) and final test data (`test_img`/`test_lab`). The pipeline:

1. verifies one-to-one image/mask pairing before creating a dataset;
2. deterministically splits only source training records into train and validation subsets using `dataset.split_seed` and `dataset.validation_fraction`;
3. applies augmentation only to training records;
4. selects checkpoints and controls early stopping only from validation crack IoU;
5. evaluates `checkpoints/best.pt` only on the untouched official test records;
6. persists split identities in `checkpoints/split_manifest.json`.

The official test set must not influence hyperparameter choice, epoch selection, or scheduler decisions. A test score obtained after repeated tuning on the same test split is no longer an unbiased final estimate.

## Data integrity

A missing mask is a hard error. Replacing it with zeros would label a real crack as background and can produce deceptively high pixel accuracy. Masks are converted to binary only after being loaded from a verified pair: intensity values over 127 become crack class 1; the rest become background class 0.

## Optimisation and metrics

The default objective is 0.7 foreground Dice plus 0.3 foreground-weighted cross-entropy. It targets thin, sparse crack structure more directly than background-inclusive Dice alone. The primary selection metric is foreground IoU.

`SegmentationMeter` accumulates a single two-class pixel confusion matrix for an entire split. It reports foreground IoU/Dice/precision/recall alongside mean IoU and pixel accuracy. Pixel accuracy must not be treated as sufficient evidence on an imbalanced crack dataset.

## Checkpoints and serving

Checkpoints store model architecture, encoder, class count, image size, split parameters, split counts, validation metrics, and model parameters. Evaluation and serving rebuild the model from checkpoint metadata. The API returns a clear unavailable state rather than output from a randomly initialised decoder.
