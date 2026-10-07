"""Integrity-checked DeepCrack data loading with train/validation/test separation."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    mask_path: Path

    @property
    def sample_id(self) -> str:
        return self.image_path.stem


def _image_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def discover_records(root: str | Path, source_split: Literal["train", "test"]) -> list[SampleRecord]:
    """Discover image/mask pairs and fail loudly for any missing annotation.

    The historic loader silently replaced a missing mask with an all-background
    mask. That corrupts supervision and is deliberately forbidden here.
    """
    root = Path(root)
    image_dir = root / f"{source_split}_img"
    mask_dir = root / f"{source_split}_lab"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Expected DeepCrack folders '{image_dir}' and '{mask_dir}'. "
            "See `python -m src.download_data --help`."
        )
    images = _image_files(image_dir)
    masks = _image_files(mask_dir)
    if not images:
        raise FileNotFoundError(f"No supported image files found in {image_dir}")
    masks_by_stem: dict[str, Path] = {}
    duplicates: list[str] = []
    for mask in masks:
        key = mask.stem.lower()
        if key in masks_by_stem:
            duplicates.append(mask.name)
        else:
            masks_by_stem[key] = mask
    if duplicates:
        raise ValueError(f"Duplicate mask stems in {mask_dir}: {duplicates[:5]}")
    records, missing = [], []
    for image in images:
        mask = masks_by_stem.get(image.stem.lower())
        if mask is None:
            missing.append(image.name)
        else:
            records.append(SampleRecord(image, mask))
    if missing:
        example = ", ".join(missing[:8])
        raise FileNotFoundError(f"{len(missing)} image(s) have no matching mask in {mask_dir}: {example}")
    return records


def split_training_records(records: Iterable[SampleRecord], validation_fraction: float, seed: int) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Create a deterministic validation subset only from DeepCrack's train split."""
    records = sorted(records, key=lambda record: record.sample_id.lower())
    if len(records) < 2:
        raise ValueError("At least two training records are required to create a validation split")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation_count = min(validation_count, len(shuffled) - 1)
    validation_ids = {record.sample_id for record in shuffled[:validation_count]}
    train_records = [record for record in records if record.sample_id not in validation_ids]
    validation_records = [record for record in records if record.sample_id in validation_ids]
    return train_records, validation_records


def get_transforms(train: bool, image_size: int) -> A.Compose:
    transforms: list[A.BasicTransform] = [A.Resize(image_size, image_size)]
    if train:
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            ]
        )
    transforms.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


class DeepCrackDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset over verified DeepCrack image/mask pairs."""

    def __init__(self, records: list[SampleRecord], transform: A.Compose) -> None:
        if not records:
            raise ValueError("Dataset received no records")
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            image_array = np.asarray(image.convert("RGB"))
        with Image.open(record.mask_path) as mask:
            mask_array = (np.asarray(mask.convert("L")) > 127).astype(np.uint8)
        transformed = self.transform(image=image_array, mask=mask_array)
        return transformed["image"], transformed["mask"].long()


def get_dataloaders(
    *,
    data_root: str | Path,
    batch_size: int,
    image_size: int,
    validation_fraction: float,
    split_seed: int,
    num_workers: int = 0,
) -> tuple[dict[SplitName, DataLoader], dict[str, object]]:
    """Return train, validation, and untouched test loaders plus a split manifest."""
    all_train_records = discover_records(data_root, "train")
    test_records = discover_records(data_root, "test")
    train_records, validation_records = split_training_records(all_train_records, validation_fraction, split_seed)
    train_ids = {record.sample_id for record in train_records}
    validation_ids = {record.sample_id for record in validation_records}
    test_ids = {record.sample_id for record in test_records}
    if train_ids & validation_ids:
        raise ValueError("Train/validation split overlap detected")

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    generator = torch.Generator().manual_seed(split_seed)
    loaders: dict[SplitName, DataLoader] = {
        "train": DataLoader(DeepCrackDataset(train_records, get_transforms(True, image_size)), shuffle=True, generator=generator, **loader_options),
        "validation": DataLoader(DeepCrackDataset(validation_records, get_transforms(False, image_size)), shuffle=False, **loader_options),
        "test": DataLoader(DeepCrackDataset(test_records, get_transforms(False, image_size)), shuffle=False, **loader_options),
    }
    manifest: dict[str, object] = {
        "dataset": "DeepCrack",
        "source_train_count": len(all_train_records),
        "source_test_count": len(test_records),
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
        "train_ids": sorted(train_ids),
        "validation_ids": sorted(validation_ids),
        "test_ids": sorted(test_ids),
    }
    return loaders, manifest


def write_split_manifest(manifest: dict[str, object], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
