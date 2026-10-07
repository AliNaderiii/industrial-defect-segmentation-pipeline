from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data_loader import discover_records, get_dataloaders, split_training_records


def write_pair(root: Path, split: str, name: str, crack: bool = False) -> None:
    (root / f"{split}_img").mkdir(parents=True, exist_ok=True)
    (root / f"{split}_lab").mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((12, 16, 3), 128, dtype=np.uint8)).save(root / f"{split}_img" / f"{name}.png")
    mask = np.zeros((12, 16), dtype=np.uint8)
    if crack:
        mask[2:4, 3:8] = 255
    Image.fromarray(mask).save(root / f"{split}_lab" / f"{name}.png")


def test_missing_mask_is_a_hard_error(tmp_path: Path) -> None:
    (tmp_path / "train_img").mkdir()
    (tmp_path / "train_lab").mkdir()
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(tmp_path / "train_img" / "orphan.png")
    with pytest.raises(FileNotFoundError, match="no matching mask"):
        discover_records(tmp_path, "train")


def test_train_validation_are_deterministic_and_test_stays_separate(tmp_path: Path) -> None:
    for index in range(5):
        write_pair(tmp_path, "train", f"train_{index}", crack=index % 2 == 0)
    for index in range(2):
        write_pair(tmp_path, "test", f"test_{index}", crack=True)
    records = discover_records(tmp_path, "train")
    train_first, validation_first = split_training_records(records, 0.4, seed=7)
    train_second, validation_second = split_training_records(records, 0.4, seed=7)
    assert [r.sample_id for r in train_first] == [r.sample_id for r in train_second]
    assert [r.sample_id for r in validation_first] == [r.sample_id for r in validation_second]
    assert {r.sample_id for r in train_first}.isdisjoint({r.sample_id for r in validation_first})

    loaders, manifest = get_dataloaders(
        data_root=tmp_path,
        batch_size=2,
        image_size=32,
        validation_fraction=0.4,
        split_seed=7,
    )
    assert len(loaders["train"].dataset) == 3
    assert len(loaders["validation"].dataset) == 2
    assert len(loaders["test"].dataset) == 2
    images, masks = next(iter(loaders["train"]))
    assert images.shape[1:] == (3, 32, 32)
    assert set(masks.unique().tolist()).issubset({0, 1})
    assert set(manifest["test_ids"]) == {"test_0", "test_1"}
