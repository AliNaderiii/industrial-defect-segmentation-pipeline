import pytest
import torch

from src.metrics import SegmentationMeter


def logits_for(labels: list[int]) -> torch.Tensor:
    logits = torch.full((1, 2, 1, len(labels)), -5.0)
    for index, label in enumerate(labels):
        logits[0, label, 0, index] = 5.0
    return logits


def test_global_confusion_matrix_and_foreground_metrics() -> None:
    meter = SegmentationMeter()
    # prediction [0, 1, 1, 0], target [0, 1, 0, 1]
    meter.update(logits_for([0, 1, 1, 0]), torch.tensor([[[0, 1, 0, 1]]]))
    result = meter.compute()
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["foreground_iou"] == pytest.approx(1 / 3)
    assert result["foreground_dice"] == pytest.approx(0.5)
    assert result["pixel_accuracy"] == pytest.approx(0.5)


def test_invalid_mask_labels_fail_loudly() -> None:
    meter = SegmentationMeter()
    with pytest.raises(ValueError, match="only 0"):
        meter.update(logits_for([0, 1]), torch.tensor([[[0, 2]]]))
