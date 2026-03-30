"""
Unit tests for evaluation metrics.

Validates Dice score, Hausdorff distance, and ACR computations.
"""

import numpy as np
import pytest
import torch

from evaluation.metrics import dice_score, hausdorff_distance, anatomical_consistency_rate, dice_score_tensor


class TestDiceScore:

    def test_perfect_match(self):
        mask = np.ones((64, 64), dtype=np.float32)
        assert dice_score(mask, mask) == pytest.approx(1.0, abs=1e-6)

    def test_no_overlap(self):
        pred = np.zeros((64, 64), dtype=np.float32)
        gt = np.ones((64, 64), dtype=np.float32)
        assert dice_score(pred, gt) < 0.01

    def test_partial_overlap(self):
        pred = np.zeros((64, 64), dtype=np.float32)
        gt = np.zeros((64, 64), dtype=np.float32)
        pred[:32, :32] = 1
        gt[:32, :32] = 1
        gt[32:, :32] = 1  # Extra FN region
        d = dice_score(pred, gt)
        assert 0.5 < d < 0.8

    def test_tensor_version(self):
        pred = torch.ones(2, 64, 64)
        target = torch.ones(2, 64, 64)
        d = dice_score_tensor(pred, target)
        assert d.item() == pytest.approx(1.0, abs=1e-5)


class TestHausdorffDistance:

    def test_identical_masks(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[20:40, 20:40] = True
        hd = hausdorff_distance(mask.astype(float), mask.astype(float))
        assert hd == pytest.approx(0.0, abs=1.0)

    def test_empty_prediction(self):
        pred = np.zeros((64, 64))
        gt = np.ones((64, 64))
        hd = hausdorff_distance(pred, gt)
        assert hd == float("inf")


class TestACR:

    def test_correct_spatial(self):
        # Liver above kidney (lower row index = superior)
        predictions = {
            "liver": np.zeros((100, 100), dtype=bool),
            "right_kidney": np.zeros((100, 100), dtype=bool),
        }
        predictions["liver"][10:30, 40:60] = True   # Superior
        predictions["right_kidney"][60:80, 40:60] = True  # Inferior

        rules = [("liver", "superior_to", "right_kidney")]
        result = anatomical_consistency_rate(predictions, spatial_rules=rules)
        assert result["spatial"] == 1.0

    def test_violated_spatial(self):
        predictions = {
            "liver": np.zeros((100, 100), dtype=bool),
            "right_kidney": np.zeros((100, 100), dtype=bool),
        }
        predictions["liver"][60:80, 40:60] = True     # Inferior (wrong!)
        predictions["right_kidney"][10:30, 40:60] = True  # Superior

        rules = [("liver", "superior_to", "right_kidney")]
        result = anatomical_consistency_rate(predictions, spatial_rules=rules)
        assert result["spatial"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
