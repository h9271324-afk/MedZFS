"""
Evaluation Metrics for MedZFS.

Implements three metrics used in the paper:
  1. Dice Similarity Coefficient (DSC) — volumetric overlap
  2. Hausdorff Distance (HD) — boundary precision
  3. Anatomical Consistency Rate (ACR) — constraint satisfaction
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def dice_score(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    smooth: float = 1e-7,
) -> float:
    """Compute Dice Similarity Coefficient.

    DSC = 2|P ∩ G| / (|P| + |G|)

    Args:
        prediction: Binary prediction mask.
        ground_truth: Binary ground truth mask.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Dice score in [0, 1].
    """
    pred = prediction.flatten().astype(np.float32)
    gt = ground_truth.flatten().astype(np.float32)
    intersection = (pred * gt).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + gt.sum() + smooth))


def hausdorff_distance(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    percentile: float = 95,
) -> float:
    """Compute Hausdorff Distance between prediction and ground truth boundaries.

    Uses the percentile variant (HD95 by default) for robustness to outliers.

    Args:
        prediction: Binary prediction mask.
        ground_truth: Binary ground truth mask.
        percentile: Percentile for robust HD computation (default 95).

    Returns:
        Hausdorff distance in pixels/voxels. Returns inf if either mask is empty.
    """
    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    if not pred.any() or not gt.any():
        return float("inf")

    # Compute surface distances
    pred_boundary = pred ^ _erode(pred)
    gt_boundary = gt ^ _erode(gt)

    if not pred_boundary.any() or not gt_boundary.any():
        return float("inf")

    # Distance transform from GT boundary
    dt_gt = distance_transform_edt(~gt_boundary)
    dt_pred = distance_transform_edt(~pred_boundary)

    # Surface distances
    surf_dist_pred_to_gt = dt_gt[pred_boundary]
    surf_dist_gt_to_pred = dt_pred[gt_boundary]

    all_distances = np.concatenate([surf_dist_pred_to_gt, surf_dist_gt_to_pred])

    return float(np.percentile(all_distances, percentile))


def _erode(mask: np.ndarray) -> np.ndarray:
    """Erode a binary mask by 1 pixel using distance transform."""
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask, iterations=1)


def anatomical_consistency_rate(
    predictions: dict,
    spatial_rules: list = None,
    hierarchical_rules: list = None,
) -> dict:
    """Compute Anatomical Consistency Rate (ACR).

    Measures the fraction of predictions satisfying anatomical constraints
    encoded in the knowledge graph.

    Args:
        predictions: Dict mapping class_name → binary prediction mask.
        spatial_rules: List of (class_a, relation, class_b) spatial constraints.
        hierarchical_rules: List of (parent, child) containment constraints.

    Returns:
        Dict with keys: spatial, hierarchical, overall (values in [0, 1]).
    """
    if spatial_rules is None:
        spatial_rules = [
            ("liver", "superior_to", "right_kidney"),
            ("spleen", "superior_to", "left_kidney"),
            ("liver", "right_of", "spleen"),
        ]

    if hierarchical_rules is None:
        hierarchical_rules = []

    total_rules = 0
    satisfied = 0

    # Check spatial rules
    spatial_satisfied = 0
    spatial_total = 0
    for class_a, relation, class_b in spatial_rules:
        if class_a not in predictions or class_b not in predictions:
            continue

        mask_a = predictions[class_a]
        mask_b = predictions[class_b]
        spatial_total += 1

        if not mask_a.any() or not mask_b.any():
            continue

        centroid_a = np.array(np.where(mask_a)).mean(axis=1)
        centroid_b = np.array(np.where(mask_b)).mean(axis=1)

        if relation == "superior_to":
            # In medical imaging, superior = lower row index
            if centroid_a[0] < centroid_b[0]:
                spatial_satisfied += 1
        elif relation == "right_of":
            if centroid_a[1] > centroid_b[1]:
                spatial_satisfied += 1
        elif relation == "left_of":
            if centroid_a[1] < centroid_b[1]:
                spatial_satisfied += 1

    # Check hierarchical rules
    hier_satisfied = 0
    hier_total = 0
    for parent, child in hierarchical_rules:
        if parent not in predictions or child not in predictions:
            continue
        hier_total += 1
        parent_mask = predictions[parent]
        child_mask = predictions[child]
        if child_mask.any():
            overlap = (parent_mask & child_mask).sum() / child_mask.sum()
            if overlap > 0.8:
                hier_satisfied += 1

    # Aggregate
    spatial_rate = spatial_satisfied / max(spatial_total, 1)
    hier_rate = hier_satisfied / max(hier_total, 1)
    total = spatial_total + hier_total
    success = spatial_satisfied + hier_satisfied
    overall = success / max(total, 1)

    return {
        "spatial": spatial_rate,
        "hierarchical": hier_rate,
        "overall": overall,
        "spatial_total": spatial_total,
        "hierarchical_total": hier_total,
    }


def dice_score_tensor(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-7) -> torch.Tensor:
    """Compute Dice score for PyTorch tensors (differentiable).

    Args:
        pred: Prediction tensor (B, H, W) or (B, 1, H, W).
        target: Ground truth tensor, same shape.
        smooth: Smoothing factor.

    Returns:
        Mean Dice score (scalar tensor).
    """
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    return dice.mean()
