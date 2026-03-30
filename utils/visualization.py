"""
Segmentation Visualization Utilities for MedZFS.

Generates overlay images, error maps, and comparison figures.
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def visualize_prediction(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    class_name: str = "",
    dice_score: float = None,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Visualize segmentation prediction with overlay and error map.

    Creates a 4-panel figure: Input, Ground Truth, Prediction, Error Map.

    Args:
        image: Input image (H, W) normalized to [0, 1].
        ground_truth: Binary ground truth mask (H, W).
        prediction: Binary prediction mask (H, W).
        class_name: Class name for the title.
        dice_score: Optional computed Dice score.
        save_path: Path to save the figure.
        show: Whether to display the figure.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("white")

    # Input image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Ground truth overlay
    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(ground_truth, alpha=0.4, cmap="Greens")
    axes[1].set_title("Ground Truth", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Prediction overlay
    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(prediction, alpha=0.4, cmap="Blues")
    title = "Prediction"
    if dice_score is not None:
        title += f" (Dice: {dice_score:.1f}%)"
    axes[2].set_title(title, fontsize=12, fontweight="bold")
    axes[2].axis("off")

    # Error map: green=TP, red=FP, yellow=FN
    error_map = np.zeros((*image.shape[:2], 3))
    tp = (prediction > 0.5) & (ground_truth > 0.5)
    fp = (prediction > 0.5) & (ground_truth < 0.5)
    fn = (prediction < 0.5) & (ground_truth > 0.5)
    error_map[tp] = [0, 0.8, 0]    # Green: true positive
    error_map[fp] = [0.9, 0, 0]    # Red: false positive
    error_map[fn] = [0.9, 0.9, 0]  # Yellow: false negative

    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(error_map, alpha=0.5)
    axes[3].set_title("Error Map (G=TP, R=FP, Y=FN)", fontsize=12, fontweight="bold")
    axes[3].axis("off")

    suptitle = f"MedZFS Segmentation: {class_name}" if class_name else "MedZFS Segmentation"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_shot_progression(
    results: Dict[int, float],
    dataset_name: str = "",
    save_path: Optional[str] = None,
):
    """Plot Dice score progression across shot settings.

    Args:
        results: Dict mapping num_shots → mean Dice score.
        dataset_name: Name of the dataset.
        save_path: Path to save the figure.
    """
    shots = sorted(results.keys())
    dices = [results[k] for k in shots]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(shots, dices, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.fill_between(shots, dices, alpha=0.15, color="#2196F3")
    ax.set_xlabel("Number of Support Examples (k)", fontsize=12)
    ax.set_ylabel("Mean Dice Score (%)", fontsize=12)
    ax.set_title(f"MedZFS Performance — {dataset_name}", fontsize=14, fontweight="bold")
    ax.set_xticks(shots)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
