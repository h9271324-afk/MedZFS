"""
Reproducibility Utilities for MedZFS.

Sets random seeds across all libraries to ensure deterministic results.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for full reproducibility.

    Sets seeds for: Python random, NumPy, PyTorch CPU, PyTorch CUDA,
    and cuDNN deterministic mode.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Random seed set to {seed}")
