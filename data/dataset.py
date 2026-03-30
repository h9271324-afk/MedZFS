"""
Episodic Medical Image Dataset for Zero-to-Few-Shot Segmentation.

Implements an episodic sampling strategy that constructs query/support pairs
from 3D medical volumes. Supports zero-shot (k=0) and few-shot (k>=1)
episodes with configurable shot settings.
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from data.transforms import MedicalTransforms


class MedicalVolumeDataset:
    """Load and manage 3D medical volumes with their segmentation masks.

    Handles NIfTI volume loading, slice extraction, and organization by
    anatomical class for episodic sampling.
    """

    def __init__(
        self,
        data_dir: str,
        classes: List[Dict],
        image_size: Tuple[int, int] = (256, 256),
        normalize: bool = True,
    ):
        """Initialize the volume dataset.

        Args:
            data_dir: Path to preprocessed data directory.
            classes: List of class definitions with name, label, description.
            image_size: Target 2D image size (H, W).
            normalize: Whether to apply intensity normalization.
        """
        self.data_dir = data_dir
        self.classes = classes
        self.image_size = image_size
        self.normalize = normalize

        # Build class-to-slices mapping
        self.class_slices = self._index_slices()

    def _index_slices(self) -> Dict[int, List[Dict]]:
        """Index all 2D slices that contain each anatomical class.

        Returns:
            Dictionary mapping class label to list of slice metadata dicts,
            where each dict has keys: volume_path, mask_path, slice_idx.
        """
        class_slices = {cls["label"]: [] for cls in self.classes}

        images_dir = os.path.join(self.data_dir, "images")
        masks_dir = os.path.join(self.data_dir, "masks")

        if not os.path.exists(images_dir):
            print(f"Warning: Images directory not found: {images_dir}")
            return class_slices

        for vol_name in sorted(os.listdir(images_dir)):
            if not vol_name.endswith((".nii", ".nii.gz")):
                continue

            vol_path = os.path.join(images_dir, vol_name)
            mask_path = os.path.join(masks_dir, vol_name)

            if not os.path.exists(mask_path):
                continue

            # Load mask to index slices containing each class
            try:
                mask_vol = nib.load(mask_path).get_fdata()
            except Exception as e:
                print(f"Warning: Could not load {mask_path}: {e}")
                continue

            for cls in self.classes:
                label = cls["label"]
                # Find slices containing this class (>50 pixels threshold)
                for z in range(mask_vol.shape[2]):
                    if np.sum(mask_vol[:, :, z] == label) > 50:
                        class_slices[label].append({
                            "volume_path": vol_path,
                            "mask_path": mask_path,
                            "slice_idx": z,
                        })

        for cls in self.classes:
            n = len(class_slices[cls["label"]])
            print(f"  Class '{cls['name']}' (label={cls['label']}): {n} slices")

        return class_slices

    def load_slice(
        self, volume_path: str, mask_path: str, slice_idx: int, target_label: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single 2D slice and its binary mask for the target class.

        Args:
            volume_path: Path to the NIfTI volume.
            mask_path: Path to the NIfTI mask.
            slice_idx: Axial slice index.
            target_label: Class label for binary mask extraction.

        Returns:
            Tuple of (image, binary_mask) as float32 numpy arrays.
        """
        img_vol = nib.load(volume_path).get_fdata()
        mask_vol = nib.load(mask_path).get_fdata()

        image = img_vol[:, :, slice_idx].astype(np.float32)
        mask = (mask_vol[:, :, slice_idx] == target_label).astype(np.float32)

        # Normalize intensity to [0, 1]
        if self.normalize:
            p1, p99 = np.percentile(image, [1, 99])
            image = np.clip(image, p1, p99)
            if p99 - p1 > 1e-8:
                image = (image - p1) / (p99 - p1)
            else:
                image = np.zeros_like(image)

        # Resize if needed
        if image.shape != tuple(self.image_size):
            from skimage.transform import resize

            image = resize(image, self.image_size, preserve_range=True).astype(
                np.float32
            )
            mask = resize(
                mask, self.image_size, order=0, preserve_range=True
            ).astype(np.float32)

        return image, mask


class EpisodicMedicalDataset(Dataset):
    """Episodic dataset for zero-to-few-shot medical image segmentation.

    Each episode consists of:
      - A query image and its ground truth mask
      - A support set of k labeled examples (k=0 for zero-shot)
      - Anatomical text description of the target class
      - Target class metadata

    The dataset dynamically samples episodes with varying shot settings.
    """

    def __init__(
        self,
        data_dir: str,
        classes: List[Dict],
        shots: List[int] = None,
        num_episodes: int = 1000,
        image_size: Tuple[int, int] = (256, 256),
        transforms: Optional[MedicalTransforms] = None,
        split: str = "train",
    ):
        """Initialize the episodic dataset.

        Args:
            data_dir: Path to preprocessed data directory.
            classes: List of class definitions with name, label, description.
            shots: List of possible shot values (e.g., [0, 1, 5, 10]).
            num_episodes: Number of episodes per epoch.
            image_size: Target 2D image size (H, W).
            transforms: Optional augmentation transforms.
            split: Data split ("train", "val", "test").
        """
        if shots is None:
            shots = [0, 1, 5, 10]
        self.shots = shots
        self.num_episodes = num_episodes
        self.transforms = transforms
        self.split = split
        self.classes = classes

        # Initialize volume dataset
        split_dir = os.path.join(data_dir, split)
        self.volume_dataset = MedicalVolumeDataset(
            data_dir=split_dir,
            classes=classes,
            image_size=image_size,
        )

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Sample a single episode.

        Args:
            idx: Episode index (not used directly; episodes are random).

        Returns:
            Dictionary containing:
              - query_image: (C, H, W) tensor
              - query_mask: (H, W) tensor
              - support_images: (k, C, H, W) tensor (empty if zero-shot)
              - support_masks: (k, H, W) tensor (empty if zero-shot)
              - class_name: str
              - class_description: str
              - class_label: int
              - num_shots: int
        """
        # Randomly select target class and shot setting
        target_class = random.choice(self.classes)
        label = target_class["label"]
        k = random.choice(self.shots)

        # Get all available slices for this class
        available_slices = self.volume_dataset.class_slices.get(label, [])

        if len(available_slices) < k + 1:
            # Fallback: if not enough slices, use zero-shot
            k = max(0, len(available_slices) - 1)

        if len(available_slices) == 0:
            # Return dummy episode if no data available
            return self._dummy_episode(target_class, k)

        # Sample query and support slices (non-overlapping)
        sampled = random.sample(available_slices, min(k + 1, len(available_slices)))
        query_meta = sampled[0]
        support_meta = sampled[1 : k + 1]

        # Load query
        query_img, query_mask = self.volume_dataset.load_slice(
            query_meta["volume_path"],
            query_meta["mask_path"],
            query_meta["slice_idx"],
            label,
        )

        # Apply transforms to query
        if self.transforms is not None and self.split == "train":
            query_img, query_mask = self.transforms(query_img, query_mask)

        # Convert to tensors: (1, H, W) for image, (H, W) for mask
        query_img = torch.from_numpy(query_img).unsqueeze(0).float()
        query_mask = torch.from_numpy(query_mask).float()

        # Repeat grayscale to 3 channels for VLM compatibility
        if query_img.shape[0] == 1:
            query_img = query_img.repeat(3, 1, 1)

        # Load support set
        support_images = []
        support_masks = []
        for s_meta in support_meta:
            s_img, s_mask = self.volume_dataset.load_slice(
                s_meta["volume_path"],
                s_meta["mask_path"],
                s_meta["slice_idx"],
                label,
            )
            s_img = torch.from_numpy(s_img).unsqueeze(0).float()
            s_mask = torch.from_numpy(s_mask).float()
            if s_img.shape[0] == 1:
                s_img = s_img.repeat(3, 1, 1)
            support_images.append(s_img)
            support_masks.append(s_mask)

        # Stack support tensors
        if len(support_images) > 0:
            support_images = torch.stack(support_images, dim=0)  # (k, C, H, W)
            support_masks = torch.stack(support_masks, dim=0)    # (k, H, W)
        else:
            h, w = query_img.shape[1], query_img.shape[2]
            support_images = torch.zeros(0, 3, h, w)
            support_masks = torch.zeros(0, h, w)

        return {
            "query_image": query_img,
            "query_mask": query_mask,
            "support_images": support_images,
            "support_masks": support_masks,
            "class_name": target_class["name"],
            "class_description": target_class["description"],
            "class_label": label,
            "num_shots": len(support_meta),
        }

    def _dummy_episode(self, target_class: Dict, k: int) -> Dict[str, torch.Tensor]:
        """Create a dummy episode when no data is available."""
        h, w = self.volume_dataset.image_size
        return {
            "query_image": torch.zeros(3, h, w),
            "query_mask": torch.zeros(h, w),
            "support_images": torch.zeros(max(k, 0), 3, h, w),
            "support_masks": torch.zeros(max(k, 0), h, w),
            "class_name": target_class["name"],
            "class_description": target_class["description"],
            "class_label": target_class["label"],
            "num_shots": 0,
        }
