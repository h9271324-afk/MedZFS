"""
Domain-Specific Medical Image Augmentations for MedZFS.

Implements augmentation transforms designed for medical images while
preserving anatomical orientation. Supports elastic deformation,
intensity jittering, and random flipping with orientation constraints.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


class MedicalTransforms:
    """Compose domain-specific augmentations for medical image segmentation.

    All transforms operate on 2D (H, W) numpy arrays and jointly transform
    both the image and its segmentation mask to maintain spatial consistency.
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize transforms from configuration.

        Args:
            config: Augmentation configuration dictionary. If None, uses
                    sensible defaults designed for medical imaging.
        """
        if config is None:
            config = {}

        aug = config.get("augmentation", config)

        # Elastic deformation parameters
        ed = aug.get("elastic_deformation", {})
        self.elastic_alpha = ed.get("alpha", 200)
        self.elastic_sigma = ed.get("sigma", 20)
        self.elastic_prob = ed.get("probability", 0.5)

        # Intensity jittering parameters
        ij = aug.get("intensity_jitter", {})
        self.brightness_range = ij.get("brightness", 0.2)
        self.contrast_range = ij.get("contrast", 0.2)
        self.jitter_prob = ij.get("probability", 0.5)

        # Random flip parameters
        rf = aug.get("random_flip", {})
        self.flip_horizontal = rf.get("horizontal", True)
        self.flip_vertical = rf.get("vertical", False)
        self.flip_prob = rf.get("probability", 0.5)

        # Random rotation
        rr = aug.get("random_rotation", {})
        self.rotation_degrees = rr.get("degrees", 15)
        self.rotation_prob = rr.get("probability", 0.3)

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the full augmentation pipeline.

        Args:
            image: Input image array of shape (H, W), float32 in [0, 1].
            mask: Binary segmentation mask of shape (H, W), float32.

        Returns:
            Augmented (image, mask) tuple.
        """
        # Elastic deformation
        if np.random.random() < self.elastic_prob:
            image, mask = self._elastic_deformation(image, mask)

        # Intensity jittering (image only)
        if np.random.random() < self.jitter_prob:
            image = self._intensity_jitter(image)

        # Random horizontal flip
        if self.flip_horizontal and np.random.random() < self.flip_prob:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # Random vertical flip (disabled by default to preserve orientation)
        if self.flip_vertical and np.random.random() < self.flip_prob:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        # Random rotation
        if np.random.random() < self.rotation_prob:
            image, mask = self._random_rotation(image, mask)

        return image, mask

    def _elastic_deformation(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random elastic deformation.

        Simulates tissue deformation commonly seen in medical imaging
        due to breathing, cardiac motion, and organ movement.

        Args:
            image: Input image (H, W).
            mask: Segmentation mask (H, W).

        Returns:
            Deformed (image, mask) tuple.
        """
        shape = image.shape
        # Generate random displacement fields
        dx = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), self.elastic_sigma
        ) * self.elastic_alpha
        dy = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), self.elastic_sigma
        ) * self.elastic_alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = [np.clip(y + dy, 0, shape[0] - 1), np.clip(x + dx, 0, shape[1] - 1)]

        # Apply deformation with appropriate interpolation orders
        image_deformed = map_coordinates(image, indices, order=3, mode="reflect")
        mask_deformed = map_coordinates(mask, indices, order=0, mode="reflect")

        return image_deformed.astype(np.float32), mask_deformed.astype(np.float32)

    def _intensity_jitter(self, image: np.ndarray) -> np.ndarray:
        """Apply random brightness and contrast jittering.

        Simulates intensity variations across scanners and acquisition
        protocols common in multi-site medical imaging studies.

        Args:
            image: Input image (H, W) in [0, 1].

        Returns:
            Jittered image.
        """
        # Brightness shift
        brightness = (np.random.random() * 2 - 1) * self.brightness_range
        image = image + brightness

        # Contrast scaling
        contrast = 1.0 + (np.random.random() * 2 - 1) * self.contrast_range
        mean = np.mean(image)
        image = (image - mean) * contrast + mean

        return np.clip(image, 0, 1).astype(np.float32)

    def _random_rotation(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply small random rotation.

        Uses a limited rotation range to preserve anatomical orientation.

        Args:
            image: Input image (H, W).
            mask: Segmentation mask (H, W).

        Returns:
            Rotated (image, mask) tuple.
        """
        from scipy.ndimage import rotate as nd_rotate

        angle = (np.random.random() * 2 - 1) * self.rotation_degrees

        image_rot = nd_rotate(image, angle, reshape=False, order=3, mode="nearest")
        mask_rot = nd_rotate(mask, angle, reshape=False, order=0, mode="nearest")

        return image_rot.astype(np.float32), mask_rot.astype(np.float32)
