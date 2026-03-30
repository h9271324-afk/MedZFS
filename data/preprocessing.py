"""
Medical Image Preprocessing Pipeline.

Handles NIfTI volume preprocessing including intensity normalization,
resampling to uniform voxel spacing, and center cropping. Designed for
Abdomen MRI, Abdomen CT, and Cardiac MRI datasets.
"""

import argparse
import os
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


class MedicalPreprocessor:
    """Preprocess 3D medical volumes for MedZFS training and evaluation.

    Performs the following steps:
      1. Load NIfTI volume and segmentation mask
      2. Resample to uniform voxel spacing
      3. Intensity normalization (percentile-based or HU windowing)
      4. Optional center cropping
      5. Save processed volumes
    """

    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        normalize: bool = True,
        crop_size: Optional[Tuple[int, int, int]] = None,
    ):
        """Initialize the preprocessor.

        Args:
            target_spacing: Target voxel spacing in mm (x, y, z).
            normalize: Whether to apply intensity normalization.
            crop_size: Optional 3D crop size (D, H, W). None = no cropping.
        """
        self.target_spacing = np.array(target_spacing)
        self.normalize = normalize
        self.crop_size = crop_size

    def process_volume(
        self,
        image_path: str,
        mask_path: Optional[str] = None,
        modality: str = "MRI",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Process a single volume and its optional mask.

        Args:
            image_path: Path to input NIfTI image volume.
            mask_path: Path to input NIfTI segmentation mask.
            modality: Imaging modality ("MRI", "CT", or "CMR").

        Returns:
            Tuple of (processed_image, processed_mask).
        """
        # Load the NIfTI volume
        img_nii = nib.load(image_path)
        image = img_nii.get_fdata().astype(np.float32)
        original_spacing = np.array(img_nii.header.get_zooms()[:3])

        # Load mask if provided
        mask = None
        if mask_path is not None and os.path.exists(mask_path):
            mask = nib.load(mask_path).get_fdata().astype(np.float32)

        # Step 1: Resample to target spacing
        image, mask = self._resample(image, mask, original_spacing)

        # Step 2: Intensity normalization
        if self.normalize:
            image = self._normalize_intensity(image, modality)

        # Step 3: Optional center cropping
        if self.crop_size is not None:
            image, mask = self._center_crop(image, mask)

        return image, mask

    def _resample(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray],
        original_spacing: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Resample volume to target voxel spacing using trilinear interpolation.

        Args:
            image: Input volume (D, H, W).
            mask: Optional segmentation mask.
            original_spacing: Original voxel spacing in mm.

        Returns:
            Resampled (image, mask) tuple.
        """
        zoom_factors = original_spacing / self.target_spacing
        resampled_image = zoom(image, zoom_factors, order=3)

        resampled_mask = None
        if mask is not None:
            # Use nearest-neighbor interpolation for masks
            resampled_mask = zoom(mask, zoom_factors, order=0)

        return resampled_image, resampled_mask

    def _normalize_intensity(
        self, image: np.ndarray, modality: str
    ) -> np.ndarray:
        """Normalize intensity values based on modality.

        For CT: clip to [-200, 300] HU then scale to [0, 1].
        For MRI/CMR: percentile-based normalization to [0, 1].

        Args:
            image: Input volume.
            modality: Imaging modality.

        Returns:
            Normalized volume with values in [0, 1].
        """
        if modality.upper() == "CT":
            # CT: Window to abdomen soft tissue
            image = np.clip(image, -200, 300)
            image = (image + 200) / 500.0
        else:
            # MRI / CMR: Percentile-based normalization
            p1, p99 = np.percentile(image[image > 0], [1, 99])
            image = np.clip(image, p1, p99)
            if p99 - p1 > 1e-8:
                image = (image - p1) / (p99 - p1)
            else:
                image = np.zeros_like(image)

        return image

    def _center_crop(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Center crop the volume to the specified size.

        Args:
            image: Input volume (D, H, W).
            mask: Optional segmentation mask.

        Returns:
            Cropped (image, mask) tuple.
        """
        d, h, w = image.shape
        cd, ch, cw = self.crop_size

        d_start = max(0, (d - cd) // 2)
        h_start = max(0, (h - ch) // 2)
        w_start = max(0, (w - cw) // 2)

        cropped_image = image[
            d_start : d_start + cd,
            h_start : h_start + ch,
            w_start : w_start + cw,
        ]

        cropped_mask = None
        if mask is not None:
            cropped_mask = mask[
                d_start : d_start + cd,
                h_start : h_start + ch,
                w_start : w_start + cw,
            ]

        return cropped_image, cropped_mask

    def process_dataset(
        self,
        input_dir: str,
        output_dir: str,
        modality: str = "MRI",
    ) -> None:
        """Process an entire dataset directory.

        Expected input structure:
            input_dir/
              images/  -> *.nii.gz volumes
              masks/   -> *.nii.gz segmentation masks

        Output structure mirrors the input structure.

        Args:
            input_dir: Path to raw data directory.
            output_dir: Path to save processed data.
            modality: Imaging modality string.
        """
        images_in = os.path.join(input_dir, "images")
        masks_in = os.path.join(input_dir, "masks")
        images_out = os.path.join(output_dir, "images")
        masks_out = os.path.join(output_dir, "masks")

        os.makedirs(images_out, exist_ok=True)
        os.makedirs(masks_out, exist_ok=True)

        vol_files = sorted([
            f for f in os.listdir(images_in)
            if f.endswith((".nii", ".nii.gz"))
        ])

        print(f"Processing {len(vol_files)} volumes from {input_dir}")

        for vol_name in vol_files:
            img_path = os.path.join(images_in, vol_name)
            mask_path = os.path.join(masks_in, vol_name)

            print(f"  Processing: {vol_name}...")

            image, mask = self.process_volume(img_path, mask_path, modality)

            # Save processed volumes as NIfTI
            out_img_path = os.path.join(images_out, vol_name)
            out_mask_path = os.path.join(masks_out, vol_name)

            nib.save(nib.Nifti1Image(image, np.eye(4)), out_img_path)
            if mask is not None:
                nib.save(nib.Nifti1Image(mask, np.eye(4)), out_mask_path)

        print(f"Done. Processed data saved to {output_dir}")


def main():
    """Command-line preprocessing entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess medical imaging datasets for MedZFS."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to raw dataset directory.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path to save processed dataset.",
    )
    parser.add_argument(
        "--target_spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
        help="Target voxel spacing in mm (x y z).",
    )
    parser.add_argument(
        "--modality", type=str, default="MRI",
        choices=["MRI", "CT", "CMR"],
        help="Imaging modality.",
    )
    parser.add_argument(
        "--normalize", action="store_true", default=True,
        help="Apply intensity normalization.",
    )
    args = parser.parse_args()

    preprocessor = MedicalPreprocessor(
        target_spacing=tuple(args.target_spacing),
        normalize=args.normalize,
    )

    preprocessor.process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        modality=args.modality,
    )


if __name__ == "__main__":
    main()
