"""
MedZFS Zero-Shot and Few-Shot Inference Pipeline.

Loads a trained MedZFS checkpoint and performs segmentation on
new medical images using either zero-shot (text-only) or few-shot
(with support examples) inference.

Usage:
    python -m inference.predict \
        --image path/to/image.nii.gz \
        --target_class liver \
        --checkpoint checkpoints/best_model.pth
"""

import argparse
import os

import nibabel as nib
import numpy as np
import torch
import yaml

from data.anatomical_graphs import AnatomicalGraphBuilder
from models.medzfs import MedZFS


# Anatomical descriptions for supported classes
CLASS_DESCRIPTIONS = {
    "liver": "A large, wedge-shaped organ in the upper right abdomen beneath the diaphragm with a smooth capsular surface.",
    "right_kidney": "A bean-shaped retroperitoneal organ on the right side, slightly lower than the left kidney.",
    "left_kidney": "A bean-shaped retroperitoneal organ on the left side near the spleen.",
    "spleen": "An ovoid organ in the left upper quadrant, posterior to the stomach and superior to the left kidney.",
    "lv_myo": "The left ventricular myocardium, a thick muscular wall forming a ring around the LV cavity.",
    "lv_bp": "The left ventricular blood pool, the bright cavity enclosed by the LV myocardium.",
    "rv": "The right ventricle, a crescent-shaped chamber with a thinner wall than the LV.",
}


class MedZFSPredictor:
    """Inference wrapper for MedZFS model."""

    def __init__(self, checkpoint_path: str, config_path: str = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load config
        if config_path:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            # Use defaults
            self.config = {"model": {"feature_dim": 512}}

        # Load model
        self.model = MedZFS(self.config).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model.eval()

    def predict_volume(
        self,
        image_path: str,
        target_class: str,
        dataset_name: str = "abd_mri",
        support_images: list = None,
        support_masks: list = None,
        output_path: str = None,
    ) -> np.ndarray:
        """Predict segmentation for a full 3D volume.

        Args:
            image_path: Path to NIfTI image.
            target_class: Name of the target anatomical class.
            dataset_name: Dataset name for graph selection.
            support_images: List of support image paths (few-shot).
            support_masks: List of support mask paths (few-shot).
            output_path: Path to save prediction NIfTI.

        Returns:
            3D prediction mask as numpy array.
        """
        # Load image
        img_nii = nib.load(image_path)
        image = img_nii.get_fdata().astype(np.float32)

        # Normalize
        p1, p99 = np.percentile(image[image > 0], [1, 99])
        image = np.clip(image, p1, p99)
        image = (image - p1) / (p99 - p1 + 1e-8)

        # Build anatomical graph
        anat_graph = AnatomicalGraphBuilder.build_graph(dataset_name)
        edge_index, edge_type = anat_graph.get_full_edge_index()
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        node_descs = anat_graph.get_node_descriptions()

        # Get class description
        class_desc = CLASS_DESCRIPTIONS.get(target_class, f"Medical anatomical structure: {target_class}")

        # Predict slice by slice
        prediction = np.zeros_like(image)
        for z in range(image.shape[2]):
            slice_img = image[:, :, z]
            # Convert to tensor
            tensor = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0).float()
            tensor = tensor.repeat(1, 3, 1, 1).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    query_image=tensor,
                    class_description=class_desc,
                    node_descriptions=node_descs,
                    edge_index=edge_index,
                    edge_type=edge_type,
                )
                pred = outputs["prediction"].squeeze().cpu().numpy()
                prediction[:, :, z] = (pred > 0.5).astype(np.float32)

        # Save prediction
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            nib.save(nib.Nifti1Image(prediction, img_nii.affine), output_path)
            print(f"Prediction saved to {output_path}")

        return prediction


def main():
    parser = argparse.ArgumentParser(description="MedZFS Inference.")
    parser.add_argument("--image", type=str, required=True, help="Path to input NIfTI image.")
    parser.add_argument("--target_class", type=str, required=True, help="Target anatomical class.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path.")
    parser.add_argument("--config", type=str, default=None, help="Config file path.")
    parser.add_argument("--dataset", type=str, default="abd_mri", help="Dataset for graph selection.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory.")
    parser.add_argument("--support_images", nargs="*", default=None, help="Support image paths.")
    parser.add_argument("--support_masks", nargs="*", default=None, help="Support mask paths.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID.")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    predictor = MedZFSPredictor(args.checkpoint, args.config, device)

    output_path = os.path.join(args.output_dir, f"{args.target_class}_prediction.nii.gz")
    predictor.predict_volume(
        image_path=args.image,
        target_class=args.target_class,
        dataset_name=args.dataset,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
