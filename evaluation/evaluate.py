"""
MedZFS Evaluation Pipeline.

Evaluates a trained model across all datasets and shot settings,
producing comprehensive result tables matching the paper format.

Usage:
    python -m evaluation.evaluate \
        --config configs/eval.yaml \
        --checkpoint checkpoints/best_model.pth \
        --shots 0 1 5 10
"""

import argparse
import os
import json
from collections import defaultdict

import numpy as np
import torch
import yaml
from tqdm import tqdm

from data.dataset import EpisodicMedicalDataset
from data.anatomical_graphs import AnatomicalGraphBuilder
from evaluation.metrics import dice_score, hausdorff_distance
from models.medzfs import MedZFS


class MedZFSEvaluator:
    """Comprehensive evaluation driver for MedZFS."""

    def __init__(self, model, device, anatomical_graph):
        self.model = model
        self.device = device
        self.anat_graph = anatomical_graph

        edge_index, edge_type = self.anat_graph.get_full_edge_index()
        self.edge_index = edge_index.to(device)
        self.edge_type = edge_type.to(device)
        self.node_descriptions = self.anat_graph.get_node_descriptions()

    @torch.no_grad()
    def evaluate_dataset(self, dataloader, shots=None, num_episodes=600):
        """Evaluate on a dataset across specified shot settings.

        Args:
            dataloader: Evaluation DataLoader.
            shots: List of shot values to evaluate.
            num_episodes: Number of evaluation episodes.

        Returns:
            Dict mapping shot → {class_name → {dice, hd}}.
        """
        if shots is None:
            shots = [0, 1, 5, 10]

        self.model.eval()
        results = {k: defaultdict(list) for k in shots}

        for batch in tqdm(dataloader, desc="Evaluating"):
            query_image = batch["query_image"].to(self.device)
            query_mask = batch["query_mask"]
            support_images = batch["support_images"].to(self.device)
            support_masks = batch["support_masks"].to(self.device)
            class_name = batch["class_name"][0]
            class_desc = batch["class_description"][0]
            k = batch["num_shots"]
            k_val = k[0].item() if isinstance(k, torch.Tensor) else k

            if k_val not in shots:
                continue

            outputs = self.model(
                query_image=query_image,
                class_description=class_desc,
                node_descriptions=self.node_descriptions,
                edge_index=self.edge_index,
                edge_type=self.edge_type,
                support_images=support_images if k_val > 0 else None,
                support_masks=support_masks if k_val > 0 else None,
            )

            pred = outputs["prediction"].cpu().numpy()
            gt = query_mask.numpy()

            for b in range(pred.shape[0]):
                pred_b = (pred[b] > 0.5).astype(np.float32)
                gt_b = gt[b].astype(np.float32)

                d = dice_score(pred_b, gt_b)
                hd = hausdorff_distance(pred_b, gt_b)

                results[k_val][class_name].append({"dice": d, "hd": hd})

        # Aggregate
        summary = {}
        for k_val in shots:
            summary[k_val] = {}
            for class_name, metrics_list in results[k_val].items():
                dices = [m["dice"] for m in metrics_list]
                hds = [m["hd"] for m in metrics_list if m["hd"] != float("inf")]
                summary[k_val][class_name] = {
                    "dice_mean": np.mean(dices) * 100 if dices else 0,
                    "dice_std": np.std(dices) * 100 if dices else 0,
                    "hd_mean": np.mean(hds) if hds else float("inf"),
                    "num_episodes": len(dices),
                }

        return summary

    def print_results(self, summary, dataset_name):
        """Print formatted results table."""
        print(f"\n{'='*60}")
        print(f"  Results: {dataset_name}")
        print(f"{'='*60}")

        for k_val in sorted(summary.keys()):
            print(f"\n  {k_val}-Shot:")
            print(f"  {'Class':<20} {'Dice (%)':>12} {'HD95':>10} {'N':>6}")
            print(f"  {'-'*50}")

            all_dices = []
            for cls_name, metrics in sorted(summary[k_val].items()):
                dice_str = f"{metrics['dice_mean']:.1f}±{metrics['dice_std']:.1f}"
                hd_str = f"{metrics['hd_mean']:.1f}" if metrics['hd_mean'] != float('inf') else "N/A"
                print(f"  {cls_name:<20} {dice_str:>12} {hd_str:>10} {metrics['num_episodes']:>6}")
                all_dices.append(metrics['dice_mean'])

            if all_dices:
                print(f"  {'MEAN':<20} {np.mean(all_dices):>12.1f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MedZFS.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 1, 5, 10])
    parser.add_argument("--datasets", nargs="+", default=["abd_mri", "abd_ct", "cmr"])
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load model
    model = MedZFS(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for dataset_name in args.datasets:
        print(f"\nEvaluating on: {dataset_name}")
        anat_graph = AnatomicalGraphBuilder.build_graph(dataset_name)
        evaluator = MedZFSEvaluator(model, device, anat_graph)

        # Load dataset-specific config
        ds_config_path = f"configs/train_{dataset_name}.yaml"
        if os.path.exists(ds_config_path):
            with open(ds_config_path) as f:
                ds_config = yaml.safe_load(f)
        else:
            ds_config = config

        data_cfg = ds_config.get("data", config.get("data", {}))
        dataset = EpisodicMedicalDataset(
            data_dir=data_cfg.get("data_dir", f"./data/processed/{dataset_name}"),
            classes=data_cfg.get("classes", []),
            shots=args.shots,
            num_episodes=config.get("evaluation", {}).get("num_episodes", 600),
            image_size=tuple(data_cfg.get("image_size", [256, 256])),
            split="test",
        )

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)
        summary = evaluator.evaluate_dataset(dataloader, shots=args.shots)
        evaluator.print_results(summary, dataset_name)
        all_results[dataset_name] = summary

    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    # Convert for JSON serialization
    json_results = {}
    for ds, k_results in all_results.items():
        json_results[ds] = {}
        for k_val, cls_results in k_results.items():
            json_results[ds][str(k_val)] = cls_results

    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
