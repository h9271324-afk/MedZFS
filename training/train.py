"""
MedZFS Training Entry Point.

Parses configuration, creates dataloaders, model, and optimizer,
then launches the three-stage training pipeline.

Usage:
    python -m training.train --config configs/train_abd_mri.yaml
    python -m training.train --config configs/train_abd_mri.yaml --stage 1
"""

import argparse
import os
import sys

import torch
import yaml

from data.dataset import EpisodicMedicalDataset
from data.transforms import MedicalTransforms
from data.anatomical_graphs import AnatomicalGraphBuilder
from models.medzfs import MedZFS
from models.loss_functions import MedZFSLoss
from training.trainer import MedZFSTrainer
from utils.seed import set_seed
from utils.logger import ExperimentLogger


def load_config(config_path: str) -> dict:
    """Load and merge YAML configuration files.

    Supports _base_ inheritance: the specified config inherits
    from a base config and overrides selected fields.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Handle base config inheritance
    if "_base_" in config:
        base_path = os.path.join(os.path.dirname(config_path), config["_base_"])
        base_config = load_config(base_path)
        # Deep merge: config overrides base_config
        merged = _deep_merge(base_config, config)
        merged.pop("_base_", None)
        return merged

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def create_dataloaders(config: dict) -> dict:
    """Create training and validation dataloaders.

    Args:
        config: Full configuration dictionary.

    Returns:
        Dict with 'train' and 'val' DataLoader objects.
    """
    data_cfg = config["data"]

    transforms = MedicalTransforms(data_cfg) if data_cfg.get("augmentation", {}).get("enabled", True) else None
    train_cfg = config.get("training", {})

    train_dataset = EpisodicMedicalDataset(
        data_dir=data_cfg["data_dir"],
        classes=data_cfg["classes"],
        shots=data_cfg.get("shots", [0, 1, 5, 10]),
        num_episodes=data_cfg.get("num_episodes", 1000),
        image_size=tuple(data_cfg.get("image_size", [256, 256])),
        transforms=transforms,
        split="train",
    )

    val_dataset = EpisodicMedicalDataset(
        data_dir=data_cfg["data_dir"],
        classes=data_cfg["classes"],
        shots=data_cfg.get("shots", [0, 1, 5, 10]),
        num_episodes=config.get("evaluation", {}).get("num_episodes", 600),
        image_size=tuple(data_cfg.get("image_size", [256, 256])),
        transforms=None,
        split="val",
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 8),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader}


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train MedZFS model.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--stage", type=int, default=0, help="Training stage (1, 2, 3 or 0 for all).")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--distributed", action="store_true", help="Use distributed training.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set seed
    seed = args.seed or config.get("training", {}).get("seed", 42)
    set_seed(seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Build anatomical graph
    dataset_name = config["data"].get("dataset_name", "abd_mri")
    anat_graph = AnatomicalGraphBuilder.build_graph(dataset_name)

    # Create model
    model = MedZFS(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create loss function
    loss_cfg = config.get("loss", {})
    criterion = MedZFSLoss(
        feature_dim=config["model"].get("feature_dim", 512),
        num_relations=config["model"].get("graph_network", {}).get("num_relation_types", 4),
        bce_weight=loss_cfg.get("segmentation", {}).get("bce_weight", 0.5),
        dice_weight=loss_cfg.get("segmentation", {}).get("dice_weight", 0.5),
        lambda_graph=loss_cfg.get("lambda_graph", 0.1),
        lambda_align=loss_cfg.get("lambda_align", 0.05),
        lambda_boundary=loss_cfg.get("lambda_boundary", 0.1),
    ).to(device)

    # Create dataloaders
    dataloaders = create_dataloaders(config)

    # Create logger
    log_cfg = config.get("logging", {})
    logger = ExperimentLogger(
        log_dir=log_cfg.get("log_dir", "./logs"),
        experiment_name=log_cfg.get("experiment_name", "medzfs"),
        backend=log_cfg.get("backend", "tensorboard"),
    )

    # Create trainer
    trainer = MedZFSTrainer(
        model=model,
        criterion=criterion,
        dataloaders=dataloaders,
        anatomical_graph=anat_graph,
        config=config,
        device=device,
        logger=logger,
    )

    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Run training
    if args.stage == 0:
        # Run all stages sequentially
        trainer.train_all_stages(lr_override=args.lr, epochs_override=args.epochs)
    else:
        trainer.train_stage(
            stage=args.stage,
            lr_override=args.lr,
            epochs_override=args.epochs,
        )

    print("Training complete!")


if __name__ == "__main__":
    main()
