"""
MedZFS Three-Stage Bilevel Training Loop.

Stage 1: Zero-shot pre-training (hallucination + graph, lr=1e-4)
Stage 2: Episodic meta-learning (hard prototype mining, lr=1e-3)
Stage 3: Joint bilevel optimization (all pathways, lr=5e-5)
"""

import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from training.scheduler import WarmupCosineScheduler


class MedZFSTrainer:
    """Three-stage bilevel trainer for MedZFS."""

    def __init__(self, model, criterion, dataloaders, anatomical_graph, config, device, logger):
        self.model = model
        self.criterion = criterion
        self.dataloaders = dataloaders
        self.anat_graph = anatomical_graph
        self.config = config
        self.device = device
        self.logger = logger
        self.global_step = 0
        self.best_dice = 0.0

        # Pre-compute graph tensors
        edge_index, edge_type = self.anat_graph.get_full_edge_index()
        self.edge_index = edge_index.to(device)
        self.edge_type = edge_type.to(device)
        self.node_descriptions = self.anat_graph.get_node_descriptions()

        # Mixed precision
        self.use_amp = config.get("training", {}).get("mixed_precision", True)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.grad_clip = config.get("training", {}).get("gradient_clip", 1.0)

        # Checkpoint directory
        self.ckpt_dir = config.get("logging", {}).get("checkpoint_dir", "./checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def _create_optimizer(self, stage: int, lr_override: Optional[float] = None):
        """Create optimizer for the given training stage."""
        stages_cfg = self.config.get("training", {}).get("stages", {})
        stage_key = f"stage{stage}"
        stage_cfg = stages_cfg.get(stage_key, {})

        lr = lr_override or stage_cfg.get("lr", 1e-4)
        wd = stage_cfg.get("weight_decay", 1e-4)

        # Stage-specific parameter groups
        if stage == 1:
            # Train hallucination + graph only
            params = list(self.model.hallucinator.parameters()) + \
                     list(self.model.graph_network.parameters()) + \
                     list(self.model.fusion.parameters()) + \
                     list(self.model.text_encoder.projection.parameters())
        elif stage == 2:
            # Add prototype miner
            params = list(self.model.hallucinator.parameters()) + \
                     list(self.model.graph_network.parameters()) + \
                     list(self.model.fusion.parameters()) + \
                     list(self.model.prototype_miner.parameters()) + \
                     list(self.model.text_encoder.projection.parameters())
        else:
            # Stage 3: All parameters including unfrozen text encoder
            self.model.text_encoder.unfreeze_backbone()
            params = [
                {"params": self.model.visual_encoder.parameters(), "lr": lr * 0.1},
                {"params": self.model.text_encoder.parameters(), "lr": lr * 0.5},
                {"params": self.model.hallucinator.parameters(), "lr": lr},
                {"params": self.model.graph_network.parameters(), "lr": lr},
                {"params": self.model.fusion.parameters(), "lr": lr},
                {"params": self.model.prototype_miner.parameters(), "lr": lr},
            ]

        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        return optimizer, stage_cfg

    def train_all_stages(self, lr_override=None, epochs_override=None):
        """Run all three training stages sequentially."""
        for stage in [1, 2, 3]:
            print(f"\n{'='*60}")
            print(f"  STAGE {stage}")
            print(f"{'='*60}")
            self.train_stage(stage, lr_override, epochs_override)

    def train_stage(self, stage: int, lr_override=None, epochs_override=None):
        """Train a single stage."""
        optimizer, stage_cfg = self._create_optimizer(stage, lr_override)
        epochs = epochs_override or stage_cfg.get("epochs", 50)
        warmup = stage_cfg.get("warmup_epochs", 5)

        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=warmup, total_epochs=epochs)

        print(f"Stage {stage}: {stage_cfg.get('description', '')}")
        print(f"  Epochs: {epochs}, LR: {optimizer.param_groups[0]['lr']:.2e}")

        for epoch in range(1, epochs + 1):
            train_metrics = self._train_epoch(optimizer, epoch, stage)
            val_metrics = self._validate_epoch(epoch, stage)

            scheduler.step()

            # Logging
            self.logger.log_metrics(train_metrics, self.global_step, prefix="train")
            self.logger.log_metrics(val_metrics, self.global_step, prefix="val")

            # Checkpoint
            log_cfg = self.config.get("logging", {})
            if epoch % log_cfg.get("save_interval", 10) == 0:
                self.save_checkpoint(f"stage{stage}_epoch{epoch}.pth")

            if val_metrics.get("dice", 0) > self.best_dice:
                self.best_dice = val_metrics["dice"]
                self.save_checkpoint(f"stage{stage}_best.pth")
                print(f"  ★ New best Dice: {self.best_dice:.4f}")

    def _train_epoch(self, optimizer, epoch, stage):
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0

        loader = self.dataloaders["train"]
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

        for batch in pbar:
            optimizer.zero_grad()

            query_image = batch["query_image"].to(self.device)
            query_mask = batch["query_mask"].to(self.device)
            support_images = batch["support_images"].to(self.device)
            support_masks = batch["support_masks"].to(self.device)
            class_desc = batch["class_description"][0]
            k = batch["num_shots"][0].item() if isinstance(batch["num_shots"], torch.Tensor) else batch["num_shots"]

            # Stage 1: zero-shot only
            if stage == 1:
                support_images = None
                support_masks = None

            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    query_image=query_image,
                    class_description=class_desc,
                    node_descriptions=self.node_descriptions,
                    edge_index=self.edge_index,
                    edge_type=self.edge_type,
                    support_images=support_images if (stage >= 2 and k > 0) else None,
                    support_masks=support_masks if (stage >= 2 and k > 0) else None,
                )

                losses = self.criterion(
                    logits=outputs["logits"],
                    target=query_mask,
                    hallucinated_prototypes=outputs["hallucinated_prototypes"],
                    hard_prototypes=outputs.get("hard_prototypes"),
                    graph_embeddings=outputs["graph_embeddings"].unsqueeze(0),
                    edge_index=self.edge_index,
                    edge_type=self.edge_type,
                )

            self.scaler.scale(losses["total"]).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()

            # Compute Dice for monitoring
            with torch.no_grad():
                pred = (outputs["prediction"] > 0.5).float()
                intersection = (pred * query_mask).sum()
                dice = (2 * intersection + 1) / (pred.sum() + query_mask.sum() + 1)

            total_loss += losses["total"].item()
            total_dice += dice.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix(loss=f"{losses['total'].item():.4f}", dice=f"{dice.item():.4f}")

        return {
            "loss": total_loss / max(num_batches, 1),
            "dice": total_dice / max(num_batches, 1),
        }

    @torch.no_grad()
    def _validate_epoch(self, epoch, stage):
        """Run validation."""
        self.model.eval()
        total_dice = 0.0
        num_batches = 0

        for batch in self.dataloaders["val"]:
            query_image = batch["query_image"].to(self.device)
            query_mask = batch["query_mask"].to(self.device)
            support_images = batch["support_images"].to(self.device)
            support_masks = batch["support_masks"].to(self.device)
            class_desc = batch["class_description"][0]
            k = batch["num_shots"][0].item() if isinstance(batch["num_shots"], torch.Tensor) else batch["num_shots"]

            outputs = self.model(
                query_image=query_image,
                class_description=class_desc,
                node_descriptions=self.node_descriptions,
                edge_index=self.edge_index,
                edge_type=self.edge_type,
                support_images=support_images if (stage >= 2 and k > 0) else None,
                support_masks=support_masks if (stage >= 2 and k > 0) else None,
            )

            pred = (outputs["prediction"] > 0.5).float()
            intersection = (pred * query_mask).sum()
            dice = (2 * intersection + 1) / (pred.sum() + query_mask.sum() + 1)
            total_dice += dice.item()
            num_batches += 1

        avg_dice = total_dice / max(num_batches, 1)
        print(f"  Val Dice: {avg_dice:.4f}")
        return {"dice": avg_dice}

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.ckpt_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "global_step": self.global_step,
            "best_dice": self.best_dice,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.global_step = ckpt.get("global_step", 0)
        self.best_dice = ckpt.get("best_dice", 0.0)
        print(f"Loaded checkpoint from {path} (step={self.global_step})")
