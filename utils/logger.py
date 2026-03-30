"""
Experiment Logger for MedZFS.

Supports TensorBoard and optionally Weights & Biases for experiment
tracking, metric logging, and visualization.
"""

import os
from typing import Optional


class ExperimentLogger:
    """Unified experiment logger supporting TensorBoard and W&B."""

    def __init__(
        self,
        log_dir: str = "./logs",
        experiment_name: str = "medzfs",
        backend: str = "tensorboard",
        wandb_config: Optional[dict] = None,
    ):
        """Initialize the logger.

        Args:
            log_dir: Directory for log files.
            experiment_name: Name of the experiment run.
            backend: Logging backend ("tensorboard" or "wandb").
            wandb_config: W&B configuration dict (project, entity).
        """
        self.backend = backend
        self.experiment_name = experiment_name
        self.writer = None

        if backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(log_dir, experiment_name)
            os.makedirs(tb_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_dir)
            print(f"TensorBoard logging to: {tb_dir}")

        elif backend == "wandb":
            try:
                import wandb
                wandb_config = wandb_config or {}
                wandb.init(
                    project=wandb_config.get("project", "medzfs"),
                    entity=wandb_config.get("entity", None),
                    name=experiment_name,
                )
                self.writer = wandb
                print("W&B logging initialized.")
            except ImportError:
                print("Warning: wandb not installed. Falling back to TensorBoard.")
                self.backend = "tensorboard"
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(log_dir, experiment_name)
                os.makedirs(tb_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=tb_dir)

    def log_metrics(self, metrics: dict, step: int, prefix: str = ""):
        """Log a dictionary of metrics.

        Args:
            metrics: Dict of metric_name → value.
            step: Global step number.
            prefix: Optional prefix (e.g., "train", "val").
        """
        for name, value in metrics.items():
            tag = f"{prefix}/{name}" if prefix else name

            if self.backend == "tensorboard" and self.writer:
                self.writer.add_scalar(tag, value, step)
            elif self.backend == "wandb" and self.writer:
                self.writer.log({tag: value}, step=step)

    def log_image(self, tag: str, image, step: int):
        """Log an image.

        Args:
            tag: Image tag/name.
            image: Image tensor (C, H, W) or numpy array.
            step: Global step.
        """
        if self.backend == "tensorboard" and self.writer:
            self.writer.add_image(tag, image, step)

    def close(self):
        """Close the logger."""
        if self.backend == "tensorboard" and self.writer:
            self.writer.close()
        elif self.backend == "wandb":
            try:
                import wandb
                wandb.finish()
            except ImportError:
                pass
