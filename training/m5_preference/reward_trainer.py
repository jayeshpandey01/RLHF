"""Supervised training loop and evaluation harness for Bradley-Terry Reward Models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import torch
from loguru import logger
from torch.utils.data import DataLoader

from training.m5_preference.reward_model import BradleyTerryLoss, RewardModel
from training.m5_preference.schemas import PreferenceBatch


class RewardTrainer:
    """Supervised trainer for Bradley-Terry reward models."""

    def __init__(
        self,
        model: RewardModel,
        loss_fn: Optional[BradleyTerryLoss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        max_grad_norm: float = 1.0,
    ):
        """Initialize the RewardTrainer.

        Args:
            model: RewardModel instance to optimize.
            loss_fn: BradleyTerryLoss criterion (default margin 0.0, l2_reg 0.001).
            optimizer: Optimizer instance (defaults to AdamW lr=1e-4).
            device: Compute device (MPS, CUDA, CPU).
            max_grad_norm: Gradient clipping norm threshold (default 1.0).
        """
        self.model = model
        self.device = device or model.device
        self.model.to(self.device)

        self.loss_fn = loss_fn or BradleyTerryLoss()
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=0.01
        )
        self.max_grad_norm = float(max_grad_norm)

    def train_step(self, batch: PreferenceBatch) -> dict[str, float]:
        """Execute a single forward-backward optimization step.

        Args:
            batch: PreferenceBatch containing chosen and rejected sequences.

        Returns:
            Dictionary of training step metrics (loss, accuracy, margin, rewards).
        """
        self.model.train()
        batch = batch.to(self.device)

        chosen_rewards = self.model(batch.chosen_input_ids, batch.chosen_attention_mask)
        rejected_rewards = self.model(batch.rejected_input_ids, batch.rejected_attention_mask)

        loss, metrics = self.loss_fn(
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            target_margins=batch.margins,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if self.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        return metrics

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate the reward model over a validation DataLoader.

        Returns:
            Dictionary of aggregated validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_margin = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                chosen_rewards = self.model(batch.chosen_input_ids, batch.chosen_attention_mask)
                rejected_rewards = self.model(batch.rejected_input_ids, batch.rejected_attention_mask)

                loss, metrics = self.loss_fn(
                    chosen_rewards=chosen_rewards,
                    rejected_rewards=rejected_rewards,
                    target_margins=batch.margins,
                )

                total_loss += metrics["loss"]
                total_acc += metrics["accuracy"]
                total_margin += metrics["mean_margin"]
                total_batches += 1

        if total_batches == 0:
            raise ValueError("Validation DataLoader is empty; cannot evaluate model.")

        return {
            "loss": total_loss / total_batches,
            "accuracy": total_acc / total_batches,
            "mean_margin": total_margin / total_batches,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 3,
        eval_every_steps: int = 50,
        early_stopping_patience: int = 3,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> dict[str, list[float]]:
        """Multi-epoch supervised training loop with validation tracking and early stopping.

        Args:
            train_loader: DataLoader yielding training PreferenceBatch instances.
            val_loader: Optional DataLoader yielding validation PreferenceBatch instances.
            epochs: Total number of training epochs.
            eval_every_steps: Interval of optimizer steps between validation runs.
            early_stopping_patience: Consecutive non-improving evaluations before early stopping.
            checkpoint_dir: Directory where best checkpoint will be stored.

        Returns:
            Dictionary containing history lists: train_loss, val_loss, val_accuracy.
        """
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        best_val_loss = float("inf")
        patience_counter = 0
        global_step = 0
        best_checkpoint_path = Path(checkpoint_dir) / "best_reward_model.pt" if checkpoint_dir else None

        for epoch in range(epochs):
            self.model.train()
            for batch in train_loader:
                step_metrics = self.train_step(batch)
                history["train_loss"].append(step_metrics["loss"])
                global_step += 1

                if val_loader is not None and global_step % eval_every_steps == 0:
                    val_metrics = self.evaluate(val_loader)
                    history["val_loss"].append(val_metrics["loss"])
                    history["val_accuracy"].append(val_metrics["accuracy"])
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs} Step {global_step}: "
                        f"Train Loss = {step_metrics['loss']:.4f}, "
                        f"Val Loss = {val_metrics['loss']:.4f}, "
                        f"Val Acc = {val_metrics['accuracy']:.4f}"
                    )

                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        patience_counter = 0
                        if best_checkpoint_path is not None:
                            self.save_checkpoint(best_checkpoint_path)
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            logger.info(
                                f"Early stopping triggered at step {global_step} "
                                f"(patience={early_stopping_patience})."
                            )
                            if best_checkpoint_path is not None and best_checkpoint_path.exists():
                                self.load_checkpoint(best_checkpoint_path)
                            return history

            # End of epoch evaluation if val_loader provided
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                    if best_checkpoint_path is not None:
                        self.save_checkpoint(best_checkpoint_path)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                        if best_checkpoint_path is not None and best_checkpoint_path.exists():
                            self.load_checkpoint(best_checkpoint_path)
                        return history

        return history

    def save_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Persist model and optimizer state to disk."""
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.model.backbone.config,
            },
            path,
        )
        logger.info(f"Saved RewardModel checkpoint to {path}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load state dict into reward model and optimizer."""
        path = Path(checkpoint_path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded RewardModel checkpoint from {path}")
