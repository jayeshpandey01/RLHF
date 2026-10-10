"""Direct Preference Optimization (DPO) trainer and evaluation loop."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from training.m5_preference.dpo_loss import DPOLoss
from training.m5_preference.log_probs import extract_batch_log_probs
from training.m5_preference.reference_model import ReferenceModelEngine
from training.m5_preference.schemas import PreferenceBatch


class DPOTrainer:
    """Supervised trainer for Direct Preference Optimization (DPO)."""

    def __init__(
        self,
        model: nn.Module,
        ref_engine: Optional[ReferenceModelEngine] = None,
        loss_fn: Optional[DPOLoss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        max_grad_norm: float = 1.0,
        length_normalize: bool = False,
    ):
        """Initialize the DPOTrainer.

        Args:
            model: Policy language model (MiniGPT) to optimize.
            ref_engine: Optional ReferenceModelEngine. If omitted and batch lacks cached
                        reference logps, DPO will run reference-free.
            loss_fn: DPOLoss criterion (default beta=0.1, sigmoid).
            optimizer: Optimizer instance (defaults to AdamW lr=5e-5).
            device: Compute device (MPS, CUDA, CPU).
            max_grad_norm: Gradient clipping threshold (default 1.0).
            length_normalize: Whether to length-normalize completion log-probs.
        """
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.ref_engine = ref_engine
        self.loss_fn = loss_fn or DPOLoss(beta=0.1)
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(), lr=5e-5, weight_decay=0.01
        )
        self.max_grad_norm = float(max_grad_norm)
        self.length_normalize = bool(length_normalize)

    def train_step(self, batch: PreferenceBatch) -> dict[str, float]:
        """Execute a single forward-backward DPO optimization step.

        Args:
            batch: PreferenceBatch containing chosen and rejected sequences.

        Returns:
            Dictionary of training step metrics (loss, chosen_rewards, rejected_rewards,
            mean_margin, accuracy, kl_divergence).
        """
        self.model.train()
        batch = batch.to(self.device)

        # 1. Forward policy model for chosen and rejected sequences
        policy_chosen_logps = extract_batch_log_probs(
            model=self.model,
            input_ids=batch.chosen_input_ids,
            attention_mask=batch.chosen_attention_mask,
            prompt_lengths=batch.prompt_lengths,
            length_normalize=self.length_normalize,
        )

        policy_rejected_logps = extract_batch_log_probs(
            model=self.model,
            input_ids=batch.rejected_input_ids,
            attention_mask=batch.rejected_attention_mask,
            prompt_lengths=batch.prompt_lengths,
            length_normalize=self.length_normalize,
        )

        # 2. Acquire reference log-probabilities (cached or via ref engine)
        if batch.ref_chosen_logps is not None and batch.ref_rejected_logps is not None:
            ref_chosen_logps = batch.ref_chosen_logps.to(self.device)
            ref_rejected_logps = batch.ref_rejected_logps.to(self.device)
        elif self.ref_engine is not None:
            ref_chosen_logps, ref_rejected_logps = self.ref_engine.compute_batch_log_probs(batch)
        else:
            ref_chosen_logps, ref_rejected_logps = None, None

        # 3. Compute DPO loss & diagnostics
        loss, metrics = self.loss_fn(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=ref_chosen_logps,
            reference_rejected_logps=ref_rejected_logps,
        )

        # 4. Backward pass and optimizer step
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if self.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate current policy on an evaluation dataloader.

        Args:
            dataloader: DataLoader yielding PreferenceBatch objects.

        Returns:
            Dictionary of aggregated evaluation metrics.
        """
        self.model.eval()
        total_metrics: dict[str, float] = {}
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(self.device)
            policy_chosen_logps = extract_batch_log_probs(
                model=self.model,
                input_ids=batch.chosen_input_ids,
                attention_mask=batch.chosen_attention_mask,
                prompt_lengths=batch.prompt_lengths,
                length_normalize=self.length_normalize,
            )
            policy_rejected_logps = extract_batch_log_probs(
                model=self.model,
                input_ids=batch.rejected_input_ids,
                attention_mask=batch.rejected_attention_mask,
                prompt_lengths=batch.prompt_lengths,
                length_normalize=self.length_normalize,
            )

            if batch.ref_chosen_logps is not None and batch.ref_rejected_logps is not None:
                ref_chosen_logps = batch.ref_chosen_logps.to(self.device)
                ref_rejected_logps = batch.ref_rejected_logps.to(self.device)
            elif self.ref_engine is not None:
                ref_chosen_logps, ref_rejected_logps = self.ref_engine.compute_batch_log_probs(batch)
            else:
                ref_chosen_logps, ref_rejected_logps = None, None

            _, step_metrics = self.loss_fn(
                policy_chosen_logps=policy_chosen_logps,
                policy_rejected_logps=policy_rejected_logps,
                reference_chosen_logps=ref_chosen_logps,
                reference_rejected_logps=ref_rejected_logps,
            )

            for k, v in step_metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            num_batches += 1

        if num_batches == 0:
            raise ValueError("Evaluation DataLoader is empty; cannot evaluate model.")

        return {k: v / num_batches for k, v in total_metrics.items()}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 3,
        eval_every_steps: int = 50,
        early_stopping_patience: int = 3,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> dict[str, list[float]]:
        """Multi-epoch supervised DPO training loop with validation tracking and early stopping.

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
        best_checkpoint_path = Path(checkpoint_dir) / "best_dpo_model.pt" if checkpoint_dir else None

        for epoch in range(epochs):
            self.model.train()
            for batch in train_loader:
                step_metrics = self.train_step(batch)
                history["train_loss"].append(step_metrics["loss"])
                global_step += 1

                if val_loader is not None and global_step % eval_every_steps == 0:
                    val_metrics = self.evaluate(val_loader)
                    history["val_loss"].append(val_metrics["loss"])
                    history["val_accuracy"].append(val_metrics.get("accuracy", 0.0))
                    logger.info(
                        f"DPO Epoch {epoch + 1}/{epochs} Step {global_step}: "
                        f"Train Loss = {step_metrics['loss']:.4f}, "
                        f"Val Loss = {val_metrics['loss']:.4f}, "
                        f"Val Acc = {val_metrics.get('accuracy', 0.0):.4f}"
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
                                f"DPO early stopping triggered at step {global_step} "
                                f"(patience={early_stopping_patience})."
                            )
                            if best_checkpoint_path is not None and best_checkpoint_path.exists():
                                self.load_checkpoint(best_checkpoint_path)
                            return history

            # End of epoch evaluation if val_loader provided
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics.get("accuracy", 0.0))
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                    if best_checkpoint_path is not None:
                        self.save_checkpoint(best_checkpoint_path)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"DPO early stopping triggered at epoch {epoch + 1}.")
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
                "config": getattr(self.model, "config", None),
            },
            path,
        )
        logger.info(f"Saved DPOTrainer checkpoint to {path}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load state dict into policy model and optimizer."""
        path = Path(checkpoint_path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded DPOTrainer checkpoint from {path}")

