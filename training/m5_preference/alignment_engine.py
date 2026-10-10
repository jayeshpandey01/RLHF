"""Dynamic Alignment Engine factory for selecting and configuring preference learning paradigms."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from loguru import logger

from training.m1_transformer.model import MiniGPT
from training.m5_preference.critic import CriticModel
from training.m5_preference.dpo_loss import DPOLoss
from training.m5_preference.dpo_trainer import DPOTrainer
from training.m5_preference.grpo_loss import GRPOLoss
from training.m5_preference.grpo_trainer import GRPOTrainer
from training.m5_preference.ppo_loss import PPOLoss
from training.m5_preference.ppo_trainer import PPOTrainer
from training.m5_preference.reference_model import ReferenceModelEngine
from training.m5_preference.reward_model import BradleyTerryLoss, RewardModel
from training.m5_preference.reward_trainer import RewardTrainer
from training.m5_preference.verifiable_reward import VerifiableRewardEngine


class AlignmentStrategy(str, Enum):
    """Supported post-training alignment strategies in Milestone 5."""

    REWARD_MODEL = "reward_model"
    DPO = "dpo"
    PPO = "ppo"
    GRPO = "grpo"


@dataclass
class AlignmentConfig:
    """Unified hyperparameters for post-training alignment paradigms."""

    strategy: AlignmentStrategy = AlignmentStrategy.DPO
    lr: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # DPO specific
    beta: float = 0.1
    dpo_loss_type: str = "sigmoid"
    label_smoothing: float = 0.0

    # PPO / GRPO specific
    clip_ratio: float = 0.2
    clip_value: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.04
    gamma: float = 1.0
    lam: float = 0.95
    group_size: int = 4

    # Reward Model specific
    rm_margin: float = 0.0
    rm_l2_reg: float = 0.001


class AlignmentEngine:
    """Unified factory pattern for instantiating post-training alignment trainers."""

    @staticmethod
    def create_trainer(
        strategy: Union[AlignmentStrategy, str],
        model: nn.Module,
        ref_model: Optional[MiniGPT] = None,
        critic: Optional[CriticModel] = None,
        reward_model: Optional[RewardModel] = None,
        reward_engine: Optional[VerifiableRewardEngine] = None,
        config: Optional[AlignmentConfig] = None,
        device: Optional[torch.device] = None,
    ) -> Any:
        """Instantiate the appropriate alignment trainer based on selected strategy.

        Args:
            strategy: Alignment strategy ("reward_model", "dpo", "ppo", "grpo").
            model: Target neural network (MiniGPT or RewardModel).
            ref_model: Optional reference policy (MiniGPT).
            critic: Optional state-value Critic network (for PPO).
            reward_model: Optional RewardModel (for PPO).
            reward_engine: Optional VerifiableRewardEngine (for GRPO).
            config: Optional AlignmentConfig hyperparameters.
            device: Compute device.

        Returns:
            Configured trainer instance (RewardTrainer, DPOTrainer, PPOTrainer, or GRPOTrainer).
        """
        if isinstance(strategy, str):
            strategy = AlignmentStrategy(strategy.lower())

        cfg = config or AlignmentConfig(strategy=strategy)
        dev = device or getattr(model, "device", torch.device("cpu"))

        logger.info(f"Instantiating AlignmentEngine for strategy: {strategy.value}")

        if strategy == AlignmentStrategy.REWARD_MODEL:
            assert isinstance(model, RewardModel), "Model must be RewardModel for REWARD_MODEL strategy"
            loss_fn = BradleyTerryLoss(margin=cfg.rm_margin, l2_reg=cfg.rm_l2_reg)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
            return RewardTrainer(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=dev,
                max_grad_norm=cfg.max_grad_norm,
            )

        elif strategy == AlignmentStrategy.DPO:
            assert isinstance(model, MiniGPT), "Model must be MiniGPT for DPO strategy"
            ref_engine = (
                ReferenceModelEngine(ref_model, device=dev) if ref_model is not None else None
            )
            loss_fn = DPOLoss(
                beta=cfg.beta,
                loss_type=cfg.dpo_loss_type,
                label_smoothing=cfg.label_smoothing,
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
            return DPOTrainer(
                model=model,
                ref_engine=ref_engine,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=dev,
                max_grad_norm=cfg.max_grad_norm,
            )

        elif strategy == AlignmentStrategy.PPO:
            assert isinstance(model, MiniGPT), "Model must be MiniGPT for PPO strategy"
            critic_net = critic or CriticModel(MiniGPT(model.config))
            loss_fn = PPOLoss(
                clip_ratio=cfg.clip_ratio,
                clip_value=cfg.clip_value,
                vf_coef=cfg.vf_coef,
                entropy_coef=cfg.entropy_coef,
            )
            actor_optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
            critic_optimizer = torch.optim.AdamW(
                critic_net.parameters(), lr=cfg.lr * 2, weight_decay=cfg.weight_decay
            )
            return PPOTrainer(
                actor=model,
                critic=critic_net,
                ref_model=ref_model,
                reward_model=reward_model,
                loss_fn=loss_fn,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                kl_coef=cfg.kl_coef,
                gamma=cfg.gamma,
                lam=cfg.lam,
                max_grad_norm=cfg.max_grad_norm,
                device=dev,
            )

        elif strategy == AlignmentStrategy.GRPO:
            assert isinstance(model, MiniGPT), "Model must be MiniGPT for GRPO strategy"
            loss_fn = GRPOLoss(
                clip_ratio=cfg.clip_ratio,
                kl_coef=cfg.kl_coef,
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
            return GRPOTrainer(
                actor=model,
                ref_model=ref_model,
                reward_engine=reward_engine or VerifiableRewardEngine(),
                loss_fn=loss_fn,
                optimizer=optimizer,
                group_size=cfg.group_size,
                max_grad_norm=cfg.max_grad_norm,
                device=dev,
            )

        else:
            raise ValueError(f"Unsupported alignment strategy: {strategy}")

