"""Preference Learning and Reinforcement Learning Package (Milestone 5)."""

from training.m5_preference.schemas import (
    PreferenceSample,
    PreferenceBatch,
)
from training.m5_preference.dataset import (
    PreferenceCollator,
    PreferenceDataset,
    create_default_preference_pairs,
)
from training.m5_preference.reward_model import (
    RewardModel,
    BradleyTerryLoss,
)
from training.m5_preference.reward_trainer import (
    RewardTrainer,
)
from training.m5_preference.log_probs import (
    extract_completion_log_probs,
    extract_batch_log_probs,
)
from training.m5_preference.dpo_loss import (
    DPOLoss,
)
from training.m5_preference.reference_model import (
    ReferenceModelEngine,
)
from training.m5_preference.dpo_trainer import (
    DPOTrainer,
)
from training.m5_preference.critic import (
    ValueHead,
    CriticModel,
)
from training.m5_preference.gae import (
    PPORolloutBatch,
    PPORolloutBuffer,
    compute_gae_advantages,
)
from training.m5_preference.ppo_loss import (
    PPOLoss,
)
from training.m5_preference.ppo_trainer import (
    PPOTrainer,
    compute_token_log_probs_and_entropy,
)
from training.m5_preference.verifiable_reward import (
    VerifiableRewardEngine,
)
from training.m5_preference.grpo_loss import (
    compute_group_advantages,
    GRPOLoss,
)
from training.m5_preference.grpo_trainer import (
    GRPOGroupBatch,
    GRPOTrainer,
)
from training.m5_preference.alignment_engine import (
    AlignmentStrategy,
    AlignmentConfig,
    AlignmentEngine,
)
from training.m5_preference.benchmark_suite import (
    AlignmentTestCase,
    AlignmentBenchmarkReport,
    AlignmentBenchmarkSuite,
    create_default_alignment_test_cases,
)

__all__ = [
    "PreferenceSample",
    "PreferenceBatch",
    "PreferenceCollator",
    "PreferenceDataset",
    "create_default_preference_pairs",
    "RewardModel",
    "BradleyTerryLoss",
    "RewardTrainer",
    "extract_completion_log_probs",
    "extract_batch_log_probs",
    "DPOLoss",
    "ReferenceModelEngine",
    "DPOTrainer",
    "ValueHead",
    "CriticModel",
    "PPORolloutBatch",
    "PPORolloutBuffer",
    "compute_gae_advantages",
    "PPOLoss",
    "PPOTrainer",
    "compute_token_log_probs_and_entropy",
    "VerifiableRewardEngine",
    "compute_group_advantages",
    "GRPOLoss",
    "GRPOGroupBatch",
    "GRPOTrainer",
    "AlignmentStrategy",
    "AlignmentConfig",
    "AlignmentEngine",
    "AlignmentTestCase",
    "AlignmentBenchmarkReport",
    "AlignmentBenchmarkSuite",
    "create_default_alignment_test_cases",
]



