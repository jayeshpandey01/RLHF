"""Verifiable rule-based reward functions for reasoning tasks (DeepSeek-R1 style)."""

from __future__ import annotations

import re
from typing import Optional, Union


class VerifiableRewardEngine:
    """Deterministic rule-based reward verification for reasoning trajectories.

    Implements:
    1. Format Reward: Enforces <think>...</think><answer>...</answer> structural XML tags.
    2. Accuracy Reward: Extracts predicted answer from <answer> or \\boxed{} and compares with ground truth.
    """

    THINK_REGEX = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    ANSWER_REGEX = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    BOXED_REGEX = re.compile(r"\\boxed\{([^}]+)\}")

    def __init__(
        self,
        format_weight: float = 0.5,
        accuracy_weight: float = 1.0,
    ):
        """Initialize the VerifiableRewardEngine.

        Args:
            format_weight: Weight assigned to valid structural format (default 0.5).
            accuracy_weight: Weight assigned to correct final answer (default 1.0).
        """
        self.format_weight = float(format_weight)
        self.accuracy_weight = float(accuracy_weight)

    def compute_format_reward(self, text: str) -> float:
        """Evaluate whether completion contains properly formed and ordered thinking and answer tags.

        Requirements:
        1. Contains <think> and </think> with non-empty content.
        2. Contains <answer> and </answer> with non-empty content.
        3. <think> precedes <answer>.

        Returns:
            1.0 if fully compliant, 0.0 otherwise.
        """
        think_match = self.THINK_REGEX.search(text)
        answer_match = self.ANSWER_REGEX.search(text)

        if not think_match or not answer_match:
            return 0.0

        think_content = think_match.group(1).strip()
        answer_content = answer_match.group(1).strip()

        if not think_content or not answer_content:
            return 0.0

        # Verify ordering: thinking block must finish before answer block begins
        if think_match.end() > answer_match.start():
            return 0.0

        return 1.0

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract final predicted answer from <answer> tags or \\boxed{} notation.

        Returns:
            Normalized extracted answer string, or None if no tag found.
        """
        # Primary check: <answer>...</answer>
        answer_match = self.ANSWER_REGEX.search(text)
        if answer_match:
            return answer_match.group(1).strip()

        # Fallback check: \boxed{...}
        boxed_match = self.BOXED_REGEX.search(text)
        if boxed_match:
            return boxed_match.group(1).strip()

        return None

    def compute_accuracy_reward(self, text: str, ground_truth: str) -> float:
        """Evaluate mathematical or exact match accuracy of extracted answer against ground truth.

        Supports:
        - String exact match (case/whitespace insensitive).
        - Numerical float equivalence (e.g. "42" == "42.0").

        Returns:
            1.0 if match confirmed, 0.0 otherwise.
        """
        predicted = self.extract_answer(text)
        if predicted is None:
            return 0.0

        pred_clean = predicted.strip().lower()
        gt_clean = str(ground_truth).strip().lower()

        # 1. Exact string match
        if pred_clean == gt_clean:
            return 1.0

        # 2. Numerical equivalence check
        try:
            val_pred = float(pred_clean)
            val_gt = float(gt_clean)
            if abs(val_pred - val_gt) < 1e-5:
                return 1.0
        except ValueError:
            pass

        return 0.0

    def score_completion(
        self,
        completion: str,
        ground_truth: Optional[str] = None,
    ) -> dict[str, float]:
        """Compute composite verifiable reward and sub-metric breakdown.

        Returns:
            Dictionary containing "total_reward", "format_reward", and "accuracy_reward".
        """
        format_r = self.compute_format_reward(completion)

        if ground_truth is not None:
            accuracy_r = self.compute_accuracy_reward(completion, ground_truth)
        else:
            accuracy_r = 0.0

        total_reward = self.format_weight * format_r + self.accuracy_weight * accuracy_r

        return {
            "total_reward": total_reward,
            "format_reward": format_r,
            "accuracy_reward": accuracy_r,
        }

