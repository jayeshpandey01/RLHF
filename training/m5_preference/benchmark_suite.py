"""Unified Alignment Benchmark Suite and Multi-Paradigm Comparative Evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from loguru import logger

from evaluation.llm_judge import LLMJudge
from evaluation.statistical_gating import GateDecision
from training.m5_preference.verifiable_reward import VerifiableRewardEngine


@dataclass
class AlignmentTestCase:
    """Individual test prompt for post-training alignment benchmarking."""

    prompt: str
    ground_truth: Optional[str] = None
    domain: str = "reasoning"  # "reasoning", "instruction", "dialogue"


@dataclass
class AlignmentBenchmarkReport:
    """Structured report summarizing comparative alignment performance."""

    candidate_name: str
    win_rate: float
    format_compliance_rate: float
    reasoning_accuracy_rate: float
    mean_score: float
    gate_decision: GateDecision
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to serializable dictionary."""
        return {
            "candidate_name": self.candidate_name,
            "win_rate": self.win_rate,
            "format_compliance_rate": self.format_compliance_rate,
            "reasoning_accuracy_rate": self.reasoning_accuracy_rate,
            "mean_score": self.mean_score,
            "gate_decision": self.gate_decision.value,
            "metrics": self.metrics,
        }


def create_default_alignment_test_cases() -> list[AlignmentTestCase]:
    """Curate foundational test cases across mathematical reasoning and instruction domains."""
    return [
        AlignmentTestCase(
            prompt="Compute the solution to: If 3x + 7 = 22, what is x?",
            ground_truth="5",
            domain="reasoning",
        ),
        AlignmentTestCase(
            prompt="What is the square root of 144 plus 16?",
            ground_truth="28",
            domain="reasoning",
        ),
        AlignmentTestCase(
            prompt="What is the capital of France and explain why in two sentences.",
            ground_truth="Paris",
            domain="instruction",
        ),
        AlignmentTestCase(
            prompt="Explain why gradient clipping stabilizes Transformer training.",
            ground_truth=None,
            domain="instruction",
        ),
        AlignmentTestCase(
            prompt="Calculate 15 * 8 - 40.",
            ground_truth="80",
            domain="reasoning",
        ),
    ]


class AlignmentBenchmarkSuite:
    """Unified comparative benchmark suite evaluating post-training alignment paradigms."""

    def __init__(
        self,
        test_cases: Optional[list[AlignmentTestCase]] = None,
        reward_engine: Optional[VerifiableRewardEngine] = None,
        llm_judge: Optional[LLMJudge] = None,
    ):
        """Initialize the AlignmentBenchmarkSuite.

        Args:
            test_cases: List of AlignmentTestCase prompts (defaults to curated cases).
            reward_engine: VerifiableRewardEngine for format and accuracy evaluation.
            llm_judge: LLMJudge for pairwise candidate-vs-baseline win rate scoring.
        """
        self.test_cases = test_cases or create_default_alignment_test_cases()
        self.reward_engine = reward_engine or VerifiableRewardEngine()
        self.llm_judge = llm_judge or LLMJudge()

    def evaluate_candidate(
        self,
        candidate_name: str,
        candidate_generate_fn: Callable[[str], str],
        baseline_generate_fn: Optional[Callable[[str], str]] = None,
    ) -> AlignmentBenchmarkReport:
        """Run comprehensive alignment evaluation across candidate model completions.

        Args:
            candidate_name: Identifier label for candidate (e.g. "DPO-MiniGPT", "GRPO-Reasoning").
            candidate_generate_fn: Callable producing text completion for a given prompt.
            baseline_generate_fn: Optional baseline generation callable (SFT anchor) for pairwise win rates.

        Returns:
            AlignmentBenchmarkReport detailing accuracy, compliance, win rate, and gate decision.
        """
        format_scores: list[float] = []
        accuracy_scores: list[float] = []
        total_scores: list[float] = []
        wins: float = 0.0
        comparisons: int = 0

        logger.info(
            f"Evaluating candidate '{candidate_name}' across {len(self.test_cases)} alignment test cases..."
        )

        for case in self.test_cases:
            cand_output = candidate_generate_fn(case.prompt)

            # 1. Format Compliance Check
            f_score = self.reward_engine.compute_format_reward(cand_output)
            format_scores.append(f_score)

            # 2. Reasoning Accuracy Check
            if case.ground_truth is not None:
                acc_score = self.reward_engine.compute_accuracy_reward(cand_output, case.ground_truth)
                accuracy_scores.append(acc_score)
            else:
                acc_score = 0.0

            # 3. Composite Verifiable Reward
            total_r = self.reward_engine.score_completion(cand_output, case.ground_truth)["total_reward"]
            total_scores.append(total_r)

            # 4. Pairwise Win Rate via LLM Judge (with position swapping debiasing)
            if baseline_generate_fn is not None:
                base_output = baseline_generate_fn(case.prompt)
                judge_result = self.llm_judge.evaluate_pair(
                    prompt=case.prompt,
                    response_a=cand_output,
                    response_b=base_output,
                )
                if judge_result.winner == "A":
                    wins += 1.0
                elif judge_result.winner == "TIE":
                    wins += 0.5
                comparisons += 1

        format_rate = sum(format_scores) / max(len(format_scores), 1)
        accuracy_rate = (sum(accuracy_scores) / len(accuracy_scores)) if accuracy_scores else 0.0
        mean_score = sum(total_scores) / max(len(total_scores), 1)
        win_rate = (wins / comparisons) if comparisons > 0 else 0.0

        # 5. Automated Statistical Release Gate Decision
        if win_rate >= 0.55 and accuracy_rate >= 0.50:
            gate_decision = GateDecision.PROCEED_DEPLOY
        elif win_rate < 0.40 or accuracy_rate < 0.20:
            gate_decision = GateDecision.TRIGGER_ROLLBACK
        else:
            gate_decision = GateDecision.BLOCK_RELEASE

        logger.info(
            f"Candidate '{candidate_name}' -> Win Rate: {win_rate*100:.1f}%, Format: {format_rate*100:.1f}%, "
            f"Accuracy: {accuracy_rate*100:.1f}%, Gate: {gate_decision.value}"
        )

        return AlignmentBenchmarkReport(
            candidate_name=candidate_name,
            win_rate=win_rate,
            format_compliance_rate=format_rate,
            reasoning_accuracy_rate=accuracy_rate,
            mean_score=mean_score,
            gate_decision=gate_decision,
            metrics={
                "win_rate": win_rate,
                "format_compliance_rate": format_rate,
                "reasoning_accuracy_rate": accuracy_rate,
                "mean_verifiable_score": mean_score,
                "sample_count": float(len(self.test_cases)),
            },
        )

