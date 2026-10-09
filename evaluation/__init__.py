"""Evaluation and Autonomous Release Gate Package (Milestone 4)."""

from evaluation.benchmark_suite import (
    GoldenTestCase,
    GoldenBenchmarkSuite,
    create_default_golden_suite_cases,
)
from evaluation.citation_verifier import (
    CitationTag,
    SentenceClaim,
    CitationMetrics,
    CitationVerifier,
)
from evaluation.llm_judge import (
    JudgeScore,
    PairwiseJudgeResult,
    LLMJudge,
)
from evaluation.statistical_gating import (
    GateDecision,
    BootstrapMetricResult,
    HypothesisTestResult,
    GateReport,
    StatisticalReleaseGate,
    compute_paired_bootstrap,
    compute_wilcoxon_signed_rank,
    compute_mcnemar_discordance,
)
from evaluation.rollback_orchestrator import (
    DeploymentState,
    RollbackOrchestrator,
)

__all__ = [
    "GoldenTestCase",
    "GoldenBenchmarkSuite",
    "create_default_golden_suite_cases",
    "CitationTag",
    "SentenceClaim",
    "CitationMetrics",
    "CitationVerifier",
    "JudgeScore",
    "PairwiseJudgeResult",
    "LLMJudge",
    "GateDecision",
    "BootstrapMetricResult",
    "HypothesisTestResult",
    "GateReport",
    "StatisticalReleaseGate",
    "compute_paired_bootstrap",
    "compute_wilcoxon_signed_rank",
    "compute_mcnemar_discordance",
    "DeploymentState",
    "RollbackOrchestrator",
]


