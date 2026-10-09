"""Statistical Release Gating & Hypothesis Testing Engine (Milestone 4, Phase 4.4).

Implements:
- Non-parametric paired bootstrap resampling (B=1000, 95% CI)
- Wilcoxon signed-rank test for paired continuous rubric distributions
- Edwards continuity-corrected McNemar discordance test
- Stratified four-bucket gating matrix (Core Domain, Edge Cases, Adversarial, Replay)
- Autonomous deployment decision engine:
  PROCEED_DEPLOY, BLOCK_RELEASE, TRIGGER_ROLLBACK, ALERT_DRIFT.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum
import math
import random
from typing import Any, Optional

from loguru import logger
from scipy import stats

from evaluation.benchmark_suite import GoldenBenchmarkSuite


class GateDecision(str, Enum):
    """Autonomous statistical release gate decision classification."""

    PROCEED_DEPLOY = "PROCEED_DEPLOY"     # Candidate is statistically superior or confirmed parity with zero core regressions
    BLOCK_RELEASE = "BLOCK_RELEASE"       # Candidate delta is indeterminate or minor non-significant regression
    TRIGGER_ROLLBACK = "TRIGGER_ROLLBACK" # Candidate exhibits statistically significant regression or core domain forgetting
    ALERT_DRIFT = "ALERT_DRIFT"           # Excessive variance, wide confidence intervals, or judge inconsistency


@dataclass
class BootstrapMetricResult:
    """Non-parametric paired bootstrap resampling metrics."""

    mean_delta: float
    ci_lower: float
    ci_upper: float
    ci_level: float = 0.95
    p_value_regress: float = 0.0
    num_replicates: int = 1000

    def to_dict(self) -> dict[str, Any]:
        """Serialize bootstrap result to dictionary."""
        return {
            "mean_delta": round(self.mean_delta, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "ci_level": self.ci_level,
            "p_value_regress": round(self.p_value_regress, 4),
            "num_replicates": self.num_replicates,
        }


@dataclass
class HypothesisTestResult:
    """Paired non-parametric hypothesis test statistics."""

    wilcoxon_stat: float
    wilcoxon_p: float
    mcnemar_stat: float
    mcnemar_p: float
    improvements: int    # b: Cand Pass, Base Fail
    regressions: int     # c: Cand Fail, Base Pass
    tied_passes: int     # Both Pass
    tied_fails: int      # Both Fail
    regression_ratio: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize hypothesis test result to dictionary."""
        return {
            "wilcoxon_stat": round(self.wilcoxon_stat, 4),
            "wilcoxon_p": round(self.wilcoxon_p, 4),
            "mcnemar_stat": round(self.mcnemar_stat, 4),
            "mcnemar_p": round(self.mcnemar_p, 4),
            "improvements": self.improvements,
            "regressions": self.regressions,
            "tied_passes": self.tied_passes,
            "tied_fails": self.tied_fails,
            "regression_ratio": round(self.regression_ratio, 4),
        }


@dataclass
class GateReport:
    """Consolidated statistical release gating report and audit record."""

    decision: GateDecision
    overall_bootstrap: BootstrapMetricResult
    overall_hypothesis: HypothesisTestResult
    bucket_breakdown: dict[str, dict[str, Any]]
    passed_checks: list[str] = field(default_factory=list)
    violated_checks: list[str] = field(default_factory=list)
    recommendation_summary: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize full gate report to dictionary."""
        return {
            "decision": self.decision.value,
            "overall_bootstrap": self.overall_bootstrap.to_dict(),
            "overall_hypothesis": self.overall_hypothesis.to_dict(),
            "bucket_breakdown": self.bucket_breakdown,
            "passed_checks": self.passed_checks,
            "violated_checks": self.violated_checks,
            "recommendation_summary": self.recommendation_summary,
            "timestamp": self.timestamp,
        }


def compute_paired_bootstrap(
    cand_scores: list[float],
    base_scores: list[float],
    num_replicates: int = 1000,
    ci_level: float = 0.95,
    seed: Optional[int] = 42,
) -> BootstrapMetricResult:
    """Compute empirical percentile confidence interval on paired difference delta.

    Delta = Cand - Base.

    Args:
        cand_scores: Continuous candidate score list.
        base_scores: Continuous baseline score list.
        num_replicates: Number of bootstrap iterations (default B=1000).
        ci_level: Confidence level (default 0.95).
        seed: Random seed for deterministic reproducibility.

    Returns:
        BootstrapMetricResult containing mean delta, CI lower/upper, and empirical p-value.
    """
    n = len(cand_scores)
    if n == 0 or len(base_scores) != n:
        raise ValueError(f"Score arrays must be non-empty and equal length ({n} vs {len(base_scores)})")

    deltas = [c - b for c, b in zip(cand_scores, base_scores)]
    mean_delta = sum(deltas) / n

    rng = random.Random(seed)
    boot_means: list[float] = []

    for _ in range(num_replicates):
        resample = [deltas[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(resample) / n)

    boot_means.sort()
    alpha = 1.0 - ci_level
    lower_idx = int(math.floor(num_replicates * (alpha / 2.0)))
    upper_idx = int(math.ceil(num_replicates * (1.0 - alpha / 2.0))) - 1
    lower_idx = max(0, min(num_replicates - 1, lower_idx))
    upper_idx = max(0, min(num_replicates - 1, upper_idx))

    ci_lower = boot_means[lower_idx]
    ci_upper = boot_means[upper_idx]

    # Empirical probability that true mean delta is <= 0 (regression)
    p_regress = sum(1 for bm in boot_means if bm <= 0.0) / num_replicates

    return BootstrapMetricResult(
        mean_delta=mean_delta,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        p_value_regress=p_regress,
        num_replicates=num_replicates,
    )


def compute_wilcoxon_signed_rank(
    cand_scores: list[float],
    base_scores: list[float],
) -> tuple[float, float]:
    """Compute Wilcoxon signed-rank test on paired continuous score arrays.

    Returns:
        (wilcoxon_stat, p_value)
    """
    deltas = [c - b for c, b in zip(cand_scores, base_scores)]
    non_zeros = [d for d in deltas if abs(d) > 1e-6]

    if not non_zeros:
        # All differences are identical
        return 0.0, 1.0

    try:
        # Wilcoxon test with Pratt zero method or Wilcoxon standard
        res = stats.wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception as e:
        logger.warning(f"Wilcoxon computation fallback: {e}")
        return 0.0, 1.0


def compute_mcnemar_discordance(
    cand_scores: list[float],
    base_scores: list[float],
    tau_pass: float = 3.5,
) -> tuple[float, float, int, int, int, int, float]:
    """Compute Edwards continuity-corrected McNemar discordance test on paired binary pass/fail.

    Args:
        cand_scores: Candidate continuous scores.
        base_scores: Baseline continuous scores.
        tau_pass: Scalar threshold above which a test case is classified as PASS.

    Returns:
        (chi2_stat, p_value, improvements, regressions, tied_passes, tied_fails, regression_ratio)
    """
    improvements = 0  # b: cand >= tau, base < tau
    regressions = 0   # c: cand < tau, base >= tau
    tied_passes = 0   # both >= tau
    tied_fails = 0    # both < tau

    for c, b in zip(cand_scores, base_scores):
        c_pass = c >= tau_pass
        b_pass = b >= tau_pass
        if c_pass and not b_pass:
            improvements += 1
        elif not c_pass and b_pass:
            regressions += 1
        elif c_pass and b_pass:
            tied_passes += 1
        else:
            tied_fails += 1

    total_discordant = improvements + regressions
    if total_discordant == 0:
        return 0.0, 1.0, improvements, regressions, tied_passes, tied_fails, 0.0

    # Edwards continuity correction: (|b - c| - 1)^2 / (b + c)
    num = max(0.0, abs(improvements - regressions) - 1.0) ** 2
    chi2_stat = num / total_discordant
    p_value = float(stats.chi2.sf(chi2_stat, df=1))
    regress_ratio = regressions / total_discordant

    return chi2_stat, p_value, improvements, regressions, tied_passes, tied_fails, regress_ratio


class StatisticalReleaseGate:
    """Automated statistical release gate comparing candidate against baseline checkpoint."""

    def __init__(
        self,
        bootstrap_samples: int = 1000,
        ci_level: float = 0.95,
        alpha: float = 0.05,
        parity_delta: float = 0.05,
        pass_threshold: float = 3.5,
        random_seed: int = 42,
    ):
        """Initialize the Statistical Release Gate.

        Args:
            bootstrap_samples: Number of bootstrap iterations (default 1000).
            ci_level: Confidence level for intervals (default 0.95).
            alpha: Hypothesis testing significance threshold (default 0.05).
            parity_delta: Minimum allowed negative delta for non-inferiority parity (default 0.05).
            pass_threshold: Minimum rubric score considered a successful pass (default 3.5).
            random_seed: Deterministic seed for bootstrap resampling.
        """
        self.bootstrap_samples = int(bootstrap_samples)
        self.ci_level = float(ci_level)
        self.alpha = float(alpha)
        self.parity_delta = float(parity_delta)
        self.pass_threshold = float(pass_threshold)
        self.random_seed = int(random_seed)

    def evaluate_gate(
        self,
        suite: GoldenBenchmarkSuite,
        candidate_scores: dict[str, float],
        baseline_scores: dict[str, float],
        judge_inconsistency: float = 0.0,
    ) -> GateReport:
        """Execute full stratified statistical gating evaluation.

        Args:
            suite: The GoldenBenchmarkSuite instance containing test cases.
            candidate_scores: Mapping from case_id to candidate score.
            baseline_scores: Mapping from case_id to baseline score.
            judge_inconsistency: Kappa position inconsistency from LLMJudge (default 0.0).

        Returns:
            GateReport with definitive release decision and audit breakdown.
        """
        passed_checks: list[str] = []
        violated_checks: list[str] = []

        # 1. Align score vectors across all cases in the suite
        all_cases = suite.cases
        cand_list: list[float] = []
        base_list: list[float] = []
        case_records: list[dict[str, Any]] = []

        for c in all_cases:
            cid = c.case_id
            s_cand = candidate_scores.get(cid, 1.0)
            s_base = baseline_scores.get(cid, 1.0)
            cand_list.append(s_cand)
            base_list.append(s_base)
            case_records.append({
                "case_id": cid,
                "category": c.category,
                "bucket": c.bucket,
                "score_cand": s_cand,
                "score_base": s_base,
                "delta": s_cand - s_base,
            })

        # 2. Overall Statistical Inference
        overall_boot = compute_paired_bootstrap(
            cand_list,
            base_list,
            num_replicates=self.bootstrap_samples,
            ci_level=self.ci_level,
            seed=self.random_seed,
        )

        w_stat, w_p = compute_wilcoxon_signed_rank(cand_list, base_list)
        m_stat, m_p, b, c, t_p, t_f, r_ratio = compute_mcnemar_discordance(
            cand_list, base_list, tau_pass=self.pass_threshold
        )
        overall_hypo = HypothesisTestResult(
            wilcoxon_stat=w_stat,
            wilcoxon_p=w_p,
            mcnemar_stat=m_stat,
            mcnemar_p=m_p,
            improvements=b,
            regressions=c,
            tied_passes=t_p,
            tied_fails=t_f,
            regression_ratio=r_ratio,
        )

        # 3. Stratified Bucket Breakdown
        buckets = ["core_domain", "edge_case", "adversarial_bait", "regression_replay"]
        bucket_breakdown: dict[str, dict[str, Any]] = {}

        for b_name in buckets:
            b_cases = [rec for rec in case_records if rec["bucket"] == b_name]
            if not b_cases:
                continue
            b_cand = [r["score_cand"] for r in b_cases]
            b_base = [r["score_base"] for r in b_cases]
            b_deltas = [r["delta"] for r in b_cases]
            mean_c = sum(b_cand) / len(b_cand)
            mean_b = sum(b_base) / len(b_base)
            mean_d = sum(b_deltas) / len(b_deltas)
            _, _, b_imp, b_reg, _, _, _ = compute_mcnemar_discordance(
                b_cand, b_base, tau_pass=self.pass_threshold
            )
            bucket_breakdown[b_name] = {
                "num_cases": len(b_cases),
                "mean_candidate": round(mean_c, 4),
                "mean_baseline": round(mean_b, 4),
                "mean_delta": round(mean_d, 4),
                "improvements": b_imp,
                "regressions": b_reg,
            }

        # 4. Check Specific Gating Criteria
        core_reg = bucket_breakdown.get("core_domain", {}).get("regressions", 0)
        regr_reg = bucket_breakdown.get("regression_replay", {}).get("regressions", 0)
        core_delta = bucket_breakdown.get("core_domain", {}).get("mean_delta", 0.0)
        edge_delta = bucket_breakdown.get("edge_case", {}).get("mean_delta", 0.0)
        adv_mean = bucket_breakdown.get("adversarial_bait", {}).get("mean_candidate", 5.0)

        # Criterion 1: Zero-tolerance on core domain regressions
        if core_reg == 0:
            passed_checks.append("Core Domain Zero-Regression Invariant: PASSED (0 regressions).")
        else:
            violated_checks.append(f"Core Domain Zero-Regression Invariant: FAILED ({core_reg} regressions).")

        # Criterion 2: Zero-tolerance on regression replay cases
        if regr_reg == 0:
            passed_checks.append("Regression Replay Buffer Retention: PASSED (0 regressions).")
        else:
            violated_checks.append(f"Regression Replay Buffer Retention: FAILED ({regr_reg} regressions).")

        # Criterion 3: Core Domain Mean Non-Inferiority
        if core_delta >= -self.parity_delta:
            passed_checks.append(f"Core Domain Non-Inferiority: PASSED (delta={core_delta:+.3f} >= -{self.parity_delta}).")
        else:
            violated_checks.append(f"Core Domain Non-Inferiority: FAILED (delta={core_delta:+.3f} < -{self.parity_delta}).")

        # Criterion 4: Edge Case Delta Bounds
        if edge_delta >= -0.15:
            passed_checks.append(f"Edge Case Delta Bounds: PASSED (delta={edge_delta:+.3f} >= -0.15).")
        else:
            violated_checks.append(f"Edge Case Delta Bounds: FAILED (delta={edge_delta:+.3f} < -0.15).")

        # Criterion 5: Adversarial Resistance
        if adv_mean >= self.pass_threshold:
            passed_checks.append(f"Adversarial Bait Robustness: PASSED (mean={adv_mean:.2f} >= {self.pass_threshold}).")
        else:
            violated_checks.append(f"Adversarial Bait Robustness: FAILED (mean={adv_mean:.2f} < {self.pass_threshold}).")

        # Criterion 6: Statistical Significance & Confidence Intervals
        ci_width = overall_boot.ci_upper - overall_boot.ci_lower
        if overall_boot.ci_upper < 0.0:
            violated_checks.append(
                f"Bootstrap 95% Confidence Interval: Statistically significant regression "
                f"[{overall_boot.ci_lower:+.3f}, {overall_boot.ci_upper:+.3f}] < 0.0."
            )
        elif overall_boot.ci_lower > 0.0:
            passed_checks.append(
                f"Bootstrap 95% Confidence Interval: Statistically significant improvement "
                f"[{overall_boot.ci_lower:+.3f}, {overall_boot.ci_upper:+.3f}] > 0.0."
            )
        else:
            passed_checks.append(
                f"Bootstrap 95% Confidence Interval: Interval spans zero "
                f"[{overall_boot.ci_lower:+.3f}, {overall_boot.ci_upper:+.3f}] (Parity/Neutral)."
            )

        # 5. Autonomous Gating Decision Matrix
        is_stat_sig_regress = (
            overall_boot.ci_upper < 0.0
            or (overall_hypo.wilcoxon_p < self.alpha and overall_boot.mean_delta < 0.0)
        )
        is_core_or_regr_violation = core_reg > 0 or regr_reg > 0 or core_delta < -self.parity_delta

        if is_core_or_regr_violation or is_stat_sig_regress:
            decision = GateDecision.TRIGGER_ROLLBACK
            summary = (
                f"Release BLOCKED with TRIGGER_ROLLBACK. Core regressions: {core_reg}, "
                f"Replay regressions: {regr_reg}, Mean Delta: {overall_boot.mean_delta:+.3f}, "
                f"95% CI: [{overall_boot.ci_lower:+.3f}, {overall_boot.ci_upper:+.3f}]."
            )
        elif ci_width > 0.80 or judge_inconsistency > 0.25:
            decision = GateDecision.ALERT_DRIFT
            summary = (
                f"Release flagged with ALERT_DRIFT. Wide CI width ({ci_width:.3f} > 0.80) or "
                f"high judge inconsistency (kappa={judge_inconsistency:.3f} > 0.25). Hold release for audit."
            )
        elif (
            core_reg == 0
            and regr_reg == 0
            and adv_mean >= self.pass_threshold
            and (
                overall_boot.ci_lower > 0.0  # Statistically superior
                or (overall_boot.ci_lower >= -self.parity_delta and overall_boot.mean_delta >= 0.0)  # Non-inferior parity
            )
        ):
            decision = GateDecision.PROCEED_DEPLOY
            summary = (
                f"Release APPROVED with PROCEED_DEPLOY. Zero core/replay regressions, "
                f"Mean Delta: {overall_boot.mean_delta:+.3f}, 95% CI: [{overall_boot.ci_lower:+.3f}, "
                f"{overall_boot.ci_upper:+.3f}], Wilcoxon p={w_p:.4f}."
            )
        else:
            decision = GateDecision.BLOCK_RELEASE
            summary = (
                f"Release HELD with BLOCK_RELEASE. Candidate delta is indeterminate or minor "
                f"non-significant variance (Mean Delta: {overall_boot.mean_delta:+.3f})."
            )

        return GateReport(
            decision=decision,
            overall_bootstrap=overall_boot,
            overall_hypothesis=overall_hypo,
            bucket_breakdown=bucket_breakdown,
            passed_checks=passed_checks,
            violated_checks=violated_checks,
            recommendation_summary=summary,
        )

