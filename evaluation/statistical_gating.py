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

from scipy.optimize import brentq
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
    power_analysis: Optional[dict[str, Any]] = None
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
            "power_analysis": self.power_analysis,
            "passed_checks": self.passed_checks,
            "violated_checks": self.violated_checks,
            "recommendation_summary": self.recommendation_summary,
            "timestamp": self.timestamp,
        }


def compute_statistical_power(
    sample_size: int,
    paired_delta_std: Optional[float],
    target_delta: Optional[float] = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Estimate paired-mean power using a noncentral-t planning model.

    This a priori estimate assumes normally distributed paired deltas. Release
    decisions are still made separately using the configured bootstrap and
    Wilcoxon analyses.
    """
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 0:
        raise ValueError("sample_size must be a non-negative integer.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1.")
    if target_delta is not None and (
        not math.isfinite(target_delta) or target_delta <= 0.0
    ):
        raise ValueError("target_delta must be a finite positive value.")
    if paired_delta_std is not None and (
        not math.isfinite(paired_delta_std) or paired_delta_std < 0.0
    ):
        raise ValueError("paired_delta_std must be finite and non-negative.")

    if paired_delta_std is None or target_delta is None:
        missing_parameters = []
        if paired_delta_std is None:
            missing_parameters.append("paired-delta standard deviation")
        if target_delta is None:
            missing_parameters.append("minimum meaningful improvement")
        return {
            "power": None,
            "mde": None,
            "sample_size": sample_size,
            "adequate_power": False,
            "target_delta": target_delta,
            "paired_delta_std": paired_delta_std,
            "warning": (
                f"Configure {' and '.join(missing_parameters)} from an independent pilot "
                "before a priori power analysis."
            ),
            "interpretation": (
                "Power was not estimated; promotion is blocked until pilot variance and "
                "a minimum meaningful improvement are configured."
            ),
        }

    if sample_size < 2:
        return {
            "power": None,
            "mde": None,
            "sample_size": sample_size,
            "adequate_power": False,
            "target_delta": target_delta,
            "paired_delta_std": paired_delta_std,
            "warning": "Sample size too small (<2) for power estimation.",
            "power_model": "paired_noncentral_t",
            "interpretation": "Power is not estimable with fewer than two paired cases.",
        }

    se = paired_delta_std / math.sqrt(sample_size)
    if se > 0.0:
        degrees_of_freedom = sample_size - 1
        critical_value = float(stats.t.ppf(1.0 - alpha / 2.0, degrees_of_freedom))

        def two_sided_power(noncentrality: float) -> float:
            return float(
                stats.nct.cdf(-critical_value, degrees_of_freedom, noncentrality)
                + stats.nct.sf(critical_value, degrees_of_freedom, noncentrality)
            )

        power = two_sided_power(target_delta / se)
        upper_effect = 1.0
        while upper_effect < 1024.0 and (
            not math.isfinite(two_sided_power(upper_effect))
            or two_sided_power(upper_effect) < 0.80
        ):
            upper_effect *= 2.0
        upper_power = two_sided_power(upper_effect)
        if not math.isfinite(upper_power) or upper_power < 0.80:
            raise ArithmeticError(
                "Could not estimate the minimum detectable effect for the configured "
                f"sample size ({sample_size}) and significance level ({alpha})."
            )
        mde = brentq(
            lambda effect: two_sided_power(effect) - 0.80,
            0.0,
            upper_effect,
        ) * se
    else:
        power = 1.0
        mde = 0.0

    adequate = power >= 0.80 and sample_size >= 30
    warning = ""
    if sample_size < 30:
        warning = (
            f"Benchmark sample size ({sample_size} cases) is below the minimum of 30 "
            "paired cases required for release."
        )
    elif not adequate:
        warning = (
            f"Estimated a priori power ({power:.2%}) is below 80% for "
            f"detecting delta={target_delta}."
        )

    return {
        "power": round(power, 4),
        "mde": round(mde, 4),
        "sample_size": sample_size,
        "adequate_power": adequate,
        "target_delta": target_delta,
        "paired_delta_std": paired_delta_std,
        "warning": warning,
        "power_model": "paired_noncentral_t",
        "interpretation": (
            "A priori paired-mean estimate using the configured independent-pilot variance "
            "and a normal paired-delta assumption; release tests are evaluated separately."
        ),
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
        edge_case_delta_threshold: float = -0.15,
        min_improvement_delta: Optional[float] = None,
        paired_delta_std: Optional[float] = None,
    ):
        """Initialize the Statistical Release Gate.

        Args:
            bootstrap_samples: Number of bootstrap iterations (default 1000).
            ci_level: Confidence level for intervals (default 0.95).
            alpha: Hypothesis testing significance threshold (default 0.05).
            parity_delta: Minimum allowed negative delta for non-inferiority parity (default 0.05).
            pass_threshold: Minimum rubric score considered a successful pass (default 3.5).
            random_seed: Deterministic seed for bootstrap resampling.
            edge_case_delta_threshold: Minimum allowed mean delta for edge cases (default -0.15).
            min_improvement_delta: Minimum meaningful positive mean delta set before evaluation.
            paired_delta_std: Paired-score standard deviation estimated from an independent pilot.
                Promotion is blocked unless both the pilot variance and a meaningful margin
                are configured.
        """
        self.bootstrap_samples = int(bootstrap_samples)
        self.ci_level = float(ci_level)
        self.alpha = float(alpha)
        self.parity_delta = float(parity_delta)
        self.pass_threshold = float(pass_threshold)
        self.random_seed = int(random_seed)
        self.edge_case_delta_threshold = float(edge_case_delta_threshold)
        if min_improvement_delta is not None and (
            not math.isfinite(min_improvement_delta) or min_improvement_delta <= 0.0
        ):
            raise ValueError("min_improvement_delta must be finite and positive.")
        self.min_improvement_delta = min_improvement_delta
        if paired_delta_std is not None and (
            not math.isfinite(paired_delta_std) or paired_delta_std < 0.0
        ):
            raise ValueError("paired_delta_std must be finite and non-negative.")
        self.paired_delta_std = paired_delta_std

    def evaluate_gate(
        self,
        suite: GoldenBenchmarkSuite,
        candidate_scores: dict[str, float],
        baseline_scores: dict[str, float],
        judge_inconsistency: float = 0.0,
        is_active_deployment: bool = False,
    ) -> GateReport:
        """Execute full stratified statistical gating evaluation.

        Args:
            suite: The GoldenBenchmarkSuite instance containing test cases.
            candidate_scores: Mapping from case_id to candidate score.
            baseline_scores: Mapping from case_id to baseline score.
            judge_inconsistency: Kappa position inconsistency from LLMJudge (default 0.0).
            is_active_deployment: Whether evaluation targets the currently active deployment
                (True -> triggers TRIGGER_ROLLBACK on regression) or an unreleased candidate
                (False -> triggers BLOCK_RELEASE on regression, preserving active model).

        Returns:
            GateReport with definitive release decision and audit breakdown.
        """
        passed_checks: list[str] = []
        violated_checks: list[str] = []

        all_cases = suite.cases
        if not all_cases:
            violated_checks.append("Evaluation benchmark suite contains 0 test cases.")
            overall_boot = compute_paired_bootstrap([0.0], [4.0], num_replicates=10, seed=self.random_seed)
            overall_hypo = HypothesisTestResult(0.0, 1.0, 0.0, 1.0, 0, 1, 0, 0, 1.0)
            return GateReport(
                decision=GateDecision.BLOCK_RELEASE,
                overall_bootstrap=overall_boot,
                overall_hypothesis=overall_hypo,
                bucket_breakdown={},
                power_analysis={"power": 0.0, "adequate_power": False, "sample_size": 0},
                passed_checks=[],
                violated_checks=violated_checks,
                recommendation_summary="Release BLOCKED: Evaluation suite has no test cases.",
            )

        # 1. Strict score validation: fail-closed on missing, malformed, or out-of-range scores
        suite_case_ids = {c.case_id for c in all_cases}
        missing_cand = sorted(cid for cid in suite_case_ids if cid not in candidate_scores)
        missing_base = sorted(cid for cid in suite_case_ids if cid not in baseline_scores)
        extra_cand = sorted(cid for cid in candidate_scores if cid not in suite_case_ids)
        extra_base = sorted(cid for cid in baseline_scores if cid not in suite_case_ids)

        invalid_cand = [
            cid for cid, s in candidate_scores.items()
            if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or s < 0.0 or s > 5.0
        ]
        invalid_base = [
            cid for cid, s in baseline_scores.items()
            if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or s < 0.0 or s > 5.0
        ]

        if missing_cand or missing_base or extra_cand or extra_base or invalid_cand or invalid_base:
            if missing_cand:
                violated_checks.append(f"Missing candidate evaluation scores for cases: {missing_cand}")
            if missing_base:
                violated_checks.append(f"Missing baseline evaluation scores for cases: {missing_base}")
            if extra_cand:
                violated_checks.append(f"Unknown candidate case IDs not in benchmark suite: {extra_cand}")
            if extra_base:
                violated_checks.append(f"Unknown baseline case IDs not in benchmark suite: {extra_base}")
            if invalid_cand:
                violated_checks.append(f"Invalid non-finite or out-of-range candidate scores for cases: {invalid_cand}")
            if invalid_base:
                violated_checks.append(f"Invalid non-finite or out-of-range baseline scores for cases: {invalid_base}")

            overall_boot = compute_paired_bootstrap([0.0], [4.0], num_replicates=10, seed=self.random_seed)
            overall_hypo = HypothesisTestResult(0.0, 1.0, 0.0, 1.0, 0, 1, 0, 0, 1.0)
            return GateReport(
                decision=GateDecision.BLOCK_RELEASE,
                overall_bootstrap=overall_boot,
                overall_hypothesis=overall_hypo,
                bucket_breakdown={},
                power_analysis={"power": 0.0, "adequate_power": False, "sample_size": len(candidate_scores)},
                passed_checks=[],
                violated_checks=violated_checks,
                recommendation_summary="Release BLOCKED: Score validation failed. Missing, malformed, or out-of-range scores.",
            )

        # 2. Align score vectors across all cases in the suite
        cand_list: list[float] = []
        base_list: list[float] = []
        case_records: list[dict[str, Any]] = []

        for c in all_cases:
            cid = c.case_id
            s_cand = float(candidate_scores[cid])
            s_base = float(baseline_scores[cid])
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

        # 3. Overall Statistical Inference & Power Analysis
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

        power_metrics = compute_statistical_power(
            sample_size=len(cand_list),
            paired_delta_std=self.paired_delta_std,
            target_delta=self.min_improvement_delta,
            alpha=self.alpha,
        )
        if not power_metrics["adequate_power"]:
            violated_checks.append(f"Statistical Sensitivity Warning: {power_metrics['warning']}")
        else:
            passed_checks.append(
                f"Statistical Sensitivity: Adequate a priori power "
                f"({power_metrics['power']:.1%}) for delta={self.min_improvement_delta}."
            )

        # 4. Stratified Bucket Breakdown & Validation
        required_buckets = ["core_domain", "edge_case", "adversarial_bait", "regression_replay"]
        bucket_breakdown: dict[str, dict[str, Any]] = {}
        missing_buckets: list[str] = []

        for b_name in required_buckets:
            b_cases = [rec for rec in case_records if rec["bucket"] == b_name]
            if not b_cases:
                missing_buckets.append(b_name)
                violated_checks.append(f"Bucket '{b_name}' has 0 test cases in evaluation suite!")
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

        # Any missing required bucket immediately blocks release
        if missing_buckets:
            summary = (
                f"Release BLOCKED: Benchmark suite missing required bucket(s): {missing_buckets}. "
                f"Stratified gating requires all 4 buckets."
            )
            return GateReport(
                decision=GateDecision.BLOCK_RELEASE,
                overall_bootstrap=overall_boot,
                overall_hypothesis=overall_hypo,
                bucket_breakdown=bucket_breakdown,
                power_analysis=power_metrics,
                passed_checks=passed_checks,
                violated_checks=violated_checks,
                recommendation_summary=summary,
            )

        # 5. Check Specific Gating Criteria
        core_reg = bucket_breakdown["core_domain"]["regressions"]
        regr_reg = bucket_breakdown["regression_replay"]["regressions"]
        core_delta = bucket_breakdown["core_domain"]["mean_delta"]
        edge_delta = bucket_breakdown["edge_case"]["mean_delta"]
        adv_mean = bucket_breakdown["adversarial_bait"]["mean_candidate"]

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
        if edge_delta >= self.edge_case_delta_threshold:
            passed_checks.append(f"Edge Case Delta Bounds: PASSED (delta={edge_delta:+.3f} >= {self.edge_case_delta_threshold:+.3f}).")
        else:
            violated_checks.append(f"Edge Case Delta Bounds: FAILED (delta={edge_delta:+.3f} < {self.edge_case_delta_threshold:+.3f}).")

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

        # 6. Autonomous Gating Decision Matrix
        is_stat_sig_regress = (
            overall_boot.ci_upper < 0.0
            or (overall_hypo.wilcoxon_p < self.alpha and overall_boot.mean_delta < 0.0)
            or (overall_hypo.mcnemar_p < self.alpha and overall_hypo.regressions > overall_hypo.improvements)
        )
        is_core_or_regr_violation = (
            core_reg > 0
            or regr_reg > 0
            or core_delta < -self.parity_delta
            or edge_delta < self.edge_case_delta_threshold
        )

        if is_core_or_regr_violation or is_stat_sig_regress:
            if is_active_deployment:
                decision = GateDecision.TRIGGER_ROLLBACK
                summary = (
                    f"ACTIVE DEPLOYMENT REGRESSION: TRIGGER_ROLLBACK. Core regressions: {core_reg}, "
                    f"Replay regressions: {regr_reg}, Edge delta: {edge_delta:+.3f}, "
                    f"Mean Delta: {overall_boot.mean_delta:+.3f}, "
                    f"95% CI: [{overall_boot.ci_lower:+.3f}, {overall_boot.ci_upper:+.3f}]."
                )
            else:
                decision = GateDecision.BLOCK_RELEASE
                summary = (
                    f"Release BLOCKED: Candidate exhibited regression or forgetting before deployment. "
                    f"Core regressions: {core_reg}, Replay regressions: {regr_reg}, "
                    f"Edge delta: {edge_delta:+.3f}, Mean Delta: {overall_boot.mean_delta:+.3f}, "
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
            and edge_delta >= self.edge_case_delta_threshold
            and power_metrics["adequate_power"]
            and (
                overall_boot.ci_lower > 0.0  # Statistically superior
                or (
                    overall_boot.ci_lower >= -self.parity_delta
                    and overall_boot.mean_delta >= self.min_improvement_delta
                )  # Non-inferior parity with minimum calibrated positive delta
            )
        ):
            decision = GateDecision.PROCEED_DEPLOY
            summary = (
                f"Release APPROVED with PROCEED_DEPLOY. Zero core/replay regressions, "
                f"Mean Delta: {overall_boot.mean_delta:+.3f} (min required: +{self.min_improvement_delta:.3f}), "
                f"95% CI: [{overall_boot.ci_lower:+.3f}, {overall_boot.ci_upper:+.3f}], Wilcoxon p={w_p:.4f}."
            )
        else:
            decision = GateDecision.BLOCK_RELEASE
            if not power_metrics["adequate_power"]:
                power_display = (
                    "not estimated"
                    if power_metrics["power"] is None
                    else f"{power_metrics['power']:.1%}"
                )
                summary = (
                    f"Release HELD with BLOCK_RELEASE. Estimated a priori power "
                    f"({power_display}) is inadequate for promotion; "
                    f"minimum required is 80% with at least 30 paired cases. "
                    f"{power_metrics['interpretation']}"
                )
            else:
                summary = (
                    f"Release HELD with BLOCK_RELEASE. Candidate delta is indeterminate or below required "
                    f"improvement margin (Mean Delta: {overall_boot.mean_delta:+.3f}, required: +{self.min_improvement_delta:.3f})."
                )

        return GateReport(
            decision=decision,
            overall_bootstrap=overall_boot,
            overall_hypothesis=overall_hypo,
            bucket_breakdown=bucket_breakdown,
            power_analysis=power_metrics,
            passed_checks=passed_checks,
            violated_checks=violated_checks,
            recommendation_summary=summary,
        )
