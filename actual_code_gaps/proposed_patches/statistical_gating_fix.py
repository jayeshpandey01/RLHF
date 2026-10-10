"""Proposed fix for StatisticalReleaseGate resolving decision loopholes (GAP-08)."""

from __future__ import annotations
from typing import Any, Optional
from loguru import logger

from evaluation.benchmark_suite import GoldenBenchmarkSuite
from evaluation.statistical_gating import (
    GateDecision,
    GateReport,
    compute_paired_bootstrap,
    compute_wilcoxon_signed_rank,
    compute_mcnemar_discordance,
    HypothesisTestResult,
)


class FixedStatisticalReleaseGate:
    """Robust statistical release gate comparing candidate against baseline checkpoint."""

    def __init__(
        self,
        bootstrap_samples: int = 1000,
        ci_level: float = 0.95,
        alpha: float = 0.05,
        parity_delta: float = 0.05,
        min_improvement_delta: float = 0.02,  # Require minimum non-zero improvement
        edge_case_delta_threshold: float = -0.15,
        pass_threshold: float = 3.5,
        random_seed: int = 42,
    ):
        self.bootstrap_samples = int(bootstrap_samples)
        self.ci_level = float(ci_level)
        self.alpha = float(alpha)
        self.parity_delta = float(parity_delta)
        self.min_improvement_delta = float(min_improvement_delta)
        self.edge_case_delta_threshold = float(edge_case_delta_threshold)
        self.pass_threshold = float(pass_threshold)
        self.random_seed = int(random_seed)

    def evaluate_gate(
        self,
        suite: GoldenBenchmarkSuite,
        candidate_scores: dict[str, float],
        baseline_scores: dict[str, float],
        judge_inconsistency: float = 0.0,
        is_live_monitoring: bool = False,  # True only for in-production deployed models
    ) -> GateReport:
        passed_checks: list[str] = []
        violated_checks: list[str] = []

        if not candidate_scores:
            violated_checks.append("Candidate evaluation score matrix is empty.")
            overall_boot = compute_paired_bootstrap([0.0], [4.0], num_replicates=10, seed=self.random_seed)
            overall_hypo = HypothesisTestResult(0.0, 1.0, 0.0, 1.0, 0, 1, 0, 0, 1.0)
            return GateReport(
                decision=GateDecision.BLOCK_RELEASE,
                overall_bootstrap=overall_boot,
                overall_hypothesis=overall_hypo,
                bucket_breakdown={},
                passed_checks=[],
                violated_checks=violated_checks,
                recommendation_summary="Release BLOCKED: Candidate evaluation scores are missing.",
            )

        # 1. Align score vectors
        all_cases = suite.cases
        cand_list: list[float] = []
        base_list: list[float] = []
        case_records: list[dict[str, Any]] = []

        for c in all_cases:
            cid = c.case_id
            if cid not in candidate_scores:
                violated_checks.append(f"Missing evaluation score for case {cid}")
            s_cand = candidate_scores.get(cid, 0.0)
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

        # 3. Stratified Bucket Breakdown & Validation
        buckets = ["core_domain", "edge_case", "adversarial_bait", "regression_replay"]
        bucket_breakdown: dict[str, dict[str, Any]] = {}

        for b_name in buckets:
            b_cases = [rec for rec in case_records if rec["bucket"] == b_name]
            if not b_cases:
                # Fix: Flag empty bucket as violated check
                violated_checks.append(f"Bucket {b_name} has zero test cases in evaluation suite!")
                continue
            b_cand = [r["score_cand"] for r in b_cases]
            b_base = [r["score_base"] for r in b_cases]
            b_deltas = [r["delta"] for r in b_cases]
            _, _, b_imp, b_reg, _, _, _ = compute_mcnemar_discordance(
                b_cand, b_base, tau_pass=self.pass_threshold
            )
            bucket_breakdown[b_name] = {
                "num_cases": len(b_cases),
                "mean_candidate": round(sum(b_cand) / len(b_cand), 4),
                "mean_baseline": round(sum(b_base) / len(b_base), 4),
                "mean_delta": round(sum(b_deltas) / len(b_deltas), 4),
                "improvements": b_imp,
                "regressions": b_reg,
            }

        core_reg = bucket_breakdown.get("core_domain", {}).get("regressions", 999)
        regr_reg = bucket_breakdown.get("regression_replay", {}).get("regressions", 999)
        core_delta = bucket_breakdown.get("core_domain", {}).get("mean_delta", -999.0)
        edge_delta = bucket_breakdown.get("edge_case", {}).get("mean_delta", -999.0)
        adv_mean = bucket_breakdown.get("adversarial_bait", {}).get("mean_candidate", 0.0)

        # 4. Check Criteria
        # Criterion 1: Core Domain Regressions
        if core_reg == 0:
            passed_checks.append("Core Domain Zero-Regression Invariant: PASSED (0 regressions).")
        else:
            violated_checks.append(f"Core Domain Zero-Regression Invariant: FAILED ({core_reg} regressions).")

        # Criterion 2: Replay Buffer Regressions
        if regr_reg == 0:
            passed_checks.append("Regression Replay Buffer Retention: PASSED (0 regressions).")
        else:
            violated_checks.append(f"Regression Replay Buffer Retention: FAILED ({regr_reg} regressions).")

        # Criterion 3: Core Delta Non-inferiority
        if core_delta >= -self.parity_delta:
            passed_checks.append(f"Core Domain Non-Inferiority: PASSED (delta={core_delta:+.3f}).")
        else:
            violated_checks.append(f"Core Domain Non-Inferiority: FAILED (delta={core_delta:+.3f}).")

        # Criterion 4: Edge Case Delta Bounds
        if edge_delta >= self.edge_case_delta_threshold:
            passed_checks.append(f"Edge Case Delta Bounds: PASSED (delta={edge_delta:+.3f}).")
        else:
            violated_checks.append(f"Edge Case Delta Bounds: FAILED (delta={edge_delta:+.3f}).")

        # Criterion 5: Adversarial Resistance
        if adv_mean >= self.pass_threshold:
            passed_checks.append(f"Adversarial Bait Robustness: PASSED (mean={adv_mean:.2f}).")
        else:
            violated_checks.append(f"Adversarial Bait Robustness: FAILED (mean={adv_mean:.2f}).")

        # 5. Robust Gating Decision Matrix
        ci_width = overall_boot.ci_upper - overall_boot.ci_lower
        is_stat_sig_regress = (
            overall_boot.ci_upper < 0.0
            or (overall_hypo.wilcoxon_p < self.alpha and overall_boot.mean_delta < 0.0)
            or (overall_hypo.mcnemar_p < self.alpha and overall_hypo.regressions > overall_hypo.improvements)
        )
        is_core_or_regr_violation = (
            core_reg > 0 or regr_reg > 0 or core_delta < -self.parity_delta or edge_delta < self.edge_case_delta_threshold
        )

        # Fix: For unreleased candidates, failure should BLOCK_RELEASE, not TRIGGER_ROLLBACK
        if is_core_or_regr_violation or is_stat_sig_regress:
            if is_live_monitoring:
                decision = GateDecision.TRIGGER_ROLLBACK
                summary = "Live monitoring detected significant regression. Reverting to anchor."
            else:
                decision = GateDecision.BLOCK_RELEASE
                summary = "Release BLOCKED: Candidate exhibits statistically significant regression or edge-case collapse."
        elif ci_width > 0.80 or judge_inconsistency > 0.25:
            decision = GateDecision.ALERT_DRIFT
            summary = "Release flagged with ALERT_DRIFT: High variance or judge inconsistency."
        elif (
            core_reg == 0
            and regr_reg == 0
            and adv_mean >= self.pass_threshold
            and edge_delta >= self.edge_case_delta_threshold
            and overall_boot.mean_delta >= self.min_improvement_delta  # Fix: Disallow no-op/noise promotion
            and overall_boot.ci_lower > -self.parity_delta
        ):
            decision = GateDecision.PROCEED_DEPLOY
            summary = f"Release APPROVED: Statistically validated improvement (mean delta={overall_boot.mean_delta:+.3f})."
        else:
            decision = GateDecision.BLOCK_RELEASE
            summary = "Release HELD: Candidate did not demonstrate sufficient statistically significant improvement."

        return GateReport(
            decision=decision,
            overall_bootstrap=overall_boot,
            overall_hypothesis=overall_hypo,
            bucket_breakdown=bucket_breakdown,
            passed_checks=passed_checks,
            violated_checks=violated_checks,
            recommendation_summary=summary,
        )

