"""Unit and Integration Tests for Milestone 4, Phase 4.4 (Statistical Release Gating)."""

import shutil
import tempfile
import unittest
from pathlib import Path

from evaluation.benchmark_suite import GoldenBenchmarkSuite
from evaluation.statistical_gating import (
    GateDecision,
    GateReport,
    HypothesisTestResult,
    StatisticalReleaseGate,
    compute_mcnemar_discordance,
    compute_paired_bootstrap,
    compute_wilcoxon_signed_rank,
)
from evaluation.rollback_orchestrator import DeploymentState, RollbackOrchestrator


class TestM4StatisticalGating(unittest.TestCase):
    """Test suite for statistical release gating, hypothesis testing, and rollback orchestration."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.suite = GoldenBenchmarkSuite.load_default_suite()
        self.gate = StatisticalReleaseGate(bootstrap_samples=500, random_seed=42)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_paired_bootstrap_resampling(self):
        """Verify bootstrap confidence intervals accurately capture paired shifts."""
        cand = [4.5, 4.8, 4.2, 4.6, 4.7]
        base = [3.5, 3.8, 3.2, 3.6, 3.7]
        boot = compute_paired_bootstrap(cand, base, num_replicates=500, seed=42)

        self.assertAlmostEqual(boot.mean_delta, 1.0, places=3)
        self.assertGreater(boot.ci_lower, 0.90)
        self.assertLessEqual(boot.ci_upper, 1.10)
        self.assertEqual(boot.p_value_regress, 0.0)

    def test_wilcoxon_signed_rank(self):
        """Verify Wilcoxon signed-rank test outputs valid statistics and p-values."""
        cand = [4.5, 4.8, 3.5, 4.6, 4.7]
        base = [3.5, 3.8, 3.6, 3.4, 3.7]
        stat, p_val = compute_wilcoxon_signed_rank(cand, base)
        self.assertGreaterEqual(stat, 0.0)
        self.assertLess(p_val, 0.20)

        # Identical arrays
        stat_tied, p_tied = compute_wilcoxon_signed_rank(cand, cand)
        self.assertEqual(stat_tied, 0.0)
        self.assertEqual(p_tied, 1.0)

    def test_mcnemar_discordance_calculation(self):
        """Verify Edwards continuity-corrected McNemar discordance test."""
        # Candidate passes 4 cases where baseline failed
        cand = [4.0, 4.2, 4.5, 4.8, 3.8]
        base = [2.0, 2.5, 2.8, 3.0, 4.0]
        stat, p_val, imp, reg, t_p, t_f, r_ratio = compute_mcnemar_discordance(
            cand, base, tau_pass=3.5
        )
        self.assertEqual(imp, 4)
        self.assertEqual(reg, 0)
        self.assertEqual(t_p, 1)
        self.assertEqual(t_f, 0)
        self.assertEqual(r_ratio, 0.0)
        self.assertGreater(stat, 0.0)

    def test_gate_proceed_deploy_on_significant_improvement(self):
        """Verify statistically superior model without regressions achieves PROCEED_DEPLOY."""
        base_scores = {c.case_id: 3.8 for c in self.suite.cases}
        cand_scores = {c.case_id: 4.6 for c in self.suite.cases}

        report = self.gate.evaluate_gate(
            suite=self.suite,
            candidate_scores=cand_scores,
            baseline_scores=base_scores,
        )

        self.assertIsInstance(report, GateReport)
        self.assertEqual(report.decision, GateDecision.PROCEED_DEPLOY)
        self.assertGreater(report.overall_bootstrap.ci_lower, 0.0)
        self.assertEqual(report.bucket_breakdown["core_domain"]["regressions"], 0)
        self.assertEqual(report.bucket_breakdown["regression_replay"]["regressions"], 0)
        self.assertTrue(any("Core Domain Zero-Regression Invariant: PASSED" in c for c in report.passed_checks))

    def test_gate_candidate_regression_blocks_release(self):
        """Verify candidate regression before deployment yields BLOCK_RELEASE, preserving active model."""
        base_scores = {c.case_id: 3.8 for c in self.suite.cases}
        cand_scores = {c.case_id: 4.5 for c in self.suite.cases}
        cand_scores["core_ml_001"] = 2.0  # Core domain regression

        # Default is_active_deployment=False (candidate evaluation)
        report = self.gate.evaluate_gate(
            suite=self.suite,
            candidate_scores=cand_scores,
            baseline_scores=base_scores,
            is_active_deployment=False,
        )

        self.assertEqual(report.decision, GateDecision.BLOCK_RELEASE)
        self.assertGreater(report.bucket_breakdown["core_domain"]["regressions"], 0)
        self.assertTrue(any("Core Domain Zero-Regression Invariant: FAILED" in v for v in report.violated_checks))
        self.assertIn("Release BLOCKED", report.recommendation_summary)

    def test_gate_active_deployment_regression_triggers_rollback(self):
        """Verify active production deployment regression yields TRIGGER_ROLLBACK."""
        base_scores = {c.case_id: 3.8 for c in self.suite.cases}
        cand_scores = {c.case_id: 4.5 for c in self.suite.cases}
        cand_scores["core_ml_001"] = 2.0

        report = self.gate.evaluate_gate(
            suite=self.suite,
            candidate_scores=cand_scores,
            baseline_scores=base_scores,
            is_active_deployment=True,
        )

        self.assertEqual(report.decision, GateDecision.TRIGGER_ROLLBACK)
        self.assertGreater(report.bucket_breakdown["core_domain"]["regressions"], 0)
        self.assertIn("TRIGGER_ROLLBACK", report.recommendation_summary)

    def test_gate_rejects_missing_candidate_scores(self):
        """Verify candidate scores missing test cases fail closed with BLOCK_RELEASE without synthesizing 0.0."""
        base_scores = {c.case_id: 3.8 for c in self.suite.cases}
        cand_scores = {c.case_id: 4.5 for c in self.suite.cases}
        del cand_scores["core_ml_001"]  # Missing score

        report = self.gate.evaluate_gate(
            suite=self.suite,
            candidate_scores=cand_scores,
            baseline_scores=base_scores,
        )
        self.assertEqual(report.decision, GateDecision.BLOCK_RELEASE)
        self.assertTrue(any("Missing candidate evaluation scores" in v for v in report.violated_checks))

    def test_gate_rejects_missing_baseline_scores(self):
        """Verify baseline scores missing test cases fail closed with BLOCK_RELEASE without synthesizing 1.0."""
        base_scores = {c.case_id: 3.8 for c in self.suite.cases}
        cand_scores = {c.case_id: 4.5 for c in self.suite.cases}
        del base_scores["core_ml_001"]

        report = self.gate.evaluate_gate(
            suite=self.suite,
            candidate_scores=cand_scores,
            baseline_scores=base_scores,
        )
        self.assertEqual(report.decision, GateDecision.BLOCK_RELEASE)
        self.assertTrue(any("Missing baseline evaluation scores" in v for v in report.violated_checks))

    def test_gate_rejects_invalid_or_out_of_range_scores(self):
        """Verify non-finite or out-of-range scores fail closed."""
        base_scores = {c.case_id: 3.8 for c in self.suite.cases}
        cand_scores = {c.case_id: 4.5 for c in self.suite.cases}
        cand_scores["core_ml_001"] = float("nan")

        report = self.gate.evaluate_gate(
            suite=self.suite,
            candidate_scores=cand_scores,
            baseline_scores=base_scores,
        )
        self.assertEqual(report.decision, GateDecision.BLOCK_RELEASE)
        self.assertTrue(any("Invalid non-finite or out-of-range" in v for v in report.violated_checks))

    def test_gate_rejects_unknown_extra_case_ids(self):
        """Verify unexpected unknown case IDs fail closed."""
        base_scores = {c.case_id: 3.8 for c in self.suite.cases}
        cand_scores = {c.case_id: 4.5 for c in self.suite.cases}
        cand_scores["unknown_bogus_case_999"] = 4.0

        report = self.gate.evaluate_gate(
            suite=self.suite,
            candidate_scores=cand_scores,
            baseline_scores=base_scores,
        )
        self.assertEqual(report.decision, GateDecision.BLOCK_RELEASE)
        self.assertTrue(any("Unknown candidate case IDs" in v for v in report.violated_checks))

    def test_gate_blocks_release_when_bucket_is_missing(self):
        """Verify missing regression_replay bucket fails closed and cannot approve release."""
        # Create a partial suite with NO regression_replay cases
        non_replay_cases = [c for c in self.suite.cases if c.bucket != "regression_replay"]
        partial_suite = GoldenBenchmarkSuite(non_replay_cases)

        base_scores = {c.case_id: 3.8 for c in partial_suite.cases}
        cand_scores = {c.case_id: 4.6 for c in partial_suite.cases}  # High scores!

        report = self.gate.evaluate_gate(
            suite=partial_suite,
            candidate_scores=cand_scores,
            baseline_scores=base_scores,
        )
        self.assertEqual(report.decision, GateDecision.BLOCK_RELEASE)
        self.assertTrue(any("regression_replay" in v for v in report.violated_checks))

    def test_gate_blocks_no_op_candidate(self):
        """Verify candidate with exact parity / no-op (mean_delta == 0.0) is blocked by positive margin."""
        base_scores = {c.case_id: 4.0 for c in self.suite.cases}
        cand_scores = {c.case_id: 4.0 for c in self.suite.cases}  # Exact tie

        report = self.gate.evaluate_gate(
            suite=self.suite,
            candidate_scores=cand_scores,
            baseline_scores=base_scores,
        )
        self.assertEqual(report.decision, GateDecision.BLOCK_RELEASE)
        self.assertAlmostEqual(report.overall_bootstrap.mean_delta, 0.0, places=4)
        self.assertIn("below required improvement margin", report.recommendation_summary)

    def test_gate_power_analysis_computation(self):
        """Verify power analysis metrics are computed and included in GateReport."""
        base_scores = {c.case_id: 3.8 for c in self.suite.cases}
        cand_scores = {c.case_id: 4.6 for c in self.suite.cases}

        report = self.gate.evaluate_gate(
            suite=self.suite,
            candidate_scores=cand_scores,
            baseline_scores=base_scores,
        )
        self.assertIsNotNone(report.power_analysis)
        self.assertIn("power", report.power_analysis)
        self.assertIn("mde", report.power_analysis)
        self.assertEqual(report.power_analysis["sample_size"], len(self.suite.cases))

    def test_gate_alert_drift_on_wide_ci_or_judge_inconsistency(self):
        """Verify evaluation volatility flags ALERT_DRIFT."""
        base_scores = {c.case_id: 3.5 for c in self.suite.cases}
        cand_scores = {c.case_id: 3.5 for c in self.suite.cases}

        # Case 1: High judge inconsistency kappa_pos = 0.35 (> 0.25)
        report_inconsistent = self.gate.evaluate_gate(
            suite=self.suite,
            candidate_scores=cand_scores,
            baseline_scores=base_scores,
            judge_inconsistency=0.35,
        )
        self.assertEqual(report_inconsistent.decision, GateDecision.ALERT_DRIFT)

    def test_gate_block_release_on_indeterminate_parity(self):
        """Verify minor non-significant delta where CI spans zero results in BLOCK_RELEASE."""
        base_scores = {c.case_id: 4.0 for c in self.suite.cases}
        # Intermixed positive and negative differences with slight negative mean (-0.0017)
        deltas_pattern = [0.05, -0.05, 0.03, -0.04, 0.02, -0.02]
        cand_scores = {
            c.case_id: 4.0 + deltas_pattern[idx % len(deltas_pattern)]
            for idx, c in enumerate(self.suite.cases)
        }

        report = self.gate.evaluate_gate(
            suite=self.suite,
            candidate_scores=cand_scores,
            baseline_scores=base_scores,
        )
        self.assertEqual(report.decision, GateDecision.BLOCK_RELEASE)
        self.assertLess(report.overall_bootstrap.ci_lower, 0.0)
        self.assertGreater(report.overall_bootstrap.ci_upper, 0.0)
        self.assertGreater(report.overall_hypothesis.wilcoxon_p, 0.05)

    def test_rollback_orchestrator_lifecycle(self):
        """Verify RollbackOrchestrator atomic promotion, candidate rejection, and hot-swap rollback."""
        orch = RollbackOrchestrator(metadata_dir=self.temp_dir)
        self.assertEqual(orch.state.active_model_id, "baseline_v0")

        # 1. Register Golden Anchor
        anchor_path = f"{self.temp_dir}/models/golden_anchor.pt"
        orch.register_golden_anchor("golden_anchor_v1", anchor_path)
        self.assertEqual(orch.state.golden_anchor_id, "golden_anchor_v1")
        self.assertEqual(orch.state.active_model_id, "golden_anchor_v1")

        # 2. Simulate PROCEED_DEPLOY
        cand_path = f"{self.temp_dir}/models/checkpoint_step100.pt"
        report_deploy = GateReport(
            decision=GateDecision.PROCEED_DEPLOY,
            overall_bootstrap=compute_paired_bootstrap([4.5], [3.5], seed=42),
            overall_hypothesis=HypothesisTestResult(
                wilcoxon_stat=1.0, wilcoxon_p=0.01, mcnemar_stat=1.0, mcnemar_p=0.01,
                improvements=1, regressions=0, tied_passes=0, tied_fails=0, regression_ratio=0.0
            ),
            bucket_breakdown={},
            recommendation_summary="Approved",
        )
        orch.execute_gate_decision(report_deploy, "cand_model_v2", cand_path)
        self.assertEqual(orch.state.active_model_id, "cand_model_v2")
        self.assertEqual(orch.state.active_checkpoint_path, cand_path)

        # 3. Simulate pre-release candidate regression on an unpromoted candidate
        # Active model must NOT roll back because cand_model_v3_bad was never deployed!
        regressed_path = f"{self.temp_dir}/models/checkpoint_step200.pt"
        report_rollback = GateReport(
            decision=GateDecision.TRIGGER_ROLLBACK,
            overall_bootstrap=compute_paired_bootstrap([2.0], [4.5], seed=42),
            overall_hypothesis=HypothesisTestResult(
                wilcoxon_stat=1.0, wilcoxon_p=0.01, mcnemar_stat=1.0, mcnemar_p=0.01,
                improvements=0, regressions=1, tied_passes=0, tied_fails=0, regression_ratio=1.0
            ),
            bucket_breakdown={},
            recommendation_summary="Core domain forgetting detected",
        )
        orch.execute_gate_decision(report_rollback, "cand_model_v3_bad", regressed_path)
        # Active model remains cand_model_v2 untouched!
        self.assertEqual(orch.state.active_model_id, "cand_model_v2")
        self.assertEqual(orch.state.active_checkpoint_path, cand_path)

        # 4. Now simulate regression on the ACTIVE model (cand_model_v2)
        orch.execute_gate_decision(report_rollback, "cand_model_v2", cand_path)
        # Active model must now roll back to golden anchor!
        self.assertEqual(orch.state.active_model_id, "golden_anchor_v1")
        self.assertEqual(orch.state.active_checkpoint_path, anchor_path)
        self.assertEqual(orch.state.last_action, "ROLLED_BACK")

        # 5. Persistence roundtrip check with a new orchestrator instance
        orch_reloaded = RollbackOrchestrator(metadata_dir=self.temp_dir)
        self.assertEqual(orch_reloaded.state.active_model_id, "golden_anchor_v1")
        self.assertEqual(len(orch_reloaded.state.history), 4)


if __name__ == "__main__":
    unittest.main()
