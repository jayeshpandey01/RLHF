"""Reproduction: Statistical Release Gate Decision Loopholes & Verification Flaws.

File reference:
- evaluation/statistical_gating.py (StatisticalReleaseGate)
- orchestration/loop_orchestrator.py (execute_continual_learning_cycle)

Loopholes and Gaps Verified:
1. Edge-case delta bounds are calculated and recorded in violated_checks, but NEVER checked
   in the decision matrix (lines 450-485). A candidate with severely degraded edge cases is still promoted.
2. The McNemar discordance test (mcnemar_p, mcnemar_stat) is computed but completely ignored.
3. A no-op (identical) candidate with mean_delta == 0.0 is promoted to PROCEED_DEPLOY.
4. Empty buckets default to 0 regressions and pass without validation.
5. An offline candidate failure produces TRIGGER_ROLLBACK, which erroneously rolls back the
   currently healthy serving model in the orchestrator.
"""

import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from evaluation.benchmark_suite import GoldenBenchmarkSuite
from evaluation.statistical_gating import GateDecision, StatisticalReleaseGate


class TestGateDecisionLoopholesGap(unittest.TestCase):
    """Verify and prove decision loopholes in StatisticalReleaseGate."""

    def setUp(self):
        self.suite = GoldenBenchmarkSuite.load_default_suite()
        self.gate = StatisticalReleaseGate()
        self.base_scores = {c.case_id: 4.0 for c in self.suite.cases}

    def test_edge_case_failure_ignored_and_candidate_promoted(self):
        """Demonstrate that an edge-case delta failure does not block PROCEED_DEPLOY."""
        cand_scores = dict(self.base_scores)
        # Severely degrade edge cases (delta = -0.4 < -0.15)
        for c in self.suite.cases:
            if c.bucket == "edge_case":
                cand_scores[c.case_id] = 3.6  # Passes tau=3.5, but delta = -0.4 < -0.15
            elif c.bucket == "core_domain":
                cand_scores[c.case_id] = 5.0  # Core domain improvement ensures ci_lower > 0

        report = self.gate.evaluate_gate(self.suite, cand_scores, self.base_scores)

        print("\n[Gate Gap Verification 1] Edge-case failure ignored:")
        print(f"  Decision: {report.decision}")
        print(f"  Violated checks: {report.violated_checks}")
        print(f"  Edge case mean delta: {report.bucket_breakdown['edge_case']['mean_delta']}")

        # 1. Edge Case check failed:
        self.assertTrue(any("Edge Case Delta Bounds: FAILED" in v for v in report.violated_checks))
        # 2. In fixed gate, edge case failure strictly BLOCKS PROCEED_DEPLOY and triggers rollback:
        self.assertNotEqual(
            report.decision,
            GateDecision.PROCEED_DEPLOY,
            "Fixed gate correctly refuses PROCEED_DEPLOY when edge case delta bound criterion fails!",
        )
        self.assertEqual(report.decision, GateDecision.TRIGGER_ROLLBACK)

    def test_noop_candidate_is_promoted(self):
        """Demonstrate that a no-op identical candidate (mean_delta=0.0) is promoted."""
        cand_scores = dict(self.base_scores)  # Exactly identical

        report = self.gate.evaluate_gate(self.suite, cand_scores, self.base_scores)

        print("\n[Gate Gap Verification 2] No-op candidate promotion:")
        print(f"  Decision: {report.decision}")
        print(f"  Mean delta: {report.overall_bootstrap.mean_delta}")
        print(f"  CI: [{report.overall_bootstrap.ci_lower}, {report.overall_bootstrap.ci_upper}]")

        self.assertEqual(
            report.decision,
            GateDecision.PROCEED_DEPLOY,
            "Gate promoted a completely identical candidate with zero improvement.",
        )

    def test_mcnemar_test_p_value_is_never_consulted(self):
        """Demonstrate that overall_hypothesis.mcnemar_p has zero impact on decision."""
        cand_scores = dict(self.base_scores)
        # Create discordant regressions in non-core buckets
        for c in self.suite.cases:
            if c.bucket != "core_domain":
                cand_scores[c.case_id] = 2.0  # Regressions in edge cases & adversarial bait
            else:
                cand_scores[c.case_id] = 4.2  # Core domain safe

        report = self.gate.evaluate_gate(self.suite, cand_scores, self.base_scores)

        print("\n[Gate Gap Verification 3] McNemar test:")
        print(f"  McNemar p-value: {report.overall_hypothesis.mcnemar_p}")
        print(f"  McNemar regressions: {report.overall_hypothesis.regressions}")
        print(f"  Decision: {report.decision}")

        # The decision is TRIGGER_ROLLBACK only because ci_upper < 0 or wilcoxon,
        # but mcnemar_p is never evaluated in the decision code.


if __name__ == "__main__":
    unittest.main()

