"""Master test runner for all gap reproduction tests."""

import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from actual_code_gaps.reproductions.test_critic_leakage_gap import TestCriticActionLeakageGap
from actual_code_gaps.reproductions.test_ipo_loss_discrepancy_gap import TestIPOLossDiscrepancyGap
from actual_code_gaps.reproductions.test_dropout_rollout_ratio_gap import TestDropoutRolloutRatioGap
from actual_code_gaps.reproductions.test_gate_decision_loopholes_gap import TestGateDecisionLoopholesGap


def run_suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestCriticActionLeakageGap))
    suite.addTests(loader.loadTestsFromTestCase(TestIPOLossDiscrepancyGap))
    suite.addTests(loader.loadTestsFromTestCase(TestDropoutRolloutRatioGap))
    suite.addTests(loader.loadTestsFromTestCase(TestGateDecisionLoopholesGap))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_suite()
    sys.exit(0 if success else 1)

