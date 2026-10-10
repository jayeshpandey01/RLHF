"""Master test runner for the Autonomous Continual Learning LLM repository."""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.run_tests import TestMilestone0
from tests.test_m1_autograd import TestAutogradEngine
from tests.test_m1_bigram import TestBigramLanguageModel
from tests.test_m1_attention import TestCausalAttentionAndKVCache
from tests.test_m1_transformer import TestMiniGPTTransformer
from tests.test_m2_scraper import TestM2Scraper
from tests.test_m2_curation import TestM2Curation
from tests.test_m3_sft import TestM3SFTDataPipeline
from tests.test_m4_benchmark_suite import TestM4BenchmarkSuite
from tests.test_m4_citation_verifier import TestM4CitationVerifier
from tests.test_m4_llm_judge import TestM4LLMJudge
from tests.test_m4_statistical_gating import TestM4StatisticalGating
from tests.test_m5_reward_model import TestM5RewardModel
from tests.test_m5_dpo import TestM5DPO
from tests.test_m5_ppo import TestM5PPO
from tests.test_m5_grpo import TestM5GRPO
from tests.test_m5_alignment_suite import TestM5AlignmentSuite
from tests.test_m6_hybrid_retrieval import TestM6HybridRetrieval
from tests.test_m6_replay_buffer import TestM6ReplayBuffer
from tests.test_m6_orchestrator import TestM6Orchestrator
from tests.test_m6_serving_gateway import TestM6ServingGateway
from tests.test_e2e_pipeline import TestE2EPipeline

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    print("======================================================================")
    print("🚀 RUNNING MASTER VERIFICATION SUITE: M0, M1, M2, M3, M4, M5, AND M6.4 COMPLETE")
    print("======================================================================")

    suite.addTests(loader.loadTestsFromTestCase(TestMilestone0))
    suite.addTests(loader.loadTestsFromTestCase(TestAutogradEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestBigramLanguageModel))
    suite.addTests(loader.loadTestsFromTestCase(TestCausalAttentionAndKVCache))
    suite.addTests(loader.loadTestsFromTestCase(TestMiniGPTTransformer))
    suite.addTests(loader.loadTestsFromTestCase(TestM2Scraper))
    suite.addTests(loader.loadTestsFromTestCase(TestM2Curation))
    suite.addTests(loader.loadTestsFromTestCase(TestM3SFTDataPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestM4BenchmarkSuite))
    suite.addTests(loader.loadTestsFromTestCase(TestM4CitationVerifier))
    suite.addTests(loader.loadTestsFromTestCase(TestM4LLMJudge))
    suite.addTests(loader.loadTestsFromTestCase(TestM4StatisticalGating))
    suite.addTests(loader.loadTestsFromTestCase(TestM5RewardModel))
    suite.addTests(loader.loadTestsFromTestCase(TestM5DPO))
    suite.addTests(loader.loadTestsFromTestCase(TestM5PPO))
    suite.addTests(loader.loadTestsFromTestCase(TestM5GRPO))
    suite.addTests(loader.loadTestsFromTestCase(TestM5AlignmentSuite))
    suite.addTests(loader.loadTestsFromTestCase(TestM6HybridRetrieval))
    suite.addTests(loader.loadTestsFromTestCase(TestM6ReplayBuffer))
    suite.addTests(loader.loadTestsFromTestCase(TestM6Orchestrator))
    suite.addTests(loader.loadTestsFromTestCase(TestM6ServingGateway))
    suite.addTests(loader.loadTestsFromTestCase(TestE2EPipeline))



    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    sys.exit(0 if result.wasSuccessful() else 1)
