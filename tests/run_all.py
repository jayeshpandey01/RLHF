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

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    print("======================================================================")
    print("🚀 RUNNING MASTER VERIFICATION SUITE: M0, M1, M2, M3, AND M4 COMPLETE")
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



    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    sys.exit(0 if result.wasSuccessful() else 1)
