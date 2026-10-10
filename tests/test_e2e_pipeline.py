"""Comprehensive End-to-End Pipeline Integration Test Suite."""

import unittest
import torch
from loguru import logger
from fastapi.testclient import TestClient

from core.hardware import get_device_strategy
from core.schemas import UserQuerySchema
from training.m1_bigram.tokenizer import CharTokenizer
from training.m1_transformer.config import GPTConfig
from training.m1_transformer.model import MiniGPT
from curation.minhash_lsh import MinHashLSH
from curation.replay_buffer import ExperienceReplayBuffer, ModelCollapseGuard
from training.m3_sft.lora import LoRAConfig, inject_lora_to_model
from training.m3_sft.dataset import CompletionMaskCollator
from training.m5_preference.verifiable_reward import VerifiableRewardEngine
from training.m5_preference.grpo_loss import compute_group_advantages
from evaluation.citation_verifier import CitationVerifier
from retrieval.schemas import RetrievalDocument
from retrieval.hybrid import HybridRetriever
from orchestration.loop_orchestrator import ContinualLearningOrchestrator
from apps.serving_manager import ModelServingManager
from apps.server import create_app


class TestE2EPipeline(unittest.TestCase):
    """End-to-End integration test covering M0 through M6."""

    def test_full_pipeline_lifecycle(self):
        # Step 1: Hardware Strategy & Schemas
        strategy = get_device_strategy()
        self.assertIsNotNone(strategy.device)
        query_contract = UserQuerySchema(query="What is Direct Preference Optimization?")
        self.assertEqual(query_contract.query, "What is Direct Preference Optimization?")

        # Step 2: MiniGPT Transformer & KV-Cache
        vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 <|>_:\n.,!?-+*/"
        tokenizer = CharTokenizer(vocab)
        config = GPTConfig(
            vocab_size=len(tokenizer),
            block_size=64,
            d_model=32,
            n_head=2,
            n_layer=2,
            dropout=0.0,
        )
        base_model = MiniGPT(config)
        sample_prompt = "What is RL?"
        input_ids = torch.tensor([tokenizer.encode(sample_prompt)], dtype=torch.long)
        generated = base_model.generate(input_ids, max_new_tokens=10, use_cache=True)
        self.assertEqual(generated.shape[1], input_ids.shape[1] + 10)

        # Step 3: Curation, MinHash LSH, & Model Collapse Guard
        lsh = MinHashLSH(num_perm=128, num_bands=32, threshold=0.80)
        sig1 = lsh.compute_signature(
            "Direct Preference Optimization trains policy weights directly without a separate reward model or critic network."
        )
        sig2 = lsh.compute_signature(
            "Direct Preference Optimization trains policy weights directly without a separate reward model or critic framework."
        )
        sim = lsh.jaccard_similarity(sig1, sig2)
        self.assertGreaterEqual(sim, 0.70)

        collapse_guard = ModelCollapseGuard()
        safe, reason, _ = collapse_guard.evaluate_text(
            "Direct Preference Optimization optimizes language models directly using preference data without reward models."
        )
        self.assertTrue(safe)
        unsafe, _, _ = collapse_guard.evaluate_text("error " * 20)
        self.assertFalse(unsafe)

        # Step 4: Stratified Experience Replay Buffer (70/30)
        replay_buf = ExperienceReplayBuffer(anchor_ratio=0.70)
        replay_buf.add_anchor_data([
            "Anchor Golden Baseline 1: Standard language syntax and conversational foundations.",
            "Anchor Golden Baseline 2: Mathematical arithmetic and formal definitions.",
        ])
        replay_buf.add_web_data([
            "Curated Web Data 1: Direct Preference Optimization paper overview.",
        ])
        batch = replay_buf.sample_batch(batch_size=3)
        self.assertEqual(len(batch), 3)

        # Step 5: SFT with LoRA & Completion-Only Masking
        lora_cfg = LoRAConfig(r=4, lora_alpha=8.0, target_modules=["qkv_proj", "out_proj"])
        lora_model = inject_lora_to_model(base_model, lora_cfg)
        chat_text = (
            "<|im_start|>user\nWhat is DPO?<|im_end|>\n"
            "<|im_start|>assistant\nDPO is Direct Preference Optimization.<|im_end|>\n"
        )
        collator = CompletionMaskCollator(tokenizer=tokenizer, max_seq_len=64)
        inp, tgt, msk = collator.collate_single(chat_text)
        self.assertIn(-100, tgt)

        # Step 6: Preference Learning & Verifiable Rewards
        verifiable_engine = VerifiableRewardEngine()
        test_reasoning = "<think>Compute 12 + 8 which is 20.</think><answer>20</answer>"
        scores = verifiable_engine.score_completion(test_reasoning, ground_truth="20")
        self.assertEqual(scores["format_reward"], 1.0)
        self.assertEqual(scores["accuracy_reward"], 1.0)
        self.assertEqual(scores["total_reward"], 1.5)

        group_rewards = torch.tensor([1.5, 0.5, 1.5, 0.5])
        advantages = compute_group_advantages(group_rewards, group_size=4)
        self.assertEqual(advantages.shape, (4,))
        self.assertGreater(advantages[0].item(), 0)
        self.assertLess(advantages[1].item(), 0)

        # Step 7: Citation Faithfulness & ALCE Verification
        verifier = CitationVerifier()
        claim_text = (
            "Direct Preference Optimization eliminates the need for reinforcement learning [1]. "
            "It expresses the reward function implicitly through the language model policy [2]."
        )
        contexts = {
            "1": "Direct Preference Optimization eliminates the need for reinforcement learning by optimizing directly.",
            "2": "Direct Preference Optimization expresses the reward function implicitly through the language model policy.",
        }
        full_context = " ".join(contexts.values())
        cit_metrics = verifier.verify_response(
            response=claim_text,
            ground_truth_context=full_context,
            reference_documents=contexts,
        )
        self.assertGreaterEqual(cit_metrics.citation_precision, 0.80)
        self.assertGreaterEqual(cit_metrics.faithfulness_score, 0.80)

        # Step 8: Hybrid Retrieval (BM25 + Dense RRF)
        retriever = HybridRetriever()
        retriever.add_document(
            RetrievalDocument(
                doc_id="doc_rlhf_1",
                text="Reinforcement Learning from Human Feedback aligns models with human intent using preference datasets.",
                url="https://arxiv.org/abs/rlhf",
                metadata={"domain": "alignment"},
            )
        )
        retriever.add_document(
            RetrievalDocument(
                doc_id="doc_dpo_2",
                text="Direct Preference Optimization trains policy weights directly without a separate reward model.",
                url="https://arxiv.org/abs/dpo",
                metadata={"domain": "alignment"},
            )
        )
        ret_results = retriever.retrieve("Direct Preference Optimization without reward models", top_k=2)
        self.assertEqual(len(ret_results), 2)
        self.assertEqual(ret_results[0].doc_id, "doc_dpo_2")

        # Step 9: Continual Loop Orchestrator & Serving Gateway E2E Flow
        serving_mgr = ModelServingManager(
            active_model="active_baseline_v1",
            anchor_model="golden_anchor_baseline",
            model_version=1,
        )
        orchestrator = ContinualLearningOrchestrator(
            active_model="active_baseline_v1",
            anchor_model="golden_anchor_baseline",
            retriever=retriever,
            replay_buffer=replay_buf,
            training_trigger_threshold=1,
        )
        app = create_app(orchestrator=orchestrator, serving_manager=serving_mgr)
        client = TestClient(app)

        # 9a. Health check
        health_resp = client.get("/health")
        self.assertEqual(health_resp.status_code, 200)
        self.assertEqual(health_resp.json()["model_version"], 1)

        # 9b. User query turn 1: Cache Miss -> Grounded RAG
        q_payload = {"query": "How does DPO work without reward models?", "top_k": 2}
        r1 = client.post("/query", json=q_payload)
        self.assertEqual(r1.status_code, 200)
        data1 = r1.json()
        self.assertFalse(data1["cached"])
        self.assertGreaterEqual(len(data1["evidence"]), 1)

        # 9c. User query turn 2: Semantic Cache Hit (< 5ms fast path)
        r2 = client.post("/query", json=q_payload)
        self.assertEqual(r2.status_code, 200)
        data2 = r2.json()
        self.assertTrue(data2["cached"])
        self.assertGreaterEqual(data2["similarity"], 0.99)

        # 9d. Ingest new query knowledge and trigger continual learning cycle
        client.post("/query", json={"query": "Explain quantum neural algorithms"})
        train_resp = client.post("/trigger_learning", json={"force": True})
        self.assertEqual(train_resp.status_code, 200)

        # 9e. Candidate Promotion & Zero-Downtime Hot-Swap
        orchestrator.candidate_model = "promoted_candidate_model_v2"
        promote_resp = client.post("/models/promote", json={"version": 2})
        self.assertEqual(promote_resp.status_code, 200)
        self.assertEqual(promote_resp.json()["new_version"], 2)
        self.assertEqual(client.get("/health").json()["model_version"], 2)

        # 9f. Rollback to Golden Anchor Baseline
        rollback_resp = client.post("/models/rollback")
        self.assertEqual(rollback_resp.status_code, 200)
        self.assertEqual(rollback_resp.json()["model_version"], 0)
        self.assertEqual(client.get("/health").json()["model_version"], 0)


if __name__ == "__main__":
    unittest.main()

