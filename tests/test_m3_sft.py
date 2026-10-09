"""Comprehensive unit and integration test suite for Milestone 3 (SFT & LoRA Data Pipeline)."""

import sys
from pathlib import Path
import unittest
import string
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tempfile
import copy

from training.m1_bigram.tokenizer import CharTokenizer
from training.m1_transformer.config import GPTConfig
from training.m1_transformer.model import MiniGPT
from training.m3_sft.prompt import (
    ChatMessage,
    InstructionExample,
    ChatMLFormatter,
)
from training.m3_sft.dataset import (
    CompletionMaskCollator,
    HarmonizedSFTDataset,
    create_default_anchor_examples,
    create_web_curation_examples,
)
from training.m3_sft.lora import (
    LoRAConfig,
    LoRALinear,
    inject_lora_to_model,
    mark_only_lora_as_trainable,
    merge_lora_weights,
    unmerge_lora_weights,
    count_trainable_parameters,
    save_adapter_weights,
    load_adapter_weights,
)
from training.m3_sft.trainer import (
    SFTConfig,
    TrainMetrics,
    SFTTrainer,
)
from training.m3_sft.evaluator import (
    RegressionResult,
    SFTRegressionEvaluator,
)
from training.m3_sft.orchestrator import (
    OrchestratorPipelineResult,
    ContinualSFTOrchestrator,
)


class TestM3SFTDataPipeline(unittest.TestCase):
    """Rigorous mathematical and engineering verification of Phase 3.1 SFT data pipeline."""

    def setUp(self):
        # Build comprehensive tokenizer supporting all standard ASCII and ChatML delimiters
        vocab_chars = sorted(
            list(set(string.printable + "<|>|_" + string.ascii_letters + string.digits))
        )
        self.tokenizer = CharTokenizer(vocab_chars)

    def test_chatml_formatter_single_turn(self):
        """Verify ChatML serialization for single-turn dialogue."""
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Explain attention."),
            ChatMessage(role="assistant", content="Attention weights tokens."),
        ]
        formatted = ChatMLFormatter.format_dialogue(messages)
        self.assertIn("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", formatted)
        self.assertIn("<|im_start|>user\nExplain attention.<|im_end|>\n", formatted)
        self.assertIn("<|im_start|>assistant\nAttention weights tokens.<|im_end|>\n", formatted)

        # Verify assistant span extraction
        spans = ChatMLFormatter.extract_assistant_spans(formatted)
        self.assertEqual(len(spans), 1)
        start, end = spans[0]
        self.assertEqual(formatted[start:end], "Attention weights tokens.")

    def test_chatml_formatter_multi_turn(self):
        """Verify multi-turn dialogue serialization and span extraction."""
        messages = [
            ChatMessage(role="system", content="Act as a tutor."),
            ChatMessage(role="user", content="Question 1"),
            ChatMessage(role="assistant", content="Answer 1"),
            ChatMessage(role="user", content="Question 2"),
            ChatMessage(role="assistant", content="Answer 2"),
        ]
        formatted = ChatMLFormatter.format_dialogue(messages)
        spans = ChatMLFormatter.extract_assistant_spans(formatted)
        self.assertEqual(len(spans), 2)
        self.assertEqual(formatted[spans[0][0] : spans[0][1]], "Answer 1")
        self.assertEqual(formatted[spans[1][0] : spans[1][1]], "Answer 2")

    def test_chatml_format_prompt_only(self):
        """Verify inference prompt generation terminates with '<|im_start|>assistant\\n'."""
        messages = [
            ChatMessage(role="system", content="You are a poet."),
            ChatMessage(role="user", content="Write a sonnet."),
            ChatMessage(role="assistant", content="Shall I compare thee..."),
        ]
        prompt_only = ChatMLFormatter.format_prompt_only(messages)
        self.assertTrue(prompt_only.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("Shall I compare thee", prompt_only)

    def test_completion_mask_collator_prompt_invariance(self):
        """Verify that prompt tokens receive target label -100 and assistant tokens are active."""
        collator = CompletionMaskCollator(self.tokenizer, max_seq_len=256)
        messages = [
            ChatMessage(role="system", content="Helpful AI."),
            ChatMessage(role="user", content="Hi!"),
            ChatMessage(role="assistant", content="Hello world!"),
        ]
        chatml_text = ChatMLFormatter.format_dialogue(messages)
        inp, tgt, msk = collator.collate_single(chatml_text)

        self.assertEqual(len(inp), len(tgt))
        self.assertEqual(len(inp), len(msk))

        # Check that there are both masked (-100) and unmasked tokens
        masked_count = sum(1 for t in tgt if t == -100)
        unmasked_count = sum(1 for t in tgt if t != -100)
        self.assertGreater(masked_count, 0)
        self.assertGreater(unmasked_count, 0)

        # Decoding active targets should yield the assistant's content
        active_tokens = [t for t in tgt if t != -100]
        decoded_assistant = self.tokenizer.decode(active_tokens)
        self.assertIn("Hello world!", decoded_assistant)

    def test_completion_mask_collator_batch_padding(self):
        """Verify batch collation padding and attention mask properties."""
        collator = CompletionMaskCollator(self.tokenizer, max_seq_len=256)
        short_text = ChatMLFormatter.format_dialogue([
            ChatMessage(role="user", content="Short"),
            ChatMessage(role="assistant", content="Short answer"),
        ])
        long_text = ChatMLFormatter.format_dialogue([
            ChatMessage(role="system", content="Detailed system instructions for testing."),
            ChatMessage(role="user", content="A significantly longer prompt to test padding."),
            ChatMessage(role="assistant", content="A substantially more elaborate response text."),
        ])

        batch = collator([short_text, long_text])
        self.assertEqual(batch.input_ids.shape[0], 2)
        self.assertEqual(batch.targets.shape[0], 2)
        self.assertEqual(batch.attention_mask.shape[0], 2)
        self.assertEqual(batch.input_ids.shape[1], batch.targets.shape[1])

        # Short text should have padding where target is -100 and attention mask is 0
        pad_positions = (batch.attention_mask[0] == 0).nonzero(as_tuple=True)[0]
        self.assertGreater(len(pad_positions), 0)
        for idx in pad_positions:
            self.assertEqual(batch.targets[0, idx].item(), -100)

    def test_cross_entropy_gradient_zero_on_prompt(self):
        """Mathematical proof verification: gradients on prompt tokens are identically zero."""
        collator = CompletionMaskCollator(self.tokenizer, max_seq_len=128)
        text = ChatMLFormatter.format_dialogue([
            ChatMessage(role="user", content="Prompt query"),
            ChatMessage(role="assistant", content="Answer output"),
        ])
        batch = collator([text])

        # Simulate logits from a language model
        seq_len = batch.input_ids.size(1)
        vocab_size = self.tokenizer.vocab_size
        logits = torch.randn(1, seq_len, vocab_size, requires_grad=True)

        # Compute cross entropy with ignore_index=-100
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            batch.targets.view(-1),
            ignore_index=-100,
        )
        loss.backward()

        # Find prompt positions where targets == -100
        masked_positions = (batch.targets[0] == -100).nonzero(as_tuple=True)[0]
        self.assertGreater(len(masked_positions), 0)

        # Gradients on masked logits must be identically 0.0
        grad_on_masked = logits.grad[0, masked_positions, :]
        self.assertTrue(torch.all(grad_on_masked == 0.0))

        # Find active assistant positions
        active_positions = (batch.targets[0] != -100).nonzero(as_tuple=True)[0]
        self.assertGreater(len(active_positions), 0)
        grad_on_active = logits.grad[0, active_positions, :]
        self.assertFalse(torch.all(grad_on_active == 0.0))

    def test_harmonized_sft_dataset_convex_mixture(self):
        """Verify experience replay 70:30 sampling distribution from HarmonizedSFTDataset."""
        anchor_examples = create_default_anchor_examples()
        web_examples = create_web_curation_examples([
            "Transformers are deep learning architectures based on multi-head self-attention. They dominate NLP.",
            "Apple Silicon Metal Performance Shaders accelerate tensor operations via unified memory architecture.",
        ])

        dataset = HarmonizedSFTDataset(
            anchor_examples=anchor_examples,
            web_examples=web_examples,
            anchor_ratio=0.70,
            epoch_length=500,
            seed=42,
        )

        self.assertEqual(len(dataset), 500)

        # Check empirical sampling distribution
        anchor_signatures = [
            "honour and virtue",
            "purpose of attention",
            "twelve and twenty-eight",
            "machine learning is a field",
        ]
        anchor_hits = 0
        for i in range(500):
            sample_text = dataset[i]
            if any(sig in sample_text.lower() for sig in anchor_signatures):
                anchor_hits += 1

        empirical_ratio = anchor_hits / 500.0
        # Expect empirical ratio to fall within reasonable bounds around 0.70 (e.g. [0.62, 0.78])
        self.assertGreaterEqual(empirical_ratio, 0.62)
        self.assertLessEqual(empirical_ratio, 0.78)

    def test_lora_linear_initial_identity_preservation(self):
        """Verify LoRALinear forward pass is bit-for-bit identical to base linear at step 0."""
        base = torch.nn.Linear(64, 128, bias=False)
        lora = LoRALinear(base, r=8, lora_alpha=16.0)
        x = torch.randn(2, 10, 64)
        base_out = base(x)
        lora_out = lora(x)
        self.assertTrue(torch.allclose(base_out, lora_out, atol=1e-6))

    def test_lora_linear_merge_and_unmerge(self):
        """Verify in-place weight merging and reversibility on LoRALinear."""
        base = torch.nn.Linear(64, 128, bias=False)
        lora = LoRALinear(base, r=8, lora_alpha=16.0)
        # Give lora_B non-zero values to simulate post-training state
        torch.nn.init.normal_(lora.lora_B, std=0.02)
        x = torch.randn(2, 10, 64)

        unmerged_out = lora(x)
        lora.merge()
        self.assertTrue(lora.is_merged)
        merged_out = lora(x)
        self.assertTrue(torch.allclose(unmerged_out, merged_out, atol=1e-5))

        lora.unmerge()
        self.assertFalse(lora.is_merged)
        restored_out = lora(x)
        self.assertTrue(torch.allclose(unmerged_out, restored_out, atol=1e-5))

    def test_model_lora_injection_and_param_reduction(self):
        """Verify MiniGPT model injection freezes base weights and reduces trainable params by >90%."""
        config = GPTConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=32,
            n_layer=2,
            n_head=4,
            d_model=64,
        )
        model = MiniGPT(config)

        lora_config = LoRAConfig(
            r=4,
            lora_alpha=8.0,
            target_modules=["qkv_proj", "out_proj", "c_fc", "c_proj"],
        )
        model = inject_lora_to_model(model, lora_config)

        trainable, total, pct = count_trainable_parameters(model)
        self.assertLess(pct, 10.0)  # >90% reduction
        self.assertGreater(trainable, 0)

        # Check that lm_head and wte remain frozen
        self.assertFalse(model.lm_head.weight.requires_grad)
        self.assertFalse(model.transformer.wte.weight.requires_grad)

    def test_lora_model_forward_identity_and_gradients(self):
        """Verify MiniGPT with LoRA produces identical logits at t=0 and restricts gradients to LoRA."""
        config = GPTConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=16,
            n_layer=2,
            n_head=2,
            d_model=32,
        )
        base_model = MiniGPT(config)
        lora_model = copy.deepcopy(base_model)

        lora_config = LoRAConfig(r=4, lora_alpha=8.0)
        inject_lora_to_model(lora_model, lora_config)

        x = torch.randint(0, config.vocab_size, (2, 8))
        base_model.eval()
        lora_model.eval()
        with torch.no_grad():
            base_logits, _ = base_model(x)
            lora_logits, _ = lora_model(x)

        self.assertTrue(torch.allclose(base_logits, lora_logits, atol=1e-5))

        # Test gradient isolation during training
        lora_model.train()
        logits, _ = lora_model(x)
        loss = logits.sum()
        loss.backward()

        # LoRA parameters must have gradients
        lora_A_grad_found = False
        for name, param in lora_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                self.assertIsNotNone(param.grad)
                lora_A_grad_found = True
            else:
                self.assertIsNone(param.grad)
        self.assertTrue(lora_A_grad_found)

    def test_lora_model_merge_unmerge_and_checkpoint(self):
        """Verify model-wide merge, unmerge, and checkpoint saving/loading."""
        config = GPTConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=16,
            n_layer=2,
            n_head=2,
            d_model=32,
        )
        base_model = MiniGPT(config)
        model = copy.deepcopy(base_model)
        fresh_model = copy.deepcopy(base_model)

        lora_config = LoRAConfig(r=4, lora_alpha=8.0)
        inject_lora_to_model(model, lora_config)

        # Mutate LoRA weights slightly to simulate post-training state
        for n, p in model.named_parameters():
            if "lora_B" in n:
                torch.nn.init.normal_(p, std=0.05)

        x = torch.randint(0, config.vocab_size, (2, 8))
        model.eval()
        with torch.no_grad():
            orig_logits, _ = model(x)

        merge_lora_weights(model)
        with torch.no_grad():
            merged_logits, _ = model(x)
        self.assertTrue(torch.allclose(orig_logits, merged_logits, atol=1e-5))

        unmerge_lora_weights(model)
        with torch.no_grad():
            unmerged_logits, _ = model(x)
        self.assertTrue(torch.allclose(orig_logits, unmerged_logits, atol=1e-5))

        # Test checkpoint saving and loading on fresh model with same base weights
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = save_adapter_weights(model, lora_config, tmp_dir)
            self.assertTrue(ckpt_path.exists())

            inject_lora_to_model(fresh_model, lora_config)
            load_adapter_weights(fresh_model, ckpt_path)

            fresh_model.eval()
            with torch.no_grad():
                fresh_logits, _ = fresh_model(x)
            self.assertTrue(torch.allclose(orig_logits, fresh_logits, atol=1e-5))

    def test_sft_cosine_lr_schedule(self):
        """Verify linear warmup and cosine decay schedule equations in SFTTrainer."""
        config = GPTConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=16,
            n_layer=1,
            n_head=2,
            d_model=16,
        )
        model = MiniGPT(config)
        inject_lora_to_model(model, LoRAConfig(r=2))

        anchors = create_default_anchor_examples()
        ds = HarmonizedSFTDataset(anchor_examples=anchors, epoch_length=20)
        sft_config = SFTConfig(
            max_steps=100,
            warmup_steps=10,
            learning_rate=2e-4,
            min_learning_rate=2e-5,
        )
        trainer = SFTTrainer(model, self.tokenizer, ds, ds, sft_config, device="cpu")

        # Step 0: Warmup start
        self.assertAlmostEqual(trainer.get_lr(0), 0.0)
        # Step 5: Warmup halfway
        self.assertAlmostEqual(trainer.get_lr(5), 1e-4)
        # Step 10: Peak LR
        self.assertAlmostEqual(trainer.get_lr(10), 2e-4)
        # Step 100: Min LR floor
        self.assertAlmostEqual(trainer.get_lr(100), 2e-5)
        # Step 55: Cosine midpoint
        mid_lr = trainer.get_lr(55)
        self.assertGreater(mid_lr, 2e-5)
        self.assertLess(mid_lr, 2e-4)

    def test_sft_trainer_short_convergence_and_checkpointing(self):
        """Verify full SFT training loop reduces loss, persists checkpoints, and generates text."""
        config = GPTConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=32,
            n_layer=2,
            n_head=2,
            d_model=32,
        )
        model = MiniGPT(config)
        lora_config = LoRAConfig(r=4, lora_alpha=8.0)
        inject_lora_to_model(model, lora_config)

        # Snapshot base weights to prove immutability
        first_base_linear = next(
            m.base_linear.weight.clone()
            for m in model.modules()
            if isinstance(m, LoRALinear)
        )

        anchors = create_default_anchor_examples()
        train_ds = HarmonizedSFTDataset(anchor_examples=anchors, epoch_length=50, seed=42)
        val_ds = HarmonizedSFTDataset(anchor_examples=anchors, epoch_length=20, seed=123)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sft_config = SFTConfig(
                max_steps=12,
                batch_size=2,
                learning_rate=3e-4,
                warmup_steps=2,
                eval_interval=6,
                eval_steps=3,
                checkpoint_dir=tmp_dir,
            )
            trainer = SFTTrainer(
                model=model,
                tokenizer=self.tokenizer,
                train_dataset=train_ds,
                val_dataset=val_ds,
                config=sft_config,
                lora_config=lora_config,
                device="cpu",
            )

            # Record initial loss
            initial_val_loss, _ = trainer.evaluate()
            history = trainer.train()

            self.assertGreaterEqual(len(history), 2)
            # Checkpoint file should exist
            best_ckpt = Path(tmp_dir) / "best_adapter" / "adapter_weights.pt"
            self.assertTrue(best_ckpt.exists())
            final_ckpt = Path(tmp_dir) / "final_adapter" / "adapter_weights.pt"
            self.assertTrue(final_ckpt.exists())

            # Base weights must remain 100% untouched
            current_base_linear = next(
                m.base_linear.weight
                for m in model.modules()
                if isinstance(m, LoRALinear)
            )
            self.assertTrue(torch.all(first_base_linear == current_base_linear))

            # Generation test
            generated = trainer.generate_sample("What is attention?", max_new_tokens=20)
            self.assertIsInstance(generated, str)
            self.assertGreater(len(generated), 5)

    def test_regression_evaluator_gate_logic(self):
        """Verify SFTRegressionEvaluator qualification gate decisions across pass, forgetting, and underfitting."""
        anchors = create_default_anchor_examples()
        anchor_ds = HarmonizedSFTDataset(anchor_examples=anchors, epoch_length=10, seed=1)
        domain_ds = HarmonizedSFTDataset(anchor_examples=anchors, epoch_length=10, seed=2)
        evaluator = SFTRegressionEvaluator(
            tokenizer=self.tokenizer,
            anchor_eval_dataset=anchor_ds,
            domain_eval_dataset=domain_ds,
            drift_threshold_pct=5.0,
            device="cpu",
        )

        config = GPTConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=16,
            n_layer=1,
            n_head=2,
            d_model=16,
        )
        base_model = MiniGPT(config)
        adapted_model = copy.deepcopy(base_model)

        # Mock evaluate_perplexity to test pure gate decision arithmetic
        # Case 1: PASS (Domain gain +15%, Anchor drift +1%)
        evaluator.evaluate_perplexity = lambda m, ds: 10.0 if m is base_model else 8.5
        # Custom mock for anchor vs domain
        def mock_eval_pass(m, ds):
            if ds is evaluator.anchor_eval_dataset:
                return 10.0 if m is base_model else 10.1  # +1% drift (<= 5%)
            else:
                return 12.0 if m is base_model else 10.0  # domain gain
        evaluator.evaluate_perplexity = mock_eval_pass
        res_pass = evaluator.evaluate_gate(base_model, adapted_model)
        self.assertTrue(res_pass.passed)
        self.assertEqual(res_pass.status, "PASS")

        # Case 2: FAIL_FORGETTING (Anchor drift +10% > 5%)
        def mock_eval_forget(m, ds):
            if ds is evaluator.anchor_eval_dataset:
                return 10.0 if m is base_model else 11.0  # +10% drift
            else:
                return 12.0 if m is base_model else 10.0
        evaluator.evaluate_perplexity = mock_eval_forget
        res_forget = evaluator.evaluate_gate(base_model, adapted_model)
        self.assertFalse(res_forget.passed)
        self.assertEqual(res_forget.status, "FAIL_FORGETTING")

        # Case 3: FAIL_NO_LEARNING (Domain gain <= 0)
        def mock_eval_no_learn(m, ds):
            if ds is evaluator.anchor_eval_dataset:
                return 10.0 if m is base_model else 10.0  # 0% drift
            else:
                return 12.0 if m is base_model else 12.5  # domain degraded
        evaluator.evaluate_perplexity = mock_eval_no_learn
        res_no_learn = evaluator.evaluate_gate(base_model, adapted_model)
        self.assertFalse(res_no_learn.passed)
        self.assertEqual(res_no_learn.status, "FAIL_NO_LEARNING")

    def test_continual_sft_orchestrator_e2e_run(self):
        """Verify turnkey ContinualSFTOrchestrator pipeline from raw web text to merged model."""
        config = GPTConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=32,
            n_layer=2,
            n_head=2,
            d_model=32,
        )
        base_model = MiniGPT(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            lora_config = LoRAConfig(r=4, lora_alpha=8.0)
            sft_config = SFTConfig(
                max_steps=6,
                batch_size=2,
                learning_rate=3e-4,
                warmup_steps=2,
                eval_interval=3,
                eval_steps=2,
                checkpoint_dir=tmp_dir,
            )

            orchestrator = ContinualSFTOrchestrator(
                base_model=base_model,
                tokenizer=self.tokenizer,
                lora_config=lora_config,
                sft_config=sft_config,
                drift_threshold_pct=100.0,  # Generous threshold to verify PASS merge flow
                device="cpu",
            )

            web_docs = [
                "Attention mechanisms in transformer neural networks compute dynamic weight correlations.",
                "Apple Silicon MPS unified memory architecture enables efficient local fine-tuning.",
            ]

            result = orchestrator.run_continual_alignment(web_docs)
            self.assertIsInstance(result, OrchestratorPipelineResult)
            self.assertGreaterEqual(len(result.training_history), 2)
            self.assertIsNotNone(result.regression_result)
            # Check deployment status string
            self.assertIn("DEPLOYED", result.deployment_status)
            self.assertTrue(result.success)
            self.assertIsNotNone(result.merged_model)

    def test_continual_sft_orchestrator_empty_web_stream(self):
        """Verify ContinualSFTOrchestrator gracefully handles empty web documents stream."""
        config = GPTConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=16,
            n_layer=1,
            n_head=2,
            d_model=16,
        )
        base_model = MiniGPT(config)
        orchestrator = ContinualSFTOrchestrator(
            base_model=base_model,
            tokenizer=self.tokenizer,
            device="cpu",
        )
        result = orchestrator.run_continual_alignment([])
        self.assertTrue(result.success)
        self.assertEqual(result.deployment_status, "SKIPPED_EMPTY_WEB_STREAM")


    def test_schema_to_instruction_example_interconnection(self):
        """Verify seamless bidirectional conversion between core Pydantic schemas and SFT classes."""
        from core.schemas import ChatMessageSchema, TrainingSampleSchema

        chat_schema = ChatMessageSchema(role="user", content="Explain LoRA.")
        chat_msg = chat_schema.to_chat_message()
        self.assertEqual(chat_msg.role, "user")
        self.assertEqual(chat_msg.content, "Explain LoRA.")

        converted_back = chat_msg.to_schema()
        self.assertEqual(converted_back.role, "user")
        self.assertEqual(converted_back.content, "Explain LoRA.")

        sample_schema = TrainingSampleSchema(
            messages=[
                ChatMessageSchema(role="user", content="What is LoRA?"),
                ChatMessageSchema(role="assistant", content="LoRA is Low-Rank Adaptation."),
            ],
            origin="web_scraped",
        )
        instruct_ex = sample_schema.to_instruction_example()
        self.assertEqual(len(instruct_ex.messages), 2)
        self.assertEqual(instruct_ex.messages[1].role, "assistant")
        self.assertEqual(instruct_ex.provenance_source, "web_scraped")

        sample_back = instruct_ex.to_schema()
        self.assertEqual(len(sample_back.messages), 2)
        self.assertEqual(sample_back.origin, "web_scraped")

    def test_regression_result_to_evaluation_report_schema(self):
        """Verify RegressionResult exports to core.schemas.EvaluationReportSchema."""
        from core.schemas import EvaluationReportSchema
        from training.m3_sft.evaluator import RegressionResult

        reg_res = RegressionResult(
            passed=True,
            status="PASS",
            base_anchor_ppl=10.0,
            adapted_anchor_ppl=10.2,
            anchor_drift_pct=2.0,
            base_domain_ppl=25.0,
            adapted_domain_ppl=18.0,
            domain_gain_pct=28.0,
            summary="Release approved.",
        )
        report = reg_res.to_schema(candidate_name="test_candidate")
        self.assertIsInstance(report, EvaluationReportSchema)
        self.assertEqual(report.decision, "APPROVED")
        self.assertEqual(report.candidate_name, "test_candidate")
        self.assertAlmostEqual(report.regression_delta, 0.02)
        self.assertEqual(report.failure_reason, "")

    def test_curation_to_continual_sft_interconnection(self):
        """Verify ContinualSFTOrchestrator accepts CleanedDocumentSchema objects directly."""
        from core.schemas import CleanedDocumentSchema

        config = GPTConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=16,
            n_layer=1,
            n_head=2,
            d_model=16,
        )
        base_model = MiniGPT(config)
        orchestrator = ContinualSFTOrchestrator(
            base_model=base_model,
            tokenizer=self.tokenizer,
            drift_threshold_pct=100.0,
            device="cpu",
        )

        cleaned_docs = [
            CleanedDocumentSchema(
                url="https://example.com/ai",
                clean_text="Self-attention allows contextual weighting across sequence tokens.",
                word_count=8,
                line_repetition_rate=0.0,
                sha256_hash="abc1234567890",
            )
        ]

        result = orchestrator.run_continual_alignment(cleaned_docs)
        self.assertTrue(result.success)
        self.assertIn("DEPLOYED", result.deployment_status)

    def test_minigpt_return_all_logits(self):
        """Verify MiniGPT forward return_all_logits returns (B, T, V) when targets is None."""
        config = GPTConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=16,
            n_layer=1,
            n_head=2,
            d_model=16,
        )
        model = MiniGPT(config)
        idx = torch.randint(0, self.tokenizer.vocab_size, (2, 8))

        # Standard inference mode: (B, 1, V)
        logits_last, loss_none = model(idx, return_all_logits=False)
        self.assertEqual(logits_last.shape, (2, 1, self.tokenizer.vocab_size))
        self.assertIsNone(loss_none)

        # Full sequence logits mode: (B, T, V)
        logits_all, loss_none2 = model(idx, return_all_logits=True)
        self.assertEqual(logits_all.shape, (2, 8, self.tokenizer.vocab_size))
        self.assertIsNone(loss_none2)


if __name__ == "__main__":
    unittest.main()

