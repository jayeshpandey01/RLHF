"""Unit and Integration Tests for Milestone 5, Phase 5.2 (Direct Preference Optimization)."""

import math
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from training.m1_bigram.tokenizer import CharTokenizer
from training.m1_transformer.model import GPTConfig, MiniGPT
from training.m5_preference.dataset import (
    PreferenceCollator,
    PreferenceDataset,
    create_default_preference_pairs,
)
from training.m5_preference.dpo_loss import DPOLoss
from training.m5_preference.dpo_trainer import DPOTrainer
from training.m5_preference.log_probs import (
    extract_batch_log_probs,
    extract_completion_log_probs,
)
from training.m5_preference.reference_model import ReferenceModelEngine
from training.m5_preference.schemas import PreferenceBatch


class TestM5DPO(unittest.TestCase):
    """Test suite for Phase 5.2 Direct Preference Optimization (DPO)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        corpus = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
            "<|>_\n:;.,!?()[]{}+-*/=@#$%^&~`'\""
        )
        self.tokenizer = CharTokenizer(corpus)

        self.config = GPTConfig(
            vocab_size=len(self.tokenizer),
            block_size=256,
            d_model=32,
            n_head=2,
            n_layer=2,
            dropout=0.0,
        )
        self.model = MiniGPT(self.config)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_log_prob_extraction_exactness(self):
        """Verify extract_completion_log_probs matches manual PyTorch calculation."""
        input_ids = torch.tensor([[5, 10, 15, 20]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        extracted_log_probs = extract_completion_log_probs(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Manual computation
        with torch.no_grad():
            logits, _ = self.model(input_ids, return_all_logits=True)
            shift_logits = logits[:, :-1, :]
            shift_targets = input_ids[:, 1:]
            manual_log_probs = F.log_softmax(shift_logits, dim=-1)
            manual_sum = torch.gather(
                manual_log_probs, dim=2, index=shift_targets.unsqueeze(-1)
            ).squeeze(-1).sum(dim=-1)

        self.assertAlmostEqual(extracted_log_probs.item(), manual_sum.item(), places=5)

    def test_prompt_masking_isolation(self):
        """Verify prompt tokens masked with -100 are completely ignored in log-probs."""
        input_ids = torch.tensor([[1, 2, 3, 10, 20, 30]], dtype=torch.long)
        labels = input_ids.clone()
        labels[:, :3] = -100  # Mask the first 3 tokens as prompt

        log_probs = extract_completion_log_probs(
            model=self.model,
            input_ids=input_ids,
            labels=labels,
        )

        # Manually compute only completion tokens at positions index 3, 4, 5
        # Tokens shifted: logits at 2 predicts 3, logits at 3 predicts 4, logits at 4 predicts 5
        with torch.no_grad():
            logits, _ = self.model(input_ids, return_all_logits=True)
            shift_logits = logits[:, 2:5, :]  # Predicts positions 3, 4, 5
            shift_targets = input_ids[:, 3:6]
            expected_sum = torch.gather(
                F.log_softmax(shift_logits, dim=-1),
                dim=2,
                index=shift_targets.unsqueeze(-1),
            ).squeeze(-1).sum(dim=-1)

        self.assertAlmostEqual(log_probs.item(), expected_sum.item(), places=5)

    def test_dpo_loss_analytical_values(self):
        """Verify DPO loss matches closed-form analytical values across known margins."""
        loss_fn = DPOLoss(beta=0.1, loss_type="sigmoid")

        # 1. Delta = 0 => loss = softplus(0) = ln(2) approx 0.693147
        policy_logp = torch.tensor([-5.0])
        ref_logp = torch.tensor([-5.0])
        loss_zero, metrics = loss_fn(
            policy_chosen_logps=policy_logp,
            policy_rejected_logps=policy_logp,
            reference_chosen_logps=ref_logp,
            reference_rejected_logps=ref_logp,
        )
        self.assertAlmostEqual(loss_zero.item(), math.log(2.0), places=5)
        self.assertAlmostEqual(metrics["mean_margin"], 0.0, places=5)

        # 2. Large positive margin: delta = +10.0 => loss approx e^-10 approx 4.54e-5
        chosen_logp = torch.tensor([50.0])
        rej_logp = torch.tensor([-50.0])
        loss_pos, _ = loss_fn(
            policy_chosen_logps=chosen_logp,
            policy_rejected_logps=rej_logp,
            reference_chosen_logps=torch.tensor([0.0]),
            reference_rejected_logps=torch.tensor([0.0]),
        )
        self.assertLess(loss_pos.item(), 1e-4)

    def test_dpo_loss_extreme_numerics(self):
        """Verify DPO loss does not NaN or Inf on extreme input margins."""
        loss_fn = DPOLoss(beta=0.1)

        huge_delta_w = torch.tensor([500.0, -500.0])
        huge_delta_l = torch.tensor([-500.0, 500.0])

        loss, metrics = loss_fn(
            policy_chosen_logps=huge_delta_w,
            policy_rejected_logps=huge_delta_l,
        )

        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_cdpo_label_smoothing(self):
        """Verify conservative DPO (cDPO) incorporates label smoothing penalty."""
        loss_clean = DPOLoss(beta=0.1, loss_type="sigmoid", label_smoothing=0.0)
        loss_smooth = DPOLoss(beta=0.1, loss_type="cdpo", label_smoothing=0.1)

        # For positive delta (clean loss is very small), label smoothing adds penalty
        pi_w = torch.tensor([10.0])
        pi_l = torch.tensor([-10.0])
        ref = torch.tensor([0.0])

        l_clean, _ = loss_clean(pi_w, pi_l, ref, ref)
        l_smooth, _ = loss_smooth(pi_w, pi_l, ref, ref)

        self.assertGreater(l_smooth.item(), l_clean.item())

    def test_ipo_loss_quadratic(self):
        """Verify Identity Preference Optimization (IPO) computes quadratic penalty on unscaled log-ratios."""
        beta = 0.2
        loss_ipo = DPOLoss(beta=beta, loss_type="ipo")

        pi_w = torch.tensor([2.0])
        pi_l = torch.tensor([1.0])
        ref = torch.tensor([0.0])

        # Azar et al. (2024) / TRL formulation:
        # h = (2.0 - 0.0) - (1.0 - 0.0) = 1.0
        # tau = 1 / (2 * 0.2) = 2.5
        # expected loss = (1.0 - 2.5)^2 = (-1.5)^2 = 2.25
        loss, metrics = loss_ipo(pi_w, pi_l, ref, ref)
        self.assertAlmostEqual(loss.item(), (1.0 - 2.5) ** 2, places=4)

    def test_reference_caching_parity(self):
        """Verify cached reference log-probabilities produce identical outputs to live engine."""
        ref_model = MiniGPT(self.config)
        ref_engine = ReferenceModelEngine(ref_model)

        collator = PreferenceCollator(self.tokenizer, max_seq_len=256)
        pairs = create_default_preference_pairs()[:2]
        batch = collator(pairs)

        # 1. Compute via live engine
        ref_w_live, ref_l_live = ref_engine.compute_batch_log_probs(batch)

        # 2. Compute via populated cache
        cached_batch = ref_engine.populate_batch_cache(batch)
        self.assertIsNotNone(cached_batch.ref_chosen_logps)
        self.assertIsNotNone(cached_batch.ref_rejected_logps)

        torch.testing.assert_close(ref_w_live, cached_batch.ref_chosen_logps)
        torch.testing.assert_close(ref_l_live, cached_batch.ref_rejected_logps)

    def test_dpo_trainer_convergence_on_synthetic_pairs(self):
        """Verify DPOTrainer converges, increases implicit margin, and achieves high accuracy."""
        device = torch.device("cpu")
        # Reference model is a frozen clone of the initial policy
        ref_model = MiniGPT(self.config)
        ref_model.load_state_dict(self.model.state_dict())
        ref_engine = ReferenceModelEngine(ref_model, device=device)

        trainer = DPOTrainer(
            model=self.model,
            ref_engine=ref_engine,
            loss_fn=DPOLoss(beta=0.1, loss_type="sigmoid"),
            optimizer=torch.optim.AdamW(self.model.parameters(), lr=1e-3),
            device=device,
        )

        collator = PreferenceCollator(self.tokenizer, max_seq_len=256)
        pairs = create_default_preference_pairs()
        batch = collator(pairs)

        # Evaluate initial state
        initial_metrics = trainer.evaluate(
            torch.utils.data.DataLoader([batch], batch_size=1, collate_fn=lambda x: x[0])
        )

        # Optimize for 20 steps
        last_metrics = initial_metrics
        for _ in range(20):
            last_metrics = trainer.train_step(batch)

        # Verify convergence
        self.assertLess(last_metrics["loss"], initial_metrics["loss"])
        self.assertGreater(last_metrics["mean_margin"], initial_metrics["mean_margin"])
        self.assertGreaterEqual(last_metrics["accuracy"], 0.80)

        # Verify checkpoint persistence
        ckpt_path = Path(self.temp_dir) / "dpo_policy.pt"
        trainer.save_checkpoint(ckpt_path)
        self.assertTrue(ckpt_path.exists())

        fresh_policy = MiniGPT(self.config)
        fresh_trainer = DPOTrainer(model=fresh_policy, device=device)
        fresh_trainer.load_checkpoint(ckpt_path)

        with torch.no_grad():
            orig_p = extract_batch_log_probs(self.model, batch.chosen_input_ids[:1], batch.chosen_attention_mask[:1], batch.prompt_lengths[:1])
            loaded_p = extract_batch_log_probs(fresh_policy, batch.chosen_input_ids[:1], batch.chosen_attention_mask[:1], batch.prompt_lengths[:1])
            self.assertAlmostEqual(orig_p.item(), loaded_p.item(), places=4)


if __name__ == "__main__":
    unittest.main()
