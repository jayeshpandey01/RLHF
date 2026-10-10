"""Unit and Integration Tests for Milestone 5, Phase 5.1 (Reward Modeling & Preferences)."""

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from core.schemas import PreferencePairSchema
from training.m1_bigram.tokenizer import CharTokenizer
from training.m1_transformer.model import GPTConfig, MiniGPT
from training.m5_preference.dataset import (
    PreferenceCollator,
    PreferenceDataset,
    create_default_preference_pairs,
)
from training.m5_preference.reward_model import BradleyTerryLoss, RewardModel
from training.m5_preference.reward_trainer import RewardTrainer
from training.m5_preference.schemas import PreferenceBatch, PreferenceSample


class TestM5RewardModel(unittest.TestCase):
    """Test suite for Phase 5.1 preference data schemas, reward modeling, and Bradley-Terry loss."""

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
        self.backbone = MiniGPT(self.config)
        self.reward_model = RewardModel(self.backbone)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_schema_roundtrip(self):
        """Verify bidirectional conversion between PreferencePairSchema and PreferenceSample."""
        sample = PreferenceSample(
            prompt="What is LoRA?",
            chosen="LoRA freezes base weights and trains rank decomposition matrices A and B.",
            rejected="LoRA retrains all parameters from scratch with AdamW.",
            margin=1.5,
            category="peft_sft",
            metadata={"annotator": "expert_1"},
        )
        schema = sample.to_schema()
        self.assertIsInstance(schema, PreferencePairSchema)
        self.assertEqual(schema.prompt, sample.prompt)
        self.assertEqual(schema.chosen, sample.chosen)
        self.assertEqual(schema.rejected, sample.rejected)
        self.assertEqual(schema.margin, 1.5)

        reconstructed = PreferenceSample.from_schema(schema)
        self.assertEqual(reconstructed.prompt, sample.prompt)
        self.assertEqual(reconstructed.margin, 1.5)

        d = sample.to_dict()
        reconstructed_dict = PreferenceSample.from_dict(d)
        self.assertEqual(reconstructed_dict.chosen, sample.chosen)

    def test_preference_dataset_and_json_io(self):
        """Verify PreferenceDataset loading, default curation, and JSON export."""
        pairs = create_default_preference_pairs()
        self.assertGreaterEqual(len(pairs), 5)
        dataset = PreferenceDataset(pairs)
        self.assertEqual(len(dataset), len(pairs))
        self.assertEqual(dataset[0].prompt, pairs[0].prompt)

        json_path = Path(self.temp_dir) / "preference_pairs.json"
        dataset.to_json(json_path)
        self.assertTrue(json_path.exists())

        loaded_dataset = PreferenceDataset.from_json(json_path)
        self.assertEqual(len(loaded_dataset), len(dataset))
        self.assertEqual(loaded_dataset[0].chosen, dataset[0].chosen)

    def test_collator_tokenization_and_padding(self):
        """Verify PreferenceCollator generates paired batch tensors with accurate masks."""
        collator = PreferenceCollator(self.tokenizer, max_seq_len=64)
        pairs = create_default_preference_pairs()[:2]
        batch = collator(pairs)

        self.assertIsInstance(batch, PreferenceBatch)
        self.assertEqual(batch.chosen_input_ids.shape, (2, 64))
        self.assertEqual(batch.chosen_attention_mask.shape, (2, 64))
        self.assertEqual(batch.rejected_input_ids.shape, (2, 64))
        self.assertEqual(batch.rejected_attention_mask.shape, (2, 64))
        self.assertEqual(batch.margins.shape, (2,))

        # Verify mask is 1 for active tokens and 0 for trailing padding
        mask_row = batch.chosen_attention_mask[0].tolist()
        num_active = sum(mask_row)
        self.assertEqual(mask_row[:num_active], [1] * num_active)
        self.assertEqual(mask_row[num_active:], [0] * (64 - num_active))

    def test_bradley_terry_loss_numerics(self):
        """Verify Bradley-Terry loss computation, margin effect, and accuracy metric."""
        loss_fn = BradleyTerryLoss(margin=0.0, l2_reg=0.0)

        # 1. Perfectly classified pairs: chosen > rejected
        r_w = torch.tensor([3.0, 2.5, 1.8])
        r_l = torch.tensor([0.5, 0.2, -0.5])
        loss_val, metrics = loss_fn(r_w, r_l)
        self.assertLess(loss_val.item(), 0.3)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertGreater(metrics["mean_margin"], 2.0)

        # 2. Completely misclassified pairs: chosen < rejected
        loss_val_bad, metrics_bad = loss_fn(r_l, r_w)
        self.assertGreater(loss_val_bad.item(), 2.0)
        self.assertEqual(metrics_bad["accuracy"], 0.0)
        self.assertLess(metrics_bad["mean_margin"], -2.0)

        # 3. Margin penalty verification
        loss_fn_margin = BradleyTerryLoss(margin=2.0, l2_reg=0.0)
        loss_with_margin, _ = loss_fn_margin(r_w, r_l)
        # Even though chosen > rejected, requiring margin 2.0 increases loss
        self.assertGreater(loss_with_margin.item(), loss_val.item())

    def test_reward_model_padding_invariance(self):
        """Verify scalar reward is invariant to padding tokens appended to sequence."""
        self.reward_model.eval()

        # Sequence 1: 15 active tokens, 10 pad tokens (total length 25)
        # Sequence 2: exact same 15 active tokens, 25 pad tokens (total length 40)
        tokens = [5, 12, 18, 22, 9, 14, 20, 11, 7, 16, 25, 30, 8, 19, 21]
        pad_id = 0

        seq1 = torch.tensor([tokens + [pad_id] * 10], dtype=torch.long)
        mask1 = torch.tensor([[1] * 15 + [0] * 10], dtype=torch.long)

        seq2 = torch.tensor([tokens + [pad_id] * 25], dtype=torch.long)
        mask2 = torch.tensor([[1] * 15 + [0] * 25], dtype=torch.long)

        with torch.no_grad():
            r1 = self.reward_model(seq1, attention_mask=mask1)
            r2 = self.reward_model(seq2, attention_mask=mask2)

        self.assertEqual(r1.shape, (1,))
        self.assertEqual(r2.shape, (1,))
        self.assertAlmostEqual(r1.item(), r2.item(), places=4)

    def test_reward_model_gradient_flow(self):
        """Verify backprop from Bradley-Terry loss flows into backbone and reward head."""
        self.reward_model.train()
        loss_fn = BradleyTerryLoss()

        seq_w = torch.randint(0, len(self.tokenizer), (2, 20))
        mask_w = torch.ones((2, 20), dtype=torch.long)
        seq_l = torch.randint(0, len(self.tokenizer), (2, 20))
        mask_l = torch.ones((2, 20), dtype=torch.long)

        r_w = self.reward_model(seq_w, mask_w)
        r_l = self.reward_model(seq_l, mask_l)
        loss, _ = loss_fn(r_w, r_l)

        loss.backward()

        self.assertIsNotNone(self.reward_model.reward_head.weight.grad)
        self.assertGreater(self.reward_model.reward_head.weight.grad.norm().item(), 0.0)
        # Check first layer transformer weight grad
        first_layer_weight = self.reward_model.backbone.transformer.h[0].attn.qkv_proj.weight
        self.assertIsNotNone(first_layer_weight.grad)
        self.assertGreater(first_layer_weight.grad.norm().item(), 0.0)

    def test_reward_trainer_convergence_on_synthetic_pairs(self):
        """Verify RewardTrainer optimizes preference accuracy and separates rewards."""
        device = torch.device("cpu")
        trainer = RewardTrainer(
            model=self.reward_model,
            loss_fn=BradleyTerryLoss(margin=0.0, l2_reg=0.001),
            optimizer=torch.optim.AdamW(self.reward_model.parameters(), lr=5e-3),
            device=device,
        )

        collator = PreferenceCollator(self.tokenizer, max_seq_len=256)
        pairs = create_default_preference_pairs()
        batch = collator(pairs)

        with torch.no_grad():
            r_w0 = self.reward_model(batch.chosen_input_ids, batch.chosen_attention_mask)
            r_l0 = self.reward_model(batch.rejected_input_ids, batch.rejected_attention_mask)
            initial_metrics = trainer.loss_fn(r_w0, r_l0, batch.margins)[1]

        # Optimize for 20 steps
        last_metrics = initial_metrics
        for _ in range(20):
            last_metrics = trainer.train_step(batch)

        # Verifications
        self.assertLess(last_metrics["loss"], initial_metrics["loss"])
        self.assertGreaterEqual(last_metrics["accuracy"], 0.80)
        self.assertGreater(last_metrics["mean_margin"], initial_metrics["mean_margin"])

        # Checkpoint persistence check
        ckpt_path = Path(self.temp_dir) / "reward_model.pt"
        trainer.save_checkpoint(ckpt_path)
        self.assertTrue(ckpt_path.exists())

        # Load into fresh reward model
        fresh_model = RewardModel(MiniGPT(self.config))
        fresh_trainer = RewardTrainer(model=fresh_model, device=device)
        fresh_trainer.load_checkpoint(ckpt_path)

        with torch.no_grad():
            orig_r = self.reward_model(batch.chosen_input_ids[:1], batch.chosen_attention_mask[:1])
            loaded_r = fresh_model(batch.chosen_input_ids[:1], batch.chosen_attention_mask[:1])
            self.assertAlmostEqual(orig_r.item(), loaded_r.item(), places=4)


if __name__ == "__main__":
    unittest.main()
