# GAP-05: Silent Fallbacks & Unwarned Objective Degradation

## 1. Landmark Academic Literature & Foundations

1. **Ouyang, L., et al. (2022)**. *Training language models to follow instructions with human feedback* (InstructGPT). Section 3: "KL Reward Regularization".
2. **Stiennon, N., et al. (2020)**. *Learning to summarize from human feedback*. NeurIPS 2020.
3. **Sculley, D., et al. (Google, 2015)**. *Hidden Technical Debt in Machine Learning Systems*. NeurIPS 2015.

---

## 2. Theoretical Necessity of Reference Models

In RLHF (PPO) and DPO:
- The reference model $\pi_{\text{ref}}$ is a frozen checkpoint of the supervised fine-tuned (SFT) model.
- Its mathematical purpose is to prevent **reward hacking** and **mode collapse** by imposing an explicit or implicit KL divergence constraint:
  $$D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$$
- Without a reference model:
  - In PPO: The policy can exploit reward model quirks by generating repetitive gibberish, punctuation loops, or nonsensical high-reward strings.
  - In DPO: Without reference log-probabilities $\log \pi_{\text{ref}}(y \mid x)$, the implicit reward formulation $r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ collapses into raw log-probabilities $\beta \log \pi_\theta(y \mid x)$, which heavily biases the policy toward short/frequent tokens (verbosity/frequency collapse) unless explicitly controlled.

---

## 3. The Code Gaps: Silent Failure Modes

### 3.1 PPO: Missing Reference Model Silently Zeroes KL Penalty
In [`training/m5_preference/ppo_trainer.py`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/ppo_trainer.py#L165-L175):
```python
        # 3. Extract reference log-probs
        if self.ref_model is not None:
            ref_log_probs, _ = compute_token_log_probs_and_entropy(self.ref_model, input_ids)
        else:
            ref_log_probs = old_log_probs.detach().clone()

        # 4. Compute token-level rewards
        rewards = torch.zeros((B, T), dtype=torch.float32, device=self.device)

        # Per-token KL penalty: -β * (log π_actor - log π_ref)
        kl_penalty = -self.kl_coef * (old_log_probs - ref_log_probs)
        rewards = rewards + kl_penalty * action_mask
```

Notice what happens when `ref_model` is not supplied:
1. `ref_log_probs = old_log_probs.detach().clone()`
2. `kl_penalty = -self.kl_coef * (old_log_probs - old_log_probs) = 0.0`
3. **The KL penalty is silently eliminated across all sequences!**
4. No warning, error, or notification is emitted.
5. The user believes they are training PPO with `kl_coef=0.05` regularisation, but the policy is actually running completely unconstrained, leading directly to reward hacking.

### 3.2 DPO: Missing Reference Log-Probs Silently Triggers Reference-Free Mode
In [`training/m5_preference/dpo_loss.py`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/dpo_loss.py#L73-L80):
```python
        # Compute implicit rewards
        if self.reference_free or reference_chosen_logps is None or reference_rejected_logps is None:
            chosen_rewards = self.beta * policy_chosen_logps
            rejected_rewards = self.beta * policy_rejected_logps
        else:
            ref_chosen = reference_chosen_logps.to(device)
            ref_rejected = reference_rejected_logps.to(device)
            chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen)
            rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected)
```
And in [`training/m5_preference/dpo_trainer.py`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/dpo_trainer.py#L88-L98):
```python
        elif self.ref_engine is not None:
            ref_chosen_logps, ref_rejected_logps = self.ref_engine.compute_batch_log_probs(batch)
        else:
            ref_chosen_logps, ref_rejected_logps = None, None
```

Notice what happens:
1. If `DPOLoss` is initialized with `reference_free=False` (default standard DPO), but `reference_chosen_logps` is `None`:
2. The condition `reference_chosen_logps is None` evaluates to `True`.
3. The loss quietly switches from standard DPO to reference-free DPO!
4. If this happens due to an upstream data loading error (e.g. batch missing cached ref logps or `ref_engine` being None), the training process silently optimizes a fundamentally different mathematical loss without notifying the engineer.

---

## 4. Software Engineering & ML Safety Impact

In high-stakes post-training pipelines, **silent fallbacks hide critical configuration errors**:
- Silent bugs do not crash unit tests (hence 100% green tests in `pytest`), but they completely invalidate scientific experiments and model alignment.
- A practitioner training with `PPOTrainer` without passing `ref_model` will observe language drift or degradation, without any log message explaining that KL regularization was disabled.

---

## 5. The Clean Fix

Fail fast and explicitly require intention when disabling reference models:

### Fix for PPO (`ppo_trainer.py`):
```python
if self.ref_model is None and self.kl_coef > 0.0:
    logger.warning(
        "PPOTrainer: ref_model was not provided, but kl_coef > 0.0. "
        "KL penalty against reference policy will be 0.0, running unconstrained RL."
    )
```

### Fix for DPO (`dpo_loss.py`):
```python
if not self.reference_free and (reference_chosen_logps is None or reference_rejected_logps is None):
    raise ValueError(
        "DPOLoss was initialized with reference_free=False, but reference log-probabilities "
        "were not provided in forward(). Pass reference log-probs or explicitly set reference_free=True."
    )
```

