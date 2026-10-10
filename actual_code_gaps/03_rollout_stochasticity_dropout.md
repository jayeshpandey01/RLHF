# GAP-03: Stochastic Dropout in Rollout Evaluation & Ratio Distortion

## 1. Landmark Academic Literature & Foundations

1. **Schulman, J., et al. (2017)**. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
2. **Shao, Z., Wang, P., Zhu, Q., et al. (2024)**. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300.
3. **Engstrom, L., Ilyas, A., Santurkar, S., et al. (2020)**. *Implementation Matters in Deep RL: A Case Study on PPO and TRPO*. ICLR 2020.
4. **Hugging Face TRL Library**. Policy rollout evaluation standards and evaluation mode enforcement.

---

## 2. Theoretical Requirement of Importance Sampling in Policy Gradients

In both PPO and GRPO, the policy gradient is estimated via importance sampling:
$$\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

The clipped surrogate objective relies on $\rho_t$ starting at **exactly 1.0**:
$$\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( \rho_t(\theta) \hat{A}_t, \; \text{clip}(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$

### The Step-Zero Parity Invariant
At step 0 of an update iteration, before any parameter update has occurred, the active policy parameters $\theta$ are identical to the rollout policy parameters $\theta_{\text{old}}$:
$$\theta \equiv \theta_{\text{old}} \implies \pi_\theta(a_t \mid s_t) \equiv \pi_{\theta_{\text{old}}}(a_t \mid s_t) \implies \rho_t = 1.0, \quad \log \rho_t = 0.0$$

Clipping is only intended to be triggered when **actual gradient descent updates** push the parameter vector $\theta$ away from $\theta_{\text{old}}$ by more than $\epsilon$ (typically $0.2$).

---

## 3. The Code Bug: Active Dropout During Rollout Scoring

In [`training/m5_preference/ppo_trainer.py`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/ppo_trainer.py#L132-L163) (`PPOTrainer.build_rollout_batch`):
```python
    @torch.no_grad()
    def build_rollout_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> PPORolloutBatch:
        ...
        # 2. Extract policy log-probs and critic state values
        old_log_probs, _ = compute_token_log_probs_and_entropy(self.actor, input_ids)
        old_values = self.critic(input_ids, attention_mask)
```
And similarly in [`training/m5_preference/grpo_trainer.py`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/grpo_trainer.py#L90-L123) (`GRPOTrainer.build_group_batch`):
```python
    @torch.no_grad()
    def build_group_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
        rewards: torch.Tensor,
    ) -> GRPOGroupBatch:
        ...
        old_log_probs, _ = compute_token_log_probs_and_entropy(self.actor, input_ids)
```

### The Problem:
1. `MiniGPT` uses `nn.Dropout` in:
   - Token and positional embedding additions (`self.transformer.drop`)
   - Feed-forward projection blocks (`MLP.dropout`)
   - Attention head projections (`CausalMultiHeadAttention.attn_dropout`, `resid_dropout`)
2. In PyTorch, **`@torch.no_grad()` does NOT disable dropout**; it only halts computational graph building. Dropout remains fully active when the module is in `.train()` mode.
3. Neither `build_rollout_batch` nor `build_group_batch` ever calls `self.actor.eval()`.
4. Therefore, when computing `old_log_probs`, random dropout masks are active.
5. In `train_step`, `self.actor.train()` is called, and `new_log_probs` are computed with a **different random dropout mask**.

### Consequence:
$$\log \pi_\theta(a_t \mid s_t; \text{mask}_2) \ne \log \pi_{\theta_{\text{old}}}(a_t \mid s_t; \text{mask}_1)$$
The importance sampling ratio $\rho_t$ is corrupt at step 0, deviating significantly from 1.0 even before any parameter updates take place!

---

## 4. Empirical Demonstration

Running [`actual_code_gaps/reproductions/test_dropout_rollout_ratio_gap.py`](file:///Users/jayesh/Documents/RLHF/actual_code_gaps/reproductions/test_dropout_rollout_ratio_gap.py):
```
[Dropout Gap Verification] Step 0 ratio with active dropout (train mode):
  Min ratio: 0.7863  (deviancy: -21.4%)
  Max ratio: 1.1173  (deviancy: +11.7%)
  Mean ratio: 0.9967

[Dropout Gap Verification] Step 0 ratio in eval mode:
  Min ratio: 1.0000
  Max ratio: 1.0000
```
With `dropout = 0.1` (the default in `GPTConfig`), the importance sampling ratio at step 0 swings between $0.78$ and $1.12$. This means:
- The PPO clipping boundaries ($[1-\epsilon, 1+\epsilon] = [0.8, 1.2]$) are immediately reached at step 0 simply from dropout noise!
- Gradients for legitimate actions are zeroed or distorted.
- Policy updates are contaminated with random Bernoulli sampling variance.

---

## 5. The Clean Fix

`build_rollout_batch` and `build_group_batch` must explicitly switch the actor (and critic) to `eval()` mode, compute deterministic log-probabilities, and restore `train()` mode.

In `ppo_trainer.py`:
```python
    @torch.no_grad()
    def build_rollout_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> PPORolloutBatch:
        ...
        self.actor.eval()
        self.critic.eval()
        try:
            old_log_probs, _ = compute_token_log_probs_and_entropy(self.actor, input_ids)
            old_values = self.critic(input_ids, attention_mask)
        finally:
            self.actor.train()
            self.critic.train()
```
And in `grpo_trainer.py`:
```python
    @torch.no_grad()
    def build_group_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
        rewards: torch.Tensor,
    ) -> GRPOGroupBatch:
        ...
        self.actor.eval()
        try:
            old_log_probs, _ = compute_token_log_probs_and_entropy(self.actor, input_ids)
        finally:
            self.actor.train()
```
This guarantees that $\rho_t \equiv 1.0$ at step 0, ensuring policy gradient updates reflect genuine policy optimization rather than dropout noise.

