# GAP-04: Absence of Online Rollout Sampling in PPO & GRPO

## 1. Landmark Academic Literature & Foundations

1. **Shao, Z., Wang, P., Zhu, Q., et al. (2024)**. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300.
2. **DeepSeek-AI (2025)**. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948.
3. **Schulman, J., et al. (2017)**. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
4. **Ouyang, L., et al. (2022)**. *Training language models to follow instructions with human feedback* (InstructGPT). NeurIPS 2022.

---

## 2. Theoretical Architecture of Online RL for LLMs

The core premise of reinforcement learning for language models—both classical PPO and modern Group Relative Policy Optimization (GRPO)—is that the policy learns from **self-generated completions sampled from its own evolving distribution**:

```
[Prompt Dataset] ──► Query q
                         │
                         ▼
             [Active Policy π_θ]
                         │
       Sampling G Completions: {o_1, o_2, ..., o_G} ~ π_θ(· | q)
                         │
                         ▼
              [Reward Function R(q, o_i)]
       (Verifiable Math/Rule or Neural Reward Model)
                         │
                         ▼
        [Advantage Estimation A_i]
          (GAE for PPO / Group Standardization for GRPO)
                         │
                         ▼
      [Policy Gradient Surrogate Update θ]
```

### In DeepSeekMath & DeepSeek-R1 (GRPO):
For each question $q$, GRPO samples a group of $G$ outputs:
$$\{o_1, o_2, \dots, o_G\} \sim \pi_{\theta_{\text{old}}}(o \mid q)$$
The reward function scores each output $r_i = \mathcal{R}(q, o_i)$, and advantages are normalized over the group:
$$A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\})}$$

Online generation is essential because:
1. It enables the model to explore new reasoning paths that do not exist in the training corpus.
2. It generates on-policy data: as the policy improves, it generates higher quality trajectories.

---

## 3. The Code Gap: Trainers Take Pre-Made Offline Sequences

In [`training/m5_preference/grpo_trainer.py`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/grpo_trainer.py#L91-L108):
```python
    @torch.no_grad()
    def build_group_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
        rewards: torch.Tensor,
    ) -> GRPOGroupBatch:
```
And in [`training/m5_preference/ppo_trainer.py`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/ppo_trainer.py#L133-L148):
```python
    @torch.no_grad()
    def build_rollout_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> PPORolloutBatch:
```

### The Discrepancy:
- Neither `PPOTrainer` nor `GRPOTrainer` implements a rollout generation loop.
- They expect the caller to pass:
  - `input_ids`: Full concatenated prompt + completion tokens already generated elsewhere.
  - `attention_mask`: Pre-padded masks.
  - `rewards`: Pre-computed scalar rewards.
- In `tests/test_m5_ppo.py` and `tests/test_m5_grpo.py`:
  - Rollouts are constructed by manually synthesizing random token tensors or static text pairs!
- Despite `docs/research.md` (lines 3200–3205) depicting an online rollout generator and calling it "Online Group Relative Policy Optimization (GRPO)":
  ```mermaid
  RolloutGen --> MultiRew --> GroupAdv --> ClippedLoss --> GRPOTrainer
  ```
  In reality, **no sampling loop, no temperature/top-p decoding, and no dynamic KV-cache generation exists in the trainers**.

---

## 4. Impact on the Pipeline

Because neither PPO nor GRPO has an online generation loop:
1. PPO and GRPO function as offline importance-weighted loss calculators over fixed static batches rather than true reinforcement learning agents.
2. Without an online rollout loop, GRPO cannot autonomously explore reasoning chains or discover novel solutions as demonstrated in DeepSeek-R1.
3. This creates a disconnect between the claims in the documentation and the reality of the code.

---

## 5. Architectural Remediation

To make GRPO truly online, `GRPOTrainer` should incorporate a `sample_group_rollouts(prompts)` method:
```python
@torch.no_grad()
def sample_group_rollouts(
    self,
    prompts: list[str],
    tokenizer,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    self.actor.eval()
    all_input_ids = []
    all_completions = []
    all_prompt_lens = []

    for prompt in prompts:
        p_ids = tokenizer.encode(prompt)
        p_len = len(p_ids)
        # Sample G independent completions per prompt
        for _ in range(self.group_size):
            # Autoregressive generation
            idx = torch.tensor([p_ids], dtype=torch.long, device=self.device)
            gen = self.actor.generate(
                idx, max_new_tokens=max_new_tokens, temperature=temperature
            )
            completion_text = tokenizer.decode(gen[0].tolist()[p_len:])
            all_input_ids.append(gen[0])
            all_completions.append(completion_text)
            all_prompt_lens.append(p_len)

    # Pad and collate into batch of shape (N * G, T)
    ...
```
This closes the gap between the theoretical design and the implementation.

