# GAP-01: PPO Critic Off-By-One & Action Information Leakage

## 1. Landmark Academic Literature & Foundations

1. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017)**. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
2. **Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015)**. *High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)*. ICLR 2016 / arXiv:1506.02438.
3. **Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., et al. (2019)**. *Fine-Tuning Language Models from Human Preferences*. arXiv:1909.08593.
4. **Ouyang, L., Wu, J., Jiang, X., et al. (2022)**. *Training language models to follow instructions with human feedback* (InstructGPT). NeurIPS 2022 / arXiv:2203.02155.
5. **Hugging Face TRL Library**. `AutoModelForCausalLMWithValueHead` & `PPOTrainer` token alignment convention.

---

## 2. Mathematical Formulation & The Causal State Invariant

In an autoregressive Markov Decision Process (MDP) for language generation:
- Given prompt $x = (w_1, \dots, w_M)$ and response tokens $y = (w_{M+1}, \dots, w_{M+T})$.
- At discrete generation step $t \in [1, T]$:
  - **State**: The prefix sequence $s_t = (x, y_{<t}) = (w_1, \dots, w_{M+t-1})$.
  - **Action**: The next token $a_t = y_t = w_{M+t} \in \mathcal{V}$, sampled from policy $\pi_\theta(a_t \mid s_t)$.
  - **State-Value Function**:
    $$V^\pi(s_t) = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{T-t} \gamma^k r_{t+k} \;\middle|\; s_t = (x, y_{<t}) \right]$$

### The Policy Gradient Baseline Subtraction Theorem
The standard policy gradient estimator with baseline subtraction states:
$$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{s_t, a_t \sim \pi} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left( Q^\pi(s_t, a_t) - V(s_t) \right) \right]$$

For $V(s_t)$ to be a valid, unbiased variance-reduction baseline:
$$\mathbb{E}_{a_t \sim \pi} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) V(s_t) \right] = V(s_t) \sum_{a_t \in \mathcal{V}} \nabla_\theta \pi_\theta(a_t \mid s_t) = V(s_t) \nabla_\theta (1) = 0$$

**CRITICAL REQUIREMENT**: $V(s_t)$ must be **conditionally independent of action $a_t$ given state $s_t$**. If $V$ depends on $a_t$, this expectation is **non-zero**, introducing severe gradient bias.

---

## 3. The Code Bug: Action Information Leakage

In [`training/m5_preference/critic.py`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/critic.py#L26-L31):
```python
class CriticModel(nn.Module):
    """Autoregressive Critic network predicting per-token state values V(s_t).

    Wraps a MiniGPT backbone architecture with a token-level ValueHead.
    For each position t in sequence x ∘ y, produces scalar value V(s_t) = V(x, y_{<t}).
    """
```
And in `forward()` ([`critic.py#L86-L88`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/critic.py#L86-L88)):
```python
    hidden = self.extract_hidden_states(input_ids)  # Shape: (B, T, d_model)
    values = self.value_head(hidden)                # Shape: (B, T)
```

### The Causal Transformer Anatomy:
In a causal decoder transformer:
- Input token at sequence index $i$ is $w_i$.
- Hidden state $h_i$ at index $i$ is computed after causal self-attention across tokens $w_0, w_1, \dots, w_i$.
- Therefore, $h_i$ incorporates the token embedding and interactions of token $w_i$!
- When the actor generates token $w_i$ (action $a$), it uses hidden state $h_{i-1}$ to predict logits: $\pi(w_i \mid w_{<i}) = \text{softmax}(W_{\text{lm}} h_{i-1})$.
- But in `critic.py`, `values[:, i]` is projected from $h_i$!
- **$h_i$ has ALREADY observed token $w_i$!**

Therefore:
$$\text{values}[:, i] = f_{\text{critic}}(w_0, \dots, w_i) = V(s_i, a_i) \quad \text{NOT} \quad V(s_i)$$

The critic at step $i$ is evaluating the state *after* the action has been chosen, effectively predicting $Q(s_i, a_i)$ or $V(s_{i+1})$, rather than $V(s_i)$.

---

## 4. Empirical Proof of Action Leakage

We constructed two sequences with the exact same prefix $s_3 = [10, 20, 30]$, but differing actions at position $3$:
- Sequence A: `[10, 20, 30, 40]`
- Sequence B: `[10, 20, 30, 95]`

Running `critic(seq)`:
```
Position 2 (state s_3 before action 3):
  val_a[0, 2] = 0.083387
  val_b[0, 2] = 0.083387
  Delta = 0.000000  (Strictly causal and action-invariant)

Position 3 (action token position):
  val_a[0, 3] = 0.126614  (with action 40)
  val_b[0, 3] = 0.051538  (with action 95)
  Delta = 0.075076  (ACTION LEAKAGE CONFIRMED!)
```
Because `val[0, 3]` changes when the action at index 3 changes, `values[:, 3]` is leaking the action!

---

## 5. Corruption of GAE Temporal Recursion

In [`training/m5_preference/gae.py`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/gae.py#L45-L58):
```python
delta = rewards[:, t] + gamma * next_values * next_masks - values[:, t]
last_gae = delta + gamma * lam * next_masks * last_gae
advantages[:, t] = last_gae
```
When `values[:, t]` is actually $V(s_t, a_t)$:
$$\delta_t = r_t + \gamma V(s_{t+1}, a_{t+1}) - V(s_t, a_t)$$
Instead of the true temporal difference error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, the TD residual computes the difference between action-values, distorting the advantage estimates and rendering the PPO surrogate updates mathematically biased.

---

## 6. The Clean Fix

To ensure `values[:, t]` represents $V(s_t) = V(x, y_{<t})$, the value at position $t$ must be derived from $h_{t-1}$.

In `critic.py`:
```python
def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    input_ids = input_ids.to(self.device)
    hidden = self.extract_hidden_states(input_ids)  # Shape: (B, T, d_model)
    raw_values = self.value_head(hidden)            # Shape: (B, T)

    # Shift values right by 1 position:
    # Value of state s_t (before token t) is predicted by hidden state at t - 1.
    # At t=0, state before prompt token 0 is unconditioned; initialize with 0.
    B, T = raw_values.size()
    shifted_values = torch.zeros_like(raw_values)
    shifted_values[:, 1:] = raw_values[:, :-1]

    if attention_mask is not None:
        shifted_values = shifted_values * attention_mask.to(self.device).float()

    return shifted_values
```

This ensures that at completion position $t$, the value estimate is derived exclusively from tokens $0 \dots t-1$, restoring exact action-invariance and unbiased GAE advantages.

