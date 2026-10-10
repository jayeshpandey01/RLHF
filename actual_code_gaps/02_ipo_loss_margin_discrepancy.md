# GAP-02: IPO Loss Margin Discrepancy & Scaling Distortion

## 1. Landmark Academic Literature & Foundations

1. **Azar, M. G., Rowland, M., Piot, B., Guo, D., et al. (Google DeepMind, 2024)**. *A General Theoretical Paradigm to Understand Learning from Human Preferences*. AISTATS 2024 / arXiv:2310.12036.
2. **Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023)**. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023 / arXiv:2305.18290.
3. **Hugging Face TRL Library**. `DPOTrainer` (`loss_type="ipo"`) reference implementation.

---

## 2. Theoretical Motivation of Identity Preference Optimization (IPO)

Standard DPO optimizes the Bradley-Terry logistic loss:
$$\mathcal{L}_{\text{DPO}}(\theta) = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

Because $\sigma(z) \to 1$ only as $z \to \infty$, DPO constantly pushes the implicit reward gap $\Delta = \beta h$ toward $+\infty$. In datasets with clear or noisy preference pairs, this causes over-optimization, probability mass displacement, and KL drift.

To solve this, **Azar et al. (Google DeepMind, 2024)** derived Identity Preference Optimization (IPO). IPO regularizes the log-likelihood ratio directly via a quadratic penalty around an exact target margin, preventing infinite margin drift.

### The True IPO Objective
Let $h_\theta(x, y_w, y_l)$ be the unscaled log-ratio difference:
$$h_\theta(x, y_w, y_l) = \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$$

In Azar et al. (2024) (Equation 17) and in Hugging Face TRL:
$$\mathcal{L}_{\text{IPO}}(\theta) = \mathbb{E}_{(x, y_w, y_l)} \left[ \left( h_\theta(x, y_w, y_l) - \frac{1}{2\tau} \right)^2 \right]$$
Where $\tau = \beta$ is the regularization coefficient.

TRL implements this directly in `DPOTrainer`:
```python
# TRL implementation:
log_ratios = (policy_chosen_logps - ref_chosen_logps) - (policy_rejected_logps - ref_rejected_logps)
losses = (log_ratios - 1 / (2 * self.beta)) ** 2
```
**Intended Target Margin**:
$$h^* = \frac{1}{2\beta}$$
For $\beta = 0.1$, the optimal log-ratio difference is $h^* = \frac{1}{2(0.1)} = 5.0$.

---

## 3. The Code Implementation Discrepancy

In [`training/m5_preference/dpo_loss.py`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/dpo_loss.py#L79-L88):
```python
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen)
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected)

        delta = chosen_rewards - rejected_rewards  # Shape: (B,)

        # Compute loss according to loss_type
        if self.loss_type == "ipo":
            # IPO: (delta - 1 / (2 * beta))^2
            tau = 1.0 / (2.0 * self.beta)
            loss = torch.mean((delta - tau) ** 2)
```

Notice what `delta` is:
$$\text{delta} = \beta \cdot ( \log \pi(y_w) - \log \pi_{\text{ref}}(y_w) ) - \beta \cdot ( \log \pi(y_l) - \log \pi_{\text{ref}}(y_l) ) = \beta \cdot h_\theta$$

Then the code computes:
$$\mathcal{L}_{\text{code}} = \left( \text{delta} - \frac{1}{2\beta} \right)^2 = \left( \beta \cdot h_\theta - \frac{1}{2\beta} \right)^2$$

### Algebraic Expansion:
Factoring out $\beta$:
$$\left( \beta \cdot h_\theta - \frac{1}{2\beta} \right)^2 = \beta^2 \left( h_\theta - \frac{1}{2\beta^2} \right)^2$$

This algebraic bug introduces **two major distortions**:

1. **Target Margin Scaled by $\frac{1}{\beta}$**:
   The minimum of the loss occurs when:
   $$\beta \cdot h_\theta = \frac{1}{2\beta} \implies h^*_{\text{code}} = \frac{1}{2\beta^2}$$
   Instead of the intended target margin $\frac{1}{2\beta}$, the code forces the policy to achieve a target margin of $\frac{1}{2\beta^2}$!
   - For $\beta = 0.1$:
     - Expected target: $h^* = \frac{1}{2(0.1)} = 5.0$
     - Actual code target: $h^*_{\text{code}} = \frac{1}{2(0.01)} = \mathbf{50.0}$!
   - A log-ratio margin of $50.0$ translates to an exponential preference probability ratio of:
     $$\frac{\pi(y_w \mid x)}{\pi(y_l \mid x)} \sim e^{50} \approx 5.18 \times 10^{21}$$
   This drives the model into extreme saturation, completely defeating the purpose of IPO!

2. **Gradient Magnitude Attenuation by $\beta^2$**:
   The derivative with respect to $h$ is:
   $$\frac{\partial \mathcal{L}_{\text{code}}}{\partial h} = 2 \beta^2 \left( h - \frac{1}{2\beta^2} \right)$$
   For $\beta = 0.1$, $\beta^2 = 0.01$. The effective gradient step is suppressed by a factor of 100!

---

## 4. Empirical Demonstration

Running [`actual_code_gaps/reproductions/test_ipo_loss_discrepancy_gap.py`](file:///Users/jayesh/Documents/RLHF/actual_code_gaps/reproductions/test_ipo_loss_discrepancy_gap.py):
```
[IPO Gap Verification] beta=0.1
  Unscaled log-ratio difference h: 2.00
  Target margin in Azar et al. / TRL: 5.00
  Target margin in dpo_loss.py:       50.00 (10x distortion!)
  Expected paper loss:                9.0000
  Actual dpo_loss.py loss:            23.0400

Loss at h=5.0 (true Azar target):     20.2500  (non-zero!)
Loss at h=50.0 (erroneous code target): 0.0000 (minimum reached at 50!)
```

---

## 5. The Clean Fix

In [`training/m5_preference/dpo_loss.py`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/dpo_loss.py#L85-L88):
```python
        # Compute unscaled log-ratio difference h
        if self.reference_free or reference_chosen_logps is None or reference_rejected_logps is None:
            log_ratios = policy_chosen_logps - policy_rejected_logps
        else:
            ref_chosen = reference_chosen_logps.to(device)
            ref_rejected = reference_rejected_logps.to(device)
            log_ratios = (policy_chosen_logps - ref_chosen) - (policy_rejected_logps - ref_rejected)

        delta = self.beta * log_ratios

        if self.loss_type == "ipo":
            # IPO (Azar et al., 2024): (h - 1 / (2 * beta))^2
            tau = 1.0 / (2.0 * self.beta)
            loss = torch.mean((log_ratios - tau) ** 2)
```
This aligns the code with Azar et al. (2024) and Hugging Face TRL, restoring the correct $h^* = \frac{1}{2\beta}$ target margin.

