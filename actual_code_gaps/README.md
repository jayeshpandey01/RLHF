# Comprehensive Research & Code Gap Analysis Dossier

This directory documents the comprehensive research audit, mathematical proofs, literature cross-examinations, and empirical validations of the **actual gaps, discrepancies, bugs, and architectural disconnects** discovered in the codebase compared against primary research literature and documentation.

---

## 1. Executive Summary & Gap Taxonomy Matrix

| Gap ID | Category | Severity | Primary Source / Landmark Reference | Affected Components | Impact Summary |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GAP-01** | Actor-Critic / RL | **CRITICAL** | Schulman et al. (2015, 2017); Ziegler et al. (2019); TRL | [`training/m5_preference/critic.py`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/critic.py#L86-L88), [`gae.py`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/gae.py#L45-L58) | **Critic Off-By-One / Action Leakage**: Value at position $t$ is computed from hidden state after observing action $y_t$. The critic evaluates $V(x, y_{\le t})$ instead of $V(x, y_{<t})$, biasing advantages and breaking policy gradient baseline subtraction. |
| **GAP-02** | Preference Loss | **HIGH** | Azar et al. (Google DeepMind, 2024); TRL `DPOTrainer` | [`training/m5_preference/dpo_loss.py`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/dpo_loss.py#L79-L88) | **IPO Target Margin Distortion**: Code computes $(\beta \Delta - \frac{1}{2\beta})^2$, driving unscaled log-ratio margin $h \to \frac{1}{2\beta^2}$ instead of $\frac{1}{2\beta}$. For $\beta=0.1$, target margin is distorted $10\times$ (from $5.0$ to $50.0$), destroying regularization. |
| **GAP-03** | RL Trajectory Scoring | **HIGH** | Schulman et al. (2017); Shao et al. (2024); TRL | [`training/m5_preference/ppo_trainer.py`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/ppo_trainer.py#L161-L163), [`grpo_trainer.py`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/grpo_trainer.py#L122-L124) | **Stochastic Dropout Active in Rollouts**: `actor.eval()` is never called during rollout log-prob computation. Dropout injects random noise ($[-21\%, +16\%]$ variance) into importance sampling ratio $\rho_t$ at step 0, triggering premature clipping. |
| **GAP-04** | Reinforcement Learning | **HIGH** | DeepSeekMath (Shao et al., 2024); DeepSeek-R1 (2025); InstructGPT | [`training/m5_preference/ppo_trainer.py`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/ppo_trainer.py#L133-L148), [`grpo_trainer.py`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/grpo_trainer.py#L91-L108) | **Absence of Online Rollout Generation**: PPO and GRPO trainers only accept static offline sequences. Neither trainer implements an autoregressive token sampling loop (`model.generate()` / group rollouts), despite documentation claiming online RL. |
| **GAP-05** | Error Handling & Safety | **MEDIUM** | Rafailov et al. (2023); Schulman et al. (2017) | [`training/m5_preference/ppo_trainer.py`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/ppo_trainer.py#L165-L175), [`dpo_loss.py`](file:///Users/jayesh/Documents/RLHF/training/m5_preference/dpo_loss.py#L73-L76) | **Silent Fallbacks & Error Masking**: In PPO, missing `ref_model` silently zeroes the KL penalty. In DPO, missing reference log-probs silently switches to reference-free mode without raising an error or warning. |
| **GAP-06** | System Architecture | **CRITICAL** | Production Continual Post-Training Architecture | [`training/m3_sft/hf_worker.py`](file:///Users/jayesh/Documents/RLHF/training/m3_sft/hf_worker.py#L58-L90), [`apps/server.py`](file:///Users/jayesh/Documents/RLHF/apps/server.py#L91-L95) | **RLHF Disconnected from Production**: DPO, PPO, GRPO, and RewardModel only run on toy MiniGPT. The live deployed pipeline (`hf_worker.py` and `server.py`) only executes basic LoRA SFT on Qwen-0.5B. Preference alignment never reaches serving. |
| **GAP-07** | Continual Learning Loop | **HIGH** | Robins (1995); Askell et al. (2021); ALCE (2023) | [`training/m3_sft/hf_worker.py`](file:///Users/jayesh/Documents/RLHF/training/m3_sft/hf_worker.py#L48-L52), [`hf_worker.py`](file:///Users/jayesh/Documents/RLHF/training/m3_sft/hf_worker.py#L344-L346) | **SFT Target is a Trivial Copy Task**: Target completion is literally `{excerpt}\n\nSource: {url}`. The benchmark rewards 4× keyword coverage + citation presence, causing the model to learn copy-paste rather than knowledge acquisition. Baseline (quantized Ollama) and Candidate (FP32 HF with 120-token cap) are not comparable. |
| **GAP-08** | Statistical Gating | **HIGH** | Efron (1979); Edwards (1948); Wilcoxon (1945) | [`evaluation/statistical_gating.py`](file:///Users/jayesh/Documents/RLHF/evaluation/statistical_gating.py#L416-L485) | **Gating Loopholes & Ignored Checks**: Edge-case delta check is appended to `violated_checks` but ignored in decision; McNemar test is never used; No-op/noisy candidate is promoted (`mean_delta >= 0`); Empty buckets default to pass; Candidate failures falsely trigger rollback on active serving model. |
| **GAP-09** | Retrieval & Security | **MEDIUM** | Karp & Rabin (1987); Robertson (2009) | [`retrieval/dense.py`](file:///Users/jayesh/Documents/RLHF/retrieval/dense.py#L12-L47), [`apps/server.py`](file:///Users/jayesh/Documents/RLHF/apps/server.py#L91-L95) | **Pseudo-Dense Hash Retrieval & Security**: "Dense" retrieval uses deterministic MD5 128-dim character n-gram hashing, not semantic neural embeddings. Ingestion and promotion endpoints lack authentication, allowing data poisoning via automatic 5-document triggers. |

---

## 2. Directory Contents

- [**01_actor_critic_ppo_critic_leakage.md**](file:///Users/jayesh/Documents/RLHF/actual_code_gaps/01_actor_critic_ppo_critic_leakage.md): Mathematical derivation, action leakage proof, and the temporal shift fix for `CriticModel`.
- [**02_ipo_loss_margin_discrepancy.md**](file:///Users/jayesh/Documents/RLHF/actual_code_gaps/02_ipo_loss_margin_discrepancy.md): Analysis of the $\frac{1}{2\beta^2}$ margin scaling error in `DPOLoss` vs Azar et al. (2024) and TRL.
- [**03_rollout_stochasticity_dropout.md**](file:///Users/jayesh/Documents/RLHF/actual_code_gaps/03_rollout_stochasticity_dropout.md): Proof of step-0 importance sampling ratio variance from unconstrained dropout.
- [**04_absence_of_online_generation.md**](file:///Users/jayesh/Documents/RLHF/actual_code_gaps/04_absence_of_online_generation.md): Detailed comparison of online rollout loops (DeepSeekMath, DeepSeek-R1) vs offline batch shortcuts in `PPOTrainer` and `GRPOTrainer`.
- [**05_silent_fallbacks_error_masking.md**](file:///Users/jayesh/Documents/RLHF/actual_code_gaps/05_silent_fallbacks_error_masking.md): Audit of silent fallback failure modes in preference training.
- [**06_rlhf_pipeline_disconnect.md**](file:///Users/jayesh/Documents/RLHF/actual_code_gaps/06_rlhf_pipeline_disconnect.md): System architecture audit documenting why Milestone 5 preference learning is completely detached from the live Ollama serving loop.
- [**07_live_continual_learning_flaws.md**](file:///Users/jayesh/Documents/RLHF/actual_code_gaps/07_live_continual_learning_flaws.md): Analysis of the live SFT copy task, gaming of keyword metrics, micro 20-step training, and evaluation baseline mismatch.
- [**08_statistical_release_gate_loopholes.md**](file:///Users/jayesh/Documents/RLHF/actual_code_gaps/08_statistical_release_gate_loopholes.md): Exhaustive analysis of decision rules, ignored McNemar and edge cases, unvetted promotions, and false rollback triggers.
- [**09_retrieval_and_security_gaps.md**](file:///Users/jayesh/Documents/RLHF/actual_code_gaps/09_retrieval_and_security_gaps.md): Pseudo-dense MD5 embedding, unauthenticated endpoints, and data poisoning risks.

### Executable Reproductions & Test Suite
The [`reproductions/`](file:///Users/jayesh/Documents/RLHF/actual_code_gaps/reproductions/) subdirectory contains runnable unit tests proving each gap:
```bash
python3 actual_code_gaps/reproductions/run_all_reproductions.py
```
- [`test_critic_leakage_gap.py`](file:///Users/jayesh/Documents/RLHF/actual_code_gaps/reproductions/test_critic_leakage_gap.py): Proves action leakage in `CriticModel`.
- [`test_ipo_loss_discrepancy_gap.py`](file:///Users/jayesh/Documents/RLHF/actual_code_gaps/reproductions/test_ipo_loss_discrepancy_gap.py): Proves 10x margin distortion in `DPOLoss`.
- [`test_dropout_rollout_ratio_gap.py`](file:///Users/jayesh/Documents/RLHF/actual_code_gaps/reproductions/test_dropout_rollout_ratio_gap.py): Proves policy ratio variance at step 0 due to active dropout.
- [`test_gate_decision_loopholes_gap.py`](file:///Users/jayesh/Documents/RLHF/actual_code_gaps/reproductions/test_gate_decision_loopholes_gap.py): Proves edge-case bypass, no-op promotion, and ignored McNemar tests.

### Proposed Code Patches
The [`proposed_patches/`](file:///Users/jayesh/Documents/RLHF/actual_code_gaps/proposed_patches/) directory provides clean, tested implementations resolving each issue.

