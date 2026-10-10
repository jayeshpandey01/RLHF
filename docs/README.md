# Research & Mathematical Foundations Library

This directory contains the theoretical research, mathematical proofs, derivations, and algorithmic architectures for the autonomous continual learning LLM system.

---

## 📖 Master Research Document

All research and mathematics are consolidated into a single, comprehensive, step-by-step master file:

👉 **[`docs/research.md`](research.md)**

### Document Structure & Contents

1. **Step 0: System Architecture & Continual Learning Dataflow** — Proposed FastAPI, hybrid retrieval, Ollama serving, quality filtering, experience replay, and model promotion flow; see the implementation-status section for current gaps.
2. **Step 1: Milestone 0 — Hardware Strategy & Data Contracts** — Auto-detection of Apple Silicon MPS (`bfloat16`), CUDA, and CPU thread strategies; Pydantic V2 data contracts.
3. **Step 2: Milestone 1 / Phase 1.1 — Reverse-Mode Autograd Engine & Gradient Verification** — Historical papers (Linnainmaa 1970, Baur-Strassen 1983, Rumelhart 1986), DAG topological DFS sorting, adjoint formulas, and central finite-difference Taylor series error bounds ($\mathcal{O}(h^2)$).
4. **Step 3: Milestone 1 / Phase 1.2 — Language Modeling Foundations, Bigram, & Numerical Loss** — Character tokenization, Maximum Likelihood Estimation, empirical perplexity, Log-Sum-Exp numerical stability trick, analytical logit gradients, and temperature sampling.
5. **Step 4: Milestone 1 / Phase 1.3 — Causal Multi-Head Self-Attention & Algorithmic KV-Cache** — Scaled dot-product attention, proof of variance preservation ($\text{Var}(QK^T/\sqrt{d_k}) = 1$), causal mask tensor algebra, multi-head projections, and KV-cache computational complexity proof ($\mathcal{O}(T^2) \to \mathcal{O}(T)$).
6. **Step 5: Milestone 1 / Phase 1.4 — Mini-GPT Pre-LayerNorm Generative Transformer** — Pre-LN vs Post-LN gradient preservation theorem (Xiong et al. 2020), GPT-2 residual variance scaling ($1/\sqrt{2L}$), Gaussian Error Linear Unit (GELU) expectation derivation, weight tying ($W_U \equiv W_E$), and AdamW decoupled weight decay optimization.
7. **Step 6: Milestone 2 — Web Ingestion, MinHash LSH, & Model Collapse Risks** — Research on recursive training, project-specific replay mixture, MinHash theorem ($\mathbb{P}[h(A)=h(B)] = J(A,B)$), and LSH bands/rows S-curve derivation.
8. **Step 7: Milestone 3 — Post-Training Alignment (SFT, DPO, & GRPO)** — SFT completion loss masking, LoRA low-rank decomposition ($W = W_0 + \frac{\alpha}{r}BA$), Bradley-Terry preference model, analytical derivation of DPO implicit reward, and DeepSeek GRPO group-relative advantage estimation without a critic model.
9. **Step 8: Milestone 4 — Evaluation, Citation Verification, & Autonomous Release Gate** — Benchmark design, citation checks, LLM judges, and statistical release criteria.
10. **Step 9: Milestone 5 — Preference Learning (Reward Model, DPO, PPO, GRPO)** — Native trainer and loss implementations with a shared alignment interface.
11. **Step 10: Milestone 6 — Retrieval, Replay, Orchestration, & Serving** — Hybrid retrieval, event processing, replay, and FastAPI components; throughput and live model integration are not yet benchmarked/configured.
12. **Research Verification and Implementation Status** — Primary papers, engineering articles, claim caveats, test evidence, and current end-to-end limitations.
