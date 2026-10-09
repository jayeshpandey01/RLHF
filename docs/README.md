# Research & Mathematical Foundations Library

This directory contains the theoretical research, mathematical proofs, derivations, and algorithmic architectures for the autonomous continual learning LLM system.

---

## 📖 Master Research Document

All research and mathematics are consolidated into a single, comprehensive, step-by-step master file:

👉 **[`docs/research.md`](research.md)**

### Document Structure & Contents

1. **Step 0: System Architecture & Continual Learning Dataflow** — FastAPI, OpenSearch BM25 + dense hybrid retrieval, Ollama serving, quality filtering, experience replay, and zero-downtime hot-swap protocol.
2. **Step 1: Milestone 0 — Hardware Strategy & Data Contracts** — Auto-detection of Apple Silicon MPS (`bfloat16`), CUDA, and CPU thread strategies; Pydantic V2 data contracts.
3. **Step 2: Milestone 1 / Phase 1.1 — Reverse-Mode Autograd Engine & Gradient Verification** — Historical papers (Linnainmaa 1970, Baur-Strassen 1983, Rumelhart 1986), DAG topological DFS sorting, adjoint formulas, and central finite-difference Taylor series error bounds ($\mathcal{O}(h^2)$).
4. **Step 3: Milestone 1 / Phase 1.2 — Language Modeling Foundations, Bigram, & Numerical Loss** — Character tokenization, Maximum Likelihood Estimation, empirical perplexity, Log-Sum-Exp numerical stability trick, analytical logit gradients, and temperature sampling.
5. **Step 4: Milestone 1 / Phase 1.3 — Causal Multi-Head Self-Attention & Algorithmic KV-Cache** — Scaled dot-product attention, proof of variance preservation ($\text{Var}(QK^T/\sqrt{d_k}) = 1$), causal mask tensor algebra, multi-head projections, and KV-cache computational complexity proof ($\mathcal{O}(T^2) \to \mathcal{O}(T)$).
6. **Step 5: Milestone 1 / Phase 1.4 — Mini-GPT Pre-LayerNorm Generative Transformer** — Pre-LN vs Post-LN gradient preservation theorem (Xiong et al. 2020), GPT-2 residual variance scaling ($1/\sqrt{2L}$), Gaussian Error Linear Unit (GELU) expectation derivation, weight tying ($W_U \equiv W_E$), and AdamW decoupled weight decay optimization.
7. **Step 6: Milestone 2 — Web Ingestion, MinHash LSH, & Model Collapse Prevention** — Shumailov et al. (*Nature* 2024) proof of variance decay under recursive self-training, experience replay 70/30 convex combination, MinHash theorem ($\mathbb{P}[h(A)=h(B)] = J(A,B)$), and LSH bands/rows S-curve derivation.
8. **Step 7: Milestone 3 — Post-Training Alignment (SFT, DPO, & GRPO)** — SFT completion loss masking, LoRA low-rank decomposition ($W = W_0 + \frac{\alpha}{r}BA$), Bradley-Terry preference model, analytical derivation of DPO implicit reward, and DeepSeek GRPO group-relative advantage estimation without a critic model.
