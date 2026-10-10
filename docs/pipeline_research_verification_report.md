# Autonomous Continual Learning LLM Post-Training Pipeline
## Comprehensive Research Review, Paper Analysis & Pipeline Verification Report

See full detailed analysis in `docs/research.md` and the master verification suite in `tests/run_all.py` and `experiments/test_e2e_pipeline.py`.

### 1. Verification Summary
- **220 / 220 master verification tests passing (100% green)** in `tests/run_all.py`.
- The master run includes `TestE2EPipeline`; all included end-to-end tests passed.

### 2. Literature Mapping Highlights
- **Autograd**: Linnainmaa (1970), Baur-Strassen Theorem (1983, $\text{Cost}(\nabla f) \le 5 \times \text{Cost}(f)$).
- **Attention & KV-Cache**: Vaswani et al. (2017) variance scaling $1/\sqrt{d_k}$, Pre-LN gradient highways (Xiong et al., 2020), Pope et al. (2022) KV-cache $\mathcal{O}(1)$ step decoding ($15.49\times$ speedup).
- **Data Curation & Collapse**: MinHash LSH (Broder, 1997; Indyk & Motwani, 1998; Lee et al., 2022), Model Collapse Guard (Shumailov et al., Nature 2024), 70/30 Anchor Experience Replay (Robins, 1995; Askell et al., 2021).
- **PEFT / LoRA**: Hu et al. (2021), completion-only masking (TRL `DataCollatorForCompletionOnlyLM` parity).
- **Preference Learning**: Bradley-Terry (1952), DPO (Rafailov et al., 2023), IPO (Azar et al., Google DeepMind 2024).
- **RL with Verifiable Rewards**: PPO & GAE (Schulman et al., 2015, 2017), GRPO & DeepSeek reverse KL (Shao et al., DeepSeekMath 2024; DeepSeek-R1 2025).
- **Evaluation & Gating**: MT-Bench LLM Judge debiasing (Zheng et al., 2023), ALCE citation precision/recall (Gao et al., 2023), Paired Bootstrap (B=1000, 95% CI), Wilcoxon & McNemar tests.
- **Hybrid Serving**: BM25 (Robertson & Zaragoza, 2009), RRF (Cormack et al., 2009), Semantic Cache (Bang et al., 2023), Reader-Writer zero-downtime hot-swap (Courtois et al., 1971).
