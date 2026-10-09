"""Golden Domain Benchmark Suite and Schema Contracts (Milestone 4 / Phase 4.1).

Implements a curated, versioned golden benchmark suite stratified across four buckets:
Core Domain, Edge Cases, Adversarial / Hallucination Bait, and Regression Replays.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional, Union
import json
import random
import re
from loguru import logger

from core.schemas import BenchmarkCaseSchema
from curation.contamination import BenchmarkContaminationGuard
from training.m3_sft.prompt import ChatMessage, ChatMLFormatter


@dataclass
class GoldenTestCase:
    """Individual benchmark test unit for domain reasoning, citations, and regression verification."""

    case_id: str
    category: Literal["ml_theory", "transformer_internals", "peft_sft", "systems_curation"]
    bucket: Literal["core_domain", "edge_case", "adversarial_bait", "regression_replay"]
    question: str
    ground_truth_context: str
    reference_answer: str
    required_citations: list[str] = field(default_factory=list)
    golden_keywords: list[str] = field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] = "medium"

    def to_schema(self) -> BenchmarkCaseSchema:
        """Convert test case into core Pydantic BenchmarkCaseSchema."""
        return BenchmarkCaseSchema(
            case_id=self.case_id,
            question=self.question,
            ground_truth_context=self.ground_truth_context,
            required_citations=list(self.required_citations),
            golden_keywords=list(self.golden_keywords),
        )

    @classmethod
    def from_schema(
        cls,
        schema: BenchmarkCaseSchema,
        category: Literal["ml_theory", "transformer_internals", "peft_sft", "systems_curation"] = "ml_theory",
        bucket: Literal["core_domain", "edge_case", "adversarial_bait", "regression_replay"] = "core_domain",
        reference_answer: str = "",
        difficulty: Literal["easy", "medium", "hard"] = "medium",
    ) -> "GoldenTestCase":
        """Reconstruct GoldenTestCase from core BenchmarkCaseSchema."""
        return cls(
            case_id=schema.case_id,
            category=category,
            bucket=bucket,
            question=schema.question,
            ground_truth_context=schema.ground_truth_context,
            reference_answer=reference_answer or schema.ground_truth_context,
            required_citations=list(schema.required_citations),
            golden_keywords=list(schema.golden_keywords),
            difficulty=difficulty,
        )

    def to_chat_messages(
        self, system_prompt: str = "You are an expert AI assistant answering questions with citations."
    ) -> list[ChatMessage]:
        """Convert test case into ChatMessage turns including context grounding."""
        user_content = (
            f"Context:\n{self.ground_truth_context}\n\n"
            f"Question: {self.question}\n"
            f"Provide an accurate answer supported by citations from the context."
        )
        return [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_content),
            ChatMessage(role="assistant", content=self.reference_answer),
        ]

    def format_prompt(
        self, system_prompt: str = "You are an expert AI assistant answering questions with citations."
    ) -> str:
        """Serialize question and context into canonical ChatML prompt string for inference."""
        messages = self.to_chat_messages(system_prompt=system_prompt)
        return ChatMLFormatter.format_prompt_only(messages)

    def evaluate_keyword_coverage(self, response: str) -> float:
        """Compute keyword recall: fraction of golden keywords present in response."""
        if not self.golden_keywords:
            return 1.0
        response_lower = response.lower()
        hits = sum(1 for kw in self.golden_keywords if kw.lower() in response_lower)
        return hits / len(self.golden_keywords)

    def evaluate_citation_presence(self, response: str) -> float:
        """Compute required citation recall: fraction of required citation tags present in response."""
        if not self.required_citations:
            return 1.0
        hits = sum(1 for cite in self.required_citations if cite in response)
        return hits / len(self.required_citations)


def create_default_golden_suite_cases() -> list[GoldenTestCase]:
    """Curate 30 standardized golden domain benchmark cases across the four-bucket distribution."""
    cases: list[GoldenTestCase] = [
        # =====================================================================
        # 1. CORE DOMAIN (12 Cases: ML Theory, Transformer, PEFT, Systems)
        # =====================================================================
        GoldenTestCase(
            case_id="core_ml_001",
            category="ml_theory",
            bucket="core_domain",
            question="State the Baur-Strassen Theorem and its implication for backpropagation complexity.",
            ground_truth_context="Baur and Strassen (1983) proved that for any algebraic circuit computing a scalar rational function f in C elementary operations, the gradient vector grad f can be computed in at most 5C operations. This guarantees reverse-mode automatic differentiation evaluates all partial derivatives in O(C) time.",
            reference_answer="The Baur-Strassen Theorem [Baur & Strassen, 1983] states that if a scalar function f is computable in C elementary operations, its entire gradient can be evaluated in at most 5C operations. This fundamental result makes reverse-mode backpropagation computationally tractable.",
            required_citations=["Baur & Strassen, 1983"],
            golden_keywords=["baur-strassen", "5c", "reverse-mode", "computational complexity"],
            difficulty="hard",
        ),
        GoldenTestCase(
            case_id="core_ml_002",
            category="ml_theory",
            bucket="core_domain",
            question="Why does decoupled weight decay in AdamW differ from standard L2 regularization?",
            ground_truth_context="Loshchilov and Hutter (2019) demonstrated that in adaptive gradient algorithms like Adam, L2 regularization becomes scaled by the inverse diagonal Hessian estimate, causing weights with large historical gradients to undergo less regularization. AdamW decouples weight decay directly from gradient updates: theta_{t+1} = (1 - eta * lambda) * theta_t - eta * m_hat / (sqrt(v_hat) + eps).",
            reference_answer="In standard Adam with L2 regularization, the penalty is added directly to gradients, which scales it inversely with adaptive second moments [Loshchilov & Hutter, 2019]. AdamW decouples weight decay by subtracting lambda * eta * theta directly from the weight vector.",
            required_citations=["Loshchilov & Hutter, 2019"],
            golden_keywords=["adamw", "decoupled weight decay", "l2 regularization", "adaptive gradient"],
            difficulty="medium",
        ),
        GoldenTestCase(
            case_id="core_ml_003",
            category="ml_theory",
            bucket="core_domain",
            question="How does gradient norm clipping prevent exploding gradients?",
            ground_truth_context="Pascanu, Mikolov, and Bengio (2013) introduced norm clipping: if the L2 norm of gradient g exceeds threshold M, the gradient is rescaled as g <- (M / ||g||_2) * g. This preserves gradient direction while bounding step magnitude.",
            reference_answer="Pascanu et al. (2013) showed that gradient clipping rescales g to (M / ||g||_2) * g when ||g||_2 > M. This guarantees that step sizes remain bounded along steep cliff surfaces while preserving descent direction.",
            required_citations=["Pascanu et al. (2013)"],
            golden_keywords=["pascanu", "norm clipping", "rescaled", "exploding gradients"],
            difficulty="easy",
        ),
        GoldenTestCase(
            case_id="core_attn_004",
            category="transformer_internals",
            bucket="core_domain",
            question="What is the mathematical formulation of scaled dot-product attention and why is sqrt(d_k) used?",
            ground_truth_context="Vaswani et al. (2017) defined Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V. The scaling factor 1/sqrt(d_k) counteracts the growth of dot product variance under large head dimensions, preventing softmax gradients from vanishing.",
            reference_answer="Vaswani et al. (2017) formulated scaled dot-product attention as Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V. Scaling by sqrt(d_k) maintains unit variance for independent zero-mean components, preventing softmax saturation.",
            required_citations=["Vaswani et al. (2017)"],
            golden_keywords=["scaled dot-product", "softmax", "variance", "vanishing gradients"],
            difficulty="medium",
        ),
        GoldenTestCase(
            case_id="core_attn_005",
            category="transformer_internals",
            bucket="core_domain",
            question="Explain how the KV-cache reduces autoregressive generation time complexity.",
            ground_truth_context="During autoregressive generation, keys and values of previous tokens are invariant. Storing K and V in memory caches avoids recomputing self-attention over historical tokens, reducing inference step complexity from O(T^2) to O(T) per newly generated token (Shazeer, 2019).",
            reference_answer="The KV-cache stores invariant past Key and Value representations in persistent buffers [Shazeer, 2019]. Instead of recomputing full sequence attention at O(T^2), incremental decoding computes only single-token queries at O(T).",
            required_citations=["Shazeer, 2019"],
            golden_keywords=["kv-cache", "autoregressive", "o(t)", "recomputing"],
            difficulty="medium",
        ),
        GoldenTestCase(
            case_id="core_attn_006",
            category="transformer_internals",
            bucket="core_domain",
            question="Compare Pre-LayerNorm and Post-LayerNorm transformer architectures in training stability.",
            ground_truth_context="Xiong et al. (2020) demonstrated that Post-LN places LayerNorm after residual additions, causing gradient norms to vanish exponentially in earlier layers without warmup. Pre-LN applies LayerNorm on input branches before multi-head attention and MLP, maintaining identity gradient flow along the residual stream.",
            reference_answer="Xiong et al. (2020) demonstrated that Pre-LN preserves identity gradient propagation along the residual highway x_{l+1} = x_l + F(LN(x_l)), eliminating the strict warm-up dependency seen in Post-LN transformers.",
            required_citations=["Xiong et al., 2020"],
            golden_keywords=["pre-layernorm", "post-layernorm", "residual highway", "warmup"],
            difficulty="hard",
        ),
        GoldenTestCase(
            case_id="core_peft_007",
            category="peft_sft",
            bucket="core_domain",
            question="Explain the low-rank parameter decomposition of LoRA and its initialization strategy.",
            ground_truth_context="Hu et al. (2021) parameterized weight adaptation as Delta W = (alpha / r) * B A, where A in R^{r x d_in} is initialized from N(0, 1/r) and B in R^{d_out x r} is initialized to zeros. At step zero, B A = 0, exactly preserving base model identity.",
            reference_answer="Hu et al. (2021) decomposed weight updates into Delta W = (alpha / r) * B A. Initializing matrix B to strictly zero ensures Delta W = 0 at step 0, preserving bit-for-bit base model outputs before optimization begins.",
            required_citations=["Hu et al. (2021)"],
            golden_keywords=["lora", "low-rank", "zero initialization", "identity preservation"],
            difficulty="medium",
        ),
        GoldenTestCase(
            case_id="core_peft_008",
            category="peft_sft",
            bucket="core_domain",
            question="Why must prompt tokens be masked to -100 during Supervised Fine-Tuning?",
            ground_truth_context="Zhou et al. (2024, LIMA) and Taori et al. (2023) established that computing cross-entropy loss over user prompt tokens forces the model to memorize user phrasing rather than learning assistant task completion. Setting target token indices to -100 zeros cross-entropy gradients over the prompt.",
            reference_answer="Setting prompt targets to -100 zeros the backpropagated cross-entropy loss [Zhou et al., 2024]. This completion-only masking forces the model to focus gradient updates strictly on assistant responses rather than memorizing prompt syntax.",
            required_citations=["Zhou et al., 2024"],
            golden_keywords=["-100", "completion-only masking", "prompt memorization", "cross-entropy"],
            difficulty="easy",
        ),
        GoldenTestCase(
            case_id="core_peft_009",
            category="peft_sft",
            bucket="core_domain",
            question="How does LoRA weight merging achieve zero-latency inference deployment?",
            ground_truth_context="Because matrix multiplication is distributive over addition, W_merged = W_0 + (alpha / r) * B A can be pre-computed and stored directly in the base linear layer weight. During serving, inference runs as a standard dense GEMM without low-rank branch overhead (Hu et al., 2021).",
            reference_answer="Due to the distributive property of linear transformations, W_merged = W_0 + (alpha / r) * B A folds the adapter directly into base weights [Hu et al., 2021]. This eliminates branch latency during production serving.",
            required_citations=["Hu et al., 2021"],
            golden_keywords=["weight merging", "zero-latency", "distributive", "dense gemm"],
            difficulty="medium",
        ),
        GoldenTestCase(
            case_id="core_sys_010",
            category="systems_curation",
            bucket="core_domain",
            question="How does MinHash LSH achieve sub-linear duplicate detection across large document corpora?",
            ground_truth_context="Broder (1997) showed that the probability of two MinHash signatures colliding under random permutations equals their Jaccard similarity: P(h(A) = h(B)) = J(A, B). Locality-Sensitive Hashing bands these hash signatures into b bands of r rows, yielding candidate collision probability P_match = 1 - (1 - s^r)^b.",
            reference_answer="Broder (1997) proved that MinHash collision probability equals Jaccard similarity. By partitioning signatures into b bands of r rows, LSH amplifies similarity differences, detecting near-duplicates in sub-linear time O(n) instead of O(n^2).",
            required_citations=["Broder, 1997"],
            golden_keywords=["minhash", "locality-sensitive hashing", "jaccard similarity", "bands"],
            difficulty="hard",
        ),
        GoldenTestCase(
            case_id="core_sys_011",
            category="systems_curation",
            bucket="core_domain",
            question="Explain the phenomenon of model collapse in recursive generative training.",
            ground_truth_context="Shumailov et al. (Nature 2024) discovered that recursively training generative language models on synthetic outputs causes progressive statistical variance collapse. Low-probability distribution tails vanish, ultimately producing degenerate gibberish. Maintaining an anchor ratio of pristine human data (>= 70%) prevents collapse.",
            reference_answer="Shumailov et al. (Nature 2024) demonstrated that training recursively on model-generated data erodes probability distribution tails. Without an anchor replay buffer of pristine human demonstrations, models undergo irreversible model collapse.",
            required_citations=["Shumailov et al., Nature 2024"],
            golden_keywords=["model collapse", "shumailov", "variance collapse", "anchor ratio"],
            difficulty="medium",
        ),
        GoldenTestCase(
            case_id="core_sys_012",
            category="systems_curation",
            bucket="core_domain",
            question="Why is bfloat16 preferred over float16 on modern hardware with Unified Memory?",
            ground_truth_context="Brain Floating Point (bfloat16) allocates 8 bits to the exponent and 7 bits to the mantissa, identically matching the dynamic dynamic range of float32 (10^{-38} to 10^{38}). This eliminates the catastrophic underflow and overflow issues frequent in float16, removing the need for loss scaling (Kalamkar et al., 2019).",
            reference_answer="bfloat16 matches float32's 8-bit dynamic exponent range while halving memory footprint [Kalamkar et al., 2019]. This eliminates numerical underflow/overflow and removes the need for loss scaling required by standard float16.",
            required_citations=["Kalamkar et al., 2019"],
            golden_keywords=["bfloat16", "exponent bits", "dynamic range", "loss scaling"],
            difficulty="easy",
        ),

        # =====================================================================
        # 2. EDGE CASES (6 Cases: Boundary conditions, empty prompts, repetition)
        # =====================================================================
        GoldenTestCase(
            case_id="edge_empty_013",
            category="systems_curation",
            bucket="edge_case",
            question="How should a text preprocessing and tokenization pipeline handle an empty input string?",
            ground_truth_context="Standard NLP pipelines require deterministic edge handling. When presented with an empty or whitespace-only document, the extractor must emit an EMPTY_CONTENT rejection code and return an empty token sequence or fallback pad token rather than crashing.",
            reference_answer="An empty or whitespace-only document must trigger an explicit EMPTY_CONTENT rejection code, safely emitting an empty token sequence or fallback pad token ID [Pipeline Spec, 2024].",
            required_citations=["Pipeline Spec, 2024"],
            golden_keywords=["empty input", "empty_content", "fallback pad", "whitespace"],
            difficulty="easy",
        ),
        GoldenTestCase(
            case_id="edge_oov_014",
            category="transformer_internals",
            bucket="edge_case",
            question="How does character-level tokenization handle out-of-vocabulary characters?",
            ground_truth_context="In a closed-vocabulary character tokenizer, characters absent from the vocabulary table stoi are either filtered out or mapped to an explicit unknown token ID <unk>. Silently crashing on unseen characters breaks continuous data ingestion.",
            reference_answer="When encountering characters absent from stoi, the tokenizer must either bypass the unseen symbol or map it to a dedicated <unk> index, preventing pipeline crashes during web ingestion.",
            required_citations=[],
            golden_keywords=["out-of-vocabulary", "stoi", "unknown token", "character tokenizer"],
            difficulty="medium",
        ),
        GoldenTestCase(
            case_id="edge_repeat_015",
            category="systems_curation",
            bucket="edge_case",
            question="What heuristic detects degenerate repetition loops in scraped web documents?",
            ground_truth_context="Raffel et al. (2020, C4) and Penedo et al. (2023, FineWeb) filter documents where the line repetition rate exceeds 0.20 or character n-gram repetition exceeds 0.30. This eliminates boilerplate navigation menus and robotic repetition loops.",
            reference_answer="Following C4 [Raffel et al., 2020], documents with a line repetition rate exceeding 20% or 10-gram repetition exceeding 30% are classified as robotic loops and quarantined.",
            required_citations=["Raffel et al., 2020"],
            golden_keywords=["line repetition rate", "c4", "quarantined", "boilerplate"],
            difficulty="medium",
        ),
        GoldenTestCase(
            case_id="edge_len_016",
            category="transformer_internals",
            bucket="edge_case",
            question="What occurs when an input sequence length exceeds the maximum positional embedding window?",
            ground_truth_context="Transformer models with learned positional embeddings (wpe) cannot evaluate positions >= block_size without index out-of-bounds errors. Robust pipelines enforce truncation to block_size or apply context sliding windows.",
            reference_answer="Position indices pos >= block_size cause CUDA/MPS tensor indexing errors. The model input must be truncated to block_size or windowed to preserve context validity.",
            required_citations=[],
            golden_keywords=["block_size", "positional embedding", "truncation", "out-of-bounds"],
            difficulty="easy",
        ),
        GoldenTestCase(
            case_id="edge_zero_active_017",
            category="peft_sft",
            bucket="edge_case",
            question="What happens when a training batch contains zero active completion tokens (all targets = -100)?",
            ground_truth_context="In PyTorch, calling cross_entropy when all target elements match ignore_index=-100 yields a NaN scalar loss under standard normalization. The training forward pass must guard against active_tokens == 0 by returning logits.sum() * 0.0.",
            reference_answer="When all target tokens equal -100, F.cross_entropy evaluates with an empty denominator, returning NaN. The model forward pass must guard with active_tokens == 0, returning logits.sum() * 0.0 to prevent gradient corruption.",
            required_citations=[],
            golden_keywords=["-100", "nan loss", "active_tokens", "guard"],
            difficulty="hard",
        ),
        GoldenTestCase(
            case_id="edge_single_tok_018",
            category="transformer_internals",
            bucket="edge_case",
            question="How does autoregressive decoding behave when conditioned on a single prefix token (T=1)?",
            ground_truth_context="With a single conditioning token, causal self-attention computes Q_1 K_1^T, where the softmax distribution over position 1 is deterministically 1.0. The KV-cache stores this initial entry and prepares the model for step 2.",
            reference_answer="For a prompt of length T=1, self-attention produces a trivial probability of 1.0 on the single token. The KV-cache registers this single key-value vector, enabling incremental rollout for subsequent tokens.",
            required_citations=[],
            golden_keywords=["t=1", "kv-cache", "softmax", "single token"],
            difficulty="medium",
        ),

        # =====================================================================
        # 3. ADVERSARIAL / HALLUCINATION BAIT (6 Cases: Distractors, False Premises)
        # =====================================================================
        GoldenTestCase(
            case_id="adv_premise_019",
            category="peft_sft",
            bucket="adversarial_bait",
            question="Why does LoRA optimize the original base linear weights W_0 during the backward pass?",
            ground_truth_context="LoRA strictly freezes the original base weights W_0 (requires_grad = False). Gradients are computed solely with respect to low-rank adapter matrices A and B (Hu et al., 2021). The premise that W_0 is optimized is false.",
            reference_answer="The premise is false. LoRA freezes the original weights W_0 completely [Hu et al., 2021]. Gradients are backpropagated strictly into low-rank adapter matrices A and B.",
            required_citations=["Hu et al., 2021"],
            golden_keywords=["premise is false", "freezes", "requires_grad = false", "strictly"],
            difficulty="medium",
        ),
        GoldenTestCase(
            case_id="adv_distract_020",
            category="ml_theory",
            bucket="adversarial_bait",
            question="Does the Adam optimizer utilize Hessian matrix inversion to compute second-order curvature?",
            ground_truth_context="Adam is a first-order stochastic optimization algorithm (Kingma & Ba, 2014). It tracks element-wise exponentially decaying averages of squared gradients v_t, not the second-order Hessian matrix. The claim that Adam computes Hessian inverses is incorrect.",
            reference_answer="No, Adam does not compute or invert the Hessian matrix. Adam is a first-order optimizer [Kingma & Ba, 2014] that approximates curvature using element-wise squared gradient moving averages v_t.",
            required_citations=["Kingma & Ba, 2014"],
            golden_keywords=["first-order", "does not compute", "hessian", "squared gradients"],
            difficulty="medium",
        ),
        GoldenTestCase(
            case_id="adv_cite_bait_021",
            category="systems_curation",
            bucket="adversarial_bait",
            question="According to Vaswani et al. (2017), how does MinHash LSH compute document shingles?",
            ground_truth_context="Vaswani et al. (2017) authored 'Attention Is All You Need', introducing the Transformer architecture. MinHash LSH was invented by Andrei Broder (1997). Vaswani et al. did not discuss MinHash LSH or document shingling.",
            reference_answer="Vaswani et al. (2017) did not study MinHash LSH; their paper introduced the Transformer architecture. MinHash shingling was developed by Andrei Broder (1997).",
            required_citations=["Broder (1997)"],
            golden_keywords=["did not study", "transformer", "broder", "minhash"],
            difficulty="hard",
        ),
        GoldenTestCase(
            case_id="adv_reverse_022",
            category="transformer_internals",
            bucket="adversarial_bait",
            question="Can bidirectional attention masks be safely used during causal language modeling?",
            ground_truth_context="Causal language modeling requires autoregressive prediction where token t can only attend to tokens <= t. Using bidirectional attention allows tokens to attend to future tokens, causing catastrophic target leakage and invalidating autoregressive generation (Radford et al., 2019).",
            reference_answer="No. Using a bidirectional mask in causal language modeling causes target leakage by exposing future tokens, completely invalidating autoregressive text generation [Radford et al., 2019].",
            required_citations=["Radford et al., 2019"],
            golden_keywords=["target leakage", "future tokens", "causal", "invalidating"],
            difficulty="easy",
        ),
        GoldenTestCase(
            case_id="adv_collapse_023",
            category="systems_curation",
            bucket="adversarial_bait",
            question="Is synthetic data generated by LLMs completely free from model collapse risks?",
            ground_truth_context="Shumailov et al. (Nature 2024) demonstrated that synthetic data contains statistical distribution truncation. Training exclusively on synthetic data accelerates model collapse, requiring real human anchor data to maintain stability.",
            reference_answer="No. Exclusively training on synthetic data accelerates model collapse due to loss of distribution tails [Shumailov et al., Nature 2024]. Human anchor data remains essential.",
            required_citations=["Shumailov et al., Nature 2024"],
            golden_keywords=["model collapse", "synthetic data", "distribution tails", "accelerates"],
            difficulty="medium",
        ),
        GoldenTestCase(
            case_id="adv_dora_024",
            category="peft_sft",
            bucket="adversarial_bait",
            question="Does standard LoRA decompose weight updates into separate magnitude and directional vectors?",
            ground_truth_context="Standard LoRA decomposes Delta W into B A without separating magnitude and direction (Hu et al., 2021). Weight-Decomposed Low-Rank Adaptation (DoRA, Liu et al., 2024) introduced explicit directional and magnitude decomposition W = m * (W_0 + Delta W) / ||W_0 + Delta W||.",
            reference_answer="Standard LoRA does not decouple magnitude and direction; it updates Delta W = (alpha / r) * B A directly [Hu et al., 2021]. Explicit magnitude-direction decomposition is introduced in DoRA [Liu et al., 2024].",
            required_citations=["Hu et al., 2021", "Liu et al., 2024"],
            golden_keywords=["standard lora does not", "dora", "magnitude", "direction"],
            difficulty="hard",
        ),

        # =====================================================================
        # 4. REGRESSION REPLAYS (6 Cases: Known Historical Failure Modes)
        # =====================================================================
        GoldenTestCase(
            case_id="regr_trunc_loss_025",
            category="peft_sft",
            bucket="regression_replay",
            question="How does window-preserving truncation prevent assistant loss erasure in long prompts?",
            ground_truth_context="Naive right-side truncation (tokens[:max_seq_len]) truncates the trailing assistant response when system and user prompts are long, leaving only prompt tokens masked to -100 and causing NaN loss. Window-preserving truncation reserves prompt budget and anchors around the assistant response.",
            reference_answer="Window-preserving truncation anchors on the assistant completion span, reserving a prompt context budget and preventing assistant response tokens from being truncated into an all-(-100) target vector [System Fix, 2026].",
            required_citations=["System Fix, 2026"],
            golden_keywords=["window-preserving", "all-(-100)", "assistant completion", "truncation"],
            difficulty="hard",
        ),
        GoldenTestCase(
            case_id="regr_weight_tie_026",
            category="transformer_internals",
            bucket="regression_replay",
            question="Why must weight tying between token embedding wte and lm_head be preserved during LoRA injection?",
            ground_truth_context="Press and Wolf (2017) introduced weight tying: lm_head.weight = wte.weight. Replacing lm_head with an un-tied linear layer duplicates parameter memory and desynchronizes input and output representations.",
            reference_answer="Press and Wolf (2017) showed weight tying ties lm_head.weight directly to wte.weight. Untying during LoRA injection breaks vocabulary representation symmetry and increases memory footprint.",
            required_citations=["Press & Wolf, 2017"],
            golden_keywords=["weight tying", "wte", "lm_head", "symmetry"],
            difficulty="medium",
        ),
        GoldenTestCase(
            case_id="regr_nan_grad_027",
            category="ml_theory",
            bucket="regression_replay",
            question="What causes gradient explosion and NaN values during unconstrained LoRA learning rates?",
            ground_truth_context="Setting LoRA learning rate eta too high without Pascanu L2 gradient clipping causes unbounded parameter steps in low-rank matrices A and B, triggering numerical overflow and NaN weights.",
            reference_answer="Excessive adapter learning rates without Pascanu gradient clipping (||g||_2 <= 1.0) cause explosive weight updates in matrix products B @ A, resulting in NaN logits [Pascanu et al., 2013].",
            required_citations=["Pascanu et al., 2013"],
            golden_keywords=["nan", "gradient clipping", "explosive", "learning rate"],
            difficulty="easy",
        ),
        GoldenTestCase(
            case_id="regr_anchor_drift_028",
            category="peft_sft",
            bucket="regression_replay",
            question="How does the anti-forgetting regression gate detect catastrophic forgetting on anchor benchmarks?",
            ground_truth_context="The regression evaluator measures completion perplexity on held-out anchor demonstrations before and after fine-tuning. If Delta PPL_anchor = (PPL_adapted - PPL_base) / PPL_base * 100 > 5.0%, the release gate triggers FAIL_FORGETTING.",
            reference_answer="The regression gate tracks relative anchor perplexity drift Delta PPL_anchor. If anchor perplexity increases by more than 5.0%, the gate flags FAIL_FORGETTING and triggers an immediate rollback [Evaluator Spec, 2026].",
            required_citations=["Evaluator Spec, 2026"],
            golden_keywords=["delta ppl_anchor", "5.0%", "fail_forgetting", "rollback"],
            difficulty="medium",
        ),
        GoldenTestCase(
            case_id="regr_prompt_bleed_029",
            category="peft_sft",
            bucket="regression_replay",
            question="What causes role bleed and prompt memorization in multi-turn ChatML fine-tuning?",
            ground_truth_context="Failing to mask prompt header tokens (<|im_start|>system and <|im_start|>user) to -100 causes the model to optimize next-token prediction over user instructions, leading to role bleed where the assistant mimics user turns.",
            reference_answer="Role bleed occurs when prompt delimiter tokens are not masked to -100. The model learns to predict user headers instead of completing assistant turns exclusively [Prompt Harmonizer Spec, 2026].",
            required_citations=["Prompt Harmonizer Spec, 2026"],
            golden_keywords=["role bleed", "prompt memorization", "<|im_start|>", "-100"],
            difficulty="medium",
        ),
        GoldenTestCase(
            case_id="regr_mps_autocast_030",
            category="systems_curation",
            bucket="regression_replay",
            question="Why must Apple Silicon MPS training use bfloat16 autocast rather than full float32 for large batches?",
            ground_truth_context="On Apple Silicon Unified Memory Architecture, running full float32 tensors doubles buffer allocations for activations and optimizer moments. Enabling torch.autocast('mps', dtype=torch.bfloat16) cuts memory bandwidth by 50% without numerical instability.",
            reference_answer="Using bfloat16 autocast on Apple Silicon MPS halves activation and optimizer state memory bandwidth compared to float32, preventing memory exhaustion on unified RAM [Hardware Strategy Spec, 2026].",
            required_citations=["Hardware Strategy Spec, 2026"],
            golden_keywords=["mps", "bfloat16", "unified memory", "autocast"],
            difficulty="easy",
        ),
    ]
    return cases


class GoldenBenchmarkSuite:
    """Production test suite container providing filtering, sampling, and contamination integration."""

    def __init__(self, cases: Optional[list[GoldenTestCase]] = None):
        self.cases: list[GoldenTestCase] = cases if cases is not None else create_default_golden_suite_cases()
        self._case_index: dict[str, GoldenTestCase] = {c.case_id: c for c in self.cases}

    @classmethod
    def load_default_suite(cls) -> "GoldenBenchmarkSuite":
        """Instantiate benchmark suite containing all 30 default golden domain cases."""
        return cls()

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> GoldenTestCase:
        return self.cases[idx]

    def get_by_id(self, case_id: str) -> Optional[GoldenTestCase]:
        """Fetch test case by unique ID."""
        return self._case_index.get(case_id)

    def filter_by_category(
        self, category: Literal["ml_theory", "transformer_internals", "peft_sft", "systems_curation"]
    ) -> "GoldenBenchmarkSuite":
        """Filter benchmark suite by specific knowledge category."""
        filtered = [c for c in self.cases if c.category == category]
        return GoldenBenchmarkSuite(filtered)

    def filter_by_bucket(
        self, bucket: Literal["core_domain", "edge_case", "adversarial_bait", "regression_replay"]
    ) -> "GoldenBenchmarkSuite":
        """Filter benchmark suite by specific behavioral bucket."""
        filtered = [c for c in self.cases if c.bucket == bucket]
        return GoldenBenchmarkSuite(filtered)

    def filter_by_difficulty(
        self, difficulty: Literal["easy", "medium", "hard"]
    ) -> "GoldenBenchmarkSuite":
        """Filter benchmark suite by problem difficulty."""
        filtered = [c for c in self.cases if c.difficulty == difficulty]
        return GoldenBenchmarkSuite(filtered)

    def sample(self, k: int, seed: int = 42) -> "GoldenBenchmarkSuite":
        """Deterministically sample k cases from the suite."""
        k = min(k, len(self.cases))
        rng = random.Random(seed)
        sampled = rng.sample(self.cases, k)
        return GoldenBenchmarkSuite(sampled)

    def to_schemas(self) -> list[BenchmarkCaseSchema]:
        """Export all test cases into core Pydantic BenchmarkCaseSchema list."""
        return [c.to_schema() for c in self.cases]

    @classmethod
    def from_schemas(cls, schemas: list[BenchmarkCaseSchema]) -> "GoldenBenchmarkSuite":
        """Instantiate benchmark suite from core BenchmarkCaseSchema collection."""
        cases = [GoldenTestCase.from_schema(s) for s in schemas]
        return cls(cases)

    def register_with_contamination_guard(self, guard: BenchmarkContaminationGuard) -> int:
        """Dynamically register all 30 benchmark cases into BenchmarkContaminationGuard.

        Extracts question, context, keywords, and reference answer n-grams to guarantee
        zero benchmark contamination during continuous web crawls.

        Returns:
            Total number of forbidden 13-grams registered.
        """
        initial_size = guard.size()
        for case in self.cases:
            combined = f"{case.question} {case.ground_truth_context} {case.reference_answer} {' '.join(case.golden_keywords)}"
            guard.add_benchmark_text(combined)
        added_count = guard.size() - initial_size
        logger.info(
            f"GoldenBenchmarkSuite: Registered {len(self.cases)} test cases "
            f"({added_count} forbidden 13-grams) into BenchmarkContaminationGuard."
        )
        return added_count

    def export_to_json(self, path: Union[Path, str]) -> Path:
        """Serialize benchmark suite to structured JSON file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(c) for c in self.cases]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"GoldenBenchmarkSuite: Exported {len(self.cases)} cases to {out_path}")
        return out_path

    @classmethod
    def load_from_json(cls, path: Union[Path, str]) -> "GoldenBenchmarkSuite":
        """Load benchmark suite from JSON file."""
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {in_path}")
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cases = [GoldenTestCase(**item) for item in data]
        return cls(cases)

    def get_summary(self) -> dict:
        """Emit detailed composition statistics of the benchmark suite."""
        category_counts = {}
        bucket_counts = {}
        difficulty_counts = {}

        for c in self.cases:
            category_counts[c.category] = category_counts.get(c.category, 0) + 1
            bucket_counts[c.bucket] = bucket_counts.get(c.bucket, 0) + 1
            difficulty_counts[c.difficulty] = difficulty_counts.get(c.difficulty, 0) + 1

        return {
            "total_cases": len(self.cases),
            "categories": category_counts,
            "buckets": bucket_counts,
            "difficulties": difficulty_counts,
            "total_keywords": sum(len(c.golden_keywords) for c in self.cases),
            "total_required_citations": sum(len(c.required_citations) for c in self.cases),
        }

