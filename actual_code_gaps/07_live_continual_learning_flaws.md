# GAP-07: Flaws in the Live Continual SFT Learning Loop

## 1. Landmark Academic Literature & Foundations

1. **Askell, A., Bai, Y., Chen, A., et al. (Anthropic, 2021)**. *A General Language Assistant as a Laboratory for Alignment*. arXiv:2112.00861.
2. **Gao, T., Yen, H., Yu, J., & Chen, D. (Princeton, 2023)**. *Enabling Large Language Models to Generate Text with Citations* (ALCE). EMNLP 2023 / arXiv:2305.14627.
3. **Penedo, G., et al. (Hugging Face, 2024)**. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*. arXiv:2406.17557.

---

## 2. Flaw 1: The SFT Target is a Trivial Copy Task

In [`training/m3_sft/hf_worker.py`](file:///Users/jayesh/Documents/RLHF/training/m3_sft/hf_worker.py#L45-L52) (`_read_web_examples`):
```python
            if text and url:
                excerpt = text[:1600]
                examples.append(
                    (
                        "Summarize the source accurately and cite it. "
                        f"Source URL: {url}\n\n{excerpt}",
                        f"{excerpt}\n\nSource: {url}",
                    )
                )
```

### The Problem:
- **Prompt**: `"Summarize the source accurately and cite it. Source URL: {url}\n\n{excerpt}"`
- **Target completion**: `f"{excerpt}\n\nSource: {url}"`
- The completion is literally a **verbatim copy** of the text that was just supplied in the prompt, with `Source: {url}` appended to the end.
- Rather than learning summarization, factual extraction, knowledge synthesis, or question answering, the model is trained on an **identity/echo function**:
  $$\text{Completion} \equiv \text{Input Excerpt}$$
- This trains the model to regurgitate whatever is in its context window without reading comprehension or semantic transformation.

---

## 3. Flaw 2: The Benchmark Rewards This Exact Copy Behavior (Teaching to the Test)

In [`training/m3_sft/hf_worker.py`](file:///Users/jayesh/Documents/RLHF/training/m3_sft/hf_worker.py#L344-L346):
```python
    @staticmethod
    def _case_score(case, answer: str) -> float:
        return 4.0 * case.evaluate_keyword_coverage(answer) + case.evaluate_citation_presence(answer)
```
Where in [`evaluation/benchmark_suite.py`](file:///Users/jayesh/Documents/RLHF/evaluation/benchmark_suite.py#L88-L100):
```python
    def evaluate_keyword_coverage(self, response: str) -> float:
        response_lower = response.lower()
        hits = sum(1 for kw in self.golden_keywords if kw.lower() in response_lower)
        return hits / len(self.golden_keywords)

    def evaluate_citation_presence(self, response: str) -> float:
        hits = sum(1 for cite in self.required_citations if cite in response)
        return hits / len(self.required_citations)
```

### The Goodhart's Law Gaming Loop:
1. The training dataset trains the candidate model to copy the provided context verbatim and append a citation line.
2. The benchmark prompt ([`hf_worker.py#L336-L341`](file:///Users/jayesh/Documents/RLHF/training/m3_sft/hf_worker.py#L336-L341)) supplies the ground truth context in the prompt and asks the model to answer using it.
3. The evaluation metric gives **4 points for keyword recall** (which a verbatim copier easily achieves by regurgitating the context) plus **1 point for citation strings**.
4. The release gate evaluates this score matrix.
5. **Result**: The candidate model scores high not because it gained reasoning capabilities or knowledge, but because the training task and the evaluation metric both reward verbatim copying. This is a classic violation of Goodhart's Law: *"When a measure becomes a target, it ceases to be a good measure."*

---

## 4. Flaw 3: Micro-Training Regime

In [`training/m3_sft/hf_worker.py`](file:///Users/jayesh/Documents/RLHF/training/m3_sft/hf_worker.py#L94-L198):
```python
max_steps = int(os.getenv("HF_SFT_MAX_STEPS", "20"))
...
training_args = TrainingArguments(
    max_steps=max_steps,              # 20 steps
    per_device_train_batch_size=1,    # Batch size 1
    gradient_accumulation_steps=4,    # Accumulation 4
    learning_rate=5e-5,
)
```
- Total training volume:
  $$N_{\text{samples}} = 20 \text{ steps} \times 4 = \mathbf{80 \text{ samples}}$$
- Fine-tuning a 500M parameter model on 80 samples for 20 steps with AdamW lr=5e-5 is virtually insufficient to adapt model weights meaningfully. It acts as a shallow prompt memorizer for a handful of examples.

---

## 5. Flaw 4: Incompatible Evaluation Baselines

In [`training/m3_sft/hf_worker.py`](file:///Users/jayesh/Documents/RLHF/training/m3_sft/hf_worker.py#L288-L333):

| Attribute | Baseline Evaluation (`_score_ollama_baseline`) | Candidate Evaluation (`_score_candidate`) |
| :--- | :--- | :--- |
| **Inference Engine** | Ollama HTTP REST API (`/api/chat`) | Local Hugging Face PyTorch `model.generate()` |
| **Quantization** | Typically 4-bit quantized (Q4_K_M) via Ollama | Full precision (FP32, `torch.float32`) |
| **Chat Template** | Ollama Modelfile internal template | Hugging Face `tokenizer.apply_chat_template` |
| **Generation Cap** | Uncapped (Ollama default) | Capped at `max_new_tokens=120` |
| **Execution Device** | Ollama runtime | PyTorch CPU / MPS |

### Consequence:
Comparing an FP32 Hugging Face candidate against a quantized Ollama baseline with different chat templates and generation limits introduces significant confounding variables:
- The measured delta $\Delta = S_{\text{cand}} - S_{\text{base}}$ reflects differences in quantization precision and token truncation rather than post-training improvement.

