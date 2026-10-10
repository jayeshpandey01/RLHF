# GAP-09: Pseudo-Dense Hash Retrieval, Security & Poisoning Gaps

## 1. Landmark Academic Literature & Foundations

1. **Karp, R. M., & Rabin, M. O. (1987)**. *Efficient randomized pattern-matching algorithms*. IBM Journal of Research and Development.
2. **Reimers, N., & Gurevych, I. (2019)**. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019.
3. **Carlini, N., et al. (2021)**. *Extracting Training Data from Large Language Models*. USENIX Security 2021.
4. **Shumailov, I., et al. (2024)**. *The Curse of Recursion: Training on Generated Data Makes Models Forget*. Nature 2024.

---

## 2. Pseudo-Dense Hash Retrieval vs Semantic Embedding

In [`retrieval/dense.py`](file:///Users/jayesh/Documents/RLHF/retrieval/dense.py#L12-L50):
```python
class SimpleHashEmbedder:
    """Deterministic character n-gram hashing baseline.

    Produces dense vectors without a pretrained model; similarity is lexical/subword-based,
    not learned semantic similarity.
    """

    def __init__(self, dim: int = 128, ngram_range: tuple = (2, 4)):
        self.dim = dim
        self.ngram_range = ngram_range

    def encode(self, text: str) -> torch.Tensor:
        ...
        for gram in ngrams:
            # Deterministic MD5 hash bucket
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 31) & 1 else -1.0
            vec[idx] += sign
        ...
        return vec / norm
```

### The Discrepancy:
- The system is documented as using **"Hybrid Retrieval: BM25 + Dense Semantic Cosine Search"**.
- In reality, the "Dense" embedding index defaults to `SimpleHashEmbedder`, which performs **MD5 character n-gram feature hashing** into a 128-dimensional vector.
- This is a randomized lexical/subword representation (similar to MinHash or feature hashing in Vowpal Wabbit), **not semantic embedding**:
  - Synonyms (e.g. "physician" and "doctor") have orthogonal hash vectors ($\cos \theta \approx 0$).
  - Paraphrased queries fail cosine lookup.
  - Semantic clustering is non-existent.
- While computationally lightweight, it does not provide neural semantic retrieval as described in RAG literature.

---

## 3. Data Poisoning Vulnerability via Automatic Triggering

In [`orchestration/loop_orchestrator.py`](file:///Users/jayesh/Documents/RLHF/orchestration/loop_orchestrator.py#L49-L50) and lines 316–318:
```python
self.training_trigger_threshold = 5

def should_trigger_continual_training(self, force: bool = False) -> Tuple[bool, str]:
    if self.new_web_samples_since_train >= self.training_trigger_threshold:
        return True, f"TRIGGERED_BUFFER_VOLUME ({self.new_web_samples_since_train} >= {self.training_trigger_threshold})"
```

### The Attack Vector:
1. Continual training is triggered automatically whenever 5 new web documents enter the buffer.
2. In [`apps/live_knowledge.py`](file:///Users/jayesh/Documents/RLHF/apps/live_knowledge.py#L72-L100), incoming user queries fetch external web documents via the `IngestionRouter` and add them to the knowledge corpus.
3. An adversary can issue 5 crafted search queries that index attacker-controlled web pages.
4. As soon as 5 pages are indexed, the system immediately launches `train_evaluate_and_package()`, fine-tuning the model on the attacker's content and attempting to hot-swap the model into serving.
5. While the `BenchmarkContaminationGuard` prevents leaking golden test strings, it does not prevent data poisoning (injecting malicious associations, backdoors, or propaganda into the general model).

---

## 4. Unauthenticated Endpoints & Operational Security

In [`apps/server.py`](file:///Users/jayesh/Documents/RLHF/apps/server.py):
- `POST /trigger_learning`: Unauthenticated endpoint allowing any client to trigger an expensive CPU/GPU training run, leading to denial-of-service (DoS) via resource exhaustion.
- `POST /models/promote`: Unauthenticated endpoint allowing any client to hot-swap the active serving model.
- `POST /models/rollback`: Unauthenticated endpoint allowing any client to revert the serving model to anchor baseline version 0.

---

## 5. Process-Local Non-Durable Job State

In [`apps/learning_jobs.py`](file:///Users/jayesh/Documents/RLHF/apps/learning_jobs.py#L13-L18):
- All training job state is stored in an in-memory dictionary `self._jobs = {}`.
- A single process crash, OOM event, or container restart completely wipes all job history and can leave orphan candidate artifacts on disk.

