"""Configuration dataclass for Mini-GPT Transformer architecture."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Architectural hyperparameters for Mini-GPT."""

    block_size: int = 128  # Maximum sequence context window length (T)
    vocab_size: int = 65  # Vocabulary size (65 unique characters in TinyShakespeare)
    n_layer: int = 4  # Number of Transformer blocks (L)
    n_head: int = 4  # Number of attention heads (d_k = d_model / n_head)
    d_model: int = 128  # Residual embedding dimension (C)
    dropout: float = 0.1  # Dropout probability
    bias: bool = False  # Whether to use bias in Linear layers (modern LLMs set False)

    def __post_init__(self) -> None:
        """Validate architectural dimension divisibility."""
        assert (
            self.d_model % self.n_head == 0
        ), f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"

