"""Character-level tokenizer for language modeling foundations."""

class CharTokenizer:
    """Character-level tokenizer mapping between characters and discrete integer indices."""

    def __init__(self, chars: list[str] | str | None = None):
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}
        if chars:
            self.build_vocab(chars)

    def build_vocab(self, text: list[str] | str) -> None:
        """Construct sorted vocabulary from text or character collection."""
        unique_chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(unique_chars)}
        self.itos = {i: ch for i, ch in enumerate(unique_chars)}

    @property
    def vocab_size(self) -> int:
        """Total number of unique tokens in vocabulary."""
        return len(self.stoi)

    @property
    def vocab(self) -> list[str]:
        """List of all unique characters in vocabulary."""
        return sorted(list(self.stoi.keys()))

    def encode(self, text: str) -> list[int]:
        """Convert string of characters into list of integer token IDs."""
        if not self.stoi:
            raise ValueError("Tokenizer vocabulary has not been built.")
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, indices: list[int]) -> str:
        """Convert list of integer token IDs back into string."""
        return "".join([self.itos.get(idx, "") for idx in indices])

