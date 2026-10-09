"""Dataset and DataLoader utilities supporting both in-memory and memory-mapped large data streaming."""

from pathlib import Path
from typing import Optional, Union
import urllib.request
import numpy as np
import torch
from loguru import logger

from training.m1_bigram.tokenizer import CharTokenizer

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class MemoryMappedTokenDataset:
    """Zero-RAM memory-mapped token dataset for arbitrarily large corpora (GBs/TBs)."""

    def __init__(self, bin_path: Union[Path, str], dtype: np.dtype = np.uint16):
        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Binary token file not found at {self.bin_path}")
        self.dtype = dtype
        self.data = np.memmap(self.bin_path, dtype=self.dtype, mode="r")
        self.length = len(self.data)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:
        arr = self.data[idx]
        return torch.from_numpy(arr.astype(np.int64))

    def get_batch(
        self, batch_size: int, block_size: int, device: Union[torch.device, str] = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample random (B, T) sequence batch with zero RAM overhead via memmap."""
        ix = np.random.randint(0, self.length - block_size, size=batch_size)
        x = np.stack([self.data[i : i + block_size] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1 : i + block_size + 1] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def export_tokens_to_binary(
    tokens: list[int], output_path: Union[Path, str], dtype: np.dtype = np.uint16
) -> Path:
    """Serialize token array to compact disk binary for high-throughput streaming."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(tokens, dtype=dtype)
    arr.tofile(out_file)
    logger.info(f"Exported {len(tokens):,} tokens to {out_file} ({out_file.stat().st_size / (1024**2):.2f} MB)")
    return out_file


def get_shakespeare_data(
    data_dir: Union[Path, str] = "data",
    export_binary: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, CharTokenizer]:
    """Load or download TinyShakespeare text, encode to tensor, and create train/val split."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    file_path = data_path / "tinyshakespeare.txt"

    if not file_path.exists():
        logger.info(f"Downloading TinyShakespeare dataset to {file_path}...")
        try:
            urllib.request.urlretrieve(SHAKESPEARE_URL, file_path)
            logger.info("Download completed successfully.")
        except Exception as e:
            logger.warning(f"Download failed ({e}). Generating fallback corpus.")
            corpus = (
                """First Citizen:\nBefore we proceed any further, hear me speak.\n\n"""
                """All:\nSpeak, speak.\n\n"""
                """First Citizen:\nYou are all resolved rather to die than to famish?\n\n"""
                """All:\nResolved. resolved.\n\n"""
                """First Citizen:\nFirst, you know Caius Marcius is chief enemy to the people.\n\n"""
                """All:\nWe know't, we know't.\n\n"""
                """First Citizen:\nLet us kill him, and we'll have corn at our own price.\nIs't a verdict?\n\n"""
                """All:\nNo more talking on't; let it be done: away, away!\n\n"""
                """Second Citizen:\nOne word, good citizens.\n"""
            ) * 500
            file_path.write_text(corpus, encoding="utf-8")

    text = file_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    encoded = tokenizer.encode(text)
    data_tensor = torch.tensor(encoded, dtype=torch.long)

    # 90% train, 10% validation split
    n = int(0.9 * len(data_tensor))
    train_data = data_tensor[:n]
    val_data = data_tensor[n:]

    if export_binary:
        train_bin = data_path / "train.bin"
        val_bin = data_path / "val.bin"
        if not train_bin.exists():
            export_tokens_to_binary(encoded[:n], train_bin)
        if not val_bin.exists():
            export_tokens_to_binary(encoded[n:], val_bin)

    logger.info(
        f"Corpus loaded: {len(text):,} chars | Vocab size: {tokenizer.vocab_size} | "
        f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens"
    )

    return train_data, val_data, tokenizer


def get_batch(
    data: Union[torch.Tensor, MemoryMappedTokenDataset],
    batch_size: int,
    block_size: int,
    device: Union[torch.device, str] = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of inputs X and shifted next-token targets Y (supports tensor and memmap)."""
    if isinstance(data, MemoryMappedTokenDataset):
        return data.get_batch(batch_size, block_size, device)

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
