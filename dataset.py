"""
dataset.py — Data Loading, Vocabulary, and DataLoader
DA6401 Assignment 3: "Attention Is All You Need"

Pipeline:
    Multi30k (HuggingFace) → spaCy tokenisation → Vocab → integer tensors → DataLoader
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
from typing import List, Tuple, Dict, Optional
import spacy


# ══════════════════════════════════════════════════════════════════════
#  VOCABULARY
# ══════════════════════════════════════════════════════════════════════

class Vocab:
    """
    Simple vocabulary class mapping tokens <-> integer indices.

    Special tokens (always at fixed indices):
        <pad>  → 0   padding / ignored in loss
        <unk>  → 1   unknown tokens not in vocab
        <sos>  → 2   start-of-sequence sentinel
        <eos>  → 3   end-of-sequence sentinel
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"

    SPECIALS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]

    def __init__(self) -> None:
        # token → index
        self.stoi: Dict[str, int] = {}
        # index → token
        self.itos: Dict[int, str] = {}
        self._add_specials()

    # ------------------------------------------------------------------
    def _add_specials(self) -> None:
        for i, tok in enumerate(self.SPECIALS):
            self.stoi[tok] = i
            self.itos[i]   = tok

    # ------------------------------------------------------------------
    def build_from_counter(self, counter: Counter, min_freq: int = 2) -> None:
        """
        Populate the vocabulary from a token frequency Counter.

        Args:
            counter  : Token → count mapping.
            min_freq : Tokens appearing fewer than this many times are dropped.
        """
        idx = len(self.stoi)          
        for token, freq in counter.most_common():
            if freq < min_freq:
                break                 # most_common is sorted descending
            if token not in self.stoi:
                self.stoi[token] = idx
                self.itos[idx]   = token
                idx += 1

    # ------------------------------------------------------------------
    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, self.UNK_TOKEN)

    def lookup_index(self, token: str) -> int:
        return self.stoi.get(token, self.stoi[self.UNK_TOKEN])

    # Convenience properties
    @property
    def pad_idx(self) -> int:  return self.stoi[self.PAD_TOKEN]

    @property
    def unk_idx(self) -> int:  return self.stoi[self.UNK_TOKEN]

    @property
    def sos_idx(self) -> int:  return self.stoi[self.SOS_TOKEN]

    @property
    def eos_idx(self) -> int:  return self.stoi[self.EOS_TOKEN]

    def __len__(self) -> int:  return len(self.stoi)

    def __repr__(self) -> str:
        return f"Vocab(size={len(self)})"


# ══════════════════════════════════════════════════════════════════════
#  MULTI30K DATASET
# ══════════════════════════════════════════════════════════════════════

class Multi30kDataset:
    """
    Wrapper around the bentrevett/multi30k HuggingFace dataset.

    Responsibilities:
        • Download and cache the dataset (train / validation / test splits)
        • Tokenise German (src) and English (tgt) with spaCy
        • Build shared-frequency vocabularies for each language
        • Convert sentences to lists of integer token indices

    Usage:
        ds = Multi30kDataset('train')
        ds.build_vocab()
        ds.process_data()

        # Access processed data
        src_sequences, tgt_sequences = ds.src_data, ds.tgt_data
        src_vocab, tgt_vocab         = ds.src_vocab, ds.tgt_vocab
    """

    # HuggingFace dataset name and split aliases
    HF_DATASET  = "bentrevett/multi30k"
    SPLIT_MAP   = {"train": "train", "valid": "validation", "test": "test"}

    def __init__(self, split: str = "train") -> None:
        """
        Args:
            split : One of 'train', 'valid', 'test'.
        """
        assert split in self.SPLIT_MAP, f"split must be one of {list(self.SPLIT_MAP)}"
        self.split = split

        # ── Load spaCy tokenisers ──────────────────────────────────────
        # Run once:  python -m spacy download de_core_news_sm
        #            python -m spacy download en_core_web_sm
        print("Loading spaCy models …")
        self.de_nlp = spacy.load("de_core_news_sm")
        self.en_nlp = spacy.load("en_core_web_sm")

        # ── Download Multi30k from HuggingFace ─────────────────────────
        print(f"Downloading Multi30k [{split}] …")
        raw = load_dataset(self.HF_DATASET)
        self.raw_data = raw[self.SPLIT_MAP[split]]

        # Will be populated by build_vocab() / process_data()
        self.src_vocab: Optional[Vocab] = None
        self.tgt_vocab: Optional[Vocab] = None
        self.src_data:  Optional[List[List[int]]] = None
        self.tgt_data:  Optional[List[List[int]]] = None

    # ------------------------------------------------------------------
    def tokenize_de(self, text: str) -> List[str]:
        """Lower-case, alphabetic German tokenisation via spaCy."""
        return [tok.text.lower() for tok in self.de_nlp.tokenizer(text)]

    def tokenize_en(self, text: str) -> List[str]:
        """Lower-case, alphabetic English tokenisation via spaCy."""
        return [tok.text.lower() for tok in self.en_nlp.tokenizer(text)]

    # ------------------------------------------------------------------
    def build_vocab(
        self,
        src_min_freq: int = 2,
        tgt_min_freq: int = 2,
    ) -> Tuple["Vocab", "Vocab"]:
        """
        Build German (src) and English (tgt) vocabularies from training data.

        Always call on the 'train' split; reuse the returned vocabs for
        validation and test splits.

        Args:
            src_min_freq : Minimum token frequency for German vocab.
            tgt_min_freq : Minimum token frequency for English vocab.

        Returns:
            (src_vocab, tgt_vocab)
        """
        src_counter: Counter = Counter()
        tgt_counter: Counter = Counter()

        print("Building vocabulary …")
        for sample in self.raw_data:
            src_counter.update(self.tokenize_de(sample["de"]))
            tgt_counter.update(self.tokenize_en(sample["en"]))

        self.src_vocab = Vocab()
        self.src_vocab.build_from_counter(src_counter, min_freq=src_min_freq)

        self.tgt_vocab = Vocab()
        self.tgt_vocab.build_from_counter(tgt_counter, min_freq=tgt_min_freq)

        print(f"  src vocab size : {len(self.src_vocab)}")
        print(f"  tgt vocab size : {len(self.tgt_vocab)}")
        return self.src_vocab, self.tgt_vocab

    # ------------------------------------------------------------------
    def set_vocab(self, src_vocab: "Vocab", tgt_vocab: "Vocab") -> None:
        """
        Inject pre-built vocabularies (for val / test splits).
        Must be called before process_data().
        """
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    # ------------------------------------------------------------------
    def _encode(self, tokens: List[str], vocab: "Vocab") -> List[int]:
        """Convert a token list to integer indices, wrapping with <sos>/<eos>."""
        return (
            [vocab.sos_idx]
            + [vocab.lookup_index(t) for t in tokens]
            + [vocab.eos_idx]
        )

    # ------------------------------------------------------------------
    def process_data(self) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Tokenise every sentence and convert to integer index lists.

        Requires src_vocab and tgt_vocab to be set first (via build_vocab
        or set_vocab).

        Returns:
            (src_data, tgt_data) — lists of integer-encoded sequences.
        """
        assert self.src_vocab is not None and self.tgt_vocab is not None, (
            "Call build_vocab() or set_vocab() before process_data()."
        )

        print(f"Processing {self.split} data …")
        self.src_data, self.tgt_data = [], []

        for sample in self.raw_data:
            src_tokens = self.tokenize_de(sample["de"])
            tgt_tokens = self.tokenize_en(sample["en"])

            self.src_data.append(self._encode(src_tokens, self.src_vocab))
            self.tgt_data.append(self._encode(tgt_tokens, self.tgt_vocab))

        print(f"  {len(self.src_data)} sentence pairs processed.")
        return self.src_data, self.tgt_data


# ══════════════════════════════════════════════════════════════════════
#  PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════

class TranslationDataset(Dataset):
    """
    PyTorch Dataset wrapping (src, tgt) integer-encoded sequences.

    Returns raw lists; padding is deferred to the collate function so
    that every batch is padded to its own max length (not the global max).
    """

    def __init__(
        self,
        src_data: List[List[int]],
        tgt_data: List[List[int]],
    ) -> None:
        assert len(src_data) == len(tgt_data)
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.src_data[idx], self.tgt_data[idx]


# ══════════════════════════════════════════════════════════════════════
#  COLLATE & DATALOADER
# ══════════════════════════════════════════════════════════════════════

def make_collate_fn(src_pad_idx: int, tgt_pad_idx: int):
    """
    Returns a collate function that pads variable-length sequences to the
    longest sequence in the batch.

    Args:
        src_pad_idx : <pad> index in the source vocabulary.
        tgt_pad_idx : <pad> index in the target vocabulary.
    """
    def collate_fn(
        batch: List[Tuple[List[int], List[int]]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_batch, tgt_batch = zip(*batch)

        # Pad each list to the max length in the batch
        src_max = max(len(s) for s in src_batch)
        tgt_max = max(len(t) for t in tgt_batch)

        src_padded = torch.tensor(
            [s + [src_pad_idx] * (src_max - len(s)) for s in src_batch],
            dtype=torch.long,
        )
        tgt_padded = torch.tensor(
            [t + [tgt_pad_idx] * (tgt_max - len(t)) for t in tgt_batch],
            dtype=torch.long,
        )
        return src_padded, tgt_padded

    return collate_fn


def get_dataloader(
    dataset:     TranslationDataset,
    src_pad_idx: int,
    tgt_pad_idx: int,
    batch_size:  int  = 128,
    shuffle:     bool = True,
    num_workers: int  = 0,
) -> DataLoader:
    """
    Wrap a TranslationDataset in a DataLoader with padding collation.

    Args:
        dataset     : TranslationDataset instance.
        src_pad_idx : Source <pad> index.
        tgt_pad_idx : Target <pad> index.
        batch_size  : Number of sentence pairs per batch.
        shuffle     : Shuffle before each epoch (should be True for training).
        num_workers : Parallel data-loading workers.

    Returns:
        DataLoader yielding (src, tgt) tensor batches.
    """
    collate_fn = make_collate_fn(src_pad_idx, tgt_pad_idx)
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        collate_fn  = collate_fn,
        num_workers = num_workers,
        pin_memory  = True,
    )


# ══════════════════════════════════════════════════════════════════════
#  CONVENIENCE: build all three splits at once
# ══════════════════════════════════════════════════════════════════════

def build_dataloaders(
    batch_size: int = 128,
    src_min_freq: int = 2,
    tgt_min_freq: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocab, Vocab]:
    """
    One-shot helper that constructs train / val / test DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    # ── Training split: build vocab ────────────────────────────────────
    train_ds = Multi30kDataset("train")
    src_vocab, tgt_vocab = train_ds.build_vocab(src_min_freq, tgt_min_freq)
    train_ds.process_data()

    # ── Validation split: reuse vocab ─────────────────────────────────
    val_ds = Multi30kDataset("valid")
    val_ds.set_vocab(src_vocab, tgt_vocab)
    val_ds.process_data()

    # ── Test split: reuse vocab ────────────────────────────────────────
    test_ds = Multi30kDataset("test")
    test_ds.set_vocab(src_vocab, tgt_vocab)
    test_ds.process_data()

    pad_src = src_vocab.pad_idx
    pad_tgt = tgt_vocab.pad_idx

    train_loader = get_dataloader(
        TranslationDataset(train_ds.src_data, train_ds.tgt_data),
        pad_src, pad_tgt, batch_size=batch_size, shuffle=True,
    )
    val_loader = get_dataloader(
        TranslationDataset(val_ds.src_data, val_ds.tgt_data),
        pad_src, pad_tgt, batch_size=batch_size, shuffle=False,
    )
    test_loader = get_dataloader(
        TranslationDataset(test_ds.src_data, test_ds.tgt_data),
        pad_src, pad_tgt, batch_size=batch_size, shuffle=False,
    )

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab


# ══════════════════════════════════════════════════════════════════════
#  QUICK SANITY CHECK
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(
        batch_size=32
    )

    src_batch, tgt_batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  src shape : {src_batch.shape}")    # [B, src_len]
    print(f"  tgt shape : {tgt_batch.shape}")    # [B, tgt_len]
    print(f"  src vocab : {src_vocab}")
    print(f"  tgt vocab : {tgt_vocab}")

    # Decode a sample sentence
    sample_src = src_batch[0].tolist()
    decoded = " ".join(src_vocab.lookup_token(i) for i in sample_src
                       if i != src_vocab.pad_idx)
    print(f"\nDecoded src[0]: {decoded}")
