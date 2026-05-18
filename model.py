"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

Implements scaled dot-product attention from Vaswani et al. 2017.

AUTOGRADER CONTRACT (do not rename):
    make_src_mask(src, pad_idx)  → torch.Tensor  [B, 1, 1, src_len]
    make_tgt_mask(tgt, pad_idx)  → torch.Tensor  [B, 1, tgt_len, tgt_len]
    class Transformer            → full model
"""

import math
import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from utils import (
    make_src_mask,
    make_tgt_mask,
    PositionalEncoding,
    MultiHeadAttention,
    greedy_decode,
    get_device,
)
from dataset import build_dataloaders, Multi30kDataset



# ══════════════════════════════════════════════════════════════════════
#  POSITIONWISE FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Two-layer MLP applied position-wise (independently to each token).

        FFN(x) = max(0, x W₁ + b₁) W₂ + b₂

    Args:
        d_model : Input/output dimension.
        d_ff    : Hidden dimension (paper uses 4 × d_model = 2048).
        dropout : Dropout between the two linear layers.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ADD & NORM  (sublayer wrapper)
# ══════════════════════════════════════════════════════════════════════

class SublayerConnection(nn.Module):
    """
    Post-LayerNorm residual connection as in the original paper:
        output = LayerNorm(x + Dropout(sublayer(x)))
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        return self.norm(x + self.dropout(sublayer(x)))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder layer.

    Sub-layers:
        1. Multi-Head Self-Attention  (with src padding mask)
        2. Position-wise Feed-Forward Network
    """

    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        d_ff:      int,
        dropout:   float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = SublayerConnection(d_model, dropout)
        self.norm2     = SublayerConnection(d_model, dropout)

    def forward(
        self,
        x:        torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x        : [B, src_len, d_model]
            src_mask : [B, 1, 1, src_len]

        Returns:
            x : [B, src_len, d_model]
        """
        x = self.norm1(x, lambda z: self.self_attn(z, z, z, mask=src_mask))
        x = self.norm2(x, self.ff)
        return x


class Encoder(nn.Module):
    """Stack of N encoder layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model:    int,
        N:          int,
        num_heads:  int,
        d_ff:       int,
        dropout:    float = 0.1,
        max_len:    int   = 256
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, dropout, max_len)
        self.layers    = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(N)
        ])
        self.scale     = math.sqrt(d_model)

    def forward(
        self,
        src:      torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src      : [B, src_len]
            src_mask : [B, 1, 1, src_len]

        Returns:
            enc_out : [B, src_len, d_model]
        """
        x = self.pos_enc(self.embedding(src) * self.scale)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder layer.

    Sub-layers:
        1. Masked Multi-Head Self-Attention  (causal / look-ahead mask)
        2. Multi-Head Cross-Attention        (query = tgt, key/value = enc_out)
        3. Position-wise Feed-Forward Network
    """

    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        d_ff:      int,
        dropout:   float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff         = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = SublayerConnection(d_model, dropout)
        self.norm2      = SublayerConnection(d_model, dropout)
        self.norm3      = SublayerConnection(d_model, dropout)

    def forward(
        self,
        x:        torch.Tensor,
        enc_out:  torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            x : [B, tgt_len, d_model]
        """
        # 1. Masked self-attention (causal mask prevents future peek)
        x = self.norm1(x, lambda z: self.self_attn(z, z, z, mask=tgt_mask))
        # 2. Cross-attention: query from decoder, key/value from encoder
        x = self.norm2(x, lambda z: self.cross_attn(z, enc_out, enc_out, mask=src_mask))
        # 3. Feed-forward
        x = self.norm3(x, self.ff)
        return x


class Decoder(nn.Module):
    """Stack of N decoder layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model:    int,
        N:          int,
        num_heads:  int,
        d_ff:       int,
        dropout:    float = 0.1,
        max_len:    int   = 256
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, dropout, max_len)
        self.layers    = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(N)
        ])
        self.scale     = math.sqrt(d_model)

    def forward(
        self,
        tgt:      torch.Tensor,
        enc_out:  torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.pos_enc(self.embedding(tgt) * self.scale)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x


# ══════════════════════════════════════════════════════════════════════
#  TRANSFORMER  (top-level autograder contract)
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence translation.

    Default hyperparameters follow a smaller config suited for Multi30k:
        d_model=256, N=3, num_heads=8, d_ff=512

    Args:
        src_vocab_size : Size of the source vocabulary.
        tgt_vocab_size : Size of the target vocabulary.
        d_model        : Embedding / model dimension.
        N              : Number of encoder and decoder layers.
        num_heads      : Number of attention heads.
        d_ff           : Feed-forward hidden dimension.
        dropout        : Dropout probability.
        max_len        : Maximum sequence length for positional encoding.
        pad_idx        : <pad> token index (for mask creation inside forward).
    """

    def __init__(
        self,
        src_vocab_size: int = 7853,
        tgt_vocab_size: int = 5893,
        d_model:    int   = 256,
        N:          int   = 4,
        num_heads:  int   = 8,
        d_ff:       int   = 512,
        dropout:    float = 0.11,
        max_len:    int   = 256,
        pad_idx:    int   = 0,
        src_vocab = None, 
        tgt_vocab = None,
    ) -> None:
        super().__init__()

        if src_vocab is None or tgt_vocab is None:
            train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(
            batch_size   = 256,
            src_min_freq = 2,
            tgt_min_freq = 2,
            )

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.dataset = Multi30kDataset()
        self.pad_idx = pad_idx

        self.encoder = Encoder(
            src_vocab_size, d_model, N, num_heads, d_ff, dropout, max_len,
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, N, num_heads, d_ff, dropout, max_len,
        )

        # Final linear projection: d_model → tgt_vocab_size
        self.output_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Xavier uniform initialisation (as recommended in the paper)."""
        file_path = "best_checkpoint.pt"

        try:
            gdown.download(id="1k7XnywpE7v6fImA-j9uLsrOsoJuzsifH", output=file_path, quiet=False)
            ckpt = torch.load(file_path, map_location="cpu")

            model_dict = {'encoder': {}, 'decoder': {}, 'output_proj': {}}
            for key, value in ckpt['model_state_dict'].items():
                ls = key.split('.')
                model_dict[key.split('.')[0]][".".join(ls[1:])] = value
            
    
            ls = [self.encoder, self.decoder, self.output_proj  ]
            ls[0].load_state_dict(model_dict['encoder'])
            ls[1].load_state_dict(model_dict['decoder'])
            ls[2].load_state_dict(model_dict['output_proj'])


            print(f"  ✓ Loaded existing checkpoint from {file_path}")
        except (FileNotFoundError, RuntimeError, gdown.exceptions.DownloadError) as e:
            print(f"  No existing checkpoint found at {file_path}. Starting fresh. {e}, ")
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
       

    # ------------------------------------------------------------------
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt:      torch.Tensor,
        enc_out:  torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(tgt, enc_out, src_mask, tgt_mask)

    # ------------------------------------------------------------------
    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full encoder → decoder forward pass.

        Args:
            src      : [B, src_len]  source token indices
            tgt      : [B, tgt_len]  target token indices (teacher-forced)
            src_mask : [B, 1, 1, src_len]       from make_src_mask()
            tgt_mask : [B, 1, tgt_len, tgt_len] from make_tgt_mask()

        Returns:
            logits : [B, tgt_len, tgt_vocab_size]  (raw, pre-softmax)
        """
        if src_mask is None:
            src_mask = make_src_mask(src, self.pad_idx)
        if tgt_mask is None:
            tgt_mask = make_tgt_mask(tgt, self.pad_idx)

        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.output_proj(dec_out)    # [B, tgt_len, vocab]

    # ------------------------------------------------------------------
    def infer(
        self,
        sentence:  str,
        device:    str = "cpu",
        max_len:   int = 100,
    ) -> str:
        """
        Translate a single German sentence string to English.

        End-to-end pipeline:
            raw string  →  spaCy tokenise  →  vocab lookup  →  greedy decode
            →  vocab reverse-lookup  →  detokenised English string

        Args:
            sentence  : Raw German sentence (e.g. "Ein Hund läuft im Park.").
            device    : Torch device string ('cpu', 'cuda', 'mps').
            max_len   : Maximum number of target tokens to generate.

        Returns:
            Translated English sentence as a whitespace-joined string.
        """

        device = get_device()
        tokens = self.dataset.tokenize_de(text = sentence)  # e.g. ["Ein", "Hund", "läuft", "im", "Park", "."]

        # ── Encode to integer indices ──────────────────────────────────
        # Wrap with <sos> and <eos>, fall back to <unk> for OOV tokens.
        indices = (
            [self.src_vocab.sos_idx]
            + [self.src_vocab.lookup_index(t) for t in tokens]
            + [self.src_vocab.eos_idx]
        )

        src = torch.tensor([indices], dtype=torch.long, device=device)
        src_mask = make_src_mask(src, pad_idx=self.pad_idx).to(device)

        # ── Greedy decode ──────────────────────────────────────────────
        pred_tensor = greedy_decode(
            model        = self,
            src          = src,
            src_mask     = src_mask,
            max_len      = max_len,
            start_symbol = self.tgt_vocab.sos_idx,
            end_symbol   = self.tgt_vocab.eos_idx,
            device       = device,
        )   # [1, out_len]

        # ── Convert indices back to tokens, strip special tokens ───────
        specials = {self.tgt_vocab.pad_idx, self.tgt_vocab.sos_idx, self.tgt_vocab.eos_idx}
        out_tokens = [
            self.tgt_vocab.lookup_token(idx)
            for idx in pred_tensor.squeeze(0).tolist()
            if idx not in specials
        ]

        return " ".join(out_tokens)


# ══════════════════════════════════════════════════════════════════════
#  FACTORY
# ══════════════════════════════════════════════════════════════════════

def make_transformer(
    src_vocab_size: int = 7853,
    tgt_vocab_size: int = 5893,
    **kwargs,
) -> Transformer:
    """
    Convenience factory to instantiate a Transformer with sensible defaults
    for the Multi30k task.

    Examples:
        model = make_transformer(src_vocab_size, tgt_vocab_size)
        model = make_transformer(src_vocab_size, tgt_vocab_size, N=6, d_model=512)
    """
    defaults = dict(d_model=256, N=5, num_heads=8, d_ff=512, dropout=0.1)
    defaults.update(kwargs)

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        **defaults,
    )

    return model


# ══════════════════════════════════════════════════════════════════════
#  QUICK SANITY CHECK
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    B, SRC_LEN, TGT_LEN = 4, 20, 18
    SRC_VOCAB, TGT_VOCAB = 7853, 5893

    src = torch.randint(1, SRC_VOCAB, (B, SRC_LEN))
    tgt = torch.randint(1, TGT_VOCAB, (B, TGT_LEN))

    model  = make_transformer(SRC_VOCAB, TGT_VOCAB)
    logits = model(src, tgt)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"logits: {tuple(logits.shape)}  params: {nparams:,}")