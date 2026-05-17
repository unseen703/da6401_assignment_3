import math
from typing import Optional, Tuple 
import torch
import torch.nn as nn
import torch.nn.functional as F 

def get_device() -> str:
    """Pick the best available torch device for training or inference."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LabelSmoothingLoss(nn.Module):
    """
    Label-smoothed cross-entropy loss.

    Instead of training against a hard one-hot target, we use a
    *smoothed* distribution:

        y_smooth[i] = (1 − ε) * one_hot[i]  +  ε / (V − 1)   for i ≠ pad
        y_smooth[pad_idx]  = 0                                  (always)

    This prevents the model from becoming over-confident (soft regulariser)
    while slightly increasing training perplexity — consistent with the
    W&B ablation required in Section 2.5.

    Why KLDivLoss?
        nn.KLDivLoss(reduction='sum') computes Σ p * log(p/q)
        which equals Σ p * (log p − log q).
        Since Σ p * log p is constant w.r.t. model parameters,
        minimising KL-divergence is equivalent to minimising cross-entropy
        against the smoothed targets.  Using KL lets us pass a soft target
        distribution as a tensor rather than hard integer indices.

    Args:
        vocab_size : Total number of output tokens (V).
        pad_idx    : Index of <pad> token — always assigned 0 probability.
        smoothing  : Smoothing factor ε (paper uses 0.1).
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx:    int,
        smoothing:  float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

        # KLDivLoss expects log-probabilities from the model side
        # and true probabilities on the target side.
        self.criterion = nn.KLDivLoss(reduction="sum")

    def forward(
        self,
        logits: torch.Tensor,   # [N, vocab_size]   raw model output
        target: torch.Tensor,   # [N]               gold token indices
    ) -> torch.Tensor:
        """
        Args:
            logits : [B * tgt_len, vocab_size]  — flattened raw logits
            target : [B * tgt_len]              — flattened gold indices

        Returns:
            Scalar loss (mean over non-pad tokens).
        """
        N = target.size(0)

        # Build smoothed target distribution: [N, vocab_size]
        smooth_targets = torch.full(
            (N, self.vocab_size),
            fill_value = self.smoothing / (self.vocab_size - 1),
            device     = logits.device,
        )
        # Place confidence mass on the correct token
        smooth_targets.scatter_(1, target.unsqueeze(1), self.confidence)
        # Zero out <pad> positions — they contribute no signal
        smooth_targets[:, self.pad_idx] = 0.0

        # Also zero entire rows where the target itself is <pad>
        pad_rows = (target == self.pad_idx)
        smooth_targets[pad_rows] = 0.0

        # KLDivLoss expects log-probabilities on the input side
        log_probs = F.log_softmax(logits, dim=-1)

        loss = self.criterion(log_probs, smooth_targets)

        # Normalise by the number of real (non-pad) tokens
        num_tokens = (~pad_rows).sum().clamp(min=1)
        return loss / num_tokens

# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS  (autograder contract)
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Build a padding mask for the encoder: positions with <pad> tokens
    get masked out (set to 0), all others remain unmasked (set to 1).

    Args:
        src     : Token indices  [B, src_len]
        pad_idx : Index of the <pad> token in the vocabulary.

    Returns:
        mask : [B, 1, 1, src_len]  — 1 where real token, 0 where pad.
    """
    # (B, 1, 1, src_len)  — broadcast across heads and query positions
    mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Build a combined padding + look-ahead (causal) mask for the decoder.
    Position i may attend only to positions ≤ i (prevents peeking at
    future tokens during training).

    Args:
        tgt     : Token indices  [B, tgt_len]
        pad_idx : Index of the <pad> token (default 0).

    Returns:
        mask : [B, 1, tgt_len, tgt_len]
               Entry (b, 0, i, j) is 1 iff position i can attend to j.
    """
    B, T = tgt.shape

    # Causal / look-ahead mask: upper triangle (excluding diagonal) = 0
    causal = torch.tril(torch.ones(T, T, device=tgt.device)).bool()  # [T, T]

    # Padding mask: positions that are <pad> cannot be attended to
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]

    # Combine: both conditions must hold
    mask = pad_mask & causal.unsqueeze(0).unsqueeze(0)  # [B, 1, T, T]

    return mask


# ══════════════════════════════════════════════════════════════════════
#  ① STANDARD SCALED DOT-PRODUCT ATTENTION
#     Vaswani et al. 2017  "Attention Is All You Need"
# ══════════════════════════════════════════════════════════════════════

class Attention(nn.Module):
    """
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    The scaling factor 1/sqrt(d_k) prevents dot products from growing
    large and pushing softmax into low-gradient saturation zones
    (Section 3.2.1 of the paper).
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        #  Store the mask for visualization
        self.attn_w: Optional[torch.Tensor] = None

    def forward(
        self,
        q: torch.Tensor,          # [B, H, T_q, d_k]
        k: torch.Tensor,          # [B, H, T_k, d_k]
        v: torch.Tensor,          # [B, H, T_k, d_k]
        mask: Optional[torch.Tensor] = None,  # broadcastable to [B, H, T_q, T_k]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output : [B, H, T_q, d_v]
            attn_w : [B, H, T_q, T_k]
        """
        d_k   = q.size(-1)
        scale = math.sqrt(d_k)

        # Raw attention scores: [B, H, T_q, T_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            # mask shape: [B, 1, 1, src_len]  or  [B, 1, T, T]
            # scores shape: [B, H, T_q, T_k]
            # Broadcasting: mask expands over the H dimension automatically.
            # Positions where mask == 0 (pad or future tokens) are set to
            # -1e9 so softmax drives their weight to ≈ 0.
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_w = F.softmax(scores, dim=-1)
        self.attn_w = attn_w.detach()
        attn_w = self.dropout(attn_w)

        output = torch.matmul(attn_w, v)          # [B, H, T_q, d_v]

        return output, self.attn_w


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Vaswani et al. 2017, Section 3.2.2).

    Splits d_model into `num_heads` parallel heads of dimension d_k = d_model / num_heads,
    applies scaled dot-product attention per head, then concatenates and projects back.

    Args:
        d_model   : Model / embedding dimension.
        num_heads : Number of parallel attention heads.
        dropout   : Dropout applied inside attention.
    """

    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        dropout:   float = 0.0
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        # Linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Instantiate scaled dot-product attention
        self.attention = Attention(dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape  [B, T, d_model]  →  [B, H, T, d_k]
        """
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape  [B, H, T, d_k]  →  [B, T, d_model]
        """
        B, H, T, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * d_k)

    def forward(
        self,
        query:  torch.Tensor,
        key:    torch.Tensor,
        value:  torch.Tensor,
        mask:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query  : [B, T_q, d_model]
            key    : [B, T_k, d_model]
            value  : [B, T_k, d_model]
            mask   : [B, 1, 1, T_k]  (src padding mask)   or
                     [B, 1, T_q, T_k] (tgt causal+padding mask).
                     0 = ignore this position, 1 = attend normally.

        Returns:
            output : [B, T_q, d_model]

        Masking note
        ────────────
        The mask is passed straight through to Attention.forward where it
        is broadcast over the H (heads) dimension.  Because we unsqueeze(1)
        when building the masks (giving dim-1 = 1), PyTorch broadcasts it
        across all H heads without any extra reshape here.  Correct shapes:

            src_mask : [B, 1, 1, src_len]  →  broadcast → [B, H, T_q, src_len]
            tgt_mask : [B, 1, T,  T      ]  →  broadcast → [B, H, T,   T      ]
        """
        Q = self._split_heads(self.W_q(query))   # [B, H, T_q, d_k]
        K = self._split_heads(self.W_k(key))     # [B, H, T_k, d_k]
        V = self._split_heads(self.W_v(value))   # [B, H, T_k, d_k]

        attention_output = self.attention(Q, K, V, mask=mask)
        if isinstance(attention_output, tuple):
            context, _ = attention_output
        else:
            context = attention_output

        output = self._merge_heads(context)       # [B, T_q, d_model]
        output = self.W_o(output)
        return output


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
#  Sinusoidal, registered as a non-trainable buffer (autograder tests this)
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (Vaswani et al. 2017, Section 3.5).

        PE(pos, 2i)   = sin(pos / 10000^{2i / d_model})
        PE(pos, 2i+1) = cos(pos / 10000^{2i / d_model})

    The encoding is *added* to the token embedding and stored as a
    buffer (not a learnable parameter) so it is saved with the model but
    not updated by the optimiser.  The autograder verifies this.
    """

    def __init__(
        self,
        d_model:  int,
        dropout:  float = 0.1,
        max_len:  int   = 5000,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build [max_len, d_model] sinusoidal table
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)   # [L, 1]
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )                                                                 # [d_model/2]

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        pe = pe.unsqueeze(0)   # [1, max_len, d_model]
        self.register_buffer("pe", pe)   # ← buffer, not parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, T, d_model]
        Returns:
            x + positional encoding  (same shape)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING  (autograder contract)
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model:        "Transformer",
    src:          torch.Tensor,
    src_mask:     torch.Tensor,
    max_len:      int,
    start_symbol: int,
    end_symbol:   int,
    device:       str = "cpu",
) -> torch.Tensor:
    """
    Autoregressive greedy decoding — generates one token per step by
    always picking argmax of the output distribution.

    Algorithm:
        1. Encode the source sequence once.
        2. Initialise decoder input ys = [<sos>].
        3. At each step, run decoder, take the last position's logit,
           argmax → next token.
        4. Append next token to ys; stop at <eos> or max_len.

    Args:
        model        : Trained Transformer (call model.eval() before this).
        src          : [1, src_len]  source token indices (single sentence).
        src_mask     : [1, 1, 1, src_len]  padding mask for src.
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : Inference device.

    Returns:
        ys : [1, out_len]  generated token indices, including <sos>.
             Stops when <eos> is produced or max_len is reached.
    """
    model.eval()

    # Derive pad_idx from the model so make_tgt_mask masks correctly even
    # if the decoded sequence ever contains a pad token index by coincidence.
    pad_idx = getattr(model, "pad_idx", 0)

    with torch.no_grad():
        # ── Encode source (done once) ──────────────────────────────────
        enc_out = model.encode(src, src_mask)    # [1, src_len, d_model]

        # ── Initialise decoder input with <sos> ───────────────────────
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            # Build causal mask for current decoder input length.
            # Pass pad_idx explicitly — consistent with training.
            tgt_mask = make_tgt_mask(ys, pad_idx=pad_idx).to(device)

            # One decoder forward step
            dec_out = model.decode(ys, enc_out, src_mask, tgt_mask)
            # dec_out : [1, cur_len, d_model]

            # Project to vocab and pick greedy argmax at last position
            logits     = model.output_proj(dec_out[:, -1, :])  # [1, vocab]
            next_token = logits.argmax(dim=-1, keepdim=True)   # [1, 1]

            ys = torch.cat([ys, next_token], dim=1)            # [1, cur_len+1]

            # Stop generation when <eos> is produced
            if next_token.item() == end_symbol:
                break

    return ys   # [1, out_len]


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES  (autograder contract)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model:     "Transformer",
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch:     int,
    path:      str = "checkpoint.pt",
) -> None:
    """
    Persist model + optimizer + scheduler state to disk.

    Saved dict structure:
        {
          'epoch'               : int,
          'model_state_dict'    : OrderedDict,
          'optimizer_state_dict': dict,
          'scheduler_state_dict': dict,
          'model_config'        : dict
        }
    """
    enc = model.encoder
    model_config = {
        "src_vocab_size": enc.embedding.num_embeddings,
        "tgt_vocab_size": model.decoder.embedding.num_embeddings,
        "d_model":        enc.embedding.embedding_dim,
        "N":              len(enc.layers),
        "num_heads":      enc.layers[0].self_attn.num_heads,
        "d_ff":           enc.layers[0].ff.linear1.out_features,
        "dropout":        enc.layers[0].self_attn.dropout.p,
        "pad_idx":        model.pad_idx,
    }

    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config":         model_config,
        },
        path,
    )
    print(f"  ✓ Checkpoint saved → {path}  (epoch {epoch})")


def load_checkpoint(
    path:      str,
    model:     "Transformer",
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler  = None,
) -> tuple:
    """
    Restore model (and optionally optimizer / scheduler) from a checkpoint.

    Returns:
        (model, optimizer, scheduler, epoch)
    """
    ckpt = torch.load(path, map_location="cpu")

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    epoch = ckpt.get("epoch", 0)
    print(f"  ✓ Checkpoint loaded ← {path}  (epoch {epoch})")
    return model, optimizer, scheduler, epoch