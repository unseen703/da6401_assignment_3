"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌──────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol,         │
  │                end_symbol, device)                                   │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                      │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                      │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └──────────────────────────────────────────────────────────────────────┘
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
import wandb
import gdown

from model       import Transformer, make_src_mask, make_tgt_mask, make_transformer
from lr_scheduler import NoamScheduler
from dataset     import build_dataloaders


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
#  Vaswani et al. 2017 Section 5.4  — ε_ls = 0.1
# ══════════════════════════════════════════════════════════════════════

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
#  TRAINING / EVALUATION EPOCH
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model:     Transformer,
    loss_fn:   nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler  = None,
    epoch_num: int  = 0,
    is_train:  bool = True,
    device:    str  = "cpu",
) -> float:
    """
    Run one full pass over the dataset.

    Teacher-forcing is used during training: the gold target sequence
    (shifted right by 1) is fed to the decoder at every step.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches [B, T].
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during evaluation).
        scheduler  : NoamScheduler (None during evaluation).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, compute gradients and update weights.
        device     : 'cpu', 'cuda', or 'mps'.

    Returns:
        avg_loss : Mean loss over all batches in the epoch.
    """
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    total_tokens = 0
    start_time   = time.time()

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch_idx, (src, tgt) in enumerate(data_iter):
            src = src.to(device)   # [B, src_len]
            tgt = tgt.to(device)   # [B, tgt_len]

            # ── Teacher forcing ───────────────────────────────────────
            # Input  to decoder : tgt[:, :-1]  (all tokens except <eos>)
            # Gold labels       : tgt[:, 1:]   (all tokens except <sos>)
            tgt_input = tgt[:, :-1]
            tgt_gold  = tgt[:, 1:]

            # ── Build masks ───────────────────────────────────────────
            src_mask = make_src_mask(src).to(device)
            tgt_mask = make_tgt_mask(tgt_input).to(device)

            # ── Forward pass ──────────────────────────────────────────
            logits = model(src, tgt_input, src_mask, tgt_mask)
            # logits : [B, tgt_len-1, vocab_size]

            # Flatten for loss computation
            B, T, V = logits.shape
            loss = loss_fn(
                logits.reshape(B * T, V),
                tgt_gold.reshape(B * T),
            )
            time.sleep(0.01) 
            # ── Backward pass ─────────────────────────────────────────
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            # ── Accumulate stats ──────────────────────────────────────
            num_tokens   = (tgt_gold != 0).sum().item()
            total_loss  += loss.item() * num_tokens
            total_tokens += num_tokens

            # ── W&B step-level logging (training only) ────────────────
            if is_train and wandb.run is not None:
                current_lr = (
                    optimizer.param_groups[0]["lr"] if optimizer else 0.0
                )
                wandb.log({
                    "train/step_loss": loss.item(),
                    "train/lr":        current_lr,
                    "train/step":      epoch_num * len(data_iter) + batch_idx,
                })

    elapsed    = time.time() - start_time
    avg_loss   = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))   # clamp to avoid overflow

    split_tag = "train" if is_train else "val"
    print(
        f"  [{split_tag}] epoch={epoch_num:02d}  "
        f"loss={avg_loss:.4f}  ppl={perplexity:.2f}  "
        f"time={elapsed:.1f}s"
    )

    # ── W&B epoch-level logging ───────────────────────────────────────
    if wandb.run is not None:
        wandb.log({
            f"{split_tag}/loss":       avg_loss,
            f"{split_tag}/perplexity": perplexity,
            "epoch":                   epoch_num,
        })

    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING  (autograder contract)
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model:        Transformer,
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

    with torch.no_grad():
        # ── Encode source (done once) ──────────────────────────────────
        enc_out = model.encode(src, src_mask)    # [1, src_len, d_model]

        # ── Initialise decoder input with <sos> ───────────────────────
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            # Build causal mask for current decoder input length
            tgt_mask = make_tgt_mask(ys).to(device)

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
#  BLEU EVALUATION  (autograder contract)
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model:            Transformer,
    test_dataloader:  DataLoader,
    tgt_vocab,
    device:           str = "cpu",
    max_len:          int = 100,
) -> float:
    """
    Compute corpus-level BLEU score on the test set using greedy decoding.

    BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between
    machine-generated and human reference translations, with a brevity
    penalty for short outputs.  Range: 0 – 100 (higher = better).

    Args:
        model           : Trained Transformer (will be set to eval mode).
        test_dataloader : DataLoader over test split; yields (src, tgt) batches.
        tgt_vocab       : Vocab object with .sos_idx, .eos_idx, .pad_idx,
                          and .lookup_token(idx) method.
        device          : Inference device.
        max_len         : Maximum tokens to generate per sentence.

    Returns:
        bleu_score : Corpus-level BLEU × 100 (float, 0–100).
    """
    from torchtext.data.metrics import bleu_score as torchtext_bleu

    model.eval()

    sos_idx = tgt_vocab.sos_idx
    eos_idx = tgt_vocab.eos_idx
    pad_idx = tgt_vocab.pad_idx

    all_hypotheses  = []   # list of token lists (predicted)
    all_references  = []   # list of list of token lists (reference; one ref each)

    with torch.no_grad():
        for src_batch, tgt_batch in test_dataloader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            for i in range(src_batch.size(0)):
                # Single-sentence decode
                src      = src_batch[i].unsqueeze(0)           # [1, src_len]
                src_mask = make_src_mask(src).to(device)

                # Run greedy decoding
                pred_indices = greedy_decode(
                    model        = model,
                    src          = src,
                    src_mask     = src_mask,
                    max_len      = max_len,
                    start_symbol = sos_idx,
                    end_symbol   = eos_idx,
                    device       = device,
                ).squeeze(0).tolist()   # [out_len]

                # Convert predicted indices → tokens (strip <sos>/<eos>/<pad>)
                hyp_tokens = [
                    tgt_vocab.lookup_token(idx)
                    for idx in pred_indices
                    if idx not in {sos_idx, eos_idx, pad_idx}
                ]

                # Convert reference indices → tokens
                ref_indices = tgt_batch[i].tolist()
                ref_tokens  = [
                    tgt_vocab.lookup_token(idx)
                    for idx in ref_indices
                    if idx not in {sos_idx, eos_idx, pad_idx}
                ]

                all_hypotheses.append(hyp_tokens)
                all_references.append([ref_tokens])  # one reference per sentence

    # torchtext bleu_score signature:
    #   bleu_score(candidate_corpus, references_corpus, max_n, weights)
    #   → float in [0, 1]
    score = torchtext_bleu(
        candidate_corpus   = all_hypotheses,
        references_corpus  = all_references,
        max_n              = 4,
        weights            = [0.25, 0.25, 0.25, 0.25],
    )

    bleu = score * 100.0
    print(f"  BLEU score : {bleu:.2f}")
    return bleu


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES  (autograder contract)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model:     Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch:     int,
    path:      str = "checkpoint.pt",
) -> None:
    """
    Persist model + optimizer + scheduler state to disk.

    The autograder calls load_checkpoint() with the saved file to
    reconstruct and evaluate the model — do NOT change the dict keys.

    Saved dict structure:
        {
          'epoch'               : int,
          'model_state_dict'    : OrderedDict,
          'optimizer_state_dict': dict,
          'scheduler_state_dict': dict,
          'model_config'        : dict   # all kwargs for Transformer(**cfg)
        }
    """
    # Gather all constructor arguments needed to rebuild the model
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
    model:     Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler  = None,
) -> int:
    """
    Restore model (and optionally optimizer / scheduler) from a checkpoint.

    Args:
        path      : Path to .pt file saved by save_checkpoint().
        model     : Transformer instance with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved.
    """
    ckpt = torch.load(path, map_location="cpu")

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    epoch = ckpt.get("epoch", 0)
    print(f"  ✓ Checkpoint loaded ← {path}  (epoch {epoch})")
    return model, optimizer,scheduler, epoch


# ══════════════════════════════════════════════════════════════════════
#  INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════

def get_device() -> str:
    """Pick the best available torch device for training or inference."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def infer_model_config_from_checkpoint(ckpt: dict) -> dict:
    """
    Reconstruct Transformer constructor kwargs from a checkpoint.

    Older checkpoints in this repo save the core model dimensions but not
    `attn_type`, `max_len`, or relative-attention kwargs.  Infer those from
    state-dict tensor names/shapes so inference can load the best checkpoint
    without requiring manual architecture flags.
    """
    cfg = dict(ckpt["model_config"])
    state = ckpt["model_state_dict"]

    cfg.setdefault("max_len", state["encoder.pos_enc.pe"].shape[1])

    if "attn_type" not in cfg:
        if any(".attention.rel_emb.weight" in key for key in state):
            cfg["attn_type"] = "relative"
        elif any(".attention.rotary." in key for key in state):
            cfg["attn_type"] = "rotary"
        else:
            cfg["attn_type"] = "standard"

    if cfg["attn_type"] == "relative" and "max_rel_dist" not in cfg:
        rel_key = next(
            key for key in state if key.endswith(".attention.rel_emb.weight")
        )
        cfg["max_rel_dist"] = (state[rel_key].shape[0] - 1) // 2

    if cfg["attn_type"] == "rotary" and "max_seq_len" not in cfg:
        cfg["max_seq_len"] = cfg["max_len"]

    return cfg


def build_model_from_checkpoint(path: str, device: str, model) -> tuple[Transformer, int, dict]:
    """
    Load checkpoint metadata, instantiate the matching Transformer, and
    restore its weights.
    """
    ckpt = torch.load(path, map_location="cpu")
    model_cfg = infer_model_config_from_checkpoint(ckpt)

    
    epoch = load_checkpoint(path, model)
    model.to(device)
    model.eval()
    return model, epoch, model_cfg


def decode_indices(indices: list[int], vocab, skip_specials: bool = True) -> str:
    """Convert token ids back to a whitespace-joined sentence."""
    specials = {vocab.pad_idx, vocab.sos_idx, vocab.eos_idx}
    tokens = [
        vocab.lookup_token(idx)
        for idx in indices
        if not skip_specials or idx not in specials
    ]
    return " ".join(tokens)


def translate_tensor(
    model: Transformer,
    src: torch.Tensor,
    tgt_vocab,
    device: str,
    max_len: int = 100,
) -> list[int]:
    """Greedy-decode a single already-tokenised source tensor."""
    src = src.unsqueeze(0).to(device) if src.dim() == 1 else src.to(device)
    src_mask = make_src_mask(src).to(device)
    pred = greedy_decode(
        model        = model,
        src          = src,
        src_mask     = src_mask,
        max_len      = max_len,
        start_symbol = tgt_vocab.sos_idx,
        end_symbol   = tgt_vocab.eos_idx,
        device       = device,
    )
    return pred.squeeze(0).tolist()


def run_inference_experiment(config: dict = None) -> None:
    """
    Load the best checkpoint and run greedy-decoding inference on Multi30k.

    By default this prints a few test-set translations and then computes BLEU
    over the full test loader.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    ckpt_path = cfg.get("inference_ckpt") or cfg["best_ckpt"]

    print("\n══ Building dataset for inference ══")
    _, _, test_loader, src_vocab, tgt_vocab = build_dataloaders(
        batch_size   = cfg["batch_size"],
        src_min_freq = cfg["src_min_freq"],
        tgt_min_freq = cfg["tgt_min_freq"],
    )

    device = get_device()
    print(f"Device : {device}")

    print(f"\n══ Loading best checkpoint: {ckpt_path} ══")
    model, epoch, model_cfg = build_model_from_checkpoint(ckpt_path, device)
    print(
        "Model : "
        f"attn_type={model_cfg['attn_type']}  "
        f"d_model={model_cfg['d_model']}  "
        f"N={model_cfg['N']}  "
        f"epoch={epoch}"
    )

    print("\n══ Sample translations ══")
    max_len = cfg.get("inference_max_len", 100)
    num_samples = cfg.get("inference_samples", 5)
    shown = 0

    with torch.no_grad():
        for src_batch, tgt_batch in test_loader:
            for i in range(src_batch.size(0)):
                src_ids = src_batch[i].tolist()
                ref_ids = tgt_batch[i].tolist()
                pred_ids = translate_tensor(
                    model    = model,
                    src      = src_batch[i],
                    tgt_vocab= tgt_vocab,
                    device   = device,
                    max_len  = max_len,
                )

                print(f"\n[{shown + 1}]")
                print(f"SRC : {decode_indices(src_ids, src_vocab)}")
                print(f"REF : {decode_indices(ref_ids, tgt_vocab)}")
                print(f"PRED: {decode_indices(pred_ids, tgt_vocab)}")

                shown += 1
                if shown >= num_samples:
                    break
            if shown >= num_samples:
                break

    if cfg.get("compute_bleu", True):
        print("\n══ BLEU on test set ══")
        try:
            evaluate_bleu(
                model           = model,
                test_dataloader = test_loader,
                tgt_vocab       = tgt_vocab,
                device          = device,
                max_len         = max_len,
            )
        except Exception as exc:
            print(f"  BLEU skipped: {exc}")


# ══════════════════════════════════════════════════════════════════════
#  ATTENTION MAP VISUALISATION  (for W&B Report Section 2.3)
# ══════════════════════════════════════════════════════════════════════

def log_attention_maps(
    model:     Transformer,
    src:       torch.Tensor,
    src_vocab,
    tgt_vocab,
    device:    str = "cpu",
    layer_idx: int = -1,
) -> None:
    """
    Extract and log attention weight heatmaps for one source sentence.

    Logs one W&B Image per attention head in the specified encoder layer.
    Use this during training (e.g., at end of each epoch) to observe
    head specialisation (Section 2.3 of the report).

    Args:
        model     : Trained Transformer in eval mode.
        src       : [1, src_len]  single tokenised source sentence.
        src_vocab : Source Vocab for decoding token labels.
        tgt_vocab : Target Vocab (needed for greedy decode labels).
        device    : Device.
        layer_idx : Which encoder layer to visualise (-1 = last).
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")   # non-interactive backend for W&B logging

    model.eval()
    src = src.to(device)
    src_mask = make_src_mask(src).to(device)

    encoder_layers = model.encoder.layers
    if not encoder_layers:
        print("  Attention maps skipped: encoder has no layers.")
        return

    resolved_layer_idx = layer_idx if layer_idx >= 0 else len(encoder_layers) + layer_idx
    if resolved_layer_idx < 0 or resolved_layer_idx >= len(encoder_layers):
        print(f"  Attention maps skipped: layer_idx={layer_idx} is out of range.")
        return

    target_attn = encoder_layers[resolved_layer_idx].self_attn
    captured_attn = {}

    def capture_attention(_module, inputs, kwargs, _output):
        q, k, _v = inputs[:3]
        mask = kwargs.get("mask", inputs[3] if len(inputs) > 3 else None)
        d_k = q.size(-1)

        if target_attn.attention.__class__.__name__ == "RotaryAttention":
            q = target_attn.attention.rope(q)
            k = target_attn.attention.rope(k)

        scores = torch.matmul(q, k.transpose(-2, -1))

        if target_attn.attention.__class__.__name__ == "RelativePositionAttention":
            T_q, T_k = q.size(2), k.size(2)
            rel_idx = target_attn.attention._relative_position_indices(T_q, T_k, q.device)
            rel_bias = target_attn.attention.rel_emb(rel_idx)
            rel_scores = torch.einsum("bhqd,qkd->bhqk", q, rel_bias)
            scores = scores + rel_scores

        scores = scores / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        captured_attn["weights"] = F.softmax(scores, dim=-1).detach()

    hook = target_attn.attention.register_forward_hook(capture_attention, with_kwargs=True)
    try:
        with torch.no_grad():
            model.encode(src, src_mask)
    finally:
        hook.remove()

    if "weights" not in captured_attn:
        print("  Attention maps skipped: no attention weights were captured.")
        return

    attn = captured_attn["weights"].squeeze(0).cpu().numpy()    # [H, src_len, src_len]
    H    = attn.shape[0]

    # Decode token labels for axis ticks
    src_tokens = [
        src_vocab.lookup_token(idx)
        for idx in src.squeeze(0).tolist()
        if idx not in {src_vocab.pad_idx}
    ]

    fig, axes = plt.subplots(2, H // 2, figsize=(H * 2, 6))
    axes      = axes.flatten()

    for h in range(H):
        ax = axes[h]
        im = ax.imshow(attn[h, :len(src_tokens), :len(src_tokens)],
                       cmap="viridis", aspect="auto")
        ax.set_title(f"Head {h + 1}", fontsize=9)
        ax.set_xticks(range(len(src_tokens)))
        ax.set_yticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(src_tokens, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f"Encoder Layer {resolved_layer_idx} — Attention Heads", fontsize=11)
    plt.tight_layout()

    if wandb.run is not None:
        wandb.log({"attention_maps": wandb.Image(fig)})

    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # Model architecture
    "d_model":      256,
    "N":            3,
    "num_heads":    8,
    "d_ff":         512,
    "dropout":      0.1,
    "max_len":      256,
    "attn_type":    "standard",   # 'standard' | 'relative' | 'rotary' | 'linear'

    # Training
    "batch_size":   128,
    "num_epochs":   15,
    "warmup_steps": 4000,
    "label_smooth": 0.1,

    # Data
    "src_min_freq": 2,
    "tgt_min_freq": 2,

    # Checkpoint
    "ckpt_path":    "checkpoint.pt",
    "best_ckpt":    "best_checkpoint.pt",
}


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment(config: dict = None) -> None:
    """
    Full end-to-end training experiment with W&B logging.

    Steps:
        1.  Init W&B
        2.  Build dataset / vocabs
        3.  Create DataLoaders  (train / val / test)
        4.  Instantiate Transformer
        5.  Instantiate Adam optimizer  (β₁=0.9, β₂=0.98, ε=1e-9)
        6.  Instantiate NoamScheduler
        7.  Instantiate LabelSmoothingLoss
        8.  Training loop with val evaluation + checkpointing
        9.  Final BLEU on held-out test set
        10. Log best checkpoint as W&B Artifact

    Args:
        config : Optional dict to override DEFAULT_CONFIG values.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # ── 1. W&B Init ───────────────────────────────────────────────────
    run = wandb.init(
        project = "da6401-a3",
        config  = cfg,
        tags    = [cfg["attn_type"], "transformer", "multi30k"],
    )
    cfg = dict(wandb.config)   # let W&B sweeps override config values

    # ── 2 & 3. Dataset & DataLoaders ──────────────────────────────────
    print("\n══ Building dataset ══")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(
        batch_size   = cfg["batch_size"],
        src_min_freq = cfg["src_min_freq"],
        tgt_min_freq = cfg["tgt_min_freq"],
    )

    # ── Device ────────────────────────────────────────────────────────
    device = get_device()
    print(f"Device : {device}")

    # ── 4. Model ──────────────────────────────────────────────────────
    print(f"\n══ Building Transformer (attn_type={cfg['attn_type']}) ══")
    attn_kwargs = {}
    if cfg["attn_type"] == "rotary":
        attn_kwargs["max_seq_len"] = cfg["max_len"]
    if cfg["attn_type"] == "relative":
        attn_kwargs["max_rel_dist"] = 32

    model = make_transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        attn_type      = cfg["attn_type"],
        d_model        = cfg["d_model"],
        N              = cfg["N"],
        num_heads      = cfg["num_heads"],
        d_ff           = cfg["d_ff"],
        dropout        = cfg["dropout"],
        max_len        = cfg["max_len"],
        pad_idx        = tgt_vocab.pad_idx,
        **attn_kwargs,
    ).to(device)

    try:
        gdown.download(id="1n4xSZDXPk7u_192-0jNnhvN-kRPcOXnX", output=cfg["best_ckpt"], quiet=False)

        model,_,_,_ = load_checkpoint(cfg['best_ckpt'], model)
        print(f"  ✓ Loaded existing checkpoint from {cfg['best_ckpt']}")
    except FileNotFoundError:
        print(f"  No existing checkpoint found at {cfg['best_ckpt']}. Starting fresh.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters : {n_params:,}")
    wandb.log({"model/n_params": n_params})

    # ── 5. Optimizer ──────────────────────────────────────────────────
    # Paper Section 5.3: β₁=0.9, β₂=0.98, ε=1e-9
    # Set lr=1.0 so the scheduler's scale IS the effective learning rate
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr     = 1.0,
        betas  = (0.9, 0.98),
        eps    = 1e-9,
    )

    # ── 6. Noam Scheduler ─────────────────────────────────────────────
    scheduler = NoamScheduler(
        optimizer,
        d_model      = cfg["d_model"],
        warmup_steps = cfg["warmup_steps"],
    )


    # ── 7. Loss Function ──────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size = len(tgt_vocab),
        pad_idx    = tgt_vocab.pad_idx,
        smoothing  = cfg["label_smooth"],
    )

    # ── 8. Training Loop ──────────────────────────────────────────────
    best_val_loss = float("inf")
    print("\n══ Training ══")

    for epoch in range(cfg["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['num_epochs']}")

        # Training pass
        train_loss = run_epoch(
            data_iter  = train_loader,
            model      = model,
            loss_fn    = loss_fn,
            optimizer  = optimizer,
            scheduler  = scheduler,
            epoch_num  = epoch,
            is_train   = True,
            device     = device,
        )
        time.sleep(0.7)  # small pause to ensure W&B logs are sent before validation
        
        # Validation pass (no gradient updates)
        val_loss = run_epoch(
            data_iter  = val_loader,
            model      = model,
            loss_fn    = loss_fn,
            optimizer  = None,
            scheduler  = None,
            epoch_num  = epoch,
            is_train   = False,
            device     = device,
        )
        
        time.sleep(1)

        # ── Save latest checkpoint ─────────────────────────────────────
        save_checkpoint(model, optimizer, scheduler, epoch, cfg["ckpt_path"])

        # ── Save best checkpoint ───────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, cfg["best_ckpt"])
            print(f"  ★ New best val loss: {best_val_loss:.4f}")

        # ── Log attention maps every 5 epochs ─────────────────────────
        if epoch % 5 == 0:
            sample_src, _ = next(iter(val_loader))
            log_attention_maps(
                model     = model,
                src       = sample_src[:1].to(device),
                src_vocab = src_vocab,
                tgt_vocab = tgt_vocab,
                device    = device,
            )

    # ── 9. Final BLEU on test set ─────────────────────────────────────
    print("\n══ Loading best checkpoint for BLEU evaluation ══")
    load_checkpoint(cfg["best_ckpt"], model)
    model.to(device)

    print("══ Evaluating on test set ══")
    bleu = evaluate_bleu(
        model           = model,
        test_dataloader = test_loader,
        tgt_vocab       = tgt_vocab,
        device          = device,
        max_len         = 100,
    )
    wandb.log({"test/bleu": bleu})

    # ── 10. Log best checkpoint as W&B Artifact ───────────────────────
    artifact = wandb.Artifact("best-transformer", type="model")
    artifact.add_file(cfg["best_ckpt"])
    run.log_artifact(artifact)

    print(f"\n══ Done. Test BLEU: {bleu:.2f} ══")
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  ABLATION RUNNERS  (for W&B Report Sections 2.1, 2.2, 2.4, 2.5)
# ══════════════════════════════════════════════════════════════════════

def run_ablation_fixed_lr() -> None:
    """Section 2.1 — Fixed LR baseline (no warm-up)."""
    config = {**DEFAULT_CONFIG, "attn_type": "standard"}

    train_loader, val_loader, _, src_vocab, tgt_vocab = build_dataloaders(
        batch_size=config["batch_size"]
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = make_transformer(len(src_vocab), len(tgt_vocab)).to(device)
    # Fixed LR: use a constant 1e-4 scheduler-free optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn   = LabelSmoothingLoss(len(tgt_vocab), tgt_vocab.pad_idx, 0.1)

    wandb.init(project="da6401-a3", name="fixed_lr_1e-4", config=config)

    for epoch in range(config["num_epochs"]):
        run_epoch(train_loader, model, loss_fn, optimizer, None,
                  epoch, is_train=True,  device=device)
        run_epoch(val_loader,   model, loss_fn, None,      None,
                  epoch, is_train=False, device=device)

    wandb.finish()


def run_ablation_no_scaling() -> None:
    """
    Section 2.2 — Attention without the 1/sqrt(d_k) scaling factor.

    To ablate this, we patch the ScaledDotProductAttention to skip
    the scale division and observe gradient norm explosion.
    """
    import model as model_module

    # Monkey-patch: remove scaling
    original_forward = model_module.ScaledDotProductAttention.forward

    def unscaled_forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1))   # NO /sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_w = torch.nn.functional.softmax(scores, dim=-1)
        return torch.matmul(attn_w, v)

    model_module.ScaledDotProductAttention.forward = unscaled_forward
    print("⚠ Attention scaling DISABLED for ablation.")

    # Run normal training for a few epochs
    run_training_experiment({**DEFAULT_CONFIG, "num_epochs": 3})

    # Restore original
    model_module.ScaledDotProductAttention.forward = original_forward
    print("Attention scaling restored.")


# ══════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DA6401 A3 — Transformer Training")
    parser.add_argument(
        "--mode",
        type    = str,
        default = "train",
        choices = ["train", "infer"],
        help    = "Run training or checkpoint inference.",
    )
    parser.add_argument(
        "--attn_type",
        type    = str,
        default = "standard",
        choices = ["standard", "relative", "rotary", "linear"],
        help    = "Attention mechanism to use.",
    )
    parser.add_argument("--d_model",      type=int,   default=256)
    parser.add_argument("--N",            type=int,   default=3)
    parser.add_argument("--num_heads",    type=int,   default=8)
    parser.add_argument("--d_ff",         type=int,   default=512)
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--num_epochs",   type=int,   default=15)
    parser.add_argument("--warmup_steps", type=int,   default=4000)
    parser.add_argument("--label_smooth", type=float, default=0.1)
    parser.add_argument(
        "--inference_ckpt",
        type    = str,
        default = None,
        help    = "Checkpoint to load for inference. Defaults to best_checkpoint.pt.",
    )
    parser.add_argument(
        "--inference_samples",
        type    = int,
        default = 5,
        help    = "Number of test-set examples to print during inference.",
    )
    parser.add_argument(
        "--inference_max_len",
        type    = int,
        default = 100,
        help    = "Maximum decoded target length during inference.",
    )
    parser.add_argument(
        "--skip_bleu",
        action  = "store_true",
        help    = "Only print sample translations; skip full test BLEU.",
    )
    parser.add_argument(
        "--ablation",
        type    = str,
        default = None,
        choices = ["fixed_lr", "no_scaling"],
        help    = "Run a specific ablation instead of the main experiment.",
    )
    args = parser.parse_args()
    cfg = vars(args)
    cfg["compute_bleu"] = not args.skip_bleu

    if args.mode == "infer":
        run_inference_experiment(cfg)
    elif args.ablation == "fixed_lr":
        run_ablation_fixed_lr()
    elif args.ablation == "no_scaling":
        run_ablation_no_scaling()
    else:
        run_training_experiment(cfg)
