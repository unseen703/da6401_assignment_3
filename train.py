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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
import wandb
import gdown

from model        import Transformer, make_transformer
from utils        import (
    LearnedPositionalEncoding,
    make_src_mask,
    make_tgt_mask,
    greedy_decode,
    save_checkpoint,
    load_checkpoint,
    LabelSmoothingLoss,
    get_device,
)
from lr_scheduler import NoamScheduler, get_lr_history, get_fixed_lr_history
from dataset      import build_dataloaders


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
    pad_idx:   int  = 0,
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
            src_mask = make_src_mask(src, pad_idx).to(device)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx).to(device)

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
    perplexity = math.exp(min(avg_loss, 100))

    split_tag = "train" if is_train else "val"
    print(
        f"  [{split_tag}] epoch={epoch_num:02d}  "
        f"loss={avg_loss:.4f}  ppl={perplexity:.2f}  "
        f"time={elapsed:.1f}s"
    )

    if wandb.run is not None:
        wandb.log({
            f"{split_tag}/loss":       avg_loss,
            f"{split_tag}/perplexity": perplexity,
            "epoch":                   epoch_num,
        })

    return avg_loss


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

    all_hypotheses  = []
    all_references  = []

    with torch.no_grad():
        for src_batch, tgt_batch in test_dataloader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            for i in range(src_batch.size(0)):
                src      = src_batch[i].unsqueeze(0)           # [1, src_len]
                src_mask = make_src_mask(src, pad_idx).to(device)

                pred_indices = greedy_decode(
                    model        = model,
                    src          = src,
                    src_mask     = src_mask,
                    max_len      = max_len,
                    start_symbol = sos_idx,
                    end_symbol   = eos_idx,
                    device       = device,
                ).squeeze(0).tolist()

                hyp_tokens = [
                    tgt_vocab.lookup_token(idx)
                    for idx in pred_indices
                    if idx not in {sos_idx, eos_idx, pad_idx}
                ]

                ref_indices = tgt_batch[i].tolist()
                ref_tokens  = [
                    tgt_vocab.lookup_token(idx)
                    for idx in ref_indices
                    if idx not in {sos_idx, eos_idx, pad_idx}
                ]

                all_hypotheses.append(hyp_tokens)
                all_references.append([ref_tokens])

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
#  INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════




def infer_model_config_from_checkpoint(ckpt: dict) -> dict:
    """
    Reconstruct Transformer constructor kwargs from a checkpoint.
    """
    cfg = dict(ckpt["model_config"])
    state = ckpt["model_state_dict"]
    cfg.setdefault("max_len", state["encoder.pos_enc.pe"].shape[1])
    return cfg


def build_model_from_checkpoint(path: str, device: str, model) -> tuple:
    """
    Load checkpoint metadata, instantiate the matching Transformer, and
    restore its weights.
    """
    ckpt = torch.load(path, map_location="cpu")
    model_cfg = infer_model_config_from_checkpoint(ckpt)
    model, _, _, epoch = load_checkpoint(path, model)
    model.to(device)
    model.eval()
    return model, epoch, model_cfg


def decode_indices(indices: list, vocab, skip_specials: bool = True) -> str:
    """Convert token ids back to a whitespace-joined sentence."""
    specials = {vocab.pad_idx, vocab.sos_idx, vocab.eos_idx}
    tokens = [
        vocab.lookup_token(idx)
        for idx in indices
        if not skip_specials or idx not in specials
    ]
    return " ".join(tokens)


def translate_tensor(
    model,
    src:      torch.Tensor,
    tgt_vocab,
    device:   str = "cpu",
    max_len:  int = 100,
) -> list:
    """Greedy-decode a single already-tokenised source tensor."""
    src = src.unsqueeze(0).to(device) if src.dim() == 1 else src.to(device)
    src_mask = make_src_mask(src, pad_idx=model.pad_idx).to(device)
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


    print(f"\n══ Loading best checkpoint: {ckpt_path} ══")
    # Build a blank model from checkpoint config, then restore weights
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = infer_model_config_from_checkpoint(ckpt)
    model = Transformer(**model_cfg).to(device)
    model, _, _, _ = load_checkpoint(ckpt_path, model)
    model.eval()
    print(
        f"Model : d_model={model_cfg['d_model']}  N={model_cfg['N']}"
    )

    print("\n══ Sample translations ══")
    max_len     = cfg.get("inference_max_len", 100)
    num_samples = cfg.get("inference_samples", 5)
    shown = 0

    with torch.no_grad():
        for src_batch, tgt_batch in test_loader:
            for i in range(src_batch.size(0)):
                src_ids  = src_batch[i].tolist()
                ref_ids  = tgt_batch[i].tolist()
                pred_ids = translate_tensor(
                    model     = model,
                    src       = src_batch[i],
                    tgt_vocab = tgt_vocab,
                    device    = device,
                    max_len   = max_len,
                )

                print(f"\n[{shown + 1}]")
                print(f"SRC : {decode_indices(src_ids,  src_vocab)}")
                print(f"REF : {decode_indices(ref_ids,  tgt_vocab)}")
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
    log_key:   str = "attention_maps",
    title:     str = None,
) -> None:
    """
    Extract and log attention weight heatmaps for one source sentence.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")

    model.eval()
    src = src.to(device)
    src_mask = make_src_mask(src, pad_idx=model.pad_idx).to(device)

    encoder_layers = model.encoder.layers
    if not encoder_layers:
        print("  Attention maps skipped: encoder has no layers.")
        return

    resolved_layer_idx = layer_idx if layer_idx >= 0 else len(encoder_layers) + layer_idx
    if resolved_layer_idx < 0 or resolved_layer_idx >= len(encoder_layers):
        print(f"  Attention maps skipped: layer_idx={layer_idx} is out of range.")
        return

    target_attn   = encoder_layers[resolved_layer_idx].self_attn
    captured_attn = {}

    def capture_attention(_module, inputs, kwargs, _output):
        q, k, _v = inputs[:3]
        mask = kwargs.get("mask", inputs[3] if len(inputs) > 3 else None)
        d_k = q.size(-1)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask , -1e9)

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

    attn = captured_attn["weights"].squeeze(0).cpu().numpy()
    H    = attn.shape[0]

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

    plot_title = title or f"Encoder Layer {resolved_layer_idx} — Attention Heads"
    plt.suptitle(plot_title, fontsize=11)
    plt.tight_layout()

    if wandb.run is not None:
        wandb.log({log_key: wandb.Image(fig)})

    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "d_model":      256,
    "N":            3,
    "num_heads":    8,
    "d_ff":         512,
    "dropout":      0.1,
    "max_len":      256,
    "batch_size":   128,
    "num_epochs":   15,
    "warmup_steps": 4000,
    "label_smooth": 0.1,
    "src_min_freq": 2,
    "tgt_min_freq": 2,
    "ckpt_path":    "checkpoint.pt",
    "best_ckpt":    "best_checkpoint.pt",
}


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment(config: dict = None) -> None:
    """
    Full end-to-end training experiment with W&B logging.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    run = wandb.init(
        project = "da6401-a3",
        config  = cfg,
        tags    = ["transformer", "multi30k"],
    )
    cfg = dict(wandb.config)

    print("\n══ Building dataset ══")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(
        batch_size   = cfg["batch_size"],
        src_min_freq = cfg["src_min_freq"],
        tgt_min_freq = cfg["tgt_min_freq"],
    )
  
    device = get_device()

    print("\n══ Building Transformer ══")
    print("cfg:", cfg)
    model = make_transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model        = cfg["d_model"],
        N              = cfg["N"],
        num_heads      = cfg["num_heads"],
        d_ff           = cfg["d_ff"],
        dropout        = cfg["dropout"],
        max_len        = cfg["max_len"],
        pad_idx        = tgt_vocab.pad_idx,
        src_vocab      = src_vocab,
        tgt_vocab      = tgt_vocab,
    ).to(device)

    print(model.infer("Ein Mann schläft auf einer Bank."))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters : {n_params:,}")
    wandb.log({"model/n_params": n_params})

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr    = 1.0,
        betas = (0.9, 0.98),
        eps   = 1e-9,
    )

    scheduler = NoamScheduler(
        optimizer,
        d_model      = cfg["d_model"],
        warmup_steps = cfg["warmup_steps"],
    )

    loss_fn = LabelSmoothingLoss(
        vocab_size = len(tgt_vocab),
        pad_idx    = tgt_vocab.pad_idx,
        smoothing  = cfg["label_smooth"],
    )

    best_val_loss = float("inf")
    print("\n══ Training ══")

    for epoch in range(cfg["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['num_epochs']}")

        train_loss = run_epoch(
            data_iter  = train_loader,
            model      = model,
            loss_fn    = loss_fn,
            optimizer  = optimizer,
            scheduler  = scheduler,
            epoch_num  = epoch,
            pad_idx    = tgt_vocab.pad_idx,
            is_train   = True,
            device     = device,
        )
        time.sleep(0.7)

        val_loss = run_epoch(
            data_iter  = val_loader,
            model      = model,
            loss_fn    = loss_fn,
            optimizer  = None,
            scheduler  = None,
            epoch_num  = epoch,
            is_train   = False,
            pad_idx    = tgt_vocab.pad_idx,
            device     = device,
        )
        time.sleep(1)

        save_checkpoint(model, optimizer, scheduler, epoch, cfg["ckpt_path"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, cfg["best_ckpt"])
            print(f"  ★ New best val loss: {best_val_loss:.4f}")

        if epoch % 5 == 0:
            sample_src, _ = next(iter(val_loader))
            log_attention_maps(
                model     = model,
                src       = sample_src[:1].to(device),
                src_vocab = src_vocab,
                tgt_vocab = tgt_vocab,
                device    = device,
            )

    print("\n══ Loading best checkpoint for BLEU evaluation ══")
    model, _, _, _ = load_checkpoint(cfg["best_ckpt"], model)
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

    artifact = wandb.Artifact("best-transformer", type="model")
    artifact.add_file(cfg["best_ckpt"])
    run.log_artifact(artifact)

    print(f"\n══ Done. Test BLEU: {bleu:.2f} ══")
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  ABLATION RUNNERS
# ══════════════════════════════════════════════════════════════════════

def run_ablation_fixed_lr() -> None:
    """Section 2.1 — Fixed LR baseline (no warm-up)."""
    config = {**DEFAULT_CONFIG}

    train_loader, val_loader, _, src_vocab, tgt_vocab = build_dataloaders(
        batch_size=config["batch_size"]
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model     = make_transformer(len(src_vocab), len(tgt_vocab), src_vocab=src_vocab, tgt_vocab=tgt_vocab).to(device)
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
    """
    import utils as utils_module

    original_forward = utils_module.Attention.forward

    def unscaled_forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1))   # NO /sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask , -1e9)
        attn_w = F.softmax(scores, dim=-1)

        return torch.matmul(attn_w, v), attn_w

    utils_module.Attention.forward = unscaled_forward
    print("⚠ Attention scaling DISABLED for ablation.")

    run_training_experiment({**DEFAULT_CONFIG, "num_epochs": 3})

    utils_module.Attention.forward = original_forward
    print("Attention scaling restored.")

# ══════════════════════════════════════════════════════════════════════
#  UTILITY — build a standard model + optimizer + loss
# ══════════════════════════════════════════════════════════════════════

def _build_experiment_stack(cfg, src_vocab, tgt_vocab, device, use_noam=True):
    """Shared boilerplate: model, optimizer, scheduler, loss_fn."""
    model = make_transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model        = cfg["d_model"],
        N              = cfg["N"],
        num_heads      = cfg["num_heads"],
        d_ff           = cfg["d_ff"],
        dropout        = cfg["dropout"],
        max_len        = cfg["max_len"],
        pad_idx        = tgt_vocab.pad_idx,
        src_vocab      = src_vocab,
        tgt_vocab      = tgt_vocab,
    ).to(device)

    if use_noam:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = NoamScheduler(
            optimizer,
            d_model      = cfg["d_model"],
            warmup_steps = cfg["warmup_steps"],
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("fixed_lr", 1e-4))
        scheduler = None

    loss_fn = LabelSmoothingLoss(
        vocab_size = len(tgt_vocab),
        pad_idx    = tgt_vocab.pad_idx,
        smoothing  = cfg.get("label_smooth", 0.1),
    )
    return model, optimizer, scheduler, loss_fn


# ══════════════════════════════════════════════════════════════════════
#  2.1 — NOAM SCHEDULER vs FIXED LEARNING RATE
# ══════════════════════════════════════════════════════════════════════

def run_experiment_21_noam_vs_fixed(config: dict = None) -> None:
    """
    Section 2.1 — Necessity of the Noam Scheduler.

    Trains two models for `num_epochs` epochs:
        • Noam schedule  (linear warm-up + inverse-sqrt decay)
        • Fixed LR       (constant 1e-4, no warm-up)

    Logs to W&B:
        • train/loss, val/loss per epoch for each run
        • LR trajectory comparison chart (matplotlib → wandb.Image)
        • Side-by-side loss overlay chart

    W&B Report talking points (auto-generated in run summary):
        • Why Transformers are sensitive to initial LR
        • How warm-up prevents early softmax saturation
    """
    cfg = {**DEFAULT_CONFIG, "num_epochs": 10, "fixed_lr": 1e-4, **(config or {})}
    device = get_device()

    # ── 0. Log LR trajectory comparison ──────────────────────────────
    wandb.init(project="da6401-a3", name="2.1_lr_schedule_comparison", config=cfg, reinit=True)

    total_steps  = cfg["num_epochs"] * 230          # ≈ batches per epoch on Multi30k
    noam_lrs     = get_lr_history(cfg["d_model"], cfg["warmup_steps"], total_steps)
    fixed_lrs    = get_fixed_lr_history(cfg["fixed_lr"], total_steps)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(noam_lrs,  label="Noam Scheduler",           color="steelblue", linewidth=2)
    ax.plot(fixed_lrs, label=f"Fixed LR ({cfg['fixed_lr']})", color="tomato",
            linestyle="--", linewidth=1.5)
    ax.axvline(cfg["warmup_steps"], color="gray", linestyle=":",
               label=f"warmup_steps = {cfg['warmup_steps']}", alpha=0.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Section 2.1 — LR Schedule Comparison")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    wandb.log({"2.1/lr_trajectory": wandb.Image(fig)})
    plt.close(fig)
    wandb.finish()

    # ── 1. Train with Noam ────────────────────────────────────────────
    train_loader, val_loader, _, src_vocab, tgt_vocab = build_dataloaders(
        batch_size   = cfg["batch_size"],
        src_min_freq = cfg["src_min_freq"],
        tgt_min_freq = cfg["tgt_min_freq"],
    )

    for run_name, use_noam in [("2.1_noam_scheduler", True), ("2.1_fixed_lr_1e-4", False)]:
        wandb.init(project="da6401-a3", name=run_name, config=cfg, reinit=True)
        model, optimizer, scheduler, loss_fn = _build_experiment_stack(
            cfg, src_vocab, tgt_vocab, device, use_noam=use_noam
        )

        for epoch in range(cfg["num_epochs"]):
            train_loss = run_epoch(
                train_loader, model, loss_fn, optimizer, scheduler,
                epoch_num=epoch, is_train=True, pad_idx=tgt_vocab.pad_idx, device=device,
            )
            val_loss = run_epoch(
                val_loader, model, loss_fn, None, None,
                epoch_num=epoch, is_train=False, pad_idx=tgt_vocab.pad_idx, device=device,
            )
            wandb.log({
                "epoch":      epoch,
                "train/loss": train_loss,
                "val/loss":   val_loss,
                "train/lr":   optimizer.param_groups[0]["lr"],
            })

        save_checkpoint(model, optimizer, scheduler or torch.optim.Adam(model.parameters()),
                        cfg["num_epochs"], f"{run_name}_best.pt")
        wandb.finish()

    print("\n✓ Section 2.1 complete. Two runs logged to W&B.")


# ══════════════════════════════════════════════════════════════════════
#  2.2 — SCALING FACTOR  1/√d_k  ABLATION
# ══════════════════════════════════════════════════════════════════════

def _make_gradient_norm_hooks(model: Transformer, max_steps: int, run_tag: str):
    """
    Register backward hooks on all Q- and K-projection weights in every
    encoder layer.  Accumulates per-step gradient L2 norms.

    Returns:
        grad_log : dict  { "Q_grad_norms": [...], "K_grad_norms": [...] }
        handles  : list of hook handles (call h.remove() after training)
    """
    grad_log = {"Q_grad_norms": [], "K_grad_norms": [], "step": []}
    step_counter = [0]
    handles = []

    for layer_idx, layer in enumerate(model.encoder.layers):
        def make_hook(which: str):
            def hook(grad):
                if step_counter[0] < max_steps:
                    norm = grad.detach().norm().item()
                    grad_log[f"{which}_grad_norms"].append(norm)
                    if which == "Q":          # count once per step
                        grad_log["step"].append(step_counter[0])
                        step_counter[0] += 1
                        wandb.log({
                            f"2.2_{run_tag}/Q_grad_norm": norm,
                            "step": step_counter[0],
                        })
                    else:
                        wandb.log({
                            f"2.2_{run_tag}/K_grad_norm": norm,
                            "step": step_counter[0],
                        })
            return hook

        h_q = layer.self_attn.W_q.weight.register_hook(make_hook("Q"))
        h_k = layer.self_attn.W_k.weight.register_hook(make_hook("K"))
        handles.extend([h_q, h_k])

    return grad_log, handles


def run_experiment_22_scaling_ablation(config: dict = None) -> None:
    """
    Section 2.2 — Ablation: The Scaling Factor 1/√d_k.

    Trains two models (3 epochs each) and logs:
        • Q / K gradient L2 norms for the first 1 000 gradient steps
        • train/loss and val/loss

    Disabling the scale is achieved by monkey-patching utils.Attention.forward
    for the "no_scale" run (same approach as run_ablation_no_scaling).
    """
    import utils as utils_module

    cfg = {**DEFAULT_CONFIG, "num_epochs": 3, **(config or {})}
    device = get_device()
    GRAD_LOG_STEPS = 1_000

    train_loader, val_loader, _, src_vocab, tgt_vocab = build_dataloaders(
        batch_size   = cfg["batch_size"],
        src_min_freq = cfg["src_min_freq"],
        tgt_min_freq = cfg["tgt_min_freq"],
    )

    original_forward = utils_module.Attention.forward

    for run_name, patch_scale in [("with_scale", False), ("no_scale", True)]:
        # Optionally disable scaling
        if patch_scale:
            def unscaled_forward(self, q, k, v, mask=None):
                scores = torch.matmul(q, k.transpose(-2, -1))   # no /sqrt(d_k)
                if mask is not None:
                    scores = scores.masked_fill(mask, -1e9)
                attn_w = F.softmax(scores, dim=-1)
                self.attn_w = attn_w.detach()
                attn_w = self.dropout(attn_w)
                return torch.matmul(attn_w, v), self.attn_w
            utils_module.Attention.forward = unscaled_forward
            print("⚠  Attention scaling DISABLED.")
        else:
            utils_module.Attention.forward = original_forward
            print("✓  Attention scaling ENABLED.")

        wandb.init(project="da6401-a3", name=f"2.2_{run_name}", config=cfg, reinit=True)
        model, optimizer, scheduler, loss_fn = _build_experiment_stack(
            cfg, src_vocab, tgt_vocab, device, use_noam=True
        )

        grad_log, hooks = _make_gradient_norm_hooks(model, GRAD_LOG_STEPS, run_name)

        for epoch in range(cfg["num_epochs"]):
            model.train()
            total_loss, total_tok = 0.0, 0
            for batch_idx, (src, tgt) in enumerate(train_loader):
                src, tgt = src.to(device), tgt.to(device)
                tgt_input, tgt_gold = tgt[:, :-1], tgt[:, 1:]
                src_mask = make_src_mask(src, tgt_vocab.pad_idx).to(device)
                tgt_mask = make_tgt_mask(tgt_input, tgt_vocab.pad_idx).to(device)

                logits = model(src, tgt_input, src_mask, tgt_mask)
                B, T, V = logits.shape
                loss = loss_fn(logits.reshape(B * T, V), tgt_gold.reshape(B * T))

                optimizer.zero_grad()
                loss.backward()              # hooks fire here
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()

                n_tok = (tgt_gold != 0).sum().item()
                total_loss += loss.item() * n_tok
                total_tok  += n_tok

            avg_loss = total_loss / max(total_tok, 1)
            val_loss = run_epoch(
                val_loader, model, loss_fn, None, None,
                epoch_num=epoch, is_train=False, pad_idx=tgt_vocab.pad_idx, device=device,
            )
            wandb.log({"epoch": epoch, "train/loss": avg_loss, "val/loss": val_loss})

        for h in hooks:
            h.remove()

        # ── Plot grad-norm comparison ─────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 4))
        steps = list(range(len(grad_log["Q_grad_norms"])))
        ax.plot(steps, grad_log["Q_grad_norms"], label="Q grad norm", color="steelblue")
        ax.plot(steps, grad_log["K_grad_norms"][:len(steps)],
                label="K grad norm", color="tomato")
        ax.set_xlabel("Training Step"); ax.set_ylabel("Gradient L2 Norm")
        ax.set_title(f"Section 2.2 — Q/K Gradient Norms ({run_name})")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        wandb.log({f"2.2/grad_norms_{run_name}": wandb.Image(fig)})
        plt.close(fig)
        wandb.finish()

    utils_module.Attention.forward = original_forward
    print("\n✓ Section 2.2 complete.")


# ══════════════════════════════════════════════════════════════════════
#  2.3 — ATTENTION ROLLOUT & HEAD SPECIALIZATION
# ══════════════════════════════════════════════════════════════════════

def run_experiment_23_attention_heads(
    model:     Transformer = None,
    src_vocab = None,
    tgt_vocab = None,
    n_samples: int = 3,
    device:    str = "cpu",
    config:    dict = None,
) -> None:
    """
    Section 2.3 — Attention Rollout & Head Specialization.

    For each of `n_samples` sentences from the validation set:
        • Logs encoder self-attention head heatmaps using log_attention_maps

    Logged to W&B:
        • 2.3/sample{sample_i}_layer{layer_i}_heads — grid of H head heatmaps

    Args:
        model     : Trained Transformer (already loaded best checkpoint).
        src_vocab : Source Vocab object.
        tgt_vocab : Target Vocab object (for pad_idx).
        n_samples : Number of validation sentences to visualise.
        device    : Inference device.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    _, val_loader, _, built_src_vocab, built_tgt_vocab = build_dataloaders(
        batch_size=cfg["batch_size"],
        src_min_freq=cfg["src_min_freq"],
        tgt_min_freq=cfg["tgt_min_freq"],
    )
    src_vocab = src_vocab or built_src_vocab
    tgt_vocab = tgt_vocab or built_tgt_vocab

    if model is None:
        device = get_device()
        model = make_transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=cfg["d_model"],
            N=cfg["N"],
            num_heads=cfg["num_heads"],
            d_ff=cfg["d_ff"],
            dropout=cfg["dropout"],
            max_len=cfg["max_len"],
            pad_idx=tgt_vocab.pad_idx,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
        ).to(device)
        try:
            model, _, _, _ = load_checkpoint(cfg.get("best_ckpt", "best_checkpoint.pt"), model)
        except Exception as e:
            print(f"  Warning: could not load checkpoint ({e}). Using random weights.")

    wandb.init(project="da6401-a3", name="2.3_attention_heads", config=cfg, reinit=True)
    model.eval()
    model.to(device)

    sample_count = 0
    for src_batch, _ in val_loader:
        for i in range(src_batch.size(0)):
            if sample_count >= n_samples:
                break

            src = src_batch[i].unsqueeze(0).to(device)    # [1, src_len]
            for layer_idx in range(len(model.encoder.layers)):
                log_attention_maps(
                    model=model,
                    src=src,
                    src_vocab=src_vocab,
                    tgt_vocab=tgt_vocab,
                    device=device,
                    layer_idx=layer_idx,
                    log_key=f"2.3/sample{sample_count+1}_layer{layer_idx}_heads",
                    title=(
                        f"Sample {sample_count + 1} | Encoder Layer {layer_idx} "
                        f"— Attention Heads"
                    ),
                )

            sample_count += 1

        if sample_count >= n_samples:
            break

    wandb.finish()
    print("\n✓ Section 2.3 complete.")


def _patch_model_with_learned_pe(model: Transformer, dropout: float, max_len: int):
    """
    Replace sinusoidal PositionalEncoding with LearnedPositionalEncoding
    in both encoder and decoder of `model` (in-place).
    """
    d_model = model.encoder.embedding.embedding_dim
    model.encoder.pos_enc = LearnedPositionalEncoding(d_model, dropout, max_len)
    model.decoder.pos_enc = LearnedPositionalEncoding(d_model, dropout, max_len)
    return model


# ══════════════════════════════════════════════════════════════════════
#  2.4 — POSITIONAL ENCODING vs LEARNED EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════

def run_experiment_24_positional_encoding(config: dict = None) -> None:
    """
    Section 2.4 — Positional Encoding vs. Learned Embeddings.

    Trains two models for `num_epochs` epochs:
        • Sinusoidal PE   (original, non-learnable)
        • Learned PE      (nn.Embedding, fully trainable)

    Logs to W&B:
        • train/loss, val/loss, val/bleu per epoch
        • Final test BLEU comparison bar chart
        • Theoretical note on extrapolation in run summary

    Deliverable for report:
        Discuss how sinusoidal encoding allows extrapolation to sequence
        lengths > max_len seen during training, while learned PE cannot.
    """
    cfg = {**DEFAULT_CONFIG, "num_epochs": 10, **(config or {})}
    device = get_device()

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(
        batch_size   = cfg["batch_size"],
        src_min_freq = cfg["src_min_freq"],
        tgt_min_freq = cfg["tgt_min_freq"],
    )

    results = {}

    for run_name, use_learned_pe in [
        ("2.4_sinusoidal_pe", False),
        ("2.4_learned_pe",    True),
    ]:
        wandb.init(project="da6401-a3", name=run_name, config=cfg, reinit=True)
        model, optimizer, scheduler, loss_fn = _build_experiment_stack(
            cfg, src_vocab, tgt_vocab, device, use_noam=True
        )

        if use_learned_pe:
            model = _patch_model_with_learned_pe(model, cfg["dropout"], cfg["max_len"])
            # Re-move to device after patching
            model.to(device)
            print(f"  Patched model with LearnedPositionalEncoding.")

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params : {n_params:,}")
        wandb.log({"model/n_params": n_params})

        best_val = float("inf")
        best_ckpt = f"{run_name}_best.pt"

        for epoch in range(cfg["num_epochs"]):
            train_loss = run_epoch(
                train_loader, model, loss_fn, optimizer, scheduler,
                epoch_num=epoch, is_train=True, pad_idx=tgt_vocab.pad_idx, device=device,
            )
            val_loss = run_epoch(
                val_loader, model, loss_fn, None, None,
                epoch_num=epoch, is_train=False, pad_idx=tgt_vocab.pad_idx, device=device,
            )
            wandb.log({"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss})

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, best_ckpt)

        # ── Evaluate BLEU ─────────────────────────────────────────────
        model, _, _, _ = load_checkpoint(best_ckpt, model)
        model.to(device)
        bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
        wandb.log({"test/bleu": bleu})
        results[run_name] = bleu
        wandb.finish()

    # ── Bar chart comparing BLEU ──────────────────────────────────────
    wandb.init(project="da6401-a3", name="2.4_bleu_comparison", reinit=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    names  = list(results.keys())
    values = list(results.values())
    bars = ax.bar(names, values, color=["steelblue", "darkorange"], edgecolor="black")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_ylabel("Test BLEU Score")
    ax.set_title("Section 2.4 — Sinusoidal vs. Learned Positional Encoding")
    ax.set_ylim(0, max(values) * 1.25)
    plt.tight_layout()
    wandb.log({"2.4/bleu_comparison": wandb.Image(fig)})
    plt.close(fig)
    wandb.finish()

    print(f"\n✓ Section 2.4 complete. Results: {results}")


# ══════════════════════════════════════════════════════════════════════
#  2.5 — LABEL SMOOTHING SENSITIVITY
# ══════════════════════════════════════════════════════════════════════

def run_experiment_25_label_smoothing(config: dict = None) -> None:
    """
    Section 2.5 — Decoder Sensitivity: Label Smoothing.

    Trains two models:
        • ε_ls = 0.1   (label-smoothed, as in the original paper)
        • ε_ls = 0.0   (hard cross-entropy, no smoothing)

    Logs to W&B:
        • train/loss, val/loss per epoch
        • Prediction Confidence: softmax probability of the gold token
          (p_correct) averaged across the batch — logged every step
        • Confidence distribution histograms at epochs 1, 5, final
        • Final test BLEU for both settings

    Reflection for report:
        Label smoothing acts as a soft regulariser that prevents the model
        from assigning full probability mass to any single token.  This
        intentionally increases training perplexity (log p_correct is lower)
        but improves generalisation by keeping the output distribution
        "spread out", which benefits BLEU.
    """
    cfg = {**DEFAULT_CONFIG, "num_epochs": 10, **(config or {})}
    device = get_device()

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(
        batch_size   = cfg["batch_size"],
        src_min_freq = cfg["src_min_freq"],
        tgt_min_freq = cfg["tgt_min_freq"],
    )

    bleu_results = {}

    for run_name, smoothing in [
        ("2.5_label_smooth_0.1", 0.1),
        ("2.5_no_label_smooth",  0.0),
    ]:
        wandb.init(project="da6401-a3", name=run_name,
                   config={**cfg, "label_smooth": smoothing}, reinit=True)

        model, optimizer, scheduler, loss_fn = _build_experiment_stack(
            {**cfg, "label_smooth": smoothing}, src_vocab, tgt_vocab, device, use_noam=True
        )

        best_val = float("inf")
        best_ckpt = f"{run_name}_best.pt"
        global_step = 0

        for epoch in range(cfg["num_epochs"]):
            model.train()
            total_loss, total_tok = 0.0, 0
            all_confidences = []

            for src, tgt in train_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input, tgt_gold = tgt[:, :-1], tgt[:, 1:]
                src_mask = make_src_mask(src, tgt_vocab.pad_idx).to(device)
                tgt_mask = make_tgt_mask(tgt_input, tgt_vocab.pad_idx).to(device)

                logits = model(src, tgt_input, src_mask, tgt_mask)
                B, T, V = logits.shape
                loss = loss_fn(logits.reshape(B * T, V), tgt_gold.reshape(B * T))

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()

                # ── Prediction confidence ─────────────────────────────
                with torch.no_grad():
                    probs = F.softmax(logits.reshape(B * T, V), dim=-1)
                    gold  = tgt_gold.reshape(B * T)
                    non_pad = gold != tgt_vocab.pad_idx
                    p_correct = probs[non_pad].gather(
                        1, gold[non_pad].unsqueeze(1)
                    ).squeeze(1)
                    mean_conf = p_correct.mean().item()
                    all_confidences.extend(p_correct.cpu().tolist())

                n_tok = (tgt_gold != 0).sum().item()
                total_loss += loss.item() * n_tok
                total_tok  += n_tok

                wandb.log({
                    "2.5/step_loss":       loss.item(),
                    "2.5/pred_confidence": mean_conf,
                    "global_step":         global_step,
                })
                global_step += 1

            # ── Epoch-level logging ───────────────────────────────────
            avg_loss = total_loss / max(total_tok, 1)
            val_loss = run_epoch(
                val_loader, model, loss_fn, None, None,
                epoch_num=epoch, is_train=False, pad_idx=tgt_vocab.pad_idx, device=device,
            )
            ppl = math.exp(min(avg_loss, 100))

            # Confidence histogram at milestone epochs
            if epoch in {0, cfg["num_epochs"] // 2, cfg["num_epochs"] - 1}:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(all_confidences, bins=50, range=(0, 1),
                        color="steelblue", edgecolor="black", alpha=0.8)
                ax.set_xlabel("Softmax Probability of Correct Token")
                ax.set_ylabel("Count")
                ax.set_title(f"Epoch {epoch+1} Confidence Dist. — {run_name}")
                plt.tight_layout()
                wandb.log({f"2.5/confidence_hist_ep{epoch+1}": wandb.Image(fig)})
                plt.close(fig)

            wandb.log({
                "epoch":          epoch,
                "train/loss":     avg_loss,
                "train/ppl":      ppl,
                "val/loss":       val_loss,
                "avg_confidence": sum(all_confidences) / max(len(all_confidences), 1),
            })

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, best_ckpt)

        # ── BLEU ──────────────────────────────────────────────────────
        model, _, _, _ = load_checkpoint(best_ckpt, model)
        model.to(device)
        bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
        wandb.log({"test/bleu": bleu})
        bleu_results[run_name] = bleu
        wandb.finish()

    # ── BLEU comparison chart ─────────────────────────────────────────
    wandb.init(project="da6401-a3", name="2.5_bleu_comparison", reinit=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    names  = [k.replace("2.5_", "") for k in bleu_results]
    values = list(bleu_results.values())
    bars = ax.bar(names, values, color=["steelblue", "tomato"], edgecolor="black")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_ylabel("Test BLEU Score")
    ax.set_title("Section 2.5 — Label Smoothing vs Standard Cross-Entropy")
    ax.set_ylim(0, max(values) * 1.25)
    plt.tight_layout()
    wandb.log({"2.5/bleu_comparison": wandb.Image(fig)})
    plt.close(fig)
    wandb.finish()

    print(f"\n✓ Section 2.5 complete. Results: {bleu_results}")


# ══════════════════════════════════════════════════════════════════════
#  TOP-LEVEL RUNNER — run all sections
# ══════════════════════════════════════════════════════════════════════

def run_all_report_experiments(config: dict = None) -> None:
    """Run all W&B report experiments sequentially."""
    print("\n" + "═" * 60)
    print("  DA6401 A3 — W&B Report Experiments")
    print("═" * 60)

    print("\n▶ Section 2.1 — Noam vs Fixed LR")
    run_experiment_21_noam_vs_fixed(config)

    print("\n▶ Section 2.2 — Scaling Factor Ablation")
    run_experiment_22_scaling_ablation(config)

    print("\n▶ Section 2.3 — Attention Head Visualisation")
    device = get_device()
    _, _, _, src_vocab, tgt_vocab = build_dataloaders(batch_size=128)
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    model = make_transformer(len(src_vocab), len(tgt_vocab),
                             src_vocab=src_vocab, tgt_vocab=tgt_vocab).to(device)
    try:
        model, _, _, _ = load_checkpoint(cfg.get("best_ckpt", "best_checkpoint.pt"), model)
    except Exception as e:
        print(f"  Warning: could not load checkpoint ({e}). Using random weights.")
    run_experiment_23_attention_heads(model, src_vocab, tgt_vocab, device=device, config=config)

    print("\n▶ Section 2.4 — Positional Encoding Comparison")
    run_experiment_24_positional_encoding(config)

    print("\n▶ Section 2.5 — Label Smoothing")
    run_experiment_25_label_smoothing(config)

    print("\n✓ All report experiments complete.")

# ══════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DA6401 A3 — Transformer Training")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "infer"])
    parser.add_argument("--d_model",      type=int,   default=256)
    parser.add_argument("--N",            type=int,   default=5)
    parser.add_argument("--num_heads",    type=int,   default=8)
    parser.add_argument("--d_ff",         type=int,   default=512)
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--num_epochs",   type=int,   default=15)
    parser.add_argument("--warmup_steps", type=int,   default=4000)
    parser.add_argument("--label_smooth", type=float, default=0.1)
    parser.add_argument("--inference_ckpt",    type=str, default=None)
    parser.add_argument("--inference_samples", type=int, default=5)
    parser.add_argument("--inference_max_len", type=int, default=100)
    parser.add_argument("--skip_bleu", action="store_true")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["fixed_lr", "no_scaling"])
    
    parser.add_argument("--report", type=str, default=None,
     choices=["2.1", "2.2", "2.3", "2.4", "2.5", "all"])
    args = parser.parse_args()
    cfg  = vars(args)
    cfg["compute_bleu"] = not args.skip_bleu


    if args.mode == "infer":
        run_inference_experiment(cfg)
    elif args.ablation == "fixed_lr":
        run_ablation_fixed_lr()
    elif args.ablation == "no_scaling":
        run_ablation_no_scaling()
    else:
        if args.report == "2.1":  run_experiment_21_noam_vs_fixed(cfg)
        elif args.report == "2.2": run_experiment_22_scaling_ablation(cfg)
        elif args.report == "2.3": run_experiment_23_attention_heads(config=cfg)
        elif args.report == "2.4": run_experiment_24_positional_encoding(cfg)
        elif args.report == "2.5": run_experiment_25_label_smoothing(cfg)
        elif args.report == "all": run_all_report_experiments(cfg)
        else:
            run_training_experiment(cfg)
