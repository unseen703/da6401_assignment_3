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

from model        import Transformer, make_transformer
from utils        import (
    make_src_mask,
    make_tgt_mask,
    greedy_decode,
    save_checkpoint,
    load_checkpoint,
    LabelSmoothingLoss,
    get_device,
)
from lr_scheduler import NoamScheduler
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
    print(f"Device : {device}")

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

    plt.suptitle(f"Encoder Layer {resolved_layer_idx} — Attention Heads", fontsize=11)
    plt.tight_layout()

    if wandb.run is not None:
        wandb.log({"attention_maps": wandb.Image(fig)})

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
    print(src_vocab)
    print(tgt_vocab)

    device = get_device()
    print(f"Device : {device}")

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

    model     = make_transformer(len(src_vocab), len(tgt_vocab)).to(device)
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
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_w = F.softmax(scores, dim=-1)
        return torch.matmul(attn_w, v), attn_w

    utils_module.Attention.forward = unscaled_forward
    print("⚠ Attention scaling DISABLED for ablation.")

    run_training_experiment({**DEFAULT_CONFIG, "num_epochs": 3})

    utils_module.Attention.forward = original_forward
    print("Attention scaling restored.")


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
        run_training_experiment(cfg)

