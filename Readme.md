# DA6401 Assignment 3 — "Attention Is All You Need"

Implementation of the Transformer architecture (Vaswani et al., 2017) for German→English translation on the Multi30k dataset, with a full W&B experiment suite covering five ablation studies.

[W&B link](https://wandb.ai/dipakkanzariya702-iitmaana/da6401-a3/reports/DA6401-Assignment-3-Transformer-for-Langauge-translation--VmlldzoxNjkzMzU2OQ?accessToken=oanq7vkyn8l11y1uqpnxod4b1ie4tc6va6z2mc1q5br1fa1z9borj9r7u2eamj0k)

[Github link](https://github.com/unseen703/da6401_assignment_3)

---

## Project Structure

```
.
├── dataset.py              # Vocab, Multi30kDataset, DataLoader pipeline
├── model.py                # Transformer, Encoder, Decoder, FFN, SublayerConnection
├── utils.py                # Attention, MultiHeadAttention, PositionalEncoding,
│                           # make_src_mask, make_tgt_mask, greedy_decode,
│                           # LabelSmoothingLoss, save/load_checkpoint
├── lr_scheduler.py         # NoamScheduler, get_lr_history, get_fixed_lr_history
├── train.py                # run_epoch, evaluate_bleu, run_training_experiment,
│                           # log_attention_maps, wandb task runners, CLI entry point
├── best_checkpoint.pt      # Best model checkpoint (auto-downloaded via gdown)
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install torch torchvision torchtext
pip install datasets transformers
pip install spacy wandb gdown matplotlib numpy
```

### 2. Download spaCy language models

```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

### 3. Log in to W&B

```bash
wandb login
```

---

## Quick Start

### Train the baseline Transformer

```bash
python train.py --mode train \
    --d_model 256 --N 4 --num_heads 8 --d_ff 512 \
    --dropout 0.13 --batch_size 128 --num_epochs 15 \
    --warmup_steps 4000 --label_smooth 0.1
```

### Run a specific W&B report experiment

```bash
# Section 2.1 — Noam Scheduler vs Fixed LR
python train.py --report 2.1

# Section 2.2 — Scaling Factor 1/√d_k Ablation
python train.py --report 2.2

# Section 2.3 — Attention Head Visualisation
python train.py --report 2.3

# Section 2.4 — Sinusoidal vs Learned Positional Encoding
python train.py --report 2.4

# Section 2.5 — Label Smoothing Sensitivity
python train.py --report 2.5

# Run all five sections sequentially
python train.py --report all
```

---

## Model Architecture

| Component | Detail |
|---|---|
| Architecture | Encoder-Decoder Transformer |
| Embedding dim (`d_model`) | 256 |
| Encoder / Decoder layers (`N`) | 4 |
| Attention heads (`num_heads`) | 8 |
| Head dim (`d_k = d_model / H`) | 32 |
| FFN hidden dim (`d_ff`) | 512 |
| Dropout | 0.13 |
| Max sequence length | 256 |
| Source vocabulary | ~7,853 tokens (German, min_freq=2) |
| Target vocabulary | ~5,893 tokens (English, min_freq=2) |
| Positional encoding | Fixed sinusoidal (Section 3.5, Vaswani et al.) |
| Weight init | Xavier uniform |
| Loss | Label-smoothed cross-entropy (ε=0.1) |

### Attention formula

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

### Noam learning rate schedule

```
lr(t) = d_model^(−0.5) × min( t^(−0.5), t × warmup_steps^(−1.5) )
```

- **Warm-up phase** (`t ≤ warmup_steps`): LR grows linearly.
- **Decay phase** (`t > warmup_steps`): LR decays as inverse square root of step.
- **Peak LR** at `t = warmup_steps`: `d_model^(−0.5) × warmup_steps^(−0.5)`.

---

## W&B Report Experiments

### Section 2.1 — Necessity of the Noam Scheduler

Compares two training runs:

| Setting | Optimizer | LR |
|---|---|---|
| Noam | Adam (β₁=0.9, β₂=0.98, ε=1e-9) | Noam schedule (warmup=4000) |
| Fixed | Adam | Constant 1e-4 |

**W&B logs:** LR trajectory chart, overlaid train/val loss curves.

**Key finding:** Early in training, Q/K/V projections are near-random. A large initial LR drives the softmax into saturation (near one-hot attention), collapsing the gradient signal. Linear warm-up keeps updates small until projections become meaningful.

---

### Section 2.2 — Ablation: The Scaling Factor 1/√d_k

Trains with and without the `/ √d_k` scaling in scaled dot-product attention.

**W&B logs:** Q and K weight gradient L2 norms for the first 1,000 steps, train/val loss.

**Key finding:** Without scaling, dot products grow as O(d_k) for random initialisation, pushing softmax into near-zero gradient regions (vanishing gradients on Q and K). With scaling, gradients remain stable throughout warm-up.

---

### Section 2.3 — Attention Rollout & Head Specialization

Extracts encoder self-attention weights for held-out validation sentences.

**W&B logs:**
- Per-head heatmaps (H subplots) for every encoder layer.
- Attention rollout map (Abnar & Zuidema, 2020): `A_rollout = A_L ⊗ … ⊗ A_1`, where each layer's mean-head attention is combined with a residual identity before matrix multiplication.

**What to look for:**
- **Local heads**: attend predominantly to the next/previous token.
- **Global heads**: capture long-range syntactic dependencies.
- **Head redundancy**: multiple heads learning near-identical patterns — common in shallower models.

---

### Section 2.4 — Positional Encoding vs Learned Embeddings

Replaces the fixed sinusoidal `PositionalEncoding` with a trainable `LearnedPositionalEncoding` (`nn.Embedding(max_len, d_model)`).

**W&B logs:** train/val loss curves, final test BLEU bar chart.

**Theoretical note:** Sinusoidal PE is defined analytically for any position index, so the model can theoretically extrapolate to sequences longer than `max_len` seen during training. Learned PE cannot — position indices beyond `max_len − 1` have no trained embedding weight.

---

### Section 2.5 — Decoder Sensitivity: Label Smoothing

Compares two loss functions:

| Setting | ε_ls | Effective target |
|---|---|---|
| Smoothed | 0.1 | `(1 − ε) × one_hot + ε / (V − 1)` |
| Hard CE | 0.0 | standard one-hot |

**W&B logs:** Per-step prediction confidence (softmax p(correct token)), confidence histograms at epochs 1, mid, and final, train/val loss, test BLEU.

**Key finding:** Label smoothing acts as a soft regulariser. It intentionally increases training perplexity (p_correct is lower) while preventing the model from over-confidently assigning all probability mass to a single token — improving BLEU on the test set.

---
## References

1. Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS. https://arxiv.org/abs/1706.03762
2. Abnar, S. & Zuidema, W. (2020). *Quantifying Attention Flow in Transformers*. ACL. https://arxiv.org/abs/2005.00928
3. Multi30k Dataset: Elliott et al. (2016). https://huggingface.co/datasets/bentrevett/multi30k