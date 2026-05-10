"""
lr_scheduler.py — Noam Learning Rate Scheduler
DA6401 Assignment 3: "Attention Is All You Need"

Reference: Vaswani et al. 2017 — https://arxiv.org/abs/1706.03762  (Section 5.3)

Formula:
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

Intuition
─────────
• During warm-up  (step ≤ warmup_steps):
      the active term is  step * warmup_steps^(-1.5)
      → LR grows *linearly* with step.

• After warm-up   (step > warmup_steps):
      the active term is  step^(-0.5)
      → LR decays as the inverse square root of the step number.

• The peak learning rate occurs exactly at step == warmup_steps and equals
      d_model^(-0.5) * warmup_steps^(-0.5)

Why this matters (W&B Report, Section 2.1)
──────────────────────────────────────────
Early in training the Q, K, V projection matrices are nearly random.
A large initial LR causes the softmax to saturate (all-or-nothing attention),
which collapses the gradient signal.  The linear warm-up keeps updates small
until the projections become meaningful, then the inverse-sqrt decay provides
a progressively finer learning signal as the model converges.
"""

import math
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


# ══════════════════════════════════════════════════════════════════════
#  NOAM SCHEDULER
# ══════════════════════════════════════════════════════════════════════

class NoamScheduler(LRScheduler):
    """
    Noam learning rate schedule from "Attention Is All You Need".

    Works as a *multiplicative* modifier on top of the base learning rate
    stored in the optimizer.  At step t:

        lr(t) = base_lr * d_model^(-0.5)
                         * min(t^(-0.5), t * warmup_steps^(-1.5))

    Typical usage (mirrors the paper's Adam config):
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=4000)

    Then per training step:
        optimizer.step()
        scheduler.step()

    Args:
        optimizer    : Wrapped optimizer. Set its lr=1.0 so that the
                       scheduler's scale factor IS the effective lr.
        d_model      : Model dimensionality (embedding size). Controls
                       the overall magnitude of the learning rate.
        warmup_steps : Number of linear warm-up steps before decay begins.
                       Paper uses 4000; shorter sequences → fewer steps.
        last_epoch   : Index of the last completed epoch/step. Default -1
                       means "start from step 0". PyTorch convention.
    """

    def __init__(
        self,
        optimizer:    optim.Optimizer,
        d_model:      int,
        warmup_steps: int,
        last_epoch:   int = -1,
    ) -> None:
        # Store hyperparameters BEFORE calling super().__init__,
        # because the parent calls get_lr() during initialisation.
        self.d_model      = d_model
        self.warmup_steps = warmup_steps

        # LRScheduler.__init__ calls self.step() once internally,
        # which in turn calls get_lr() → _get_lr_scale().
        super().__init__(optimizer, last_epoch=last_epoch)

    # ------------------------------------------------------------------
    def _get_lr_scale(self) -> float:
        """
        Compute the Noam scale factor for the current step.

        step = self.last_epoch + 1   (shift by 1 to avoid step=0
                                      which would cause division by zero
                                      in the step^(-0.5) term)

        Returns:
            float : Scalar multiplier applied to each param-group's base lr.
        """
        step = self.last_epoch + 1   # 1-indexed step number

        # Two candidate schedules — take the smaller one (min):
        #   warm-up ramp  : step * warmup_steps^(-1.5)
        #   decay slope   : step^(-0.5)
        warmup_ramp = step * (self.warmup_steps ** -1.5)
        decay_slope = step ** -0.5

        scale = (self.d_model ** -0.5) * min(decay_slope, warmup_ramp)
        return scale

    # ------------------------------------------------------------------
    def get_lr(self) -> list:
        """
        Return the new learning rate for every parameter group.

        PyTorch's scheduler machinery calls this once per .step().
        We multiply each group's *base* lr (set at optimizer creation)
        by the Noam scale factor.

        Returns:
            list[float] : One lr value per optimizer param group.
        """
        scale = self._get_lr_scale()
        return [base_lr * scale for base_lr in self.base_lrs]


# ══════════════════════════════════════════════════════════════════════
#  HELPER — simulate LR trajectory  (do NOT modify; used by autograder)
# ══════════════════════════════════════════════════════════════════════

def get_lr_history(
    d_model:      int,
    warmup_steps: int,
    total_steps:  int,
) -> list:
    """
    Simulate the LR trajectory of NoamScheduler for `total_steps` steps.

    Creates a throw-away 1-parameter model + Adam optimizer so the
    scheduler can be exercised without real training.

    Args:
        d_model      : Model dimensionality.
        warmup_steps : Warm-up steps.
        total_steps  : Number of steps to simulate.

    Returns:
        list[float] : LR value recorded *before* each optimizer.step(),
                      length == total_steps.
    """
    dummy_model = torch.nn.Linear(1, 1)
    optimizer   = optim.Adam(dummy_model.parameters(), lr=1.0)
    scheduler   = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    history = []
    for _ in range(total_steps):
        history.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    return history


# ══════════════════════════════════════════════════════════════════════
#  OPTIONAL: Fixed LR baseline (for W&B Report Section 2.1 comparison)
# ══════════════════════════════════════════════════════════════════════

def get_fixed_lr_history(fixed_lr: float, total_steps: int) -> list:
    """
    Simulate a constant-LR baseline for comparison plots.

    Returns:
        list[float] : `fixed_lr` repeated `total_steps` times.
    """
    return [fixed_lr] * total_steps


# ══════════════════════════════════════════════════════════════════════
#  VISUAL CHECK — run:  python lr_scheduler.py
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    D_MODEL      = 512
    WARMUP_STEPS = 4000
    TOTAL_STEPS  = 20_000
    FIXED_LR     = 1e-4

    lrs       = get_lr_history(D_MODEL, WARMUP_STEPS, TOTAL_STEPS)
    fixed_lrs = get_fixed_lr_history(FIXED_LR, TOTAL_STEPS)

    # ── Verify autograder properties ──────────────────────────────────
    warmup_region = lrs[:WARMUP_STEPS]
    decay_region  = lrs[WARMUP_STEPS:]

    assert all(
        warmup_region[i] <= warmup_region[i + 1]
        for i in range(len(warmup_region) - 1)
    ), "LR not monotonically increasing during warm-up!"

    assert all(
        decay_region[i] >= decay_region[i + 1]
        for i in range(len(decay_region) - 1)
    ), "LR not monotonically decreasing after warm-up!"

    # Theoretical peak: d_model^(-0.5) * warmup^(-0.5)
    theoretical_peak = (D_MODEL ** -0.5) * (WARMUP_STEPS ** -0.5)
    actual_peak      = max(lrs)
    print(f"Theoretical peak lr : {theoretical_peak:.6f}")
    print(f"Actual peak lr      : {actual_peak:.6f}")
    print(f"Peak at step        : {lrs.index(actual_peak)}")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lrs,       label="Noam Scheduler",      color="steelblue", linewidth=2)
    ax.plot(fixed_lrs, label=f"Fixed LR ({FIXED_LR})", color="tomato",
            linestyle="--", linewidth=1.5)
    ax.axvline(WARMUP_STEPS, color="gray", linestyle=":", alpha=0.8,
               label=f"warmup_steps = {WARMUP_STEPS}")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"Noam LR Schedule  (d_model={D_MODEL})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("lr_schedule.png", dpi=150)
    plt.show()
    print("Plot saved → lr_schedule.png")
