"""Training helpers that need no PyTorch: schedules, batch sizing, health.

Keeping them free of torch lets them be tested, and reasoned about, without
the worker process.
"""

import dataclasses
import math
from typing import Any

import numpy as np

AUTO = "auto"
MIN_BATCH = 32
MAX_BATCH = 512
ROWS_PER_BATCH = 32
EXPLODE_FACTOR = 100.0
EXPLODE_FLOOR = 1e3
VANISH_NORM = 1e-7
STALL_IMPROVEMENT = 0.01


def resolve_batch_size(setting: int | str, rows: int) -> int:
    """Turns a batch-size setting into a number of rows per step.

    `auto` aims for roughly `rows / 32` steps per epoch rounded to a power of
    two, between 32 and 512: small data gets small batches (more updates),
    large data gets large ones (better hardware use).

    Args:
        setting: A positive integer or `auto`.
        rows: Number of training rows.

    Returns:
        Rows per optimisation step, never more than `rows`.

    Raises:
        ValueError: For any other setting.
    """
    if setting == AUTO:
        target = max(rows / ROWS_PER_BATCH, 1.0)
        size = 2 ** round(math.log2(target))
        return int(min(max(size, MIN_BATCH), MAX_BATCH, max(rows, 1)))
    if isinstance(setting, int) and setting > 0:
        return min(setting, max(rows, 1))
    raise ValueError(
        f"batch size must be a positive integer or 'auto': {setting!r}"
    )


def learning_rate_at(
    step: int, total_steps: int, base: float, warmup_steps: int, kind: str
) -> float:
    """Learning rate for an optimisation step.

    Args:
        step: Zero-based step index.
        total_steps: Steps in the whole run.
        base: Peak learning rate.
        warmup_steps: Steps over which the rate rises linearly from zero.
        kind: `constant` or `cosine` (decay to zero after the warm-up).

    Returns:
        The learning rate.

    Raises:
        ValueError: For an unknown schedule.
    """
    if kind not in {"constant", "cosine"}:
        raise ValueError(f"unknown learning-rate schedule '{kind}'")
    if warmup_steps and step < warmup_steps:
        return base * (step + 1) / warmup_steps
    if kind == "constant":
        return base
    span = max(total_steps - warmup_steps, 1)
    progress = min((step - warmup_steps) / span, 1.0)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclasses.dataclass
class TrainingMonitor:
    """Watches losses and gradient norms for signs of a broken run.

    Attributes:
        losses: Mean training loss of each epoch.
        norms: Gradient norm (before clipping) of each optimisation step.
        clipped: Steps whose gradient was clipped.
        nonfinite: Whether a non-finite loss or gradient was seen.
    """

    losses: list[float] = dataclasses.field(default_factory=list)
    norms: list[float] = dataclasses.field(default_factory=list)
    clipped: int = 0
    nonfinite: bool = False

    def step(self, loss: float, norm: float, limit: float | None) -> bool:
        """Records one optimisation step.

        Args:
            loss: Loss of the batch.
            norm: Gradient norm before clipping.
            limit: Clipping threshold, if any.

        Returns:
            False when the run has diverged and should stop.
        """
        if not (math.isfinite(loss) and math.isfinite(norm)):
            self.nonfinite = True
            return False
        self.norms.append(norm)
        if limit is not None and norm > limit:
            self.clipped += 1
        return True

    def epoch(self, loss: float) -> None:
        """Records the mean training loss of an epoch."""
        self.losses.append(loss)

    def report(self) -> dict[str, Any]:
        """Summarizes the run and names the failure mode, if any.

        Returns:
            A verdict (`ok`, `diverged`, `exploding`, `vanishing` or
            `stalled`) with the statistics behind it.
        """
        norms = np.asarray(self.norms)
        median = float(np.median(norms)) if norms.size else 0.0
        peak = float(np.max(norms)) if norms.size else 0.0
        improvement = 0.0
        if len(self.losses) > 1 and abs(self.losses[0]) > 0:
            improvement = (self.losses[0] - self.losses[-1]) / abs(
                self.losses[0]
            )
        if self.nonfinite:
            verdict = "diverged"
        elif median > 0 and (
            peak > EXPLODE_FACTOR * median or median > EXPLODE_FLOOR
        ):
            verdict = "exploding"
        elif median < VANISH_NORM:
            verdict = "vanishing"
        elif improvement < STALL_IMPROVEMENT and len(self.losses) > 2:
            verdict = "stalled"
        else:
            verdict = "ok"
        return {
            "verdict": verdict,
            "steps": int(norms.size),
            "median_gradient_norm": median,
            "peak_gradient_norm": peak,
            "clipped_steps": self.clipped,
            "loss_improvement": improvement,
        }
