"""Deployment streams with known truth (Phase 13, WP-13.1).

Every stream returns the true change point, the true per-batch risk and the
true labelling mechanism, so a monitor's false-alarm rate, detection delay and
bias can be measured against something other than itself. All draws come from
`numpy.random.default_rng(seed)` and the seed is part of the returned record.

World. A case has a covariate score `z ~ N(mu_t, 1)` and a loss
`l ~ Bernoulli(r_t)`. The model's loss rate `r_t` is the quantity a monitor has
to track:

* `abrupt`: `r_t = base` before the change point and `base + delta` from it;
* `gradual`: the rise is linear over `ramp` batches after the change point;
* `covariate`: `mu_t` moves by `shift` at the change point, `r_t` is unchanged
  (benign: `P(l | z)` does not depend on `z`);
* `concept`: `r_t` rises while `mu_t` stays put;
* `none`: nothing changes.

Labels. A case's loss is observed only if it is selected:

* `full`: every case, with delay `window` batches (outcome independent);
* `mar`: with probability `pi(z) = clip(logistic(a * z + b), floor, 1)`,
  depending on the covariate only; `pi` is logged;
* `mnar`: depending on the loss (`pi_loss` for a loss of 1, `pi_ok` for 0),
  not logged;
* an audit slice labels a fraction `epsilon` of cases uniformly, whatever the
  rule says; its draws are logged as `audit`.

Delay. Outcome independent: `window` batches for all. Outcome dependent: a
loss of 1 arrives `slow` batches after a loss of 0 would.

Feedback. `approval` streams make the covariate mean of batch `t + 1` move by
`gamma` times the share of model-decided approvals in batch `t`; a fraction
`bypass` of cases skips the model (random decision).

Challenger sequences. Challenger `i` has a true advantage `advantage[i]` over
the champion; losses are comonotone through a shared latent `u_t` per case, so
the challengers' paired differences are dependent.
"""

from collections.abc import Sequence
import dataclasses
import math
from typing import Any

import numpy as np

KINDS = ("none", "abrupt", "gradual", "covariate", "concept")
LABELS = ("full", "mar", "mnar")


@dataclasses.dataclass(frozen=True, slots=True)
class StreamSpec:
    """What a stream looks like.

    Attributes:
        kind: One of `KINDS`.
        batches: Number of batches.
        size: Cases per batch.
        base: Loss rate before any change.
        delta: Rise of the loss rate (loss units).
        shift: Move of the covariate mean (covariate shift).
        change: Batch index at which the change starts.
        ramp: Batches over which a gradual rise completes.
        labels: One of `LABELS`.
        window: Batches after which an outcome has matured.
        slow: Extra delay of a loss of 1 (0 means outcome independent).
        epsilon: Fraction of cases labelled by the uniform audit slice.
        floor: Smallest selection probability of the MAR rule.
        seed: Seed of the stream.
    """

    kind: str = "none"
    batches: int = 200
    size: int = 50
    base: float = 0.10
    delta: float = 0.10
    shift: float = 2.0
    change: int = 100
    ramp: int = 20
    labels: str = "full"
    window: int = 1
    slow: int = 0
    epsilon: float = 0.0
    floor: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        """Validates the specification."""
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if self.labels not in LABELS:
            raise ValueError(f"labels must be one of {LABELS}")
        if self.batches < 1 or self.size < 1 or self.window < 0:
            raise ValueError("batches and size must be positive")
        if not 0.0 < self.base < 1.0 or not 0.0 <= self.epsilon < 1.0:
            raise ValueError("base must be in (0, 1) and epsilon in [0, 1)")


@dataclasses.dataclass(frozen=True)
class Stream:
    """A simulated deployment stream and its truth.

    Per-case arrays have shape `(batches, size)`.

    Attributes:
        spec: The specification.
        z: Covariate scores.
        loss: True losses (0 or 1).
        risk: True loss rate of each batch (conditional on the stream).
        pi: Logged probability that the case was labelled (all ones for
            `full`; the audit-slice rate for `mnar`, which is the only part
            that is known).
        selected: Whether the case's label is ever observed.
        audit: Whether the case was in the audit slice.
        arrival: Batch index at which the label arrives (`batches + window`
            + slack when never observed).
        change_point: True change point, or None for a stable stream.
        true_pi: The probability that really selected the case.
    """

    spec: StreamSpec
    z: np.ndarray
    loss: np.ndarray
    risk: np.ndarray
    pi: np.ndarray
    selected: np.ndarray
    audit: np.ndarray
    arrival: np.ndarray
    change_point: int | None
    true_pi: np.ndarray

    def mean_risk(self) -> float:
        """True mean loss rate over the whole stream."""
        return float(self.risk.mean())


def risk_path(spec: StreamSpec) -> np.ndarray:
    """True loss rate of each batch."""
    index = np.arange(spec.batches)
    path = np.full(spec.batches, spec.base, dtype=float)
    after = index >= spec.change
    if spec.kind == "abrupt" or spec.kind == "concept":
        path[after] = spec.base + spec.delta
    elif spec.kind == "gradual":
        part = np.clip((index - spec.change + 1) / max(spec.ramp, 1), 0.0, 1.0)
        path = spec.base + spec.delta * part
    return path


def covariate_path(spec: StreamSpec) -> np.ndarray:
    """Covariate mean of each batch."""
    path = np.zeros(spec.batches)
    if spec.kind == "covariate":
        path[np.arange(spec.batches) >= spec.change] = spec.shift
    return path


def logistic(value: np.ndarray) -> np.ndarray:
    """The logistic function."""
    return 1.0 / (1.0 + np.exp(-value))


def simulate(spec: StreamSpec) -> Stream:
    """Draws one stream.

    Args:
        spec: The specification.

    Returns:
        The stream with its truth.
    """
    rng = np.random.default_rng(spec.seed)
    shape = (spec.batches, spec.size)
    risk = risk_path(spec)
    z = covariate_path(spec)[:, None] + rng.standard_normal(shape)
    loss = (rng.random(shape) < risk[:, None]).astype(float)
    audit = rng.random(shape) < spec.epsilon
    if spec.labels == "full":
        true_pi = np.ones(shape)
        logged = np.ones(shape)
    elif spec.labels == "mar":
        rule = np.clip(logistic(1.5 * z - 0.5), spec.floor, 1.0)
        true_pi = 1.0 - (1.0 - rule) * (1.0 - spec.epsilon)
        logged = true_pi.copy()
    else:
        rule = np.where(loss > 0, 0.9, 0.3)
        true_pi = 1.0 - (1.0 - rule) * (1.0 - spec.epsilon)
        logged = np.full(shape, spec.epsilon)
    selected = audit | (rng.random(shape) < true_pi)
    if spec.labels == "full":
        selected = np.ones(shape, dtype=bool)
    delay = spec.window + spec.slow * (loss > 0)
    arrival = np.arange(spec.batches)[:, None] + delay + np.zeros(shape, int)
    arrival = np.where(selected, arrival, spec.batches + spec.window + 10_000)
    change = None if spec.kind == "none" else spec.change
    return Stream(
        spec, z, loss, risk, logged, selected, audit, arrival, change, true_pi
    )


@dataclasses.dataclass(frozen=True)
class FeedbackStream:
    """An approval loop in which the model's decisions move later covariates.

    Attributes:
        z: Covariate scores, shape `(batches, size)`.
        bypass: Cases that skipped the model (random decision).
        approved: Decisions taken.
        gamma: True strength of the loop (0 means no loop).
    """

    z: np.ndarray
    bypass: np.ndarray
    approved: np.ndarray
    gamma: float


def feedback_stream(
    batches: int,
    size: int,
    gamma: float,
    bypass: float,
    seed: int,
) -> FeedbackStream:
    """Draws an approval loop.

    Batch `t` has a covariate mean proportional to `gamma` times the approval
    share of batch `t - 1` above one half.
    A model-decided case is approved when `z > 0`; a bypass case by a coin
    flip, so the next-period covariates of bypass cases do not feed the model
    (the bypass group is the unaffected comparison).

    Args:
        batches: Number of batches.
        size: Cases per batch.
        gamma: Strength of the loop; 0 for a loop-free world.
        bypass: Fraction of cases that skip the model.
        seed: Seed.
    """
    rng = np.random.default_rng(seed)
    shape = (batches, size)
    skipped = rng.random(shape) < bypass
    z = np.zeros(shape)
    approved = np.zeros(shape, dtype=bool)
    share = 0.5
    for step in range(batches):
        mean = gamma * (share - 0.5) * 2.0 * 3.0
        base = rng.standard_normal(size)
        z[step] = mean + base
        coin = rng.random(size) < 0.5
        approved[step] = np.where(skipped[step], coin, z[step] > 0)
        share = float(approved[step].mean())
    return FeedbackStream(z, skipped, approved, gamma)


@dataclasses.dataclass(frozen=True)
class ChallengerSequence:
    """Dependent challengers against one champion.

    Attributes:
        advantage: True advantage of each challenger (champion loss rate
            minus the challenger's), in loss units.
        champion: Champion losses, shape `(batches, size)`.
        challengers: Challenger losses, shape `(count, batches, size)`.
    """

    advantage: np.ndarray
    champion: np.ndarray
    challengers: np.ndarray


def challenger_sequence(
    advantages: Sequence[float],
    batches: int,
    size: int,
    base: float,
    seed: int,
) -> ChallengerSequence:
    """Draws challengers whose losses are comonotone with the champion's.

    With a shared latent `u ~ U(0, 1)` per case, champion loss is
    `1[u < base]` and challenger `i` has loss `1[u < base - advantage_i]`, so
    every pair differs only on cases with `u` between the two rates and the
    challengers are dependent through `u`.

    Args:
        advantages: True advantage of each challenger.
        batches: Matured batches per challenger.
        size: Cases per batch.
        base: Champion loss rate.
        seed: Seed.
    """
    rng = np.random.default_rng(seed)
    latent = rng.random((batches, size))
    gains = np.asarray(advantages, dtype=float)
    rates = np.clip(base - gains, 0.0, 1.0)
    champion = (latent < base).astype(float)
    challengers = (latent[None] < rates[:, None, None]).astype(float)
    return ChallengerSequence(gains, champion, challengers)


def true_selective_risk(answered: np.ndarray, stream: Stream) -> float:
    """Realised loss rate over answered cases (the simulator's truth)."""
    count = float(answered.sum())
    if count == 0.0:
        return math.nan
    return float((stream.loss * answered).sum() / count)


def draw_calibration(count: int, r_star: float, seed: int) -> dict[str, Any]:
    """A calibration set for selective prediction with a known risk curve.

    The score `g` is uniform on `(0, 1)`; a case with score `g` has loss
    probability `0.30 * (1 - g) ** 2`, decreasing in `g`. The oracle selective
    risk of answering the cases with `g >= tau` is `oracle_risk(tau)`.

    Args:
        count: Number of cases.
        r_star: Target risk (carried for the caller's convenience).
        seed: Seed.
    """
    rng = np.random.default_rng(seed)
    score = rng.random(count)
    loss = (rng.random(count) < 0.30 * (1.0 - score) ** 2).astype(float)
    return {"score": score, "loss": loss, "r_star": r_star}


def oracle_risk(tau: float) -> float:
    """True selective risk ``E[loss | g >= tau]`` for `draw_calibration`.

    ``integral_tau^1 0.3 (1 - g)^2 dg / (1 - tau) = 0.1 (1 - tau)^2``.
    """
    return 0.1 * (1.0 - tau) ** 2
