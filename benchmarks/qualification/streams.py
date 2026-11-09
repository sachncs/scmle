"""Deployment streams with known truth.

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

import numpy

KINDS = ("none", "abrupt", "gradual", "covariate", "concept")
LABELS = ("full", "mar", "mnar")
# probability of approving a skipped case
FAIR_COIN = 0.5


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
        dependence: How much a case's loss probability moves with its
            covariate: `p = clip(r_t + dependence * (logistic(z) - 0.5))`.
            The mean over the unshifted covariate is `r_t`; with a nonzero
            value, selecting labels on `z` biases the naive mean.
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
    dependence: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        """Validates the specification.

        Raises:
            ValueError: If `kind` or `labels` is unknown, `batches`, `size` or
                `window` is out of range, or `base` is not in (0, 1) or
                `epsilon` not in [0, 1).
        """
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
    z: numpy.ndarray
    loss: numpy.ndarray
    risk: numpy.ndarray
    pi: numpy.ndarray
    selected: numpy.ndarray
    audit: numpy.ndarray
    arrival: numpy.ndarray
    change_point: int | None
    true_pi: numpy.ndarray

    def mean_risk(self) -> float:
        """True mean loss rate over the whole stream.

        Returns:
            The average of the per-batch true risk.
        """
        return float(self.risk.mean())


def risk_path(spec: StreamSpec) -> numpy.ndarray:
    """True loss rate of each batch.

    Args:
        spec: The stream specification.

    Returns:
        An array of length `batches`: constant `base` for `none` and
        `covariate`, a step to `base + delta` at the change point for
        `abrupt` and `concept`, and a linear ramp over `ramp` batches for
        `gradual`.
    """
    index = numpy.arange(spec.batches)
    path = numpy.full(spec.batches, spec.base, dtype=float)
    after = index >= spec.change
    if spec.kind in {"abrupt", "concept"}:
        path[after] = spec.base + spec.delta
    elif spec.kind == "gradual":
        part = numpy.clip(
            (index - spec.change + 1) / max(spec.ramp, 1), 0.0, 1.0
        )
        path = spec.base + spec.delta * part
    return path


def covariate_path(spec: StreamSpec) -> numpy.ndarray:
    """Covariate mean of each batch.

    Args:
        spec: The stream specification.

    Returns:
        An array of length `batches`: zero, plus `shift` from the change
        point on for `covariate` streams.
    """
    path = numpy.zeros(spec.batches)
    if spec.kind == "covariate":
        path[numpy.arange(spec.batches) >= spec.change] = spec.shift
    return path


def logistic(value: numpy.ndarray) -> numpy.ndarray:
    """The logistic function.

    Args:
        value: Input array.

    Returns:
        `1 / (1 + exp(-value))`, elementwise.
    """
    return 1.0 / (1.0 + numpy.exp(-value))


def marginal_risk(base: float, dependence: float) -> float:
    """Mean loss rate of a stable stream, integrated over the covariate.

    With `dependence > 0` the per-case probability is clipped to [0, 1], so
    its mean over `z ~ N(0, 1)` differs a little from `base`; this integrates
    it with Gauss-Hermite nodes.

    Args:
        base: Loss rate before any change.
        dependence: How much a case's loss probability moves with its
            covariate.

    Returns:
        The mean per-case loss probability over a standard normal covariate.
    """
    nodes, weights = numpy.polynomial.hermite_e.hermegauss(200)
    chance = numpy.clip(base + dependence * (logistic(nodes) - 0.5), 0.0, 1.0)
    return float(numpy.sum(weights * chance) / numpy.sum(weights))


def simulate(spec: StreamSpec) -> Stream:
    """Draws one stream.

    Args:
        spec: The specification.

    Returns:
        The stream with its truth.
    """
    rng = numpy.random.default_rng(spec.seed)
    shape = (spec.batches, spec.size)
    risk = risk_path(spec)
    z = covariate_path(spec)[:, None] + rng.standard_normal(shape)
    chance = numpy.clip(
        risk[:, None] + spec.dependence * (logistic(z) - 0.5), 0.0, 1.0
    )
    loss = (rng.random(shape) < chance).astype(float)
    audit = rng.random(shape) < spec.epsilon
    if spec.labels == "full":
        true_pi = numpy.ones(shape)
        logged = numpy.ones(shape)
    elif spec.labels == "mar":
        rule = numpy.clip(logistic(1.5 * z - 0.5), spec.floor, 1.0)
        true_pi = 1.0 - (1.0 - rule) * (1.0 - spec.epsilon)
        logged = true_pi.copy()
    else:
        rule = numpy.where(loss > 0, 0.9, 0.3)
        true_pi = 1.0 - (1.0 - rule) * (1.0 - spec.epsilon)
        logged = numpy.full(shape, spec.epsilon)
    selected = audit | (rng.random(shape) < true_pi)
    if spec.labels == "full":
        selected = numpy.ones(shape, dtype=bool)
    delay = spec.window + spec.slow * (loss > 0)
    arrival = (
        numpy.arange(spec.batches)[:, None] + delay + numpy.zeros(shape, int)
    )
    arrival = numpy.where(
        selected, arrival, spec.batches + spec.window + 10_000
    )
    change = None if spec.kind == "none" else spec.change
    return Stream(
        spec, z, loss, risk, logged, selected, audit, arrival, change, true_pi
    )


@dataclasses.dataclass(frozen=True)
class FeedbackStream:
    """An approval loop in which decisions change a case's next features.

    Attributes:
        z: Covariate scores at decision time, shape `(batches, size)`.
        next_z: The same cases' features one period later.
        bypass: Cases that skipped the model (random decision).
        approved: Decisions taken.
        gamma: True strength of the loop (0 means no loop).
    """

    z: numpy.ndarray
    next_z: numpy.ndarray
    bypass: numpy.ndarray
    approved: numpy.ndarray
    gamma: float


def feedback_stream(
    batches: int,
    size: int,
    gamma: float,
    bypass: float,
    seed: int,
) -> FeedbackStream:
    """Draws an approval loop.

    A model-decided case is approved when `z > 0`; a bypass case by a coin
    flip, independent of `z` (the randomized, logged slice). An approval moves
    the case's next-period feature by `gamma`: `next_z = z + gamma * approved
    + noise`. With `gamma = 0` decisions do not touch the data and the bypass
    and model-decided groups have the same next-period distribution; with
    `gamma > 0` they differ, because the model approves selectively and the
    coin does not.

    Args:
        batches: Number of batches.
        size: Cases per batch.
        gamma: Strength of the loop; 0 for a loop-free world.
        bypass: Fraction of cases that skip the model.
        seed: Seed.

    Returns:
        The simulated loop with arrays of shape `(batches, size)`.
    """
    rng = numpy.random.default_rng(seed)
    shape = (batches, size)
    skipped = rng.random(shape) < bypass
    z = rng.standard_normal(shape)
    coin = rng.random(shape) < FAIR_COIN
    approved = numpy.where(skipped, coin, z > 0)
    next_z = z + gamma * approved + 0.5 * rng.standard_normal(shape)
    return FeedbackStream(z, next_z, skipped, approved, gamma)


@dataclasses.dataclass(frozen=True)
class ChallengerSequence:
    """Dependent challengers against one champion.

    Attributes:
        advantage: True advantage of each challenger (champion loss rate
            minus the challenger's), in loss units.
        champion: Champion losses, shape `(batches, size)`.
        challengers: Challenger losses, shape `(count, batches, size)`.
    """

    advantage: numpy.ndarray
    champion: numpy.ndarray
    challengers: numpy.ndarray


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

    Returns:
        The champion's and challengers' simulated losses with their true
        advantages.
    """
    rng = numpy.random.default_rng(seed)
    latent = rng.random((batches, size))
    gains = numpy.asarray(advantages, dtype=float)
    rates = numpy.clip(base - gains, 0.0, 1.0)
    champion = (latent < base).astype(float)
    challengers = (latent[None] < rates[:, None, None]).astype(float)
    return ChallengerSequence(gains, champion, challengers)


def true_selective_risk(answered: numpy.ndarray, stream: Stream) -> float:
    """Realised loss rate over answered cases (the simulator's truth).

    Args:
        answered: Boolean mask of the cases the system answered, with the
            shape of the stream's losses.
        stream: The simulated stream holding the true losses.

    Returns:
        The mean true loss over the answered cases; NaN when none were
        answered.
    """
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

    Returns:
        A dict with the arrays `score` and `loss` (0 or 1) of length `count`
        and the target risk `r_star`.
    """
    rng = numpy.random.default_rng(seed)
    score = rng.random(count)
    loss = (rng.random(count) < 0.30 * (1.0 - score) ** 2).astype(float)
    return {"score": score, "loss": loss, "r_star": r_star}


def oracle_risk(tau: float) -> float:
    """True selective risk ``E[loss | g >= tau]`` for `draw_calibration`.

    ``integral_tau^1 0.3 (1 - g)^2 dg / (1 - tau) = 0.1 (1 - tau)^2``.

    Args:
        tau: Score threshold; cases with a score at or above it are answered.

    Returns:
        The expected loss rate of the answered cases.
    """
    return 0.1 * (1.0 - tau) ** 2
