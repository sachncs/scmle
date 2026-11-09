"""Delayed, partial and selected labels.

Three separate problems, one estimator each.

Delay. A case has *matured* when its outcome window has elapsed since the
prediction (`is_matured`). Maturity is decided by prediction time, never by
whether a label arrived: when slow outcomes are the bad ones, the labels that
have arrived by now are biased towards the good ones, and a matured-by-arrival
estimate inherits that bias.

Selection. If a label is observed only for a subset chosen by a rule, the mean
loss over labelled cases is biased (verification bias; Begg and Greenes 1983,
from memory, not read). With the probability ``pi`` that a case is labelled
*logged at decision time*, the Horvitz-Thompson mean ``(1/n) sum_i obs_i *
loss_i / pi_i`` is unbiased for the mean loss over all `n` cases without a
model of the selection. It needs ``pi_i >= epsilon > 0`` and the logged ``pi``
to be the true one; `check_randomization` tests the uniformity of an audit
slice's own draws. Each term is at most ``bound / epsilon``, so a batch mean is
rescaled by ``epsilon / bound`` into ``[0, 1]`` before it goes into a betting
monitor (wider intervals are the price).

Without a logged propensity and without an audit slice the selection cannot be
tested and is not assumed away: the stream is `unverified` and `permits_action`
is False, so no retrain, promotion or rollback may rest on it. With an audit
slice only, the estimate uses the audit-labelled cases alone, each with weight
``1 / audit_rate``.

Feedback. A model's decisions can change the data that later trains and
evaluates it. Only a randomized bypass slice (a logged fraction of cases that
skip the model) can show it: `bypass_test` compares next-period features of
bypass and model-decided cases with a kernel two-sample test. Without a
bypass slice the loop is undetectable and `FEEDBACK_UNVERIFIED` says so.
"""

from collections.abc import Sequence
import dataclasses
import enum

import numpy
import pandas
from scipy import stats

from scmle.contracts import qualification
from scmle.data.robustness import shift

COMPLETE = "complete"
LOGGED_PROPENSITY = "logged_propensity"
AUDIT_SLICE = "audit_slice"
UNVERIFIED = "unverified"
MECHANISMS = (COMPLETE, LOGGED_PROPENSITY, AUDIT_SLICE, UNVERIFIED)
FEEDBACK_UNVERIFIED = (
    "no randomized bypass slice: a feedback loop cannot be detected from "
    "logs, retraining on the model's own decisions is unverified"
)


class Mechanism(enum.StrEnum):
    """How the labels of a batch were selected.

    Attributes:
        COMPLETE: Every label was observed.
        LOGGED_PROPENSITY: The probability of observing each label was logged.
        AUDIT_SLICE: A randomized audit slice was labelled.
        UNVERIFIED: The selection cannot be tested.
    """

    COMPLETE = COMPLETE
    LOGGED_PROPENSITY = LOGGED_PROPENSITY
    AUDIT_SLICE = AUDIT_SLICE
    UNVERIFIED = UNVERIFIED


def is_matured(batch: int, now: int, window: int) -> bool:
    """Reports whether a case predicted in `batch` has matured at batch `now`.

    Args:
        batch: Batch in which the case was predicted.
        now: Current batch.
        window: Batches an outcome needs to be known.

    Returns:
        True if at least `window` batches have passed.

    Raises:
        ValueError: If `window` is negative.
    """
    if window < 0:
        raise ValueError("window must not be negative")
    return now - batch >= window


def permits_action(mechanism: Mechanism) -> bool:
    """Reports whether a retrain, promotion or rollback may rest on this stream.

    Args:
        mechanism: The stream's label mechanism.

    Returns:
        False only for `unverified`.
    """
    return mechanism is not Mechanism.UNVERIFIED


@dataclasses.dataclass(frozen=True, slots=True)
class BatchEstimate:
    """The risk estimate of one matured batch.

    Attributes:
        mechanism: How the labels were selected.
        cases: Matured cases in the batch.
        labelled: Cases whose label was used.
        mean: Estimated mean loss over all matured cases, in loss units
            (None when unverified).
        scaled: `mean` rescaled into ``[0, 1]`` for a betting monitor (None
            when unverified).
        scale: The upper bound of `mean` (``bound / epsilon``).
    """

    mechanism: Mechanism
    cases: int
    labelled: int
    mean: float | None
    scaled: float | None
    scale: float


def classify(
    observed: numpy.ndarray,
    propensity: numpy.ndarray | None,
    audit: numpy.ndarray | None,
    epsilon: float,
) -> Mechanism:
    """Determines the label mechanism a batch supports.

    Args:
        observed: Whether each matured case's label was observed.
        propensity: Logged probability of observing each label, or None.
        audit: Whether each case was in the audit slice, or None.
        epsilon: The smallest propensity the policy declares.

    Returns:
        `complete` when every label was observed; `logged_propensity` when every
        propensity is logged and at least `epsilon`; `audit_slice` when an audit
        slice exists and `epsilon > 0`; otherwise `unverified`.
    """
    if bool(numpy.all(observed)):
        return Mechanism.COMPLETE
    if propensity is not None and epsilon > 0.0:
        values = numpy.asarray(propensity, dtype=float)
        if numpy.all(numpy.isfinite(values)) and numpy.all(
            values >= epsilon - 1e-12
        ):
            return Mechanism.LOGGED_PROPENSITY
    if audit is not None and epsilon > 0.0 and bool(numpy.any(audit)):
        return Mechanism.AUDIT_SLICE
    return Mechanism.UNVERIFIED


def estimate_batch(
    loss: numpy.ndarray,
    observed: numpy.ndarray,
    propensity: numpy.ndarray | None,
    audit: numpy.ndarray | None,
    epsilon: float,
    bound: float = 1.0,
    signed: bool = False,
) -> BatchEstimate:
    """Estimates the Horvitz-Thompson mean loss of one matured batch.

    Args:
        loss: Loss of each matured case; entries of unobserved cases are ignored
            (they may be NaN).
        observed: Whether each label was observed.
        propensity: Logged probability of observing each label, or None.
        audit: Whether each case was in the audit slice, or None.
        epsilon: Smallest propensity (and audit rate) declared by the policy.
        bound: Upper bound of a loss.
        signed: Allow values in ``[-bound, bound]`` (paired differences).

    Returns:
        The estimate. For `audit_slice` the cases outside the slice are ignored
        and the audit rate is `epsilon`.

    Raises:
        ValueError: On misaligned inputs, or a loss outside ``[0, bound]``.
    """
    losses = numpy.asarray(loss, dtype=float)
    seen = numpy.asarray(observed, dtype=bool)
    if losses.shape != seen.shape or losses.size == 0:
        raise ValueError("loss and observed must be aligned and non-empty")
    if propensity is not None and numpy.shape(propensity) != seen.shape:
        raise ValueError("propensity must be aligned with the cases")
    if audit is not None and numpy.shape(audit) != seen.shape:
        raise ValueError("audit must be aligned with the cases")
    used = losses[seen]
    floor = -bound if signed else 0.0
    if (
        numpy.any(~numpy.isfinite(used))
        or numpy.any(used < floor)
        or numpy.any(used > bound)
    ):
        raise ValueError("observed losses must be finite and within [0, bound]")
    mechanism = classify(seen, propensity, audit, epsilon)
    cases = int(seen.size)
    if mechanism is Mechanism.UNVERIFIED:
        return BatchEstimate(mechanism, cases, 0, None, None, bound)
    if mechanism is Mechanism.COMPLETE:
        mean = float(used.mean())
        return BatchEstimate(
            mechanism, cases, int(seen.sum()), mean, mean / bound, bound
        )
    scale = bound / epsilon
    if mechanism is Mechanism.LOGGED_PROPENSITY:
        weights = 1.0 / numpy.asarray(propensity, dtype=float)
        total = float(
            numpy.sum(numpy.where(seen, losses, 0.0) * weights * seen)
        )
        counted = int(seen.sum())
    else:
        slice_mask = numpy.asarray(audit, dtype=bool) & seen
        total = float(numpy.sum(numpy.where(slice_mask, losses, 0.0))) / epsilon
        counted = int(slice_mask.sum())
    mean = total / cases
    return BatchEstimate(
        mechanism, cases, counted, mean, min(mean / scale, 1.0), scale
    )


@dataclasses.dataclass(frozen=True, slots=True)
class RandomizationCheck:
    """Result of the audit-slice uniformity test.

    Attributes:
        audited: Cases that were audited.
        cases: Cases logged.
        rate: Declared audit rate.
        p_value: Two-sided exact binomial p-value.
        passed: Whether the p-value is above the level.
    """

    audited: int
    cases: int
    rate: float
    p_value: float
    passed: bool


def check_randomization(
    audit: numpy.ndarray, rate: float, alpha: float = 0.001
) -> RandomizationCheck:
    """Tests that the audit slice's draws have the declared rate.

    A two-sided exact binomial test of the number of audited cases against
    `rate`. It catches a slice that is not drawn at the declared rate; it cannot
    show independence from the covariates or the outcome.

    Args:
        audit: Whether each logged case was audited.
        rate: Declared audit rate, in ``(0, 1)``.
        alpha: Level of the test.

    Returns:
        The audited and logged counts, the declared rate, the p-value, and
        whether the p-value is above `alpha`.

    Raises:
        ValueError: If `rate` is not strictly between 0 and 1.
    """
    flags = numpy.asarray(audit, dtype=bool)
    if not 0.0 < rate < 1.0:
        raise ValueError("rate must be strictly between 0 and 1")
    result = stats.binomtest(int(flags.sum()), int(flags.size), rate)
    return RandomizationCheck(
        int(flags.sum()),
        int(flags.size),
        rate,
        float(result.pvalue),
        bool(result.pvalue > alpha),
    )


def bypass_test(
    bypass_features: pandas.DataFrame,
    model_features: pandas.DataFrame,
    seed: int,
    permutations: int = 199,
    alpha: float = 0.05,
) -> qualification.Check:
    """Tests whether the model's decisions shift the next-period features.

    Compares next-period features of cases whose decision bypassed the model
    (randomized, logged) with those of model-decided cases, with a kernel
    two-sample test (`scmle.data.robustness.shift.mmd_test`). A small p-value is
    evidence that the model's decisions change the data it will see.

    Args:
        bypass_features: Next-period features of the bypass slice.
        model_features: Next-period features of model-decided cases.
        seed: Seed of the permutation test.
        permutations: Number of permutations.
        alpha: Level of the test.

    Returns:
        The check from the two-sample test.
    """
    return shift.mmd_test(
        model_features,
        bypass_features,
        seed,
        permutations=permutations,
        alpha=alpha,
    )


def feedback_verdict(
    bypass_rate: float, check: qualification.Check | None
) -> str:
    """Summarizes the feedback-loop evidence.

    Args:
        bypass_rate: Share of cases that bypassed the model.
        check: The result of `bypass_test`, or None.

    Returns:
        `unverified` without a bypass slice or check, `loop_detected` if the
        p-value is at most 0.05, otherwise `no_loop_detected`.
    """
    if bypass_rate <= 0.0 or check is None:
        return UNVERIFIED
    if check.p_value is not None and check.p_value <= 0.05:
        return "loop_detected"
    return "no_loop_detected"


def stream_mechanism(batches: Sequence[BatchEstimate]) -> Mechanism:
    """Finds the weakest mechanism over a window of batches.

    Args:
        batches: The batch estimates.

    Returns:
        The mechanism that comes last in the order complete, logged propensity,
        audit slice, unverified; `complete` for no batches.
    """
    order = {m: i for i, m in enumerate(Mechanism)}
    weakest = Mechanism.COMPLETE
    for item in batches:
        if order[item.mechanism] > order[weakest]:
            weakest = item.mechanism
    return weakest
