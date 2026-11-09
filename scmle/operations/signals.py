"""Windowed monitors: service health, feature quality, drift, delayed labels.

`SignalMonitor` turns raw events into `Signal` rows over a sliding window:

* infrastructure: server error rate, availability and p95 latency from
  request events (every request is counted; none is sampled);
* data quality: missing-rate increase, unseen categories, the share of
  requests rejected as invalid, and the noise-adjusted PSI of each feature
  against the training reference (`scmle.monitoring.reference`), from feature
  rows (sampled at ``sample_rate``);
* model performance: the mean loss of cases whose outcome has arrived
  (delayed labels) against ``reference_loss`` plus a tolerance, using the same
  names as `scmle.governance.loop.LoopPolicy`.

A window with too few samples is ``insufficient_data``: it is reported and
never treated as healthy. Nothing here decides what to do; the classifier
(`scmle.operations.classify`) and the loop do.
"""

from collections import deque
from collections.abc import Sequence
import dataclasses
import enum
import threading
from typing import Any

import numpy as np
import pandas as pd

from scmle.contracts import observability
from scmle.monitoring import monitor as monitor_module
from scmle.monitoring import reference as reference_module

INVALID_STATUSES = frozenset({400, 409, 413, 415, 422})


class Category(enum.StrEnum):
    """Where a fault lives."""

    INFRASTRUCTURE = "infrastructure"
    DATA_QUALITY = "data_quality"
    MODEL_PERFORMANCE = "model_performance"


class Status(enum.StrEnum):
    """State of one signal in its window."""

    OK = "ok"
    BREACHED = "breached"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclasses.dataclass(frozen=True, slots=True)
class Signal:
    """One measurement over a window and how it compares with its limit.

    Attributes:
        name: Metric name.
        category: Where a breach would point.
        status: ok, breached or insufficient_data.
        value: Observed value (None when it could not be computed or
            there were too few samples to judge).
        threshold: The limit it was compared with.
        direction: ``max`` if larger is worse, ``min`` if smaller is worse.
        samples: Observations behind the value.
        detail: Extra facts, such as the worst features.
    """

    name: str
    category: Category
    status: Status
    value: float | None
    threshold: float
    direction: str
    samples: int
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def breached(self) -> bool:
        """Whether the signal crossed its limit."""
        return self.status is Status.BREACHED

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready record.

        Returns:
            The signal's fields as a dict, with enums as their values.
        """
        return {
            "name": self.name,
            "category": self.category.value,
            "status": self.status.value,
            "value": self.value,
            "threshold": self.threshold,
            "direction": self.direction,
            "samples": self.samples,
            "detail": dict(self.detail),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class WindowPolicy:
    """The window every monitor looks at.

    Attributes:
        window_seconds: Length of the sliding window.
        min_samples: Fewest observations a signal needs to be judged.
        sample_rate: Share of feature rows kept, in (0, 1].
        seed: Seed of the row sampler (deterministic).
    """

    window_seconds: float = 300.0
    min_samples: int = 30
    sample_rate: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        """Validates ranges.

        Raises:
            ValueError: If the window length or minimum sample count is not
                positive, or the sample rate is outside (0, 1].
        """
        if self.window_seconds <= 0 or self.min_samples < 1:
            raise ValueError("window_seconds and min_samples must be positive")
        if not 0.0 < self.sample_rate <= 1.0:
            raise ValueError("sample_rate must be in (0, 1]")


@dataclasses.dataclass(frozen=True, slots=True)
class SignalThresholds:
    """Limits per signal. The defaults are starting points, not findings.

    Attributes:
        server_error_rate: Largest tolerated share of 5xx.
        availability: Smallest tolerated share of requests that were not
            5xx or 429.
        latency_p95_seconds: Largest tolerated p95 latency.
        invalid_request_rate: Largest tolerated share of rejected inputs.
        missing_rate_increase: Largest rise of a feature's missing share over
            its reference.
        drift: Thresholds shared with `scmle.monitoring.monitor`; the
            ``psi_high`` limit applies to the largest feature PSI and the
            ``unseen_category_rate`` limit to the largest unseen-category
            rate.
        reference_loss: Mean loss at training time.
        loss_tolerance: Largest tolerated rise of the windowed mean loss
            over ``reference_loss``; the loss limit is their sum.
    """

    server_error_rate: float = 0.05
    availability: float = 0.99
    latency_p95_seconds: float = 1.0
    invalid_request_rate: float = 0.05
    missing_rate_increase: float = 0.05
    drift: monitor_module.MonitoringThresholds = dataclasses.field(
        default_factory=monitor_module.MonitoringThresholds
    )
    reference_loss: float = 0.1
    loss_tolerance: float = 0.05


KIND_OF = {
    "server_error_rate": observability.SignalKind.ERROR,
    "availability": observability.SignalKind.ERROR,
    "latency_p95_seconds": observability.SignalKind.LATENCY,
    "invalid_request_rate": observability.SignalKind.DATA_QUALITY,
    "feature_missing_increase": observability.SignalKind.DATA_QUALITY,
    "unseen_category_rate": observability.SignalKind.DATA_QUALITY,
    "feature_psi_max": observability.SignalKind.DRIFT,
    "windowed_loss": observability.SignalKind.ERROR,
}


def judge(
    name: str,
    category: Category,
    value: float | None,
    threshold: float,
    direction: str,
    samples: int,
    minimum: int,
    **detail: Any,
) -> Signal:
    """Compares a value with its limit, or reports too little data.

    A ``max`` signal breaches when the value is above the limit and a ``min``
    signal when it is below it; a value equal to the limit is ok.

    Args:
        name: Signal name.
        category: Category the signal belongs to.
        value: The measured value, or None if it could not be computed.
        threshold: The limit.
        direction: ``max`` if the value must not exceed the limit, ``min``
            if it must not fall below it.
        samples: Number of samples behind the value.
        minimum: Fewest samples for a judgement.
        **detail: Extra facts kept on the signal.

    Returns:
        The signal: insufficient data if the value is None or the samples
        are fewer than the minimum (its value is dropped to None when the
        samples are too few), otherwise breached or ok against the limit.
        A direction other than ``max`` or ``min`` is always ok.
    """
    if value is None or samples < minimum:
        status = Status.INSUFFICIENT_DATA
        value = None if samples < minimum else value
    elif (direction == "max" and value > threshold) or (
        direction == "min" and value < threshold
    ):
        status = Status.BREACHED
    else:
        status = Status.OK
    return Signal(
        name, category, status, value, threshold, direction, samples, detail
    )


class SignalMonitor:
    """Collects events and evaluates every signal over a sliding window."""

    def __init__(
        self,
        policy: WindowPolicy = WindowPolicy(),  # noqa: B008
        thresholds: SignalThresholds = SignalThresholds(),  # noqa: B008
        profile: reference_module.ReferenceProfile | None = None,
    ) -> None:
        """Creates a monitor.

        Args:
            policy: Window, sample size and sampling rate.
            thresholds: Limits.
            profile: Training reference for drift and feature quality; without
                it the feature-missing, unseen-category and drift signals are
                not produced (the infrastructure, invalid-request and
                performance signals still are).
        """
        self.policy = policy
        self.thresholds = thresholds
        self.profile = profile
        self.requests: deque[tuple[float, str, int, float]] = deque()
        self.features: deque[tuple[float, pd.DataFrame]] = deque()
        self.outcomes: deque[tuple[float, np.ndarray]] = deque()
        self.sampler = np.random.default_rng(policy.seed)
        self.lock = threading.Lock()

    def record_request(
        self, at: float, route: str, status: int, seconds: float
    ) -> None:
        """Records one finished request (epoch seconds ``at``).

        Args:
            at: Time of the request, in epoch seconds.
            route: The route served.
            status: The HTTP status.
            seconds: Duration of the request in seconds.
        """
        with self.lock:
            self.requests.append((at, route, status, seconds))

    def observe_features(self, at: float, frame: pd.DataFrame) -> int:
        """Keeps a sample of the rows of a scored batch; returns the count.

        Args:
            at: Time of the batch, in epoch seconds.
            frame: The scored rows; each is kept independently with
                probability equal to the policy's sample rate.

        Returns:
            The number of rows kept.
        """
        if frame.empty:
            return 0
        keep = self.sampler.random(len(frame)) < self.policy.sample_rate
        kept = frame.loc[keep].reset_index(drop=True)
        if not kept.empty:
            with self.lock:
                self.features.append((at, kept))
        return len(kept)

    def observe_outcomes(
        self, at: float, losses: Sequence[float] | np.ndarray
    ) -> None:
        """Records losses in [0, 1] of cases whose outcome has arrived.

        An empty sequence records nothing.

        Args:
            at: Time the outcomes arrived, in epoch seconds.
            losses: The loss of each case.

        Raises:
            ValueError: If a loss is outside [0, 1] or not finite.
        """
        values = np.asarray(list(losses), dtype=float)
        if values.size == 0:
            return
        if (
            not np.all(np.isfinite(values))
            or values.min() < 0
            or values.max() > 1
        ):
            raise ValueError("losses must be finite and within [0, 1]")
        with self.lock:
            self.outcomes.append((at, values))

    def prune(self, now: float) -> None:
        """Drops recorded events older than the window.

        Events before ``now - window_seconds`` are dropped from the front of
        each queue; the scan stops at the first event inside the window, so
        events are assumed to be recorded in time order. Does not take the
        lock.

        Args:
            now: The current time, in epoch seconds.
        """
        start = now - self.policy.window_seconds
        for queue in (self.requests, self.features, self.outcomes):
            while queue and queue[0][0] < start:
                queue.popleft()

    def evaluate(self, now: float) -> tuple[Signal, ...]:
        """Evaluates every signal over the window ending at ``now``.

        Prunes events older than the window first. Events stamped after
        ``now`` are kept but not counted.

        Args:
            now: The current time, in epoch seconds.

        Returns:
            The signals in order: server error rate, availability and p95
            latency; the invalid-request rate; the feature-missing,
            unseen-category and drift signals when a reference profile was
            given; the windowed loss.
        """
        with self.lock:
            self.prune(now)
            requests = [r for r in self.requests if r[0] <= now]
            frames = [f for at, f in self.features if at <= now]
            outcomes = [o for at, o in self.outcomes if at <= now]
        found = self.service_signals(requests)
        found += self.data_signals(frames, requests)
        found += self.performance_signals(outcomes)
        return tuple(found)

    def service_signals(
        self, requests: list[tuple[float, str, int, float]]
    ) -> list[Signal]:
        """Judges server error rate, availability and p95 latency.

        Args:
            requests: The requests in the window, as (time, route, status,
                seconds).

        Returns:
            The server error rate (share of 5xx), availability (share not 5xx
            or 429) and 95th percentile latency signals, in that order. All
            have no value when there are no requests.
        """
        size = len(requests)
        limits = self.thresholds
        minimum = self.policy.min_samples
        statuses = np.array([r[2] for r in requests], dtype=int)
        seconds = np.array([r[3] for r in requests], dtype=float)
        server = float(np.mean(statuses >= 500)) if size else None
        unavailable = (
            float(np.mean((statuses >= 500) | (statuses == 429)))
            if size
            else None
        )
        p95 = float(np.percentile(seconds, 95)) if size else None
        return [
            judge(
                "server_error_rate",
                Category.INFRASTRUCTURE,
                server,
                limits.server_error_rate,
                "max",
                size,
                minimum,
            ),
            judge(
                "availability",
                Category.INFRASTRUCTURE,
                None if unavailable is None else 1.0 - unavailable,
                limits.availability,
                "min",
                size,
                minimum,
            ),
            judge(
                "latency_p95_seconds",
                Category.INFRASTRUCTURE,
                p95,
                limits.latency_p95_seconds,
                "max",
                size,
                minimum,
            ),
        ]

    def data_signals(
        self,
        frames: list[pd.DataFrame],
        requests: list[tuple[float, str, int, float]],
    ) -> list[Signal]:
        """Judges rejected inputs, feature quality and drift.

        Args:
            frames: The sampled feature rows in the window.
            requests: The requests in the window, as (time, route, status,
                seconds).

        Returns:
            The invalid-request-rate signal (share of requests with status
            400, 409, 413, 415 or 422) and, when a reference profile exists,
            the signals for the largest increase in missing share over the
            reference (a feature absent from the rows counts as entirely
            missing), the largest unseen-category rate and the largest
            noise-adjusted PSI. The feature signals have no value until the
            sampled rows number at least the minimum.
        """
        limits = self.thresholds
        minimum = self.policy.min_samples
        size = len(requests)
        invalid = (
            float(np.mean([r[2] in INVALID_STATUSES for r in requests]))
            if size
            else None
        )
        found = [
            judge(
                "invalid_request_rate",
                Category.DATA_QUALITY,
                invalid,
                limits.invalid_request_rate,
                "max",
                size,
                minimum,
            )
        ]
        if self.profile is None:
            return found
        rows = pd.concat(frames, ignore_index=True) if frames else None
        count = 0 if rows is None else len(rows)
        missing: dict[str, float] = {}
        drift: dict[str, float] = {}
        unseen: dict[str, float] = {}
        if rows is not None and count >= minimum:
            for name, reference in self.profile.features.items():
                if name not in rows.columns:
                    missing[name] = 1.0 - reference.missing_rate
                    continue
                column = rows[name]
                missing[name] = (
                    float(column.isna().mean()) - reference.missing_rate
                )
                drift[name] = reference.adjusted_psi(column)
                unseen[name] = reference.unseen_rate(column)
        worst_missing = max(missing.values()) if missing else None
        worst_drift = max(drift.values()) if drift else None
        worst_unseen = max(unseen.values()) if unseen else None
        ranked = sorted(drift, key=lambda key: drift[key], reverse=True)[:3]
        found += [
            judge(
                "feature_missing_increase",
                Category.DATA_QUALITY,
                worst_missing,
                limits.missing_rate_increase,
                "max",
                count,
                minimum,
                features=sorted(
                    key
                    for key, value in missing.items()
                    if value > limits.missing_rate_increase
                ),
            ),
            judge(
                "unseen_category_rate",
                Category.DATA_QUALITY,
                worst_unseen,
                limits.drift.unseen_category_rate,
                "max",
                count,
                minimum,
                features=sorted(
                    key
                    for key, value in unseen.items()
                    if value > limits.drift.unseen_category_rate
                ),
            ),
            judge(
                "feature_psi_max",
                Category.DATA_QUALITY,
                worst_drift,
                limits.drift.psi_high,
                "max",
                count,
                minimum,
                worst=[(key, round(drift[key], 4)) for key in ranked],
            ),
        ]
        return found

    def performance_signals(self, outcomes: list[np.ndarray]) -> list[Signal]:
        """Judges the mean loss of cases whose outcome arrived.

        Args:
            outcomes: The loss arrays recorded in the window.

        Returns:
            The windowed-loss signal, limited by the reference loss plus the
            tolerance.
        """
        limits = self.thresholds
        values = np.concatenate(outcomes) if outcomes else np.array([])
        mean = float(values.mean()) if values.size else None
        return [
            judge(
                "windowed_loss",
                Category.MODEL_PERFORMANCE,
                mean,
                limits.reference_loss + limits.loss_tolerance,
                "max",
                int(values.size),
                self.policy.min_samples,
                reference_loss=limits.reference_loss,
            )
        ]


def zero_one_losses(
    predictions: Sequence[Any], labels: Sequence[Any]
) -> np.ndarray:
    """Loss 1 where a prediction differs from its label, else 0.

    Args:
        predictions: The predictions.
        labels: The observed labels, one per prediction.

    Returns:
        An array of 0.0 and 1.0.

    Raises:
        ValueError: If the lengths differ.
    """
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must align")
    return np.array(
        [
            0.0 if p == y else 1.0
            for p, y in zip(predictions, labels, strict=True)
        ]
    )


def as_metric_samples(
    signals: Sequence[Signal],
    timestamp: str,
    labels: dict[str, str] | None = None,
) -> list[observability.MetricSample]:
    """Converts signals that have a value to `MetricSample` rows.

    Args:
        signals: The evaluated signals.
        timestamp: Timestamp for every sample.
        labels: Labels added to every sample.

    Returns:
        One sample for each signal that has a value, in the signals' order,
        with the unit ``s`` for names containing ``seconds`` and ``ratio``
        otherwise.

    Raises:
        KeyError: If a signal's name has no metric kind.
    """
    return [
        observability.MetricSample(
            name=signal.name,
            kind=KIND_OF[signal.name],
            value=float(signal.value),
            unit="ratio" if "seconds" not in signal.name else "s",
            timestamp=timestamp,
            labels=dict(labels or {}),
        )
        for signal in signals
        if signal.value is not None
    ]
