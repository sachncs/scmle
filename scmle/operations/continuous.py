"""Continuous evaluation on top of the closed loop.

What this adds to `scmle.governance.loop.Loop`, and what it leaves alone:

* **Feeding.** Delayed labels and observed business outcomes become losses in
  [0, 1] and go to `Loop.mature`, for the model in service and for each
  challenger on the same cases. Raw labels and outcomes are kept.
* **Retraining policy.** `RetrainPolicy` says when retraining is *recommended*
  (the default) or *triggered*. Either way the question reaches a person
  through `Loop.escalate`, the certified path. The loop's own alarm still does
  not gate retraining (`alarm_gates_retrain` stays False because its
  detection-delay gates missed their floors); nothing here sets it or claims
  otherwise. A trigger only trains a *challenger*: the challenger still has to
  win the loop's e-process, pass the gate and be approved before it serves.
* **History.** Dataset versions (by content digest), training runs, proposals,
  comparisons, replacements and post-promotion checks are appended to a
  hash-chained file, so what a model was trained on can be recovered.
* **Replacement.** `replace` refuses unless the loop recorded a promotion
  proposal for the challenger, the promotion gate admits the build (signed
  bundle, independent approval, scan, approver, staging record for
  production) and the release registry confirms the runtime.
* **After promotion.** `post_promotion_check` compares selected operational
  and business metrics before and after with bootstrap intervals and says
  whether the model helped, with the uncertainty, or that it cannot tell.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import hashlib
import json
import math
import pathlib
import time
from typing import Any

import numpy
import pandas
from scipy import stats

from scmle import durability
from scmle.deploy import control
from scmle.deploy import gate
from scmle.governance import deployment_ledger
from scmle.governance import never_list
from scmle.governance import release_state
import scmle.governance.loop
import scmle.monitoring.champion
from scmle.operations import signals

RECOMMEND = "recommend"
TRIGGER = "trigger"
MODES = (RECOMMEND, TRIGGER)
RETRAIN_QUESTION = "retrain_recommendation"
DEFAULT_IF_UNANSWERED = "no_retrain"
HISTORY_FILE = "training_history.jsonl"
LABELS_FILE = "retained_outcomes.jsonl"
DATASET_DIRECTORY = "datasets"
DATA_FAULTS = frozenset(
    {"invalid_request_rate", "feature_missing_increase", "unseen_category_rate"}
)


def canonical_csv(frame: pandas.DataFrame) -> bytes:
    """The bytes a dataset version is computed from.

    Args:
        frame: The dataset.

    Returns:
        The UTF-8 CSV of the frame with columns sorted by name, no index
        and newline line endings.
    """
    ordered = frame.reindex(sorted(frame.columns), axis=1)
    return ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")


def dataset_version(frame: pandas.DataFrame) -> str:
    """``sha256:<hex>`` of the canonical CSV of a dataset.

    Args:
        frame: The dataset.

    Returns:
        The version string.
    """
    return "sha256:" + hashlib.sha256(canonical_csv(frame)).hexdigest()


def load_dataset(
    directory: str | pathlib.Path, version: str
) -> pandas.DataFrame:
    """Reads a preserved dataset back and checks its digest.

    Args:
        directory: The directory holding the preserved datasets.
        version: The dataset version, as returned by `dataset_version`. The
            file read is named after it, with the colon replaced by a hyphen.

    Returns:
        The stored dataset.

    Raises:
        ValueError: If the stored copy no longer has the digest it is named
            for.
    """
    path = pathlib.Path(directory) / DATASET_DIRECTORY
    frame = pandas.read_csv(
        path / (version.replace(":", "-") + ".csv"),
        float_precision="round_trip",
    )
    if dataset_version(frame) != version:
        raise ValueError(f"the stored copy of {version} has changed")
    return frame


class TrainingHistory:
    """An append-only, hash-chained record of what was trained and decided."""

    def __init__(self, path: str | pathlib.Path) -> None:
        """Opens (or starts) the history at ``path``.

        Args:
            path: The history file.
        """
        self.path = pathlib.Path(path)

    def rows(self) -> list[dict[str, Any]]:
        """Every row, verified.

        Returns:
            The rows, oldest first; empty if the file does not exist.

        Raises:
            ValueError: If the chain is broken.
        """
        if not self.path.exists():
            return []
        found = []
        previous = "0" * 64
        for number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines()
        ):
            row = json.loads(line)
            content = {
                k: row[k] for k in ("seq", "at", "kind", "facts", "prev")
            }
            digest = hashlib.sha256(
                json.dumps(content, sort_keys=True).encode()
            ).hexdigest()
            if (
                row["seq"] != number
                or row["prev"] != previous
                or (row["hash"] != digest)
            ):
                raise ValueError(f"training history broken at row {number}")
            previous = row["hash"]
            found.append(row)
        return found

    def record(self, kind: str, at: float, **facts: Any) -> dict[str, Any]:
        """Appends one row and returns it.

        Args:
            kind: The kind of event.
            at: Time of the event, in epoch seconds; stored rounded to three
                decimals.
            **facts: Facts recorded with the event; values that JSON cannot
                encode are stored as their string form.

        Returns:
            The row, with its sequence number, previous hash and own hash.
        """
        rows = self.rows()
        content = {
            "seq": len(rows),
            "at": round(at, 3),
            "kind": kind,
            "facts": json.loads(json.dumps(facts, default=str)),
            "prev": rows[-1]["hash"] if rows else "0" * 64,
        }
        row = {
            **content,
            "hash": hashlib.sha256(
                json.dumps(content, sort_keys=True).encode()
            ).hexdigest(),
        }
        durability.append_text(
            self.path, json.dumps(row, sort_keys=True) + "\n"
        )
        return row

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        """Rows of one kind, oldest first.

        Args:
            kind: The kind of event.

        Returns:
            The verified rows of that kind.
        """
        return [row for row in self.rows() if row["kind"] == kind]


@dataclasses.dataclass(frozen=True, slots=True)
class RetrainPolicy:
    """When retraining is recommended or triggered.

    A *primary* reason is needed: a windowed loss above the reference plus
    tolerance with at least ``min_outcomes`` outcomes, an L1 alarm of the loop,
    or a model older than ``max_age_days``. Input drift supports a primary
    reason and is never enough alone. Retraining is held back while an
    infrastructure fault or a data-quality fault is open, because both would
    teach the next model the fault, and during ``cooldown_batches`` after the
    last proposal.

    Attributes:
        mode: ``recommend`` (default) or ``trigger``.
        min_outcomes: Fewest arrived outcomes behind a performance reason.
        max_age_days: Age that alone justifies a proposal, or None.
        cooldown_batches: Batches between proposals.
    """

    mode: str = RECOMMEND
    min_outcomes: int = 100
    max_age_days: float | None = None
    cooldown_batches: int = 20

    def __post_init__(self) -> None:
        """Validates the choice and the bounds.

        Raises:
            ValueError: If the mode is unknown, the minimum outcomes is below
                one, the cooldown is negative, or the maximum age is not
                positive.
        """
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.min_outcomes < 1 or self.cooldown_batches < 0:
            raise ValueError("min_outcomes >= 1 and cooldown_batches >= 0")
        if self.max_age_days is not None and self.max_age_days <= 0:
            raise ValueError("max_age_days must be positive")


@dataclasses.dataclass(frozen=True, slots=True)
class RetrainDecision:
    """What the policy decided for one review.

    Attributes:
        action: ``none``, ``recommend`` or ``trigger``.
        reasons: Primary and supporting reasons.
        held_back: Why a proposal was not made though a reason existed.
        escalation_id: The pending escalation carrying the question.
        challenger: Name of the challenger a trigger started.
        alarm_gates_retrain: Copied from the loop policy, for the record.
    """

    action: str
    reasons: tuple[str, ...]
    held_back: str | None
    escalation_id: str | None
    challenger: str | None
    alarm_gates_retrain: bool

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready record.

        Returns:
            The decision's fields as a dict.
        """
        return dataclasses.asdict(self)


def judge_retraining(
    policy: RetrainPolicy,
    rows: Sequence[signals.Signal],
    alarm_since_last: bool,
    age_days: float | None,
    batches_since_proposal: int | None,
) -> tuple[list[str], list[str], str | None]:
    """The pure part of the policy.

    Args:
        policy: The retraining policy.
        rows: The current window's signals.
        alarm_since_last: Whether the loop's alarm fired since the last
            proposal.
        age_days: Age of the model in days, or None.
        batches_since_proposal: Batches since the last proposal, or None if
            there was none.

    Returns:
        Primary reasons, supporting reasons and the reason for holding back
        (None if nothing holds a proposal back).
    """
    by_name = {row.name: row for row in rows}
    primary: list[str] = []
    support: list[str] = []
    loss = by_name.get("windowed_loss")
    if (
        loss is not None
        and loss.breached
        and loss.samples >= policy.min_outcomes
    ):
        primary.append(
            f"windowed loss {loss.value:.3f} is above {loss.threshold:.3f} "
            f"over {loss.samples} outcomes"
        )
    if alarm_since_last:
        primary.append("the loop's L1 alarm fired")
    if (
        policy.max_age_days is not None
        and age_days is not None
        and age_days > policy.max_age_days
    ):
        primary.append(f"the model is {age_days:.0f} days old")
    psi = by_name.get("feature_psi_max")
    if psi is not None and psi.breached:
        support.append("input drift is above its threshold")
    held = None
    infra = [
        r.name
        for r in rows
        if r.breached and r.category.value == "infrastructure"
    ]
    data = [r.name for r in rows if r.breached and r.name in DATA_FAULTS]
    if infra:
        held = "an infrastructure fault is open: " + ", ".join(infra)
    elif data:
        held = "a data-quality fault is open: " + ", ".join(data)
    elif (
        batches_since_proposal is not None
        and batches_since_proposal < policy.cooldown_batches
    ):
        held = (
            f"the last proposal was {batches_since_proposal} batches ago "
            f"(cooldown {policy.cooldown_batches})"
        )
    return primary, support, held


@dataclasses.dataclass(frozen=True, slots=True)
class Indicator:
    """A metric watched after a promotion.

    Attributes:
        name: Metric name.
        kind: ``operational`` or ``business``.
        better: ``higher`` or ``lower``.
        margin: Smallest difference that matters, in the metric's unit.
        primary: Whether the model must improve it to count as helping.
    """

    name: str
    kind: str
    better: str
    margin: float = 0.0
    primary: bool = False

    def __post_init__(self) -> None:
        """Validates the choices.

        Raises:
            ValueError: If the kind is not operational or business, `better`
                is not higher or lower, or the margin is negative or not finite.
        """
        if self.kind not in {"operational", "business"}:
            raise ValueError("kind must be operational or business")
        if self.better not in {"higher", "lower"}:
            raise ValueError("better must be higher or lower")
        if self.margin < 0 or not math.isfinite(self.margin):
            raise ValueError("margin must be finite and not negative")


def bootstrap_difference(
    before: numpy.ndarray,
    after: numpy.ndarray,
    alpha: float,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile interval of mean(after) - mean(before).

    Args:
        before: Observations of the earlier period.
        after: Observations of the later period.
        alpha: Error level; the interval spans the alpha/2 to 1 - alpha/2
            percentiles.
        resamples: Number of bootstrap resamples.
        seed: Seed of the resampling.

    Returns:
        The lower and upper bounds of the interval.
    """
    rng = numpy.random.default_rng(seed)
    first = rng.choice(before, size=(resamples, len(before))).mean(axis=1)
    second = rng.choice(after, size=(resamples, len(after))).mean(axis=1)
    gap = second - first
    low, high = numpy.quantile(gap, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def post_promotion_check(
    specs: Sequence[Indicator],
    before: Mapping[str, Sequence[float]],
    after: Mapping[str, Sequence[float]],
    *,
    alpha: float = 0.05,
    min_samples: int = 30,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Did the promoted model help? An interval and a verdict per metric.

    The error level is split over the metrics (Bonferroni), so the intervals
    hold together at ``1 - alpha``. A metric is ``improved`` or ``worse`` only
    when its whole interval lies beyond ``margin`` on that side; otherwise it
    is ``inconclusive``, as it is when either period has fewer than
    ``min_samples`` observations. The overall verdict is ``harmed`` if any
    metric is worse, ``helped`` if there is at least one primary metric, every
    primary metric improved and none is worse, otherwise ``inconclusive``.

    This is a comparison of two periods. It cannot separate the model from
    anything else that changed between them; the statement says so.

    Args:
        specs: The metrics to judge.
        before: Observations of each metric in the earlier period.
        after: Observations of each metric in the later period.
        alpha: Joint error level over all metrics.
        min_samples: Fewest observations in each period for a verdict.
        resamples: Number of bootstrap resamples.
        seed: Seed of the resampling.

    Returns:
        The overall verdict, one record per metric in the order of `specs`
        (verdict, interval and sample counts; a metric with too few samples
        has a reason and no interval instead of the other fields) and a
        statement of what the comparison shows.

    Raises:
        ValueError: If a metric has no values for a period or ``specs`` is
            empty.
    """
    if not specs:
        raise ValueError("name at least one metric")
    adjusted = alpha / len(specs)
    results = []
    for spec in specs:
        if spec.name not in before or spec.name not in after:
            raise ValueError(f"no values for {spec.name} in both periods")
        old = numpy.asarray(before[spec.name], dtype=float)
        new = numpy.asarray(after[spec.name], dtype=float)
        if min(len(old), len(new)) < min_samples:
            results.append(
                {
                    "metric": spec.name,
                    "kind": spec.kind,
                    "verdict": "inconclusive",
                    "reason": f"fewer than {min_samples} observations "
                    "in a period",
                    "samples_before": len(old),
                    "samples_after": len(new),
                    "interval": None,
                }
            )
            continue
        low, high = bootstrap_difference(old, new, adjusted, resamples, seed)
        sign = 1.0 if spec.better == "higher" else -1.0
        good_low, good_high = sorted((sign * low, sign * high))
        if good_low > spec.margin:
            verdict = "improved"
        elif good_high < -spec.margin:
            verdict = "worse"
        else:
            verdict = "inconclusive"
        results.append(
            {
                "metric": spec.name,
                "kind": spec.kind,
                "better": spec.better,
                "margin": spec.margin,
                "primary": spec.primary,
                "verdict": verdict,
                "interval": [round(low, 6), round(high, 6)],
                "confidence": round(1.0 - adjusted, 4),
                "samples_before": len(old),
                "samples_after": len(new),
                "descriptive_difference": round(
                    float(new.mean() - old.mean()), 6
                ),
            }
        )
    verdicts = {r["metric"]: r["verdict"] for r in results}
    primaries = [s.name for s in specs if s.primary]
    if "worse" in verdicts.values():
        overall = "harmed"
    elif primaries and all(verdicts[n] == "improved" for n in primaries):
        overall = "helped"
    else:
        overall = "inconclusive"
    return {
        "overall": overall,
        "metrics": results,
        "statement": (
            f"Overall: {overall}. Each interval is the change in the mean "
            f"from the period before to the period after, at joint "
            f"confidence {1 - alpha:.0%} over {len(specs)} metrics. This "
            "compares two periods; it cannot separate the model from "
            "seasonality, traffic mix or other changes, and a holdback or "
            "randomised split would be needed to attribute the change to the "
            "model."
        ),
    }


def holdback_assignment(
    unit_ids: Sequence[Any], fraction: float, salt: str
) -> numpy.ndarray:
    """Which units stay on the previous model: a keyed hash of the unit id.

    The side of a unit depends only on its id and the `salt` (for example the
    release id), so every replica of the service and any later audit agree
    without sharing state, and a unit keeps its side for the life of the
    release. The hash is uniform, so each unit is held back independently
    with probability `fraction`.

    Args:
        unit_ids: Identifiers of the units.
        fraction: Share of units to hold back, strictly between 0 and 1.
        salt: Value mixed into the hash, for example the release id.

    Returns:
        A boolean array, True for each unit that stays on the previous
        model.

    Raises:
        ValueError: If `fraction` is not strictly between 0 and 1.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie strictly between 0 and 1")
    draws = numpy.array(
        [
            int.from_bytes(
                hashlib.sha256(f"{salt}:{unit}".encode()).digest()[:8], "big"
            )
            / 2.0**64
            for unit in unit_ids
        ]
    )
    return draws < fraction


def holdback_check(
    specs: Sequence[Indicator],
    holdback: Mapping[str, Sequence[float]],
    promoted: Mapping[str, Sequence[float]],
    *,
    expected_fraction: float,
    alpha: float = 0.05,
    min_samples: int = 30,
    resamples: int = 2000,
    seed: int = 0,
    ratio_level: float = 0.001,
) -> dict[str, Any]:
    """Did the promoted model help, against units kept on the old model?

    The comparison is concurrent: units were assigned to the previous model
    (`holdback`) or to the promoted one at random by `holdback_assignment`,
    so seasonality, traffic mix and other changes affect both sides alike and
    the difference is attributable to the model, to the extent that the
    assignment was random and the units do not influence each other.

    The check first compares the observed split with `expected_fraction` by an
    exact two-sided binomial test at `ratio_level` (a sample-ratio check). The
    group sizes are taken from the first metric in `specs`. A split that does
    not match means the assignment was not what was declared: the overall
    verdict is then capped at ``inconclusive`` (the per-metric verdicts are
    left as computed) and the statement says so.
    Otherwise the metric intervals and verdicts are those of
    `post_promotion_check`, with the holdback as the reference period.

    Args:
        specs: The metrics to judge.
        holdback: Observations of each metric for the units kept on the
            previous model.
        promoted: Observations of each metric for the units on the promoted
            model.
        expected_fraction: The declared share of units held back, strictly
            between 0 and 1.
        alpha: Joint error level over all metrics.
        min_samples: Fewest observations in each group for a verdict.
        resamples: Number of bootstrap resamples.
        seed: Seed of the resampling.
        ratio_level: Level of the sample-ratio test.

    Returns:
        The result of `post_promotion_check` with the design, the
        sample-ratio check and a statement added; the overall verdict is
        capped at inconclusive when the split does not match.

    Raises:
        ValueError: As `post_promotion_check`, and for an
            ``expected_fraction`` outside the open unit interval.
    """
    if not 0.0 < expected_fraction < 1.0:
        raise ValueError("expected_fraction must lie strictly between 0 and 1")
    first = specs[0].name if specs else ""
    held = len(holdback.get(first, ()))
    moved = len(promoted.get(first, ()))
    total = held + moved
    p_value = (
        float(stats.binomtest(held, total, expected_fraction).pvalue)
        if total
        else 1.0
    )
    flag = p_value < ratio_level
    result = post_promotion_check(
        specs,
        holdback,
        promoted,
        alpha=alpha,
        min_samples=min_samples,
        resamples=resamples,
        seed=seed,
    )
    overall = "inconclusive" if flag else result["overall"]
    result["design"] = "concurrent_holdback"
    result["overall"] = overall
    result["ratio_check"] = {
        "holdback_units": held,
        "promoted_units": moved,
        "expected_fraction": expected_fraction,
        "p_value": p_value,
        "level": ratio_level,
        "flag": flag,
    }
    result["statement"] = (
        f"Overall: {overall}. Each interval is the difference in the mean "
        "between units served by the promoted model and units kept on the "
        f"previous model at the same time, at joint confidence "
        f"{1 - alpha:.0%} over {len(specs)} metrics. "
        + (
            "The observed split differs from the declared fraction "
            f"(p = {p_value:.2g}), so the assignment was not what was "
            "declared and no verdict is given. "
            if flag
            else "The change is attributable to the model if units were "
            "assigned at random and do not influence one another. "
        )
        + "The check does not look at metrics that were not selected."
    )
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class Replacement:
    """What an attempt to replace the production model did.

    Attributes:
        done: Whether the runtime confirmed the new model.
        reasons: Why not, when it was not done.
        outcome: The release registry's outcome, when promotion was tried.
    """

    done: bool
    reasons: tuple[str, ...]
    outcome: release_state.Outcome | None = None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready record.

        Returns:
            Whether it was done, the reasons and the release outcome (or None)
            as a dict.
        """
        return {
            "done": self.done,
            "reasons": list(self.reasons),
            "outcome": None if self.outcome is None else self.outcome.to_dict(),
        }


class ContinuousEvaluation:
    """Feeds the loop, applies the retraining policy and guards replacement."""

    def __init__(
        self,
        loop: scmle.governance.loop.Loop,
        directory: str | pathlib.Path,
        policy: RetrainPolicy = RetrainPolicy(),  # noqa: B008
        retrain: Callable[[], str | None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Creates the evaluation around an opened loop.

        Args:
            loop: The closed loop of the deployment.
            directory: Where history, retained outcomes and datasets live.
            policy: The retraining policy.
            retrain: Hook that trains and stages a challenger and returns its
                name, or None if it started none. Without it a trigger or an
                approved retrain starts no challenger.
            clock: Callable that returns the current time in epoch seconds.
        """
        self.loop = loop
        self.directory = pathlib.Path(directory)
        self.policy = policy
        self.retrain = retrain
        self.clock = clock
        self.history = TrainingHistory(self.directory / HISTORY_FILE)
        self.seen_alarms = sum(
            1 for r in loop.ledger.rows if r["kind"] == deployment_ledger.ALARM
        )
        proposals = self.history.of_kind("retrain_decision")
        self.last_proposal_batch: int | None = (
            proposals[-1]["facts"].get("batch") if proposals else None
        )

    # -- feeding -----------------------------------------------------------

    def retain(self, kind: str, values: Mapping[str, Any]) -> None:
        """Keeps raw labels or outcomes with the batch they arrived at.

        Args:
            kind: The kind of values, for example ``business_outcomes``.
            values: The raw values by case id.
        """
        line = json.dumps(
            {
                "at": round(self.clock(), 3),
                "batch": self.loop.batch,
                "kind": kind,
                "values": {str(k): v for k, v in values.items()},
            },
            sort_keys=True,
            default=str,
        )
        durability.append_text(self.directory / LABELS_FILE, line + "\n")

    def feed_labels(
        self,
        predictions: Mapping[str, Any],
        labels: Mapping[str, Any],
        challengers: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> int:
        """Delayed labels become 0/1 losses for the loop.

        Args:
            predictions: The serving model's prediction by case id.
            labels: The true label by case id, as it arrives.
            challengers: For each challenger, its prediction by case id on
                the same cases.

        Returns:
            Cases that matched a pending batch.
        """
        losses = {
            str(case): 0.0 if predictions[case] == truth else 1.0
            for case, truth in labels.items()
            if case in predictions
        }
        paired: dict[str, dict[str, float]] = {}
        for name, theirs in (challengers or {}).items():
            paired[name] = {
                str(case): 0.0 if theirs[case] == truth else 1.0
                for case, truth in labels.items()
                if case in theirs
            }
        self.retain("labels", labels)
        return self.loop.mature(losses, challengers=paired or None)

    def feed_outcomes(
        self,
        outcomes: Mapping[str, float],
        cap: float,
        challengers: Mapping[str, Mapping[str, float]] | None = None,
    ) -> int:
        """Business outcomes (a cost, higher is worse) become losses.

        The loss is ``min(outcome / cap, 1)``; ``cap`` is the outcome that is
        as bad as it gets for the purpose of the loop and is declared by the
        owner of the metric, not inferred.

        Args:
            outcomes: The outcome of each case, by case id.
            cap: The outcome that counts as the worst possible loss.
            challengers: The outcomes the challengers would have had, by
                challenger name and case id.

        Returns:
            The number of cases the loop matched to earlier batches.

        Raises:
            ValueError: If ``cap`` is not positive or an outcome is negative.
        """
        if cap <= 0 or any(v < 0 for v in outcomes.values()):
            raise ValueError("cap must be positive and outcomes not negative")
        losses = {str(k): min(v / cap, 1.0) for k, v in outcomes.items()}
        paired = {
            name: {str(k): min(v / cap, 1.0) for k, v in theirs.items()}
            for name, theirs in (challengers or {}).items()
        }
        self.retain("business_outcomes", outcomes)
        return self.loop.mature(losses, challengers=paired or None)

    def advance(self) -> scmle.governance.loop.Outcome:
        """Steps the loop and writes new proposals to the history.

        Returns:
            The loop's step result; new promotion proposals are also written to
            the history.
        """
        before = len(self.loop.ledger.rows)
        result = self.loop.step()
        for row in self.loop.ledger.rows[before:]:
            if row["kind"] == deployment_ledger.PROMOTION_PROPOSAL:
                self.history.record(
                    "promotion_proposal", self.clock(), **row["body"]
                )
        return result

    # -- retraining --------------------------------------------------------

    def review(
        self,
        rows: Sequence[signals.Signal],
        age_days: float | None = None,
    ) -> RetrainDecision:
        """Applies the policy to a window of signals.

        A proposal always goes through `Loop.escalate`. In ``trigger`` mode a
        challenger is also started through `start_challenger` at once (which
        starts none without a retrain hook or while one is already queued).
        A review that makes no proposal is written to the history as
        ``retrain_review`` and changes no state; a proposal is written as
        ``retrain_decision`` and moves the alarm count and the cooldown
        reference to the present.

        Args:
            rows: The window's signals.
            age_days: Age of the model in days, if known.

        Returns:
            The decision: action ``none`` (with the reason for holding back
            if there was a primary reason), or the policy's mode with the
            pending escalation's id and, in trigger mode, the challenger's
            name if one was started.
        """
        alarms = sum(
            1
            for r in self.loop.ledger.rows
            if r["kind"] == deployment_ledger.ALARM
        )
        fresh_alarm = alarms > self.seen_alarms
        batch = self.loop.batch
        since = (
            None
            if self.last_proposal_batch is None
            else batch - self.last_proposal_batch
        )
        primary, support, held = judge_retraining(
            self.policy, rows, fresh_alarm, age_days, since
        )
        gates = self.loop.policy.alarm_gates_retrain
        reasons = tuple(primary + support)
        if not primary or held is not None:
            decision = RetrainDecision(
                "none", reasons, held if primary else None, None, None, gates
            )
            self.history.record(
                "retrain_review",
                self.clock(),
                batch=batch,
                **decision.to_dict(),
            )
            return decision
        self.seen_alarms = alarms
        pending = self.loop.escalate(
            RETRAIN_QUESTION,
            {"reasons": list(reasons), "mode": self.policy.mode},
            evidence=("retained_outcomes.jsonl", HISTORY_FILE),
            proposed_default=DEFAULT_IF_UNANSWERED,
        )
        identity = pending.id if pending is not None else self.pending_id()
        challenger = None
        if self.policy.mode == TRIGGER:
            challenger = self.start_challenger(reasons)
        decision = RetrainDecision(
            self.policy.mode, reasons, None, identity, challenger, gates
        )
        self.last_proposal_batch = batch
        self.history.record(
            "retrain_decision", self.clock(), batch=batch, **decision.to_dict()
        )
        return decision

    def pending_id(self) -> str | None:
        """Id of the pending retrain question, if one is open.

        Returns:
            The identifier of the open retrain escalation, or None.
        """
        for item in self.loop.pending_escalations():
            if item.question.get("action") == RETRAIN_QUESTION:
                return item.id
        return None

    def start_challenger(self, reasons: Sequence[str]) -> str | None:
        """Trains a challenger with the hook and registers it with the loop.

        Args:
            reasons: Why the challenger is trained, recorded in the ledger.

        Returns:
            The challenger's name, or None if there is no hook or the loop
            already has a challenger waiting.
        """
        if self.retrain is None or self.loop.queue:
            return None
        name = self.retrain()
        if name is None:
            return None
        self.loop.ledger.commit(
            deployment_ledger.RETRAIN,
            {
                "reason": "operations retraining policy",
                "mode": self.policy.mode,
                "batch": self.loop.batch,
                "challenger": name,
                "reasons": list(reasons),
            },
        )
        if name not in self.loop.ledger.registered:
            self.loop.register_challenger(name)
        return name

    def run_approved_retrain(self, escalation_id: str) -> str | None:
        """Trains a challenger once a person has approved the recommendation.

        Args:
            escalation_id: The retrain escalation that was answered.

        Returns:
            The challenger's name, or None if none was started.

        Raises:
            ValueError: If the ledger holds no attestation for this
                escalation whose decision approves or overrides. Only the
                ledger row is looked at; its signature and expiry are not
                checked here.
        """
        for row in self.loop.ledger.rows:
            body = row["body"]
            if (
                row["kind"] == deployment_ledger.ATTESTATION
                and body.get("escalation_id") == escalation_id
                and deployment_ledger.ReplyDecision(body["decision"])
                in never_list.ALLOWING
            ):
                return self.start_challenger(("approved recommendation",))
        raise ValueError(f"{escalation_id} has no approving reply")

    # -- history -----------------------------------------------------------

    def record_training(
        self,
        name: str,
        frame: pandas.DataFrame,
        metrics: Mapping[str, float],
        reason: str,
        preserve: bool = True,
    ) -> dict[str, Any]:
        """Records a training run and keeps the exact dataset.

        Args:
            name: Model or challenger name.
            frame: The training data as used.
            metrics: Evaluation metrics of the run, recorded as given.
            reason: Why it was trained.
            preserve: Whether to write a copy of the dataset (as its
                canonical CSV, unless one with the same version already
                exists) under the datasets folder of the directory.

        Returns:
            The history row.
        """
        version = dataset_version(frame)
        if preserve:
            path = (
                self.directory
                / DATASET_DIRECTORY
                / (version.replace(":", "-") + ".csv")
            )
            if not path.exists():
                durability.write_atomically(path, canonical_csv(frame))
        return self.history.record(
            "training",
            self.clock(),
            model=name,
            dataset_version=version,
            rows=len(frame),
            columns=sorted(frame.columns),
            metrics=dict(metrics),
            reason=reason,
            preserved=preserve,
        )

    def promotion_proposals(self) -> list[str]:
        """Challengers the loop's e-process proposed, oldest first.

        Returns:
            The names of the challengers the loop proposed for promotion, in
            order of proposal.
        """
        return [
            str(r["body"]["name"])
            for r in self.loop.ledger.rows
            if r["kind"] == deployment_ledger.PROMOTION_PROPOSAL
        ]

    # -- replacement -------------------------------------------------------

    def replace(
        self,
        challenger: str,
        release: release_state.ReleaseRegistry,
        plane: control.ControlPlane,
        candidate: gate.Candidate,
        key: str | bytes,
        *,
        health: release_state.Health | None = None,
        steps: tuple[int, ...] = (100,),
        comparison: scmle.monitoring.champion.Comparison | None = None,
    ) -> Replacement:
        """Puts a challenger into an environment, or says why not.

        Refuses unless (1) the loop recorded a promotion proposal for the
        challenger, (2) the promotion gate admits the candidate and (3) the
        release registry has a valid approval for it. If an offline
        `comparison` is given its decision must be ``promote``. The registry
        changes only after the runtime confirms. A refusal or an unconfirmed
        rollout does not raise; it returns ``done=False``. Invalid `steps` or
        a candidate container digest that is not a digest raise the registry's
        ``ConfigError``. Every outcome is written to the history.

        Args:
            challenger: Name of the challenger.
            release: The release registry.
            plane: The control plane of the platform.
            candidate: The candidate the gate judges.
            key: Key that verifies the approval and the gate evidence.
            health: Called after each traffic step; reasons stop the rollout.
            steps: Traffic shares in percent to pass through, increasing and
                ending in 100.
            comparison: An offline comparison; if given, its decision must be
                ``promote``.

        Returns:
            Whether the replacement was done and, if not, the reasons.
        """
        reasons = []
        if challenger not in self.promotion_proposals():
            reasons.append(
                f"the loop has no promotion proposal for {challenger!r}"
            )
        if comparison is not None and comparison.decision != "promote":
            reasons.append("the offline comparison does not favour it")
        decision = gate.admit(candidate, key)
        reasons.extend(decision.reasons)
        if reasons:
            self.history.record(
                "replacement_refused",
                self.clock(),
                challenger=challenger,
                reasons=reasons,
            )
            return Replacement(done=False, reasons=tuple(reasons))
        outcome = release.promote(
            candidate.model.model,
            candidate.model.version,
            plane,
            candidate.environment,
            candidate.container_digest,
            steps=steps,
            health=health,
            signing_key=key,
        )
        self.history.record(
            "replacement" if outcome.done else "replacement_refused",
            self.clock(),
            challenger=challenger,
            environment=candidate.environment,
            outcome=outcome.to_dict(),
        )
        return Replacement(outcome.done, outcome.reasons, outcome)

    def check_after_promotion(
        self,
        specs: Sequence[Indicator],
        before: Mapping[str, Sequence[float]],
        after: Mapping[str, Sequence[float]],
        **options: Any,
    ) -> dict[str, Any]:
        """Runs `post_promotion_check` and records the result.

        Args:
            specs: The metrics to judge.
            before: Observations of each metric in the earlier period.
            after: Observations of each metric in the later period.
            **options: Further options of `post_promotion_check`.

        Returns:
            The result of `post_promotion_check`.
        """
        result = post_promotion_check(specs, before, after, **options)
        self.history.record("post_promotion_check", self.clock(), **result)
        return result
