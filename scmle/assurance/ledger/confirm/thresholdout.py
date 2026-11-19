"""Thresholdout: the confirmation holdout reused for adaptive queries.

The reusable holdout of Dwork, Feldman, Hardt, Pitassi, Reingold and Roth
(Science 349, 2015). An analyst asks queries, each a function that scores
every row in [0, 1]; the answer estimates the population mean of that score.
The training answer (the mean on `X`) is returned unless it differs from the
holdout mean by more than a noisy threshold ``T + Lap(2 sigma)``; then the
holdout mean plus ``Lap(sigma)``, rounded to the reporting `step`, is
returned and one unit of the overfitting `budget` is spent. After `queries`
answers or `budget` overfitting answers the holdout refuses further queries.

Derivation (every threshold is a variable). The guarantee is the
description-length argument of the same paper: a transcript is the sequence
of answers, each either "train" or a holdout value on the `step` grid of
``L = floor(1 / step) + 1`` levels, with at most `budget` holdout values, so
there are at most ``K = sum_{k <= budget} C(queries, k) L^k`` transcripts.
For a fixed analyst each transcript fixes the queries asked, so at most
``K * queries`` queries can ever be asked, and Hoeffding with a union bound
gives, with probability ``1 - confidence / 2``, every holdout mean within

    sampling = sqrt((ln K + ln(4 queries / confidence)) / (2 size))

of its population mean, however the analyst adapts. The noise is
independent of the data, so it does not enter that count. For one answer,
with probability ``1 - confidence / 2`` its noise satisfies
``|Lap(b)| <= b g`` with ``g = ln(2 / confidence)``. A train answer is then
within ``threshold + 2 noise g + sampling`` of the truth and a holdout answer
within ``noise g + step / 2 + sampling``. The threshold is set so that the
first equals the `tolerance`, with ``noise = threshold / RATIO`` (the
threshold-to-noise ratio of the paper's experiment, ``T = 0.04``,
``sigma = 0.01``):

    threshold = (tolerance - sampling) / (1 + 2 g / RATIO)

so each answer is within `tolerance` of its population mean with
probability at least ``1 - confidence``, the reuse error rate. A holdout
smaller than `minimum` rows cannot meet the tolerance and is refused.

Assumptions: the holdout rows are independent draws from the population of
interest, independent of the training rows and of the analyst except
through the answers; scores lie in [0, 1].

Resilience: with a `journal`, every answer is written (atomically, fsynced)
before it is returned, with the counters and the noise state, so an
interrupted session resumes with its budget spent, never reset. A journal
written for other rows or other terms is refused.
"""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy
import pandas

from scmle import durability
from scmle.assurance.ledger import usage
from scmle.assurance.ledger.confirm import records
import scmle.assurance.ledger.ledger
from scmle.contracts import errors
import scmle.contracts.records

TOLERANCE = 0.1
CONFIDENCE = 0.05
QUERIES = 100
BUDGET = 10
STEP = 0.01
RATIO = 4.0
TRAIN = "train"
HOLDOUT = "holdout"
CLAIM = "thresholdout"
KIND = "reuse"
DOCUMENT = "thresholdout.json"
ANSWERED = "answered"
EXHAUSTED = "exhausted"
VERSION = 1

Rows = tuple[pandas.DataFrame, pandas.Series]
Query = Callable[[pandas.DataFrame, pandas.Series], Any]


def transcripts(queries: int, budget: int, step: float) -> float:
    """Counts, in logarithm, the transcripts a session can produce.

    Args:
        queries: Largest number of answers.
        budget: Largest number of holdout answers.
        step: Reporting step of a holdout answer.

    Returns:
        ``ln K`` with ``K = sum_{k <= budget} C(queries, k) L^k`` and
        ``L = floor(1 / step) + 1`` levels.
    """
    levels = math.floor(1.0 / step) + 1
    terms = [
        math.lgamma(queries + 1)
        - math.lgamma(k + 1)
        - math.lgamma(queries - k + 1)
        + k * math.log(levels)
        for k in range(budget + 1)
    ]
    peak = max(terms)
    return peak + math.log(sum(math.exp(t - peak) for t in terms))


def floor(confidence: float, step: float) -> float:
    """The smallest slack ``tolerance - sampling`` the rounding allows.

    Args:
        confidence: Per-answer error probability.
        step: Reporting step of a holdout answer.

    Returns:
        The slack at which a holdout answer's bound ``noise g + step / 2``
        equals a train answer's ``threshold + 2 noise g``.
    """
    spread = math.log(2.0 / confidence)
    return (1 + 2 * spread / RATIO) * (step / 2) / (1 + spread / RATIO)


@dataclasses.dataclass(frozen=True, slots=True)
class Terms:
    """The parameters of a reusable holdout and the values derived from them.

    Attributes:
        size: Rows in the holdout.
        tolerance: Largest error of an answer, in score units (policy: a
            tenth of the [0, 1] range, the resolution at which a model
            score is acted on).
        confidence: Largest probability that an answer misses by more than
            `tolerance` (the plan's ``alpha``).
        queries: Largest number of answers.
        budget: Largest number of holdout (overfitting) answers.
        step: Reporting step of a holdout answer.
    """

    size: int
    tolerance: float = TOLERANCE
    confidence: float = CONFIDENCE
    queries: int = QUERIES
    budget: int = BUDGET
    step: float = STEP

    def __post_init__(self) -> None:
        """Validates the parameters and that the holdout is large enough.

        Raises:
            InputError: If a parameter is outside its range, or `size` is
                below `minimum` for the other parameters.
        """
        for name in ("tolerance", "confidence"):
            value = getattr(self, name)
            if not (isinstance(value, float | int) and 0.0 < value < 1.0):
                raise errors.InputError(f"{name} must be within (0, 1)")
        if not 0.0 < self.step <= 1.0:
            raise errors.InputError("step must be within (0, 1]")
        for name in ("size", "queries", "budget"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise errors.InputError(f"{name} must be an integer")
            if value < 1:
                raise errors.InputError(f"{name} must be at least 1")
        if self.budget > self.queries:
            raise errors.InputError("budget must not exceed queries")
        least = minimum(
            self.tolerance,
            self.confidence,
            self.queries,
            self.budget,
            self.step,
        )
        if self.size < least:
            raise errors.InputError(
                f"a reusable holdout of {self.size} rows cannot keep answers "
                f"within {self.tolerance:g}; it needs at least {least} rows "
                "(or a larger tolerance, fewer queries or a smaller budget)"
            )

    @property
    def sampling(self) -> float:
        """Largest gap between a holdout mean and its population mean."""
        count = transcripts(self.queries, self.budget, self.step)
        spread = count + math.log(4 * self.queries / self.confidence)
        return math.sqrt(spread / (2 * self.size))

    @property
    def threshold(self) -> float:
        """The threshold ``T`` on the train-holdout gap."""
        spread = math.log(2.0 / self.confidence)
        return (self.tolerance - self.sampling) / (1 + 2 * spread / RATIO)

    @property
    def noise(self) -> float:
        """The Laplace scale ``sigma`` of a holdout answer."""
        return self.threshold / RATIO

    def to_dict(self) -> Mapping[str, Any]:
        """Returns the parameters and the derived values, JSON-ready.

        Returns:
            The six parameters plus `sampling`, `threshold` and `noise`.
        """
        return {
            "size": self.size,
            "tolerance": self.tolerance,
            "confidence": self.confidence,
            "queries": self.queries,
            "budget": self.budget,
            "step": self.step,
            "sampling": self.sampling,
            "threshold": self.threshold,
            "noise": self.noise,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Terms":
        """Rebuilds the terms from `to_dict`; the derived values are recomputed.

        Args:
            record: A `to_dict` mapping.

        Returns:
            The terms.

        Raises:
            InputError: If a parameter is missing or invalid.
        """
        names = ("size", "tolerance", "confidence", "queries", "budget", "step")
        missing = [n for n in names if n not in record]
        if missing:
            raise errors.InputError(f"thresholdout terms lack {missing}")
        return cls(**{n: record[n] for n in names})


def minimum(
    tolerance: float = TOLERANCE,
    confidence: float = CONFIDENCE,
    queries: int = QUERIES,
    budget: int = BUDGET,
    step: float = STEP,
) -> int:
    """The fewest holdout rows that keep every answer within the tolerance.

    Args:
        tolerance: Largest error of an answer.
        confidence: Per-answer error probability.
        queries: Largest number of answers.
        budget: Largest number of holdout answers.
        step: Reporting step of a holdout answer.

    Returns:
        The smallest size whose `sampling` leaves the slack `floor` needs.

    Raises:
        InputError: If no size suffices (the step is too coarse for the
            tolerance).
    """
    slack = tolerance - floor(confidence, step)
    if slack <= 0.0:
        raise errors.InputError(
            f"step {step:g} is too coarse for tolerance {tolerance:g}"
        )
    count = transcripts(queries, budget, step)
    spread = count + math.log(4 * queries / confidence)
    return math.ceil(spread / (2 * slack * slack))


@dataclasses.dataclass(frozen=True, slots=True)
class Answer:
    """One answer of a reusable holdout.

    Attributes:
        index: Position of the query, from 0.
        value: The estimate of the query's population mean.
        origin: `train` or `holdout`, the set the value came from.
        left: Overfitting budget left after the answer.
    """

    index: int
    value: float
    origin: str
    left: int

    def to_dict(self) -> Mapping[str, Any]:
        """Returns the answer as a JSON-ready record.

        Returns:
            A mapping of every field to its value.
        """
        return dataclasses.asdict(self)


def score(query: Query, rows: Rows, name: str) -> numpy.ndarray:
    """Evaluates a query on rows and checks its scores.

    Args:
        query: Scores every row (features, target) in [0, 1].
        rows: The features and target.
        name: Which rows, for the message.

    Returns:
        One finite score per row, in [0, 1].

    Raises:
        InputError: If the scores are not one number per row in [0, 1].
    """
    features, target = rows
    values = numpy.asarray(query(features, target), dtype=float)
    if values.shape != (len(target),):
        raise errors.InputError(
            f"a query must score each of the {len(target)} {name} rows once"
        )
    if not numpy.all(numpy.isfinite(values)):
        raise errors.InputError("a query's scores must be finite")
    if numpy.any(values < 0.0) or numpy.any(values > 1.0):
        raise errors.InputError("a query's scores must lie in [0, 1]")
    return values


class Thresholdout:
    """A holdout that answers adaptive queries within a derived tolerance.

    Attributes:
        terms: The parameters and derived values.
        train: The training rows (features, target).
        holdout: The holdout rows (features, target).
        journal: The file each answer is written to before it is returned,
            or None.
        asked: Answers given so far.
        spent: Holdout answers given so far.
        answers: Every answer, in order.
        generator: The noise source.
        seal: The hashes binding the session to its rows and terms.
    """

    def __init__(
        self,
        train: Rows,
        holdout: Rows,
        terms: Terms | None = None,
        *,
        seed: int = 0,
        journal: str | Path | None = None,
    ) -> None:
        """Binds the rows and resumes from the journal when it exists.

        Args:
            train: The training rows (features, target).
            holdout: The holdout rows (features, target).
            terms: Parameters; by default the module defaults for the
                holdout's size.
            seed: Seed of the noise.
            journal: File the session is kept in, or None.

        Raises:
            InputError: If `terms.size` is not the holdout's size, or the
                defaults do not fit the holdout (see `Terms`).
            LeakageError: If the journal belongs to other rows or terms.
        """
        size = len(holdout[1])
        self.terms = Terms(size) if terms is None else terms
        if self.terms.size != size:
            raise errors.InputError(
                f"the terms are for {self.terms.size} rows, the holdout has "
                f"{size}"
            )
        self.train = train
        self.holdout = holdout
        self.journal = None if journal is None else Path(journal)
        self.asked = 0
        self.spent = 0
        self.answers: list[Answer] = []
        self.generator = numpy.random.default_rng(seed)
        self.seal = {
            "train": usage.fingerprint(*train),
            "holdout": usage.fingerprint(*holdout),
            "terms": scmle.assurance.ledger.ledger.digest(
                dict(self.terms.to_dict())
            ),
        }
        if self.journal is not None and self.journal.exists():
            self.resume(self.journal)

    @property
    def left(self) -> int:
        """Overfitting budget left."""
        return self.terms.budget - self.spent

    def resume(self, path: Path) -> None:
        """Restores the counters, answers and noise state from a journal.

        Args:
            path: The journal.

        Raises:
            LeakageError: If the journal is unreadable or belongs to other
                rows or terms.
        """
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as failure:
            raise errors.LeakageError(
                f"the reusable holdout's journal {path} is unreadable "
                f"({failure}); it is not reset"
            ) from failure
        if not isinstance(stored, dict) or stored.get("seal") != self.seal:
            raise errors.LeakageError(
                f"the journal {path} belongs to other rows or terms"
            )
        self.asked = int(stored["asked"])
        self.spent = int(stored["spent"])
        self.answers = [Answer(**a) for a in stored["answers"]]
        self.generator.bit_generator.state = stored["state"]

    def record(self) -> Mapping[str, Any]:
        """Returns the session as a JSON-ready record.

        Returns:
            The version, terms, seal, counters, answers and noise state.
        """
        return {
            scmle.contracts.records.KEY: VERSION,
            "terms": dict(self.terms.to_dict()),
            "seal": self.seal,
            "asked": self.asked,
            "spent": self.spent,
            "answers": [dict(a.to_dict()) for a in self.answers],
            "state": self.generator.bit_generator.state,
        }

    def query(self, query: Query) -> Answer:
        """Answers one query.

        The answer is journalled before it is returned; if the query or the
        write fails, nothing is spent and the noise state is restored.

        Args:
            query: Scores every row (features, target) in [0, 1].

        Returns:
            The answer.

        Raises:
            LeakageError: If `queries` answers or `budget` holdout answers
                were already given.
            InputError: If the query's scores are invalid.
        """
        if self.asked >= self.terms.queries:
            raise errors.LeakageError(
                f"the reusable holdout gave its {self.terms.queries} answers"
            )
        if self.spent >= self.terms.budget:
            raise errors.LeakageError(
                f"the reusable holdout spent its budget of {self.terms.budget}"
            )
        fit = float(score(query, self.train, TRAIN).mean())
        held = float(score(query, self.holdout, HOLDOUT).mean())
        before = self.generator.bit_generator.state
        answer = self.decide(fit, held)
        self.answers.append(answer)
        self.asked += 1
        self.spent += answer.origin == HOLDOUT
        try:
            self.save()
        except BaseException:
            self.answers.pop()
            self.asked -= 1
            self.spent -= answer.origin == HOLDOUT
            self.generator.bit_generator.state = before
            raise
        return answer

    def decide(self, fit: float, held: float) -> Answer:
        """Chooses the train or the noisy holdout answer.

        Args:
            fit: The query's mean on the training rows.
            held: The query's mean on the holdout rows.

        Returns:
            The answer (not yet counted).
        """
        terms = self.terms
        margin = terms.threshold + self.generator.laplace(0.0, 2 * terms.noise)
        if abs(fit - held) <= margin:
            return Answer(self.asked, fit, TRAIN, self.left)
        noisy = held + self.generator.laplace(0.0, terms.noise)
        value = min(1.0, max(0.0, round(noisy / terms.step) * terms.step))
        return Answer(self.asked, value, HOLDOUT, self.left - 1)

    def save(self) -> None:
        """Writes the session to the journal, if there is one."""
        if self.journal is not None:
            text = json.dumps(self.record(), sort_keys=True) + "\n"
            durability.write_atomically(self.journal, text.encode("utf-8"))


@dataclasses.dataclass(frozen=True, slots=True)
class Session:
    """An adaptive analysis run on the single opening of `C`.

    Attributes:
        train: The training rows (features, target), from `X`.
        analyst: Asks its queries of the holdout and returns JSON-ready
            findings; it may stop early or when the budget is spent.
        tolerance: Largest error of an answer.
        confidence: Per-answer error probability.
        queries: Largest number of answers.
        budget: Largest number of holdout answers.
        step: Reporting step of a holdout answer.
        seed: Seed of the noise.
        journal: File the session is kept in, or None.
    """

    train: Rows
    analyst: Callable[[Thresholdout], Mapping[str, Any]]
    tolerance: float = TOLERANCE
    confidence: float = CONFIDENCE
    queries: int = QUERIES
    budget: int = BUDGET
    step: float = STEP
    seed: int = 0
    journal: str | Path | None = None

    def plan(self) -> Mapping[str, Any]:
        """Returns what is registered before `C` opens.

        Returns:
            The parameters, the seed and the fingerprint of the train rows.
        """
        return {
            "tolerance": self.tolerance,
            "confidence": self.confidence,
            "queries": self.queries,
            "budget": self.budget,
            "step": self.step,
            "seed": self.seed,
            "train": usage.fingerprint(*self.train),
        }

    def docket(self) -> records.Docket:
        """Registers the session as a claim run on the single opening.

        Returns:
            A docket whose run reuses the opened rows through Thresholdout
            and writes the transcript as `DOCUMENT`.
        """
        commitment = records.Commitment(
            CLAIM,
            KIND,
            (
                "adaptive queries on C answered by Thresholdout, each within "
                f"{self.tolerance:g} of its population mean with probability "
                f"at least {1 - self.confidence:g}"
            ),
            scmle.assurance.ledger.ledger.digest(dict(self.plan())),
        )
        return records.Docket((commitment,), self.run)

    def run(
        self, features: pandas.DataFrame, target: pandas.Series
    ) -> records.Output:
        """Runs the analyst against the opened rows.

        Args:
            features: The opened features of `C`.
            target: The opened target of `C`.

        Returns:
            The claim's status, `answered` or `exhausted` (the analyst ran
            into the budget), and the transcript document.
        """
        terms = Terms(
            len(target),
            self.tolerance,
            self.confidence,
            self.queries,
            self.budget,
            self.step,
        )
        guard = Thresholdout(
            self.train,
            (features, target),
            terms,
            seed=self.seed,
            journal=self.journal,
        )
        status = ANSWERED
        try:
            findings = dict(self.analyst(guard))
        except errors.LeakageError as failure:
            # handled: the analyst ran into the budget; the answers given
            # are kept and the status says so
            status = EXHAUSTED
            findings = {"stopped": str(failure)}
        document = {**guard.record(), "findings": findings, "status": status}
        document.pop("state")
        return records.Output({CLAIM: status}, {DOCUMENT: document})
