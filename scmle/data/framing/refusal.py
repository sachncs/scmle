"""LLM slot-filling interface, ambiguity score and the refusal rule.

No LLM client ships with the package. A `SlotSampler` is the interface a
caller implements; `ReplaySampler` replays recorded responses so tests make
no network call. `k` drafts are drawn, each slot gets a canonical value, and
the ambiguity score is

    A = max over critical slots of (1 - m_s),

with `m_s` the share of drafts agreeing with the modal value (mean pairwise
Jaccard for the population). The draft is refused when `A > tau`. `tau` is the
largest value whose upper Clopper-Pearson bound on the error rate among
accepted drafts, on a calibration split, is at most `epsilon`. Self-
consistency is agreement, not correctness: a consistently wrong model passes.
Refusal protects only as far as disagreement predicts error, which the
held-out corpus measures. No such corpus ships with the package, so no `tau`
is on file and the LLM path is descriptive.
"""

from collections import Counter
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import hashlib
import itertools
import json
import re
from typing import Any
from typing import Protocol

import numpy
import pandas
from scipy import stats

from scmle.contracts import errors
from scmle.contracts import signing

CRITICAL_SLOTS = ("target", "prediction_moment", "unit", "population")
SLOT_NAMES = (
    "question",
    "target",
    "prediction_moment",
    "unit",
    "population",
    "deployment_population",
    "summary_measure",
)
DEFAULT_K = 10
DEFAULT_EPSILON = 0.05
PREDICATE = re.compile(
    r"^\s*(\w+)\s*(==|!=|<=|>=|<|>)\s*('[^']*'|\"[^\"]*\"|-?\d+(?:\.\d+)?)\s*$"
)


class SlotSampler(Protocol):
    """Draws slot proposals for a request (an LLM adapter)."""

    model_id: str

    def sample(self, prompt: str, index: int) -> Mapping[str, Any]:
        """Returns one JSON-like proposal for the draw `index`.

        Args:
            prompt: The prompt text.
            index: Position of the draw among the `k` draws.

        Returns:
            A mapping from slot name to proposed value.
        """
        ...


def prompt_for(request_text: str, columns: Sequence[str]) -> str:
    """Builds the prompt text, hashed with the samples into the stored draft.

    Args:
        request_text: The request in plain language.
        columns: Names of the data columns, listed in the prompt.

    Returns:
        The prompt asking for the charter slots as JSON.
    """
    return (
        "Fill these charter slots as JSON: "
        + ", ".join(SLOT_NAMES)
        + ". Columns: "
        + ", ".join(columns)
        + ". Request: "
        + request_text
    )


class ReplaySampler:
    """Replays recorded proposals; never touches the network."""

    def __init__(
        self, model_id: str, recorded: Sequence[Mapping[str, Any]]
    ) -> None:
        """Stores the proposals, served in order of draw index.

        Args:
            model_id: Name of the model the proposals are attributed to.
            recorded: The proposals, one per draw.
        """
        self.model_id = model_id
        self.recorded = [dict(r) for r in recorded]

    def sample(self, prompt: str, index: int) -> Mapping[str, Any]:  # noqa: ARG002 -- implements the SlotSampler protocol
        """Returns the recorded proposal for `index`.

        Args:
            prompt: The prompt text (not used; the replay is fixed).
            index: Position of the draw.

        Returns:
            The proposal recorded at position `index`.

        Raises:
            ConfigError: If fewer than `index + 1` proposals were recorded.
        """
        if index >= len(self.recorded):
            raise errors.ConfigError("no recorded sample for this draw")
        return self.recorded[index]


def draw(
    sampler: SlotSampler, prompt: str, k: int = DEFAULT_K
) -> list[dict[str, Any]]:
    """Draws `k` proposals and keeps only the known slot names in each.

    Args:
        sampler: The source of proposals.
        prompt: The prompt text.
        k: Number of draws.

    Returns:
        One mapping per draw, holding only the entries whose key is in
        `SLOT_NAMES`.
    """
    out = []
    for index in range(k):
        proposal = sampler.sample(prompt, index)
        out.append({s: proposal[s] for s in SLOT_NAMES if s in proposal})
    return out


def samples_hash(model_id: str, prompt: str, samples: Sequence[Any]) -> str:
    """Hashes the model id, the prompt and the raw samples together.

    Args:
        model_id: Name of the model.
        prompt: The prompt text.
        samples: The raw proposals.

    Returns:
        The SHA-256 hex digest of their canonical payload.
    """
    body = {"model": model_id, "prompt": prompt, "samples": list(samples)}
    return hashlib.sha256(signing.canonical_payload(body)).hexdigest()


def population_ids(
    frame: pandas.DataFrame, predicate: Any
) -> frozenset[int] | None:
    """Row positions a simple predicate selects, or None if it is invalid.

    The grammar is `column op literal` joined by `and`; anything else is
    rejected, so a proposal is never evaluated as code.

    Args:
        frame: The data the predicate refers to.
        predicate: The predicate text; empty, None, `true` and `all` select
            every row.

    Returns:
        The positions of the selected rows, or None if a part does not
        match the grammar, names a missing column, or compares a column to a
        value of an incompatible type.
    """
    text = str(predicate or "").strip()
    mask = numpy.ones(len(frame), dtype=bool)
    if text and text.lower() not in {"true", "all"}:
        for part in re.split(r"\s+and\s+", text):
            match = PREDICATE.match(part)
            if match is None or match.group(1) not in frame.columns:
                return None
            column, op, raw = match.groups()
            value: Any = raw[1:-1] if raw[0] in "'\"" else float(raw)
            series = frame[column]
            try:
                mask &= {
                    "==": series == value,
                    "!=": series != value,
                    "<": series < value,
                    "<=": series <= value,
                    ">": series > value,
                    ">=": series >= value,
                }[op].to_numpy()
            except TypeError:
                return None
    return frozenset(int(i) for i in numpy.flatnonzero(mask))


def canonical(frame: pandas.DataFrame, slot: str, value: Any) -> Any:
    """Puts a slot value in a canonical, hashable form.

    Args:
        frame: The data, used to evaluate a population predicate.
        slot: The slot name.
        value: The proposed value.

    Returns:
        For `unit`, the sorted tuple of column names; for `population`, the
        set of selected row positions (None if the predicate is invalid);
        for any other slot, the text of the value, or None if it is None.
    """
    if slot == "unit":
        keys = [value] if isinstance(value, str) else list(value or [])
        return tuple(sorted(str(k) for k in keys))
    if slot == "population":
        return population_ids(frame, value)
    return None if value is None else str(value)


def jaccard(first: frozenset[int], second: frozenset[int]) -> float:
    """Computes the Jaccard similarity of two row sets.

    Args:
        first: One set of row positions.
        second: The other set.

    Returns:
        The size of the intersection over the size of the union, or 1.0 for
        two empty sets.
    """
    union = len(first | second)
    return 1.0 if union == 0 else len(first & second) / union


@dataclasses.dataclass(frozen=True, slots=True)
class Consensus:
    """Modal slot values across the drafts, and the ambiguity score.

    Attributes:
        values: Modal raw value per slot (population: first modal draw).
        agreement: `m_s` per critical slot.
        ambiguity: `A`, the largest disagreement on a critical slot.
        conflicts: Critical slots with disagreement above zero.
    """

    values: dict[str, Any]
    agreement: dict[str, float]
    ambiguity: float
    conflicts: tuple[str, ...]


def consensus(
    frame: pandas.DataFrame, samples: Sequence[Mapping[str, Any]]
) -> Consensus:
    """Computes canonical values, agreement and ambiguity of `samples`.

    Args:
        frame: The data, used to evaluate population predicates.
        samples: The proposals drawn for one request.

    Returns:
        The raw value of each slot (the most common canonical value's first
        raw form; for `population`, the first proposal that parses), the
        agreement `m_s` of each critical slot, the ambiguity score (the
        largest `1 - m_s`), and the critical slots whose agreement is below
        1.

    Raises:
        ValueError: If there are no samples.
    """
    if not samples:
        raise ValueError("at least one sample is needed")
    values: dict[str, Any] = {}
    agreement: dict[str, float] = {}
    for slot in SLOT_NAMES:
        raw = [s.get(slot) for s in samples]
        canon = [canonical(frame, slot, v) for v in raw]
        if slot == "population":
            sets = [c for c in canon if c is not None]
            invalid = len(canon) - len(sets)
            pairs = list(itertools.combinations(sets, 2))
            mean = (
                float(numpy.mean([jaccard(p, q) for p, q in pairs]))
                if pairs
                else 1.0
            )
            share = mean * len(sets) / len(canon) if canon else 0.0
            if invalid == len(canon):
                share = 0.0
            first = next(
                (r for r, c in zip(raw, canon, strict=True) if c is not None),
                None,
            )
            values[slot] = first
        else:
            counts = Counter(canon)
            modal, hits = counts.most_common(1)[0]
            share = hits / len(canon)
            values[slot] = next(
                r for r, c in zip(raw, canon, strict=True) if c == modal
            )
        if slot in CRITICAL_SLOTS:
            agreement[slot] = share
    ambiguity = max(1.0 - m for m in agreement.values())
    conflicts = tuple(s for s, m in agreement.items() if m < 1.0)
    return Consensus(values, agreement, ambiguity, conflicts)


def clopper_pearson_upper(
    errors_count: int, n: int, level: float = 0.95
) -> float:
    """Computes a one-sided upper Clopper-Pearson bound on an error rate.

    Args:
        errors_count: Number of errors observed.
        n: Number of trials.
        level: Confidence level of the bound.

    Returns:
        The upper bound; 1.0 when there are no trials or every trial is an
        error.
    """
    if n < 1:
        return 1.0
    if errors_count >= n:
        return 1.0
    return float(stats.beta.ppf(level, errors_count + 1, n - errors_count))


@dataclasses.dataclass(frozen=True, slots=True)
class Calibration:
    """`tau` and the risk-coverage curve behind it.

    Attributes:
        tau: Largest ambiguity accepted, or None if no value met `epsilon`.
        epsilon: Tolerated error rate among accepted drafts.
        n_calibration: Problems in the calibration split.
        curve: (tau, coverage, error rate, CP upper bound) per candidate.
    """

    tau: float | None
    epsilon: float
    n_calibration: int
    curve: tuple[tuple[float, float, float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        """Returns the calibration as a JSON-ready record.

        Returns:
            The risk-coverage certificate: `tau`, `epsilon`,
            `n_calibration` and the curve as lists.
        """
        return {
            "tau": self.tau,
            "epsilon": self.epsilon,
            "n_calibration": self.n_calibration,
            "risk_coverage": [list(c) for c in self.curve],
        }


def refuse(ambiguity: float, tau: float | None) -> bool:
    """Decides whether to refuse a draft.

    Args:
        ambiguity: The ambiguity score `A` of the draft.
        tau: The calibrated largest accepted ambiguity, or None if there is
            none.

    Returns:
        True when there is no calibrated `tau` or `ambiguity > tau`.
    """
    return tau is None or ambiguity > tau


def dump(value: Any) -> str:
    """Serialises a value as stable JSON text, for artifacts.

    Args:
        value: The value to serialise. Sets and other values JSON cannot
            encode are converted with `list`.

    Returns:
        Indented JSON with sorted keys.
    """
    return json.dumps(value, indent=2, sort_keys=True, default=list)
