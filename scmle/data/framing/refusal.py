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
held-out corpus measures. That corpus does not exist yet, so no `tau` is on
file and the LLM path is descriptive.
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

import numpy as np
import pandas as pd
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
        """Returns one JSON-like proposal for the draw `index`."""


def prompt_for(request_text: str, columns: Sequence[str]) -> str:
    """The prompt text; hashed with the samples into the stored draft."""
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
        """Stores the proposals, served in order of draw index."""
        self.model_id = model_id
        self.recorded = [dict(r) for r in recorded]

    def sample(self, prompt: str, index: int) -> Mapping[str, Any]:
        """Returns the recorded proposal for `index`."""
        if index >= len(self.recorded):
            raise errors.ConfigError("no recorded sample for this draw")
        return self.recorded[index]


def draw(
    sampler: SlotSampler, prompt: str, k: int = DEFAULT_K
) -> list[dict[str, Any]]:
    """Draws `k` proposals and keeps only those with known slot names."""
    out = []
    for index in range(k):
        proposal = sampler.sample(prompt, index)
        out.append({s: proposal[s] for s in SLOT_NAMES if s in proposal})
    return out


def samples_hash(model_id: str, prompt: str, samples: Sequence[Any]) -> str:
    """SHA-256 over model id, prompt and the raw samples."""
    body = {"model": model_id, "prompt": prompt, "samples": list(samples)}
    return hashlib.sha256(signing.canonical_payload(body)).hexdigest()


def population_ids(
    frame: pd.DataFrame, predicate: Any
) -> frozenset[int] | None:
    """Row positions a simple predicate selects, or None if it is invalid.

    The grammar is `column op literal` joined by `and`; anything else is
    rejected, so a proposal is never evaluated as code.
    """
    text = str(predicate or "").strip()
    mask = np.ones(len(frame), dtype=bool)
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
    return frozenset(int(i) for i in np.flatnonzero(mask))


def canonical(frame: pd.DataFrame, slot: str, value: Any) -> Any:
    """Canonical, hashable form of a slot value."""
    if slot == "unit":
        keys = [value] if isinstance(value, str) else list(value or [])
        return tuple(sorted(str(k) for k in keys))
    if slot == "population":
        return population_ids(frame, value)
    return None if value is None else str(value)


def jaccard(first: frozenset[int], second: frozenset[int]) -> float:
    """Jaccard similarity of two row sets (1 for two empty sets)."""
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
    frame: pd.DataFrame, samples: Sequence[Mapping[str, Any]]
) -> Consensus:
    """Canonical values, agreement and ambiguity of `samples`."""
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
                float(np.mean([jaccard(p, q) for p, q in pairs]))
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
    """One-sided upper Clopper-Pearson bound on an error rate."""
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
        """JSON-ready record (the risk-coverage certificate)."""
        return {
            "tau": self.tau,
            "epsilon": self.epsilon,
            "n_calibration": self.n_calibration,
            "risk_coverage": [list(c) for c in self.curve],
        }


def calibrate_tau(
    ambiguity: Sequence[float],
    wrong: Sequence[bool],
    epsilon: float = DEFAULT_EPSILON,
    level: float = 0.95,
) -> Calibration:
    """Largest `tau` whose accepted drafts have CP upper bound <= `epsilon`.

    Args:
        ambiguity: Ambiguity score of each calibration problem.
        wrong: Whether the accepted draft would be wrong for each problem.
        epsilon: Tolerated error rate among accepted drafts.
        level: Confidence of the one-sided bound.

    Returns:
        The calibration. `tau` is None when no candidate meets `epsilon`.
    """
    scores = np.asarray(ambiguity, dtype=float)
    bad = np.asarray(wrong, dtype=bool)
    if scores.shape != bad.shape or scores.size == 0:
        raise ValueError("ambiguity and wrong must be equal, non-empty")
    curve = []
    best: float | None = None
    for tau in sorted(set(scores.tolist())):
        accepted = scores <= tau
        n = int(accepted.sum())
        wrong_n = int(bad[accepted].sum())
        bound = clopper_pearson_upper(wrong_n, n, level)
        curve.append((tau, n / scores.size, wrong_n / n, bound))
        if bound <= epsilon:
            best = tau
    return Calibration(best, epsilon, scores.size, tuple(curve))


def refuse(ambiguity: float, tau: float | None) -> bool:
    """Whether to refuse: no calibrated `tau`, or `A > tau`."""
    return tau is None or ambiguity > tau


def dump(value: Any) -> str:
    """Stable JSON text, for artifacts."""
    return json.dumps(value, indent=2, sort_keys=True, default=list)
