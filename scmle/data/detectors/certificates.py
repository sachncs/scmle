"""Detector certificates: measured operating characteristics tied to code.

A check may gate a run only if its operating characteristic has been measured
and is on file. A `DetectorCertificate` records, per defect type, how
many planted defects a detector found and how many clean rows it flagged, with
Wilson intervals. It is keyed by a hash of the detector's code, so a code
change without re-certification leaves the old certificate stale, and the
check falls back to descriptive.

Plants are synthetic, so recall measured on them is optimistic; the Wilson
lower bound states how well the count supports the floor, and results are
kept per defect type, never pooled into one number.
"""

import ast
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import enum
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import re
import textwrap
import tokenize
from typing import Any
from typing import Self

from scipy import stats

from scmle.contracts import evidence
from scmle.contracts import qualification

SCHEMA_VERSION = 1
DEFAULT_CONFIDENCE = qualification.DEFAULT_CONFIDENCE
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
INTERVAL_TOLERANCE = 1e-9
DOCSTRING_OWNERS = (
    ast.Module,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


class CertificateError(ValueError):
    """A certificate is missing, malformed or does not permit gating."""


class StaleCertificateError(CertificateError):
    """A certificate exists but was issued for different code."""


class GateStatus(enum.StrEnum):
    """Whether a check may act on its result."""

    GATING = "gating"
    DESCRIPTIVE = "descriptive"


class CertificateState(enum.StrEnum):
    """How the store relates to a detector's current code."""

    CURRENT = "current"
    STALE = "stale"
    MISSING = "missing"


def wilson_interval(
    successes: int, n: int, confidence: float = DEFAULT_CONFIDENCE
) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion.

    Args:
        successes: Observed successes, between 0 and `n`.
        n: Trials, at least 1.
        confidence: Level of the interval, in (0, 1).

    Returns:
        (low, high), clipped to [0, 1].

    Raises:
        ValueError: On a count that is not an integer in range or a
            confidence outside (0, 1).
    """
    qualification.count(n, "n", minimum=1)
    qualification.count(successes, "successes")
    if successes > n:
        raise ValueError("successes must be between 0 and n")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1)")
    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    p = successes / n
    scale = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / scale
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / scale
    # The interval contains p exactly; clamp away rounding at the edges.
    return max(0.0, min(centre - half, p)), min(1.0, max(centre + half, p))


def docstring_lines(tree: ast.Module) -> set[int]:
    """Finds the lines occupied only by a docstring statement.

    Args:
        tree: Parsed module.

    Returns:
        The 1-based line numbers of every docstring that shares its lines
        with no other statement.
    """
    statements = [n for n in ast.walk(tree) if isinstance(n, ast.stmt)]
    lines: set[int] = set()
    for owner in ast.walk(tree):
        if not isinstance(owner, DOCSTRING_OWNERS) or not owner.body:
            continue
        first = owner.body[0]
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            continue
        span = range(first.lineno, (first.end_lineno or first.lineno) + 1)
        shared = any(
            other is not first and other.lineno in span for other in statements
        )
        if not shared:
            lines.update(span)
    return lines


def hash_source(source: str) -> str:
    """SHA-256 of source text with comments, docstrings and blanks removed.

    The text itself is hashed, not a syntax tree, so the value does not
    depend on the Python version that computes it. Reformatting the code
    changes the hash, which is deliberate: any edit needs re-certification.
    Source that is not valid Python raises `SyntaxError`.

    Args:
        source: Python source for a function, class or module.

    Returns:
        64 lowercase hexadecimal characters.
    """
    text = textwrap.dedent(source)
    tree = ast.parse(text)
    lines = text.splitlines()
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            row, column = token.start
            lines[row - 1] = lines[row - 1][:column]
    dropped = docstring_lines(tree)
    kept = [
        line.rstrip()
        for number, line in enumerate(lines, start=1)
        if line.strip() and number not in dropped
    ]
    return hashlib.sha256("\n".join(kept).encode("utf-8")).hexdigest()


def code_hash(*objects: Any) -> str:
    """Combined hash of the source of functions, classes or modules.

    Only the given objects' own source is hashed, not what they call, so a
    detector's certificate must list every callable that affects its result.

    Args:
        *objects: Objects whose source `inspect` can read.

    Returns:
        64 lowercase hexadecimal characters; order matters.

    Raises:
        ValueError: If no object is given.
    """
    if not objects:
        raise ValueError("code_hash needs at least one object")
    parts = [hash_source(inspect.getsource(item)) for item in objects]
    return hashlib.sha256("\n".join(parts).encode("ascii")).hexdigest()


def corpus_hash(parts: Iterable[str]) -> str:
    """Hashes the corpus file hashes a run used, ignoring their order.

    Args:
        parts: SHA-256 hex digests of the corpus files.

    Returns:
        64 lowercase hexadecimal characters.

    Raises:
        ValueError: If no part is given or a part is not a SHA-256 digest.
    """
    members = sorted(parts)
    if not members:
        raise ValueError("corpus_hash needs at least one part")
    for part in members:
        require_digest(part, "corpus part")
    return hashlib.sha256("\n".join(members).encode("ascii")).hexdigest()


def require_digest(value: Any, name: str) -> str:
    """Returns `value` if it is a lowercase SHA-256 hex digest.

    Args:
        value: The value to check.
        name: Name of the value, used in the error message.

    Returns:
        `value`, unchanged.

    Raises:
        ValueError: If `value` is not a 64-character lowercase hex string.
    """
    if not isinstance(value, str) or not SHA256_HEX.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class CertificationPolicy:
    """Pre-registered floors a detector must clear to be certified.

    Attributes:
        recall_floor: Minimum allowed lower Wilson bound of recall.
        false_alarm_ceiling: Maximum allowed upper Wilson bound of the
            false-alarm rate on a clean copy.
        confidence: Level of both intervals, in (0, 1).
    """

    recall_floor: float
    false_alarm_ceiling: float
    confidence: float = DEFAULT_CONFIDENCE

    def __post_init__(self) -> None:
        """Validates the thresholds.

        Raises:
            ValueError: If a floor or ceiling is not a probability, or
                `confidence` is not strictly between 0 and 1.
        """
        qualification.probability(self.recall_floor, "recall_floor")
        qualification.probability(
            self.false_alarm_ceiling, "false_alarm_ceiling"
        )
        confidence = qualification.finite(self.confidence, "confidence")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be within (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        """Returns the policy as a JSON-ready record.

        Returns:
            A mapping of every field to its value.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a policy, rejecting unknown or missing fields.

        Args:
            record: A mapping holding exactly the fields of the policy.

        Returns:
            The policy. A record with unexpected or missing fields, or
            invalid values, raises `ValueError`.
        """
        qualification.require_fields(cls, record)
        return cls(**record)


def interval_pair(value: Any, name: str) -> tuple[float, float]:
    """Parses a stored (low, high) interval.

    Args:
        value: A list or tuple of two numbers.
        name: Name of the field, used in the error message.

    Returns:
        The low and high limits as floats.

    Raises:
        ValueError: If `value` is not a pair, or a limit is not a finite
            number.
    """
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{name} must hold (low, high)")
    return (
        qualification.finite(value[0], name),
        qualification.finite(value[1], name),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class DefectOutcome:
    """Measured result for one defect type, with its verdict.

    Attributes:
        certificate: Counts of planted defects found and clean rows flagged.
        recall_interval: Wilson interval for recall.
        false_alarm_interval: Wilson interval for the false-alarm rate.
        passed: Whether both bounds meet the policy.
    """

    certificate: qualification.Certificate
    recall_interval: tuple[float, float]
    false_alarm_interval: tuple[float, float]
    passed: bool

    @classmethod
    def measure(
        cls,
        certificate: qualification.Certificate,
        policy: CertificationPolicy,
    ) -> Self:
        """Computes the intervals and verdict for `certificate`.

        Args:
            certificate: Counts of planted defects found and clean rows
                flagged.
            policy: The floors to judge against.

        Returns:
            The outcome. It passes when the lower Wilson bound of recall is
            at least the recall floor and the upper Wilson bound of the
            false-alarm rate is at most the false-alarm ceiling.
        """
        recall = wilson_interval(
            certificate.detected, certificate.planted, policy.confidence
        )
        alarms = wilson_interval(
            certificate.false_alarms, certificate.clean_rows, policy.confidence
        )
        passed = (
            recall[0] >= policy.recall_floor
            and alarms[1] <= policy.false_alarm_ceiling
        )
        return cls(certificate, recall, alarms, passed)

    def to_dict(self) -> dict[str, Any]:
        """Returns the outcome as a JSON-ready record.

        Returns:
            A mapping of the certificate, the two intervals as lists and
            the verdict.
        """
        return {
            "certificate": self.certificate.to_dict(),
            "recall_interval": list(self.recall_interval),
            "false_alarm_interval": list(self.false_alarm_interval),
            "passed": self.passed,
        }

    @classmethod
    def from_dict(
        cls, record: Mapping[str, Any], policy: CertificationPolicy
    ) -> Self:
        """Rebuilds an outcome and checks it against its own counts.

        Args:
            record: A mapping holding exactly the fields of the outcome.
            policy: The policy the verdict is recomputed under.

        Returns:
            The outcome as recomputed from the counts.

        Raises:
            CertificateError: If a stored interval or verdict disagrees with
                the value recomputed from the counts.
        """
        qualification.require_fields(cls, record)
        certificate = qualification.Certificate.from_dict(record["certificate"])
        expected = cls.measure(certificate, policy)
        for name in ("recall_interval", "false_alarm_interval"):
            stored = interval_pair(record[name], name)
            recomputed = getattr(expected, name)
            if any(
                abs(a - b) > INTERVAL_TOLERANCE
                for a, b in zip(stored, recomputed, strict=True)
            ):
                raise CertificateError(
                    f"{name} does not match the recorded counts"
                )
        if record["passed"] is not expected.passed:
            raise CertificateError(
                "passed does not match the recorded counts and policy"
            )
        return expected


@dataclasses.dataclass(frozen=True, slots=True)
class DetectorCertificate:
    """Certificate for one detector version on one corpus and seed set.

    Attributes:
        detector_id: Detector that was measured.
        code_hash: Hash of the code that was measured.
        corpus_hash: Hash of the data the plants were injected into.
        seeds: Run seeds the injections used.
        policy: Floors the outcomes were judged against.
        outcomes: One outcome per defect type.
        created: Caller-supplied ISO timestamp, or None. Never read from
            the clock, so the file is reproducible.
    """

    detector_id: str
    code_hash: str
    corpus_hash: str
    seeds: tuple[int, ...]
    policy: CertificationPolicy
    outcomes: tuple[DefectOutcome, ...]
    created: str | None = None

    def __post_init__(self) -> None:
        """Validates the record.

        Raises:
            ValueError: If an id or hash is invalid, `seeds` or `outcomes`
                is not a non-empty tuple, a seed is not a non-negative
                integer, defect types repeat, or an outcome belongs to a
                different detector.
        """
        evidence.require_text(self.detector_id, "detector_id")
        require_digest(self.code_hash, "code_hash")
        require_digest(self.corpus_hash, "corpus_hash")
        qualification.require_tuple(self.seeds, "seeds")
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        for seed in self.seeds:
            qualification.count(seed, "seeds")
        qualification.require_tuple(self.outcomes, "outcomes")
        if not self.outcomes:
            raise ValueError("outcomes must not be empty")
        kinds = [o.certificate.defect_type for o in self.outcomes]
        if len(set(kinds)) != len(kinds):
            raise ValueError("defect types must be unique")
        for outcome in self.outcomes:
            if outcome.certificate.detector_id != self.detector_id:
                raise ValueError("outcome belongs to a different detector")

    @property
    def key(self) -> tuple[str, str, str, tuple[int, ...]]:
        """The detector id, code hash, corpus hash and sorted seeds."""
        return (
            self.detector_id,
            self.code_hash,
            self.corpus_hash,
            tuple(sorted(self.seeds)),
        )

    @property
    def certified_defect_types(self) -> tuple[str, ...]:
        """The defect types whose outcome passed the policy."""
        return tuple(
            o.certificate.defect_type for o in self.outcomes if o.passed
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns the certificate as a JSON-ready record.

        Returns:
            A mapping of every field to its value, with the policy and the
            outcomes as records and the seeds as a list.
        """
        return {
            "detector_id": self.detector_id,
            "code_hash": self.code_hash,
            "corpus_hash": self.corpus_hash,
            "seeds": list(self.seeds),
            "policy": self.policy.to_dict(),
            "outcomes": [o.to_dict() for o in self.outcomes],
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a certificate, verifying every stored verdict.

        Args:
            record: A mapping holding exactly the fields of the certificate.

        Returns:
            The certificate. A record that is invalid, or whose stored
            verdict disagrees with its counts, raises `ValueError` or
            `CertificateError`.
        """
        qualification.require_fields(cls, record)
        policy = CertificationPolicy.from_dict(record["policy"])
        return cls(
            detector_id=record["detector_id"],
            code_hash=record["code_hash"],
            corpus_hash=record["corpus_hash"],
            seeds=tuple(record["seeds"]),
            policy=policy,
            outcomes=tuple(
                DefectOutcome.from_dict(item, policy)
                for item in record["outcomes"]
            ),
            created=record["created"],
        )


def certify(
    detector_id: str,
    code_digest: str,
    corpus_digest: str,
    seeds: Sequence[int],
    results: Sequence[qualification.Certificate],
    policy: CertificationPolicy,
    created: str | None = None,
) -> DetectorCertificate:
    """Judges measured results against a policy.

    Args:
        detector_id: Detector that was measured.
        code_digest: `code_hash` of the measured code.
        corpus_digest: `corpus_hash` of the data used.
        seeds: Run seeds of the injections.
        results: One measured `Certificate` per defect type.
        policy: Pre-registered floors.
        created: Optional ISO timestamp supplied by the caller.

    Returns:
        The certificate, with a verdict per defect type. An empty `results`
        or a repeated defect type raises `ValueError`.
    """
    outcomes = tuple(DefectOutcome.measure(r, policy) for r in results)
    return DetectorCertificate(
        detector_id=detector_id,
        code_hash=code_digest,
        corpus_hash=corpus_digest,
        seeds=tuple(seeds),
        policy=policy,
        outcomes=outcomes,
        created=created,
    )


def demote(
    decision: qualification.Decision, status: GateStatus
) -> qualification.Decision:
    """Limits a descriptive check to a warning at most.

    Args:
        decision: The decision the check reached.
        status: Whether the check may gate.

    Returns:
        `decision` when the check is gating or the decision is not a block
        or quarantine; otherwise a warning.
    """
    if status is GateStatus.GATING:
        return decision
    if decision in (
        qualification.Decision.BLOCK,
        qualification.Decision.QUARANTINE,
    ):
        return qualification.Decision.WARN
    return decision


@dataclasses.dataclass(frozen=True, slots=True)
class AuditReport:
    """Gating detectors that lack a usable certificate.

    Attributes:
        missing: Detectors with no certificate at all.
        stale: Detectors whose certificates are for other code.
        uncertified: Detectors with a current certificate that does not
            pass for the required defect type.
    """

    missing: tuple[str, ...]
    stale: tuple[str, ...]
    uncertified: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when every gating detector has a usable certificate."""
        return not (self.missing or self.stale or self.uncertified)


class CertificateStore:
    """The set of certificates written as `detector_certificates.json`."""

    def __init__(self, items: Iterable[DetectorCertificate] = ()) -> None:
        """Creates a store, keeping the last certificate for each key.

        Args:
            items: Certificates to add, in order.
        """
        self.by_key: dict[tuple[Any, ...], DetectorCertificate] = {}
        for item in items:
            self.add(item)

    @property
    def certificates(self) -> tuple[DetectorCertificate, ...]:
        """The stored certificates, ordered by key."""
        return tuple(self.by_key[key] for key in sorted(self.by_key))

    def add(self, item: DetectorCertificate) -> None:
        """Stores a certificate, replacing one with the same key.

        Args:
            item: The certificate to store.
        """
        self.by_key[item.key] = item

    def current(
        self, detector_id: str, code_digest: str
    ) -> tuple[DetectorCertificate, ...]:
        """Finds the certificates issued for exactly this code.

        Args:
            detector_id: The detector.
            code_digest: Hash of the detector's current code.

        Returns:
            The matching certificates, ordered by key.
        """
        return tuple(
            c
            for c in self.certificates
            if c.detector_id == detector_id and c.code_hash == code_digest
        )

    def state(self, detector_id: str, code_digest: str) -> CertificateState:
        """Finds how the store relates to a detector's current code.

        Args:
            detector_id: The detector.
            code_digest: Hash of the detector's current code.

        Returns:
            `current` if a certificate matches the code, `stale` if the
            detector has certificates only for other code, otherwise
            `missing`.
        """
        if self.current(detector_id, code_digest):
            return CertificateState.CURRENT
        if any(c.detector_id == detector_id for c in self.certificates):
            return CertificateState.STALE
        return CertificateState.MISSING

    def status(
        self, detector_id: str, code_digest: str, defect_type: str
    ) -> GateStatus:
        """Finds whether a detector may gate on a defect type.

        Args:
            detector_id: The detector.
            code_digest: Hash of the detector's current code.
            defect_type: The defect type.

        Returns:
            `gating` when a certificate for the current code passed for
            `defect_type`, otherwise `descriptive`.
        """
        for item in self.current(detector_id, code_digest):
            if defect_type in item.certified_defect_types:
                return GateStatus.GATING
        return GateStatus.DESCRIPTIVE

    def mark_gating(
        self, detector_id: str, code_digest: str, defect_type: str
    ) -> GateStatus:
        """Returns `GATING`, or raises when the detector may not gate.

        Args:
            detector_id: The detector.
            code_digest: Hash of the detector's current code.
            defect_type: The defect type.

        Returns:
            `GateStatus.GATING`.

        Raises:
            StaleCertificateError: The certificate is for other code.
            CertificateError: There is no certificate, or it does not pass
                for `defect_type`.
        """
        state = self.state(detector_id, code_digest)
        if state is CertificateState.MISSING:
            raise CertificateError(
                f"no certificate for detector '{detector_id}'"
            )
        if state is CertificateState.STALE:
            raise StaleCertificateError(
                f"certificate for '{detector_id}' is for different code"
            )
        if self.status(detector_id, code_digest, defect_type) is not (
            GateStatus.GATING
        ):
            raise CertificateError(
                f"'{detector_id}' is not certified for '{defect_type}'"
            )
        return GateStatus.GATING

    def to_dict(self) -> dict[str, Any]:
        """Returns the store as a JSON-ready document.

        Returns:
            A mapping of the schema version and the certificates as
            records.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "certificates": [c.to_dict() for c in self.certificates],
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Self:
        """Rebuilds a store, verifying every stored verdict.

        Args:
            document: A mapping with `schema_version` and `certificates`.

        Returns:
            The store.

        Raises:
            CertificateError: On an unknown schema version.
            ValueError: If the document has fields other than
                `schema_version` and `certificates`. A certificate that is
                invalid, or whose stored verdict disagrees with its counts,
                also raises `ValueError` or `CertificateError`.
        """
        if document.get("schema_version") != SCHEMA_VERSION:
            raise CertificateError(
                f"unsupported schema_version: {document.get('schema_version')}"
            )
        extra = sorted(set(document) - {"schema_version", "certificates"})
        if extra:
            raise ValueError(f"unexpected field(s): {extra}")
        return cls(
            DetectorCertificate.from_dict(item)
            for item in document["certificates"]
        )

    def save(self, path: Path) -> None:
        """Writes stable JSON, so equal stores give equal bytes.

        Args:
            path: File to write.
        """
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        path.write_text(text + "\n", encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: Path) -> Self:
        """Reads and verifies a store written by `save`.

        Args:
            path: File to read.

        Returns:
            The store. A missing file raises `OSError`, invalid JSON raises
            `json.JSONDecodeError`, and a document `from_dict` rejects
            raises the error it raises.
        """
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def audit(
    store: CertificateStore, gating: Mapping[str, tuple[str, str]]
) -> AuditReport:
    """Checks every gating detector has a current, passing certificate.

    Args:
        store: Certificates on file.
        gating: Detector id -> (current code hash, defect type it gates on).

    Returns:
        Which detectors are missing, stale or uncertified.
    """
    missing: list[str] = []
    stale: list[str] = []
    uncertified: list[str] = []
    for detector_id in sorted(gating):
        digest, defect = gating[detector_id]
        state = store.state(detector_id, digest)
        if state is CertificateState.MISSING:
            missing.append(detector_id)
        elif state is CertificateState.STALE:
            stale.append(detector_id)
        elif store.status(detector_id, digest, defect) is not (
            GateStatus.GATING
        ):
            uncertified.append(detector_id)
    return AuditReport(tuple(missing), tuple(stale), tuple(uncertified))
