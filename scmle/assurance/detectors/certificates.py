"""Detector certificates: measured operating characteristics tied to code.

A check may gate a run only if its operating characteristic has been measured
and is on file. A `Certification` records, per defect type, how
many planted defects a detector found and how many clean rows it flagged, with
Wilson intervals, or exact Clopper-Pearson intervals when the policy says
`exact`. It is keyed by a hash of the detector's code, so a code
change without re-certification leaves the old certificate stale, and the
check falls back to descriptive.

Plants are synthetic, so recall measured on them is optimistic; the Wilson
lower bound states how well the count supports the floor, and results are
kept per defect type, never pooled into one number.
"""

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import enum
import json
from pathlib import Path
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.assurance.detectors.hashing import code_hash
from scmle.assurance.detectors.hashing import corpus_hash
from scmle.assurance.detectors.hashing import hash_source
from scmle.assurance.detectors.hashing import require_digest
from scmle.assurance.detectors.intervals import interval
from scmle.assurance.detectors.intervals import METHODS
from scmle.assurance.detectors.intervals import wilson_interval
from scmle.contracts import errors
from scmle.contracts import evidence
from scmle.contracts import qualification
from scmle.contracts import records

# The hashing names live in `hashing` and the interval names in `intervals`;
# both stay importable from here.
__all__ = [
    "DIGESTS",
    "INTERVAL_TOLERANCE",
    "METHODS",
    "PAIR_LENGTH",
    "SCHEMA_VERSION",
    "Audit",
    "Authority",
    "CertificateError",
    "Certification",
    "Freshness",
    "Outcome",
    "StaleError",
    "Standard",
    "Store",
    "audit",
    "certify",
    "code_hash",
    "corpus_hash",
    "demote",
    "flatten",
    "hash_source",
    "interval",
    "nest",
    "require_digest",
    "wilson_interval",
]

SCHEMA_VERSION = 1
DIGESTS = {"code_hash": "code", "corpus_hash": "corpus"}
INTERVAL_TOLERANCE = 1e-9
PAIR_LENGTH = 2


class CertificateError(ValueError):
    """A certificate is missing, malformed or does not permit gating."""


class StaleError(CertificateError):
    """A certificate exists but was issued for different code."""


class Authority(enum.StrEnum):
    """Whether a check may act on its result."""

    GATING = "gating"
    DESCRIPTIVE = "descriptive"


class Freshness(enum.StrEnum):
    """How the store relates to a detector's current code."""

    CURRENT = "current"
    STALE = "stale"
    MISSING = "missing"


def stamp(layout: records.Layout) -> records.Layout:
    """Gives a version-2 policy the interval method it was judged by.

    Args:
        layout: The policy at version 2.

    Returns:
        The policy at version 3, with method `wilson`.
    """
    return {**layout, "method": "wilson"}


def unstamp(layout: records.Layout) -> records.Layout:
    """Removes the interval method from a policy, back to version 2.

    Args:
        layout: The policy at version 3.

    Returns:
        The policy at version 2.

    Raises:
        InputError: If the policy uses exact intervals, which version 2
            cannot hold.
    """
    if layout.get("method", "wilson") != "wilson":
        raise errors.InputError("an exact policy has no version-2 layout")
    return {k: v for k, v in layout.items() if k != "method"}


@dataclasses.dataclass(frozen=True, slots=True)
class Standard:
    """Pre-registered floors a detector must clear to be certified.

    A policy carries its own schema version, like the `Certificate` counts
    it judges; a stored policy without one is version 1.

    Attributes:
        recall: Minimum allowed lower bound of recall (the recall floor).
        alarms: Maximum allowed upper bound of the false-alarm rate on a
            clean copy (the false-alarm ceiling).
        confidence: Level of both two-sided intervals, in (0, 1); each
            bound is one-sided at `(1 - confidence) / 2`.
        method: How the intervals are computed, one of `METHODS`; a stored
            policy from before version 3 is `wilson`.
        migrations: The policy's version steps, oldest first.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={"recall_floor": "recall", "false_alarm_ceiling": "alarms"}
        ),
        records.Migration(change=stamp, revert=unstamp),
    )

    recall: float
    alarms: float
    confidence: float = qualification.DEFAULT_CONFIDENCE
    method: str = "wilson"

    def __post_init__(self) -> None:
        """Validates the thresholds.

        Raises:
            ValueError: If a floor or ceiling is not a probability,
                `confidence` is not strictly between 0 and 1, or the method
                is unknown.
        """
        qualification.probability(self.recall, "recall")
        qualification.probability(self.alarms, "alarms")
        confidence = qualification.finite(self.confidence, "confidence")
        if not 0.0 < confidence < 1.0:
            raise errors.InputError("confidence must be within (0, 1)")
        if self.method not in METHODS:
            raise errors.InputError(f"method must be one of {METHODS}")

    def to_dict(self) -> dict[str, Any]:
        """Returns the policy as a JSON-ready record.

        Returns:
            A mapping of the schema version and every field to its value.
        """
        return {
            "schema_version": records.ORIGIN + len(self.migrations),
            **dataclasses.asdict(self),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a policy, rejecting unknown or missing fields.

        Args:
            record: A mapping holding exactly the fields of the policy, at
                any supported schema version.

        Returns:
            The policy. A record with unexpected or missing fields, or
            invalid values, raises `ValueError`.
        """
        return cls(**qualification.require(cls, record))


@dataclasses.dataclass(frozen=True, slots=True)
class Outcome:
    """Measured result for one defect type, with its verdict.

    An outcome carries its own schema version; a stored outcome without
    one is version 1.

    Attributes:
        certificate: Counts of planted defects found and clean rows flagged.
        recall: Interval for recall, by the policy's method.
        alarms: Interval for the false-alarm rate, by the policy's method.
        passed: Whether both bounds meet the policy.
        migrations: The outcome's version steps, oldest first.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "recall_interval": "recall",
                "false_alarm_interval": "alarms",
            }
        ),
    )

    certificate: qualification.Certificate
    recall: tuple[float, float]
    alarms: tuple[float, float]
    passed: bool

    @classmethod
    def measure(
        cls,
        certificate: qualification.Certificate,
        policy: Standard,
    ) -> Self:
        """Computes the intervals and verdict for `certificate`.

        Args:
            certificate: Counts of planted defects found and clean rows
                flagged.
            policy: The floors to judge against.

        Returns:
            The outcome. It passes when the lower bound of recall is at
            least the recall floor and the upper bound of the false-alarm
            rate is at most the false-alarm ceiling, both by the policy's
            method.
        """
        recall = interval(
            certificate.detected,
            certificate.planted,
            policy.confidence,
            policy.method,
        )
        alarms = interval(
            certificate.alarms,
            certificate.clean,
            policy.confidence,
            policy.method,
        )
        passed = recall[0] >= policy.recall and alarms[1] <= policy.alarms
        return cls(certificate, recall, alarms, passed)

    def to_dict(self) -> dict[str, Any]:
        """Returns the outcome as a JSON-ready record.

        Returns:
            A mapping of the certificate, the two intervals as lists and
            the verdict.
        """
        return {
            "schema_version": records.ORIGIN + len(self.migrations),
            "certificate": self.certificate.to_dict(),
            "recall": list(self.recall),
            "alarms": list(self.alarms),
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any], policy: Standard) -> Self:
        """Rebuilds an outcome and checks it against its own counts.

        Args:
            record: A mapping holding exactly the fields of the outcome, at
                any supported schema version.
            policy: The policy the verdict is recomputed under.

        Returns:
            The outcome as recomputed from the counts.

        Raises:
            CertificateError: If a stored interval or verdict disagrees with
                the value recomputed from the counts.
        """
        record = qualification.require(cls, record)
        certificate = qualification.Certificate.from_dict(record["certificate"])
        expected = cls.measure(certificate, policy)
        for name in ("recall", "alarms"):
            limits = record[name]
            if (
                not isinstance(limits, list | tuple)
                or len(limits) != PAIR_LENGTH
            ):
                raise errors.InputError(f"{name} must hold (low, high)")
            stored = (
                qualification.finite(limits[0], name),
                qualification.finite(limits[1], name),
            )
            recomputed = getattr(expected, name)
            if any(
                abs(a - b) > INTERVAL_TOLERANCE
                for a, b in zip(stored, recomputed, strict=True)
            ):
                raise CertificateError(
                    f"{name} interval does not match the recorded counts"
                )
        if record["passed"] is not expected.passed:
            raise CertificateError(
                "passed does not match the recorded counts and policy"
            )
        return expected


def nest(layout: records.Layout) -> records.Layout:
    """Moves a renamed version-1 certificate's hashes under ``digests``.

    Args:
        layout: The certificate after its other keys were renamed.

    Returns:
        The certificate with ``code_hash`` and ``corpus_hash`` nested as
        ``digests.code`` and ``digests.corpus``. The policy and outcomes
        carry their own versions and are not touched.
    """
    rest = {k: v for k, v in layout.items() if k not in DIGESTS}
    rest["digests"] = {
        new: layout[old] for old, new in DIGESTS.items() if old in layout
    }
    return rest


def flatten(layout: records.Layout) -> records.Layout:
    """Undoes `nest`.

    Args:
        layout: A version-2 certificate.

    Returns:
        The certificate with its digests as version-1 keys.
    """
    rest = {k: v for k, v in layout.items() if k != "digests"}
    digests = layout.get("digests", {})
    rest.update(
        {old: digests[new] for old, new in DIGESTS.items() if new in digests}
    )
    return rest


@dataclasses.dataclass(frozen=True, slots=True)
class Certification:
    """Certificate for one detector version on one corpus and seed set.

    A certificate carries its own schema version, as do its policy and
    outcomes; a stored certificate without one is version 1. Nothing is
    re-hashed on loading, so a migrated certificate keeps the code hash it
    was issued for.

    Attributes:
        detector: Detector that was measured.
        digests: Hashes keyed by the hashed thing: ``code``, the code that
            was measured (the staleness key), and ``corpus``, the data the
            plants were injected into.
        seeds: Run seeds the injections used.
        policy: Floors the outcomes were judged against.
        outcomes: One outcome per defect type.
        created: Caller-supplied ISO timestamp, or None. Never read from
            the clock, so the file is reproducible.
        migrations: The certificate's version steps, oldest first.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={"detector_id": "detector"}, change=nest, revert=flatten
        ),
    )

    detector: str
    digests: Mapping[str, str]
    seeds: tuple[int, ...]
    policy: Standard
    outcomes: tuple[Outcome, ...]
    created: str | None = None

    def __post_init__(self) -> None:
        """Validates the record.

        Raises:
            ValueError: If an id or hash is invalid, `digests` is not exactly
                ``code`` and ``corpus``, `seeds` or `outcomes` is not a
                non-empty tuple, a seed is not a non-negative integer,
                defect types repeat, or an outcome belongs to a different
                detector.
        """
        evidence.require_text(self.detector, "detector")
        if set(self.digests) != {"code", "corpus"}:
            raise errors.InputError("digests must hold code and corpus")
        require_digest(self.digests["code"], "code digest")
        require_digest(self.digests["corpus"], "corpus digest")
        object.__setattr__(self, "digests", dict(self.digests))
        qualification.tupled(self.seeds, "seeds")
        if not self.seeds:
            raise errors.InputError("seeds must not be empty")
        for seed in self.seeds:
            qualification.count(seed, "seeds")
        qualification.tupled(self.outcomes, "outcomes")
        if not self.outcomes:
            raise errors.InputError("outcomes must not be empty")
        kinds = [o.certificate.defect for o in self.outcomes]
        if len(set(kinds)) != len(kinds):
            raise errors.InputError("defect types must be unique")
        for outcome in self.outcomes:
            if outcome.certificate.detector != self.detector:
                raise errors.InputError(
                    "outcome belongs to a different detector"
                )

    @property
    def key(self) -> tuple[str, str, str, tuple[int, ...]]:
        """The detector id, code hash, corpus hash and sorted seeds."""
        return (
            self.detector,
            self.digests["code"],
            self.digests["corpus"],
            tuple(sorted(self.seeds)),
        )

    @property
    def certified_defect_types(self) -> tuple[str, ...]:
        """The defect types whose outcome passed the policy."""
        return tuple(o.certificate.defect for o in self.outcomes if o.passed)

    def to_dict(self) -> dict[str, Any]:
        """Returns the certificate as a JSON-ready record.

        Returns:
            A mapping of every field to its value, with the policy and the
            outcomes as records and the seeds as a list.
        """
        return {
            "schema_version": records.ORIGIN + len(self.migrations),
            "detector": self.detector,
            "digests": dict(self.digests),
            "seeds": list(self.seeds),
            "policy": self.policy.to_dict(),
            "outcomes": [o.to_dict() for o in self.outcomes],
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a certificate, verifying every stored verdict.

        Args:
            record: A mapping holding exactly the fields of the certificate,
                at any supported schema version.

        Returns:
            The certificate. A record that is invalid, or whose stored
            verdict disagrees with its counts, raises `ValueError` or
            `CertificateError`.
        """
        record = qualification.require(cls, record)
        policy = Standard.from_dict(record["policy"])
        digests = record["digests"]
        if not isinstance(digests, Mapping):
            raise errors.InputError("digests must be a mapping")
        return cls(
            detector=record["detector"],
            digests=digests,
            seeds=tuple(record["seeds"]),
            policy=policy,
            outcomes=tuple(
                Outcome.from_dict(item, policy) for item in record["outcomes"]
            ),
            created=record["created"],
        )


def certify(
    detector_id: str,
    code_digest: str,
    corpus_digest: str,
    seeds: Sequence[int],
    results: Sequence[qualification.Certificate],
    policy: Standard,
    created: str | None = None,
) -> Certification:
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
    outcomes = tuple(Outcome.measure(r, policy) for r in results)
    return Certification(
        detector=detector_id,
        digests={"code": code_digest, "corpus": corpus_digest},
        seeds=tuple(seeds),
        policy=policy,
        outcomes=outcomes,
        created=created,
    )


def demote(
    decision: qualification.Decision, status: Authority
) -> qualification.Decision:
    """Limits a descriptive check to a warning at most.

    Args:
        decision: The decision the check reached.
        status: Whether the check may gate.

    Returns:
        `decision` when the check is gating or the decision is not a block
        or quarantine; otherwise a warning.
    """
    if status is Authority.GATING:
        return decision
    if decision in (
        qualification.Decision.BLOCK,
        qualification.Decision.QUARANTINE,
    ):
        return qualification.Decision.WARN
    return decision


@dataclasses.dataclass(frozen=True, slots=True)
class Audit:
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


class Store:
    """The set of certificates written as `detector_certificates.json`.

    The document's own layout is unchanged since version 1; each
    certificate in it carries its own schema version.
    """

    def __init__(self, items: Iterable[Certification] = ()) -> None:
        """Creates a store, keeping the last certificate for each key.

        Args:
            items: Certificates to add, in order.
        """
        self.by_key: dict[tuple[Any, ...], Certification] = {}
        for item in items:
            self.add(item)

    @property
    def certificates(self) -> tuple[Certification, ...]:
        """The stored certificates, ordered by key."""
        return tuple(self.by_key[key] for key in sorted(self.by_key))

    def add(self, item: Certification) -> None:
        """Stores a certificate, replacing one with the same key.

        Args:
            item: The certificate to store.
        """
        self.by_key[item.key] = item

    def current(
        self, detector_id: str, code_digest: str
    ) -> tuple[Certification, ...]:
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
            if c.detector == detector_id and c.digests["code"] == code_digest
        )

    def state(self, detector_id: str, code_digest: str) -> Freshness:
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
            return Freshness.CURRENT
        if any(c.detector == detector_id for c in self.certificates):
            return Freshness.STALE
        return Freshness.MISSING

    def status(
        self, detector_id: str, code_digest: str, defect_type: str
    ) -> Authority:
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
                return Authority.GATING
        return Authority.DESCRIPTIVE

    def authorise(
        self, detector_id: str, code_digest: str, defect_type: str
    ) -> Authority:
        """Returns `GATING`, or raises when the detector may not gate.

        Args:
            detector_id: The detector.
            code_digest: Hash of the detector's current code.
            defect_type: The defect type.

        Returns:
            `Authority.GATING`.

        Raises:
            StaleError: The certificate is for other code.
            CertificateError: There is no certificate, or it does not pass
                for `defect_type`.
        """
        state = self.state(detector_id, code_digest)
        if state is Freshness.MISSING:
            raise CertificateError(
                f"no certificate for detector '{detector_id}'"
            )
        if state is Freshness.STALE:
            raise StaleError(
                f"certificate for '{detector_id}' is for different code"
            )
        if self.status(detector_id, code_digest, defect_type) is not (
            Authority.GATING
        ):
            raise CertificateError(
                f"'{detector_id}' is not certified for '{defect_type}'"
            )
        return Authority.GATING

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

        Each certificate migrates itself from the version it was stored at.

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
            raise errors.InputError(f"unexpected field(s): {extra}")
        return cls(
            Certification.from_dict(item) for item in document["certificates"]
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


def audit(store: Store, gating: Mapping[str, tuple[str, str]]) -> Audit:
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
        if state is Freshness.MISSING:
            missing.append(detector_id)
        elif state is Freshness.STALE:
            stale.append(detector_id)
        elif store.status(detector_id, digest, defect) is not (
            Authority.GATING
        ):
            uncertified.append(detector_id)
    return Audit(tuple(missing), tuple(stale), tuple(uncertified))
