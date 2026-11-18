"""Draft statuses, slot thresholds, the request and the framing result."""

from collections.abc import Mapping
import dataclasses
import hashlib
import re
from typing import Any
from typing import ClassVar

from scmle.assurance.charter import contracts
from scmle.assurance.framing import provenance
import scmle.assurance.framing.questions
from scmle.contracts import records
from scmle.contracts import signing

ISSUED = "issued"
WITH_QUESTIONS = "issued_with_questions"
REFUSED = "refused"
MAX_TARGET_MISSING = 0.2
MIN_DISTINCT_VALUES = 2
MIN_SUBSTRING_CHARS = 4
MIN_POPULATION_SHARE = 0.01
UNIT_DUPLICATE_TOLERANCE = 0.01
TARGET_WORDS = ("target", "label", "outcome", "y")
CAUSAL_WORDS = re.compile(
    r"\b(cause|causal|effect of|impact of|because of|treatment effect)\b",
    re.IGNORECASE,
)
BACKTICKED = re.compile(r"`([^`]+)`")
UNSPECIFIED = "unspecified (open question)"
CRITICAL_FIELDS = ("target", "prediction_moment", "unit")
MAX_MOMENT_VIOLATIONS = 0
COST_RANGE = (0.05, 0.5)


@dataclasses.dataclass(frozen=True, slots=True)
class Offer:
    """What the data source offers.

    Attributes:
        uri: Where the data lives.
        columns: Column names the source has.
        key: Declared key columns, if any.
        time: Column holding when the outcome is known, if any.
        migrations: The stored layout's version steps, oldest first; version
            1 spelled `key` and `time` as `primary_key` and
            `target_time_column`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={"primary_key": "key", "target_time_column": "time"}
        ),
    )

    uri: str
    columns: tuple[str, ...]
    key: tuple[str, ...] = ()
    time: str | None = None

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Offer":
        """Builds the description from a JSON object of any stored version.

        Args:
            record: A mapping with `uri` and `columns`, and optionally `key`
                and `time` (version 1: `primary_key` and
                `target_time_column`; no `schema_version` key means 1).

        Returns:
            The description. A missing `uri` or `columns` raises `KeyError`.
        """
        layout = records.load(cls.__name__, cls.migrations, record)
        return cls(
            uri=str(layout["uri"]),
            columns=tuple(layout["columns"]),
            key=tuple(layout.get("key", ())),
            time=layout.get("time"),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Request:
    """A plain-language request and the source it is about.

    Attributes:
        text: The request in plain language.
        source: What the data source offers.
    """

    text: str
    source: Offer


@dataclasses.dataclass(frozen=True, slots=True)
class Probe:
    """One LLM-free check of a draft against `X`.

    Attributes:
        name: The check.
        ok: Whether it passed.
        blocking: Whether a failure stops the draft.
        detail: Counts behind the result.
    """

    name: str
    ok: bool
    blocking: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Returns the check as a JSON-ready record.

        Returns:
            A mapping of every field to its value.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class Draft:
    """The result of `frame`.

    Attributes:
        status: `issued`, `issued_with_questions` or `refused`.
        charter: The unsigned draft charter, or None when refused.
        fields: Source of each drafted field.
        checks: Data-side checks.
        questions: Tipping-point questions for a person.
        robust: Unknowns whose plausible range leaves the verdict unchanged.
        open_items: Slots left at a placeholder, to be answered.
        baseline: The baseline measured on `X`, if a decision column exists.
        refusal: Why the draft was refused, if it was.
        llm: Model id, prompt, raw samples, hash and ambiguity, if used.
        discarded: What was dropped and why (the LLM path, for example).
        migrations: The written layout's version steps, oldest first;
            version 2 spells `open_items` as `pending`. The digest
            (`sha`) is computed on the version-1 layout.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(renames={"open_items": "pending"}),
    )

    status: str
    charter: contracts.AnalysisCharter | None
    fields: dict[str, provenance.Filling]
    checks: tuple[Probe, ...]
    questions: tuple[scmle.assurance.framing.questions.Question, ...]
    robust: tuple[dict[str, Any], ...]
    open_items: tuple[str, ...]
    baseline: dict[str, Any] | None
    refusal: dict[str, Any] | None
    llm: dict[str, Any] | None
    discarded: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Returns the `framing_draft.json` document.

        Returns:
            A JSON-ready mapping of the status, the charter and its hash
            (both None when refused), the source of each field, the checks,
            the open items, the LLM record, what was discarded, and a label
            that the draft is unverified.
        """
        return {
            records.KEY: records.ORIGIN + len(self.migrations),
            "status": self.status,
            "charter": None if self.charter is None else self.charter.to_dict(),
            "charter_hash": (
                None if self.charter is None else self.charter.charter_hash
            ),
            "fields": {k: v.value for k, v in sorted(self.fields.items())},
            "checks": [c.to_dict() for c in self.checks],
            "pending": list(self.open_items),
            "llm": self.llm,
            "discarded": list(self.discarded),
            "label": (
                "a drafted charter is a draft: nothing here shows it is "
                "correct; llm fields are unverified until a person signs"
            ),
        }

    @property
    def sha(self) -> str:
        """The SHA-256 hex digest of the draft document, anchored at version 1.

        The document enters in its version-1 layout (`anchor`) and the
        charter in its own (`AnalysisCharter.anchor`), so renaming a draft,
        charter or cost-model key leaves the digest of a registered draft
        unchanged.
        """
        document = self.anchor()
        if self.charter is not None:
            document["charter"] = self.charter.anchor()
        return hashlib.sha256(signing.canonicalise(document)).hexdigest()

    def anchor(self) -> records.Layout:
        """Builds the document in its version-1 layout.

        Returns:
            The `framing_draft.json` document as version 1 wrote it
            (`open_items`), with `schema_version` 1.
        """
        current = self.to_dict()
        del current[records.KEY]
        return {
            records.KEY: records.ORIGIN,
            **records.downgrade(self.migrations, current),
        }
