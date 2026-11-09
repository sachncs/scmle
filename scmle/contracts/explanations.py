"""Portable, reviewable evidence contracts for model explanations."""

from collections.abc import Mapping
import datetime
import json
import math
from typing import Any
from typing import Literal
from urllib.parse import urlparse

import pydantic

EXPLANATION_SCHEMA_VERSION = 1
PrivacyStatus = Literal["allowed", "redacted", "denied"]


def require_text(value: str, field_name: str) -> str:
    """Returns non-empty text for a required contract field.

    Args:
        value: Candidate field value.
        field_name: Name used in the validation error.

    Returns:
        The stripped text.

    Raises:
        ValueError: If the value is empty after stripping whitespace.
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty")
    return stripped


def validate_json_value(value: Any, field_name: str = "value") -> object:
    """Validates and normalizes a value accepted by the evidence contract.

    Only JSON-native values are accepted. In particular, non-finite floats,
    non-string object keys, NumPy scalars, and arbitrary objects are rejected
    instead of being silently stringified into evidence.

    Args:
        value: Candidate JSON value.
        field_name: Path used in validation errors.

    Returns:
        A JSON-native value containing only dictionaries and lists.

    Raises:
        TypeError: If the value is not JSON-native.
        ValueError: If a float is not finite.
    """
    if value is None or type(value) is bool or type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite float")
        return value
    if type(value) is str:
        return value
    if isinstance(value, (list, tuple)):
        return [
            validate_json_value(item, f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{field_name} has a non-string key: {key!r}")
            normalized[key] = validate_json_value(item, f"{field_name}.{key}")
        return normalized
    raise TypeError(
        f"{field_name} must contain only JSON-native values, got "
        f"{type(value).__name__}"
    )


class Contract(pydantic.BaseModel):
    """Immutable base model shared by explanation evidence records."""

    model_config = pydantic.ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class DatasetSnapshot(Contract):
    """Identity of the dataset state used to produce an explanation."""

    identifier: str
    version: str
    digest: str

    @pydantic.field_validator("identifier", "version", "digest")
    @classmethod
    def validate_identity(cls, value: str, info: Any) -> str:
        """Rejects blank snapshot identity components.

        Args:
            value: The component text.
            info: The validation context, naming the field.

        Returns:
            The text with surrounding whitespace removed; blank text is
            rejected.
        """
        return require_text(value, f"dataset_snapshot.{info.field_name}")


class FeatureGraphReference(Contract):
    """Version and fingerprint of the feature transformation graph."""

    version: int = pydantic.Field(ge=1)
    fingerprint: str

    @pydantic.field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        """Rejects a missing feature-graph fingerprint.

        Args:
            value: The fingerprint.

        Returns:
            The fingerprint with surrounding whitespace removed.
        """
        return require_text(value, "feature_graph.fingerprint")


class Method(Contract):
    """Algorithm identity and resolved parameters for an explanation."""

    name: str
    parameters: dict[str, Any]

    @pydantic.field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Rejects an unnamed explanation method.

        Args:
            value: The method name.

        Returns:
            The name with surrounding whitespace removed.
        """
        return require_text(value, "method.name")

    @pydantic.field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, value: Any) -> dict[str, object]:
        """Validates method parameters without lossy coercion.

        Args:
            value: The parameters as supplied.

        Returns:
            The parameters as a JSON-native mapping.

        Raises:
            ValueError: If the parameters are not JSON-native or not an object.
        """
        try:
            normalized = validate_json_value(value, "method.parameters")
        except (TypeError, ValueError) as error:
            raise ValueError(str(error)) from error
        if not isinstance(normalized, dict):
            raise ValueError("method.parameters must be a JSON object")
        return normalized


class PrivacyDecision(Contract):
    """Decision describing whether explanation content may be disclosed."""

    decision: PrivacyStatus
    reason: str
    redacted_fields: tuple[str, ...] = ()

    @pydantic.field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        """Requires an auditable reason for the privacy decision.

        Args:
            value: The reason.

        Returns:
            The reason with surrounding whitespace removed.
        """
        return require_text(value, "privacy_decision.reason")

    @pydantic.field_validator("redacted_fields")
    @classmethod
    def validate_redacted_fields(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Rejects blank names in the redaction record.

        Args:
            value: The names of the redacted fields.

        Returns:
            The names with surrounding whitespace removed.
        """
        return tuple(
            require_text(field, "privacy_decision.redacted_fields")
            for field in value
        )


class EvidenceLink(Contract):
    """Reference to supporting evidence for an explanation record."""

    relation: str
    uri: str
    digest: str | None = None

    @pydantic.field_validator("relation", "uri")
    @classmethod
    def validate_text(cls, value: str, info: Any) -> str:
        """Requires a non-empty relation and URI.

        A blank value is rejected with a `ValueError`.

        Args:
            value: The relation or URI text.
            info: The validation context, naming the field.

        Returns:
            The text with surrounding whitespace removed.
        """
        return require_text(value, f"evidence_link.{info.field_name}")

    @pydantic.field_validator("uri")
    @classmethod
    def validate_uri(cls, value: str) -> str:
        """Rejects URI schemes that cannot be safely rendered as links.

        Args:
            value: The URI.

        Returns:
            `value`, unchanged.

        Raises:
            ValueError: If the scheme is data, javascript or vbscript.
        """
        scheme = urlparse(value).scheme.casefold()
        if scheme in {"data", "javascript", "vbscript"}:
            raise ValueError(f"unsupported evidence link scheme: {scheme}")
        return value

    @pydantic.field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str | None) -> str | None:
        """Normalizes an optional evidence digest.

        Args:
            value: The digest, or None.

        Returns:
            The digest with surrounding whitespace removed, or None.
        """
        return None if value is None else require_text(value, "link.digest")


class Evidence(Contract):
    """Portable evidence record for one global or local explanation.

    The record deliberately stores references and JSON-native outputs rather
    than Python objects. This makes the evidence portable across processes,
    safe to persist, and suitable for rendering by a trusted HTML viewer.
    ``canonical_json`` escapes HTML-sensitive characters while preserving a
    deterministic key and separator order.

    Attributes:
        model_fingerprint: Content identity of the model being explained.
        bundle_digest: Digest of the packaged model bundle.
        input_fingerprint: Content identity of the explained input.
        dataset_snapshot: Dataset identity and version used for context.
        feature_graph: Feature graph version and fingerprint.
        method: Explanation method and resolved parameters.
        output: JSON-native explanation result.
        timestamp: Time the explanation was generated; normalized to UTC.
        software_version: Version of the software that generated the record.
        uncertainty: JSON-native uncertainty evidence.
        stability: JSON-native stability evidence.
        limitations: Known limitations that must travel with the result.
        causal_warning: Explicit causal-versus-associational warning.
        privacy_decision: Disclosure and redaction decision.
        evidence_links: Supporting artifact or report references.
        schema_version: Version of the evidence record layout; at least one.
    """

    schema_version: int = pydantic.Field(
        default=EXPLANATION_SCHEMA_VERSION, ge=1
    )
    model_fingerprint: str
    bundle_digest: str
    input_fingerprint: str
    dataset_snapshot: DatasetSnapshot
    feature_graph: FeatureGraphReference
    method: Method
    output: Any
    timestamp: datetime.datetime
    software_version: str
    uncertainty: dict[str, Any]
    stability: dict[str, Any]
    limitations: tuple[str, ...]
    causal_warning: str
    privacy_decision: PrivacyDecision
    evidence_links: tuple[EvidenceLink, ...]

    @pydantic.field_validator(
        "model_fingerprint",
        "bundle_digest",
        "input_fingerprint",
        "software_version",
    )
    @classmethod
    def validate_identity(cls, value: str, info: Any) -> str:
        """Rejects blank top-level identity fields.

        Args:
            value: The field text.
            info: The validation context, naming the field.

        Returns:
            The text with surrounding whitespace removed.
        """
        return require_text(value, info.field_name)

    @pydantic.field_validator(
        "output", "uncertainty", "stability", mode="before"
    )
    @classmethod
    def validate_json_fields(cls, value: Any, info: Any) -> object:
        """Rejects values that cannot be safely serialized as JSON.

        Args:
            value: The output, uncertainty or stability value.
            info: The validation context, naming the field.

        Returns:
            The value as JSON-native data.

        Raises:
            ValueError: If the value is not JSON-native; this wraps the
                `TypeError` or `ValueError` of the JSON validation.
        """
        try:
            return validate_json_value(value, info.field_name)
        except (TypeError, ValueError) as error:
            raise ValueError(str(error)) from error

    @pydantic.field_validator("uncertainty", "stability")
    @classmethod
    def validate_measurement_object(
        cls, value: dict[str, object], info: Any
    ) -> dict[str, object]:
        """Requires an explicit uncertainty and stability object.

        Args:
            value: The uncertainty or stability mapping.
            info: The validation context, naming the field.

        Returns:
            `value`, unchanged.

        Raises:
            ValueError: If the mapping is empty.
        """
        if not value:
            raise ValueError(f"{info.field_name} must not be empty")
        return value

    @pydantic.field_validator("limitations")
    @classmethod
    def validate_limitations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Requires at least one limitation to prevent silent overclaiming.

        Args:
            value: The limitations.

        Returns:
            The limitations with surrounding whitespace removed.

        Raises:
            ValueError: If there is none, or an item is blank.
        """
        if not value:
            raise ValueError("limitations must contain at least one item")
        return tuple(require_text(item, "limitations") for item in value)

    @pydantic.field_validator("causal_warning")
    @classmethod
    def validate_causal_warning(cls, value: str) -> str:
        """Requires an explicit causal interpretation warning.

        Args:
            value: The warning text.

        Returns:
            The warning with surrounding whitespace removed.
        """
        return require_text(value, "causal_warning")

    @pydantic.field_validator("evidence_links")
    @classmethod
    def validate_evidence_links(
        cls, value: tuple[EvidenceLink, ...]
    ) -> tuple[EvidenceLink, ...]:
        """Requires at least one independently addressable evidence link.

        Args:
            value: The evidence links.

        Returns:
            `value`, unchanged.

        Raises:
            ValueError: If there is no link.
        """
        if not value:
            raise ValueError("evidence_links must contain at least one item")
        return value

    @pydantic.field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: datetime.datetime) -> datetime.datetime:
        """Requires an aware timestamp and stores it in UTC.

        Args:
            value: The timestamp.

        Returns:
            The same instant in UTC.

        Raises:
            ValueError: If the timestamp has no timezone.
        """
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include timezone information")
        return value.astimezone(datetime.UTC)

    def to_dict(self) -> dict[str, object]:
        """Returns a JSON-native, deterministic evidence mapping.

        Returns:
            The evidence as JSON-native data.

        Raises:
            TypeError: If the serialized evidence is not an object, which
                cannot happen in practice because a model dump is always a
                mapping.
        """
        record = self.model_dump(mode="json")
        normalized = validate_json_value(record, "evidence")
        if not isinstance(normalized, dict):
            raise TypeError("serialized explanation evidence must be an object")
        return normalized

    def canonical_json(self) -> str:
        """Returns deterministic JSON safe to embed in an HTML document."""
        payload = json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return (
            payload.replace("&", "\\u0026")
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
        )
