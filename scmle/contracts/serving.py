"""Versioned request and response schemas of the prediction service.

Version 1 is the only version. A caller may name the version it was written
against (`schema_version`) and the model schema it expects
(`expected_schema_digest`); the service refuses a version or a schema it does
not serve with a typed error instead of guessing. Every response carries the
schema version, the model version and the digest of the exact artifact.
"""

from collections.abc import Mapping
import dataclasses
import hashlib
import json
from typing import Any

from scmle.contracts import errors

SCHEMA_VERSION = "1"
SUPPORTED_VERSIONS = (SCHEMA_VERSION,)
PREDICT_PATHS = ("/predict", "/v1/predict")
NULL_POLICIES = ("impute", "reject")
UNSEEN_POLICIES = ("report", "reject")


class SchemaChangeError(errors.ScmleError):
    """The caller and the service disagree about the model's input schema."""

    category = "schema_change"


class PolicyError(errors.DataError):
    """Input is well formed but the service policy forbids scoring it."""

    category = "data_policy"


@dataclasses.dataclass(frozen=True, slots=True)
class ScoreRequest:
    """A parsed prediction request."""

    records: list[dict[str, Any]]
    schema_version: str = SCHEMA_VERSION
    expected_schema_digest: str | None = None


def schema_digest(schema: Mapping[str, Any]) -> str:
    """Returns the `sha256:` digest of a bundle's input schema.

    Args:
        schema: The input schema.

    Returns:
        ``sha256:`` followed by the hex digest of the schema's compact,
        key-sorted JSON.
    """
    text = json.dumps(
        schema, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_request(payload: object, max_records: int) -> ScoreRequest:
    """Validates a decoded JSON body against request schema version 1.

    Args:
        payload: The decoded JSON body.
        max_records: Largest number of records one request may hold.

    Returns:
        The request: its records, schema version and the digest the caller
        expects, if given.

    Raises:
        ValueError: If the body has unknown fields or no records.
        OverflowError: If it holds more than `max_records` records.
        SchemaChangeError: If it names an unsupported schema version.
        TypeError: If the body is not an object, the records are not a list
            of objects, or the expected digest is not text.
    """
    if not isinstance(payload, dict):
        raise TypeError("the request body must be a JSON object")
    version = payload.get("schema_version", SCHEMA_VERSION)
    if version not in SUPPORTED_VERSIONS:
        raise SchemaChangeError(
            f"unsupported request schema_version {version!r}",
            {"supported": list(SUPPORTED_VERSIONS)},
        )
    unknown = sorted(
        set(payload) - {"schema_version", "records", "expected_schema_digest"}
    )
    if unknown:
        raise ValueError(f"unknown request fields: {unknown}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise TypeError("'records' must be a list")
    if not records:
        raise ValueError("records must not be empty")
    if len(records) > max_records:
        raise OverflowError(f"request contains more than {max_records} records")
    if not all(isinstance(record, dict) for record in records):
        raise TypeError("every record must be an object")
    expected = payload.get("expected_schema_digest")
    if expected is not None and not isinstance(expected, str):
        raise TypeError("'expected_schema_digest' must be text")
    return ScoreRequest(records, str(version), expected)


REQUEST_JSON_SCHEMA: dict[str, Any] = {
    "$id": "scmle.serving.request.v1",
    "type": "object",
    "required": ["records"],
    "additionalProperties": False,
    "properties": {
        "schema_version": {"enum": list(SUPPORTED_VERSIONS)},
        "expected_schema_digest": {"type": "string"},
        "records": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "object"},
        },
    },
}

RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "$id": "scmle.serving.response.v1",
    "type": "object",
    "required": [
        "schema_version",
        "request_id",
        "model_version",
        "artifact_digest",
        "schema_digest",
        "record_count",
        "predictions",
        "schema_valid",
        "diagnostics",
        "rows",
    ],
    "properties": {
        "schema_version": {"enum": list(SUPPORTED_VERSIONS)},
        "request_id": {"type": "string"},
        "model_version": {"type": "string"},
        "artifact_digest": {"pattern": "^sha256:[0-9a-f]{64}$"},
        "schema_digest": {"pattern": "^sha256:[0-9a-f]{64}$"},
        "record_count": {"type": "integer"},
        "predictions": {"type": "array"},
        "classes": {"type": "array"},
        "probabilities": {"type": "array"},
        "out_of_distribution": {"type": "array"},
        "schema_valid": {"type": "boolean"},
        "diagnostics": {"type": "object"},
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["missing", "unseen"],
            },
        },
    },
}

ERROR_JSON_SCHEMA: dict[str, Any] = {
    "$id": "scmle.serving.error.v1",
    "type": "object",
    "required": ["schema_version", "request_id", "error", "category"],
}

RESPONSE_KEYS = tuple(RESPONSE_JSON_SCHEMA["required"])
