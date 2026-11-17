"""Digest, manifest and signature checks of a modality package."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import hmac
import io
import json
from pathlib import Path

import joblib
import numpy
from sklearn.linear_model import LogisticRegression

from scmle import envelope
from scmle.contracts import errors
from scmle.contracts import signing
from scmle.pipeline.modality_lifecycle import constants
from scmle.pipeline.modality_lifecycle import containers


def file_digest(path: Path) -> str:
    """Computes the SHA-256 digest of a file.

    Args:
        path: File to hash.

    Returns:
        The hexadecimal digest of the file contents.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_feature_names(feature_names: object) -> list[str]:
    """Checks the feature names that a package manifest declares.

    Args:
        feature_names: The manifest's `feature_names` entry.

    Returns:
        The same list.

    Raises:
        ValueError: If it is not a non-empty list of distinct, non-blank
            strings.
    """
    if not isinstance(feature_names, list) or not feature_names:
        raise errors.InputError("modality package feature_names are invalid")
    if any(
        not isinstance(name, str) or not name.strip() for name in feature_names
    ) or len(set(feature_names)) != len(feature_names):
        raise errors.InputError("modality package feature_names are invalid")
    return feature_names


def authenticate(
    signature: object, unsigned: Mapping[str, object], signing_key: str | bytes
) -> None:
    """Checks a package manifest's signature over its unsigned content.

    Args:
        signature: The manifest's `signature` entry.
        unsigned: The manifest without its signature.
        signing_key: Key the signature is checked with.

    Raises:
        ValueError: If the signature is missing, unsupported, empty or does
            not verify.
    """
    if (
        not isinstance(signature, dict)
        or signature.get("algorithm") != signing.SIGNATURE_ALGORITHM
    ):
        raise errors.InputError(
            "modality package signature is missing or unsupported"
        )
    value = signature.get("value")
    if not isinstance(value, str) or not value.strip():
        raise errors.InputError("modality package signature is empty")
    expected = signing.mac(unsigned, signing_key)
    if not hmac.compare_digest(value, expected):
        raise errors.InputError("modality package signature does not verify")


def verify_package_manifest(
    manifest: object, signing_key: str | bytes
) -> tuple[list[str], str]:
    """Checks a package manifest's content and signature.

    Args:
        manifest: The parsed `manifest.json`.
        signing_key: Key the manifest signature is checked with.

    Returns:
        The feature names the package was trained on and the model digest
        that the manifest records.

    Raises:
        ValueError: If the manifest is not an object, or its schema version,
            modality, dataset hash, feature names, model digest, evaluation
            or signature is missing, malformed or does not verify.
    """
    if not isinstance(manifest, Mapping):
        raise errors.InputError("modality package manifest must be an object")
    unsigned = dict(manifest)
    signature = unsigned.pop("signature", None)
    schema_version = unsigned.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != (
        constants.PACKAGE_SCHEMA_VERSION
    ):
        raise errors.InputError("unsupported modality package schema version")
    modality = unsigned.get("modality")
    if not isinstance(modality, str) or not modality.strip():
        raise errors.InputError("modality package modality is invalid")
    dataset_hash = unsigned.get("dataset_hash")
    if not isinstance(dataset_hash, str) or not dataset_hash.strip():
        raise errors.InputError("modality package dataset_hash is invalid")
    feature_names = require_feature_names(unsigned.get("feature_names"))
    model_digest = unsigned.get("model_sha256")
    if not isinstance(model_digest, str) or not containers.hashlike(
        model_digest
    ):
        raise errors.InputError("modality package model_sha256 is invalid")
    if not isinstance(unsigned.get("evaluation"), Mapping):
        raise errors.InputError("modality package evaluation is invalid")
    authenticate(signature, unsigned, signing_key)
    return feature_names, model_digest


def vet(loaded: object, feature_count: int) -> None:
    """Checks a deserialized package classifier against its feature schema.

    Args:
        loaded: The deserialized payload.
        feature_count: Number of features the manifest declares.

    Raises:
        TypeError: If the payload is not a logistic regression.
        ValueError: If the model's feature count differs, it has fewer than
            two classes or it holds non-finite values.
    """
    if not isinstance(loaded, LogisticRegression):
        raise errors.KindError(
            "modality package does not contain the expected classifier"
        )
    if getattr(loaded, "n_features_in_", None) != feature_count:
        raise errors.InputError(
            "modality package model and feature schema do not match"
        )
    if len(getattr(loaded, "classes_", ())) < constants.MIN_CLASS_COUNT:
        raise errors.InputError(
            "modality package classifier has fewer than two classes"
        )
    if (
        not numpy.isfinite(loaded.coef_).all()
        or not numpy.isfinite(loaded.intercept_).all()
    ):
        raise errors.InputError(
            "modality package classifier contains non-finite values"
        )


def load_package(
    path: str | Path, signing_key: str | bytes
) -> LogisticRegression:
    """Verifies a signed package before deserializing its estimator.

    Args:
        path: Package directory holding `manifest.json` and `model.joblib`.
        signing_key: Key the manifest signature is checked with.

    Returns:
        The classifier stored in the package.

    Raises:
        ValueError: If the package is incomplete, its manifest, signature,
            digest or feature schema is invalid, or the classifier has fewer
            than two classes or non-finite values.
    """
    target = Path(path)
    manifest_path = target / "manifest.json"
    model_path = target / "model.joblib"
    if not manifest_path.is_file() or not model_path.is_file():
        raise errors.InputError("modality package is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_names, model_digest = verify_package_manifest(manifest, signing_key)
    payload = envelope.fetch(model_path, model_digest)
    loaded = joblib.load(io.BytesIO(payload))
    vet(loaded, len(feature_names))
    return loaded
