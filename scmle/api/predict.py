"""Prediction from a saved model bundle, without the training project."""

import pathlib
from typing import Any

import numpy
import pandas

from scmle.artifacts import bundle
import scmle.inference.mapping


def load_model(
    directory: str | pathlib.Path,
    trusted: bool = False,
    signing_key: str | bytes | None = None,
) -> bundle.ModelBundle:
    """Loads a saved model bundle.

    Bundles are pickle-based, so loading is refused unless the signature
    verifies or the caller vouches for the source. An unauthenticated,
    tampered or incompatible bundle raises `ArtifactError`, and so does
    `trusted=True` when `$SCMLE_ENV` is `production`.

    Args:
        directory: Directory written by `RunResult.save`.
        trusted: Load an unsigned bundle you vouch for.
        signing_key: HMAC key verifying the bundle signature; defaults to
            `$SCMLE_SIGNING_KEY`.

    Returns:
        The bundle, after its checksums and signature checks.
    """
    return bundle.ModelBundle.load(
        directory, trusted=trusted, signing_key=signing_key
    )


def predict(
    directory: str | pathlib.Path,
    data: str | pathlib.Path | pandas.DataFrame,
    trusted: bool = False,
    signing_key: str | bytes | None = None,
    mapping: scmle.inference.mapping.ColumnMapping | None = None,
) -> numpy.ndarray:
    """Loads a bundle and predicts labels or values.

    Args:
        directory: Saved bundle directory.
        data: DataFrame or path to a supported tabular dataset.
        trusted: Load an unsigned bundle you vouch for.
        signing_key: HMAC key verifying the bundle signature.
        mapping: Explicit column-name policy; exact names if omitted.

    Returns:
        Predictions, one per row (one column per target for multi-target
        models).
    """
    model = load_model(directory, trusted, signing_key)
    return model.predict(data, mapping)


def score(
    directory: str | pathlib.Path,
    data: str | pathlib.Path | pandas.DataFrame,
    trusted: bool = False,
    signing_key: str | bytes | None = None,
    mapping: scmle.inference.mapping.ColumnMapping | None = None,
) -> dict[str, Any]:
    """Loads a bundle and returns predictions with schema diagnostics.

    Args:
        directory: Saved bundle directory.
        data: DataFrame or path to a supported tabular dataset.
        trusted: Load an unsigned bundle you vouch for.
        signing_key: HMAC key verifying the bundle signature.
        mapping: Explicit column-name policy; exact names if omitted.

    Returns:
        A dict with `predictions`, `model_version`, `schema_valid` and
        `diagnostics`; classifiers also get `classes` and `probabilities`,
        and a bundle with a novelty detector gets `out_of_distribution`.
    """
    model = load_model(directory, trusted, signing_key)
    return model.score(data, mapping)
