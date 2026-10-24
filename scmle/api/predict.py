"""Prediction from a saved model bundle, without the training project."""

import pathlib
from typing import Any

import numpy as np
import pandas as pd

from scmle.artifacts import bundle as bundle_module
from scmle.inference import mapping as mapping_module


def load_model(
    directory: str | pathlib.Path,
    trusted: bool = False,
    signing_key: str | bytes | None = None,
) -> bundle_module.ModelBundle:
    """Loads a saved model bundle.

    Bundles are pickle-based, so loading is refused unless the signature
    verifies or the caller vouches for the source.

    Args:
        directory: Directory written by `RunResult.save`.
        trusted: Load an unsigned bundle you vouch for.
        signing_key: HMAC key verifying the bundle signature.

    Returns:
        The verified bundle.
    """
    return bundle_module.ModelBundle.load(
        directory, trusted=trusted, signing_key=signing_key
    )


def predict(
    directory: str | pathlib.Path,
    data: str | pathlib.Path | pd.DataFrame,
    trusted: bool = False,
    signing_key: str | bytes | None = None,
    mapping: mapping_module.ColumnMapping | None = None,
) -> np.ndarray:
    """Loads a bundle and predicts labels or values.

    Args:
        directory: Saved bundle directory.
        data: DataFrame or path to a supported tabular dataset.
        trusted: Load an unsigned bundle you vouch for.
        signing_key: HMAC key verifying the bundle signature.
        mapping: Explicit column-name policy; exact names if omitted.

    Returns:
        Predictions, one per row.
    """
    model = load_model(directory, trusted, signing_key)
    return model.predict(data, mapping)


def score(
    directory: str | pathlib.Path,
    data: str | pathlib.Path | pd.DataFrame,
    trusted: bool = False,
    signing_key: str | bytes | None = None,
    mapping: mapping_module.ColumnMapping | None = None,
) -> dict[str, Any]:
    """Loads a bundle and returns predictions with schema diagnostics.

    Args:
        directory: Saved bundle directory.
        data: DataFrame or path to a supported tabular dataset.
        trusted: Load an unsigned bundle you vouch for.
        signing_key: HMAC key verifying the bundle signature.
        mapping: Explicit column-name policy; exact names if omitted.

    Returns:
        Predictions, probabilities for classifiers, model version and
        whether the input satisfied the schema.
    """
    model = load_model(directory, trusted, signing_key)
    return model.score(data, mapping)
