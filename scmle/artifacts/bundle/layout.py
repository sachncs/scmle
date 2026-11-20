"""The files of a bundle directory, its format and file checksums."""

import hashlib
import pathlib

# Format 2 signs and checksums the schema alongside every serialized model
# component. Format 1 bundles did not authenticate schema.json and are refused
# by the loader rather than silently weakening the trust boundary. Formats 3
# to 7 changed the module paths or class names stored in the pickled members:
# 3 for the class renames, 4 for the package layering (register phase 3,
# DES-13), 5 for the single-word names of lane data_assurance (phase 6:
# `SkewedLogTransformer` is `Logarithm`, `AsinhTarget` is `Asinh`), 6 for
# those of lane models (phase 6: the estimator, interval and calibrator
# classes such as `CvPlusEnsemble`, `QuantileIntervals`, `StoredShiftWeights`
# and `DirichletCalibrator`, and their attributes), 7 for those of lane rest
# (phase 6: the preparation wrappers, resamplers and selector such as
# `FlatPredictionClassifier`, `RandomOverSampler` and `SparseSelector`, and
# their attributes).
BUNDLE_FORMAT = 7
MODEL_FILE = "pipeline.joblib"
SCHEMA_FILE = "schema.json"
REFERENCE_FILE = "reference.json"
CALIBRATOR_FILE = "calibrator.joblib"
CONFORMAL_FILE = "conformal.json"
QUANTILE_FILE = "quantile.joblib"
CV_PLUS_FILE = "cv_plus.joblib"
SHIFT_WEIGHTS_FILE = "shift_weights.joblib"
COVERAGE_FILE = "coverage_report.json"
NOVELTY_FILE = "novelty.joblib"
HISTORY_FILE = "series_history.parquet"
MANIFEST_FILE = "model_manifest.json"


def shasum(path: pathlib.Path) -> str:
    """Returns the SHA-256 of a file, streamed.

    Args:
        path: The file.

    Returns:
        The hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
