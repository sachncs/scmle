"""Model bundle: a validated, portable inference pipeline.

SECURITY: bundles are serialized with joblib (pickle). Loading executes code,
so only load bundles from sources you trust. Checksums detect corruption or
tampering after saving, not malice at the source.
"""

from collections.abc import Sequence
import contextlib
import dataclasses
import hashlib
import hmac
import importlib
import importlib.metadata
import json
import os
import pathlib
import platform
import tempfile
import threading
from typing import Any
from typing import Self

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn import calibration
from sklearn import frozen

import scmle
from scmle import durability
from scmle.contracts import artifacts as artifact_contracts
from scmle.contracts import data as data_contracts
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.readers import interchange
from scmle.data.readers import tabular
from scmle.evaluation import conformal as conformal_module
from scmle.evaluation import cv_plus as cv_plus_module
from scmle.evaluation import dirichlet
from scmle.evaluation import quantile as quantile_module
from scmle.evaluation import reliability
from scmle.evaluation import shift_weights as shift_weights_module
from scmle.explainability import counterfactual as counterfactual_module
from scmle.explainability import local
from scmle.features import feature_graph
from scmle.inference import mapping as mapping_module
from scmle.inference import novelty as novelty_module
from scmle.inference import onnx_export
from scmle.inference import packaging
from scmle.inference import validation
from scmle.models import thresholds
from scmle.monitoring import reference as reference_module

# Format 2 signs and checksums the schema alongside every serialized model
# component. Format 1 bundles did not authenticate schema.json and are refused
# by the loader rather than silently weakening the trust boundary.
BUNDLE_FORMAT = 2
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
PARITY_TOLERANCE = 1e-9
SIGNATURE_FILE = "signature.json"
SIGNING_KEY_ENV = "SCMLE_SIGNING_KEY"
SIGNATURE_ALGORITHM = "hmac-sha256"
MANIFEST_FILE = "model_manifest.json"
BINARY_CLASSES = 2
MAX_KNOWN_CATEGORIES = 200


TRACKED_LIBRARIES = (
    "numpy",
    "scipy",
    "joblib",
    "lightgbm",
    "xgboost",
    "catboost",
    "imblearn",
)


def installed_distributions() -> dict[str, str]:
    """Every installed distribution and its version (for the lock)."""
    found = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            found[name.lower().replace("_", "-")] = distribution.version
    return dict(sorted(found.items()))


def installed_versions() -> dict[str, str]:
    """Versions of the tracked libraries that are installed."""
    found = {}
    for name in TRACKED_LIBRARIES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        found[name] = str(getattr(module, "__version__", "unknown"))
    return found


def brier(
    probabilities: np.ndarray, labels: np.ndarray, classes: tuple[Any, ...]
) -> float:
    """Mean squared error between probabilities and one-hot labels."""
    truth = (labels[:, None] == np.asarray(classes)[None, :]).astype(float)
    return float(np.mean(np.sum((probabilities - truth) ** 2, axis=1)))


def file_sha256(path: pathlib.Path) -> str:
    """Returns the SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def minor_version(version: str) -> list[str]:
    """Returns the `[major, minor]` components of a version string."""
    return version.split(".")[:2]


def resolve_key(signing_key: str | bytes | None) -> bytes | None:
    """Returns the signing key from the argument or the environment."""
    key = (
        signing_key
        if signing_key is not None
        else os.environ.get(SIGNING_KEY_ENV)
    )
    if key is None or key == "":
        return None
    return key.encode() if isinstance(key, str) else key


def sign_manifest(manifest_bytes: bytes, key: bytes) -> str:
    """HMAC-SHA256 of the manifest (which carries every file checksum)."""
    return hmac.new(key, manifest_bytes, hashlib.sha256).hexdigest()


def verify_signature(
    root: pathlib.Path, signing_key: str | bytes | None
) -> bool:
    """Checks the manifest signature.

    Args:
        root: Bundle directory.
        signing_key: HMAC key (defaults to the environment).

    Returns:
        True when a signature exists and verifies; False when the bundle
        is unsigned or no key is available.

    Raises:
        ArtifactError: If a signature exists but does not verify.
    """
    signature_path = root / SIGNATURE_FILE
    key = resolve_key(signing_key)
    if not signature_path.exists() or key is None:
        return False
    try:
        record = json.loads(signature_path.read_text(encoding="utf-8"))
        algorithm = record.get("algorithm")
        recorded = record.get("signature", "")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        AttributeError,
    ) as error:
        raise errors.ArtifactError(
            "invalid bundle signature metadata"
        ) from error
    if algorithm != SIGNATURE_ALGORITHM:
        raise errors.ArtifactError(
            f"unsupported bundle signature algorithm {algorithm!r}"
        )
    expected = sign_manifest((root / MANIFEST_FILE).read_bytes(), key)
    if not hmac.compare_digest(recorded, expected):
        raise errors.ArtifactError(
            "bundle signature is invalid: the bundle was modified or "
            "signed with a different key"
        )
    return True


@dataclasses.dataclass(frozen=True, slots=True)
class BundleInfo:
    """Provenance and compatibility metadata of a bundle.

    Attributes:
        model_name: Selected model family.
        model_version: Version label of this bundle.
        metric: Optimized metric.
        format: Bundle format version.
        scmle_version: Framework version that built it.
        python: Python version that built it.
        sklearn: scikit-learn version that built it.
        pandas: pandas version that built it.
        files: SHA-256 per stored file.
        model_params: Hyperparameter overrides of the selected model.
        strategy: Data strategy of the selected candidate.
        decision_threshold: Optimized binary decision threshold, if any.
        libraries: Versions of the other libraries the pipeline depends on
            (numpy, scipy, joblib and any installed model library).
        calibration: Description of the probability calibration applied at
            inference, empty if none.
        environment: Every distribution installed when the bundle was built,
            by name, so the environment can be rebuilt.
        platform: Operating system and architecture that built it.
    """

    model_name: str
    model_version: str
    metric: str
    format: int = BUNDLE_FORMAT
    scmle_version: str = scmle.__version__
    python: str = dataclasses.field(default_factory=platform.python_version)
    sklearn: str = sklearn.__version__
    pandas: str = pd.__version__
    files: dict[str, str] = dataclasses.field(default_factory=dict)
    model_params: dict[str, Any] = dataclasses.field(default_factory=dict)
    strategy: str = "default"
    decision_threshold: float | None = None
    libraries: dict[str, str] = dataclasses.field(
        default_factory=lambda: installed_versions()
    )
    calibration: str = ""
    environment: dict[str, str] = dataclasses.field(default_factory=dict)
    platform: str = ""

    def environment_lock(self) -> str:
        """The build environment as a pinned requirements listing."""
        header = [
            "# scmle environment lock",
            f"# python {self.python}",
            f"# platform {self.platform or 'unknown'}",
            f"# scmle {self.scmle_version}",
        ]
        pins = [
            f"{name}=={version}" for name, version in self.environment.items()
        ]
        return "\n".join([*header, *pins]) + "\n"

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        return dataclasses.asdict(self)

    def compatibility_problems(self) -> list[str]:
        """Lists dependency mismatches with the running environment."""
        problems = []
        if minor_version(self.sklearn) != minor_version(sklearn.__version__):
            problems.append(
                f"scikit-learn {self.sklearn} (installed {sklearn.__version__})"
            )
        running = platform.python_version()
        if minor_version(self.python) != minor_version(running):
            problems.append(f"python {self.python} (running {running})")
        if minor_version(self.pandas) != minor_version(pd.__version__):
            problems.append(
                f"pandas {self.pandas} (installed {pd.__version__})"
            )
        current = installed_versions()
        for library, built in sorted(self.libraries.items()):
            now = current.get(library)
            if now is not None and minor_version(built) != minor_version(now):
                problems.append(f"{library} {built} (installed {now})")
        return problems


class ModelBundle:
    """A fitted pipeline plus the schema and metadata needed to serve it."""

    def __init__(
        self,
        pipeline: Any,
        schema: artifact_contracts.BundleSchema,
        info: BundleInfo,
        reference: reference_module.ReferenceProfile | None = None,
        history: pd.DataFrame | None = None,
        calibrator: Any = None,
        conformal: conformal_module.ConformalCalibrator | None = None,
        novelty: novelty_module.NoveltyDetector | None = None,
        quantile: quantile_module.QuantileIntervals | None = None,
    ) -> None:
        """Initializes the bundle.

        Args:
            pipeline: Fitted preprocessing + estimator pipeline.
            schema: Input contract.
            info: Provenance metadata.
            reference: Training-time distributions for monitoring.
            history: Recent rows per entity that series features look back
                on, so records scored alone still get lag / rolling values.
            calibrator: Probability calibrator fitted on labeled rows after
                training (see `calibrate`), applied by `predict_proba`.
            conformal: Out-of-fold nonconformity scores that turn
                predictions into intervals or sets with a coverage level.
            novelty: Detector that flags rows unlike the training data.
            quantile: Low and high quantile models with calibration, for
                intervals whose width follows the row (regression).
        """
        self.pipeline = pipeline
        self.schema = schema
        self.info = info
        self.reference = reference
        self.history = history
        self.calibrator = calibrator
        self.conformal = conformal
        self.novelty = novelty
        self.quantile = quantile
        # The fold models behind `predict_interval(method="cv_plus")`; set
        # by the run or by `load`, kept out of the constructor signature.
        self.cv_plus: cv_plus_module.CvPlusEnsemble | None = None
        # Calibration weights for covariate shift, attached on demand by
        # `attach_shift_weights`.
        self.shift_weights: shift_weights_module.StoredShiftWeights | None = (
            None
        )
        # Serializes `remember`, which rewrites `history` from itself.
        self.history_lock = threading.RLock()

    @classmethod
    def create(
        cls,
        pipeline: Any,
        plan: data_contracts.FeaturePlan,
        dev_features: pd.DataFrame,
        target: str,
        task: str,
        metric: str,
        model_name: str,
        model_version: str = "v1",
        model_params: dict[str, Any] | None = None,
        strategy: str = "default",
        decision_threshold: float | None = None,
        reference: reference_module.ReferenceProfile | None = None,
        graph_fingerprint: str = "",
        group_column: str | None = None,
        time_column: str | None = None,
        history_source: pd.DataFrame | None = None,
        class_labels: tuple[Any, ...] | None = None,
    ) -> Self:
        """Builds a bundle from a fitted pipeline and its training data.

        Args:
            pipeline: Fitted pipeline.
            plan: Feature columns used.
            dev_features: Training features (for dtypes and categories).
            target: Target column name.
            task: Task type value.
            metric: Optimized metric name.
            model_name: Selected model family.
            model_version: Bundle version label.
            model_params: Hyperparameter overrides of the selected model.
            strategy: Data strategy of the selected candidate.
            decision_threshold: Optimized binary decision threshold, if any.
            reference: Training-time distributions for monitoring.
            graph_fingerprint: Hash of the feature graph, if one was used.
            group_column: Entity column for series features.
            time_column: Ordering column for series features.
            history_source: All rows known at training time; the recent ones
                per entity are stored as the model's series history.
            class_labels: Labels of the classes in probability-column order
                when the pipeline works on codes (ordinal tasks).

        Returns:
            The bundle.
        """
        computed = plan.derived_names
        series_names = {n.name for n in plan.series}
        numeric = tuple(c for c in plan.numeric if c not in computed)
        categorical = tuple(c for c in plan.categorical if c not in computed)
        extra = tuple(
            c
            for c in plan.extra_inputs
            if c not in computed and c not in numeric + categorical
        )
        series_columns = tuple(
            c for c in (*plan.features, *plan.extra_inputs) if c in series_names
        )
        known = {
            column: sorted(map(str, dev_features[column].dropna().unique()))[
                :MAX_KNOWN_CATEGORIES
            ]
            for column in categorical
        }
        classes = getattr(pipeline, "classes_", None)
        schema = artifact_contracts.BundleSchema(
            target=target,
            task=task,
            numeric=numeric,
            categorical=categorical,
            dtypes={
                c: str(dev_features[c].dtype) for c in (*numeric, *categorical)
            },
            known_categories=known,
            classes=class_labels
            if class_labels is not None
            else None
            if classes is None
            else tuple(np.asarray(classes).tolist()),
            extra_inputs=extra,
            series=tuple(n.model_dump(mode="json") for n in plan.series),
            series_columns=series_columns,
            graph_fingerprint=graph_fingerprint,
            group_column=group_column,
            time_column=time_column,
            targets=tuple(getattr(pipeline, "target_columns", ())),
            optional_inputs=(
                (pipeline.column,)
                if isinstance(pipeline, thresholds.GroupThresholdClassifier)
                else ()
            ),
        )
        info = BundleInfo(
            model_name,
            model_version,
            metric,
            model_params=model_params or {},
            strategy=strategy,
            decision_threshold=decision_threshold,
            environment=installed_distributions(),
            platform=platform.platform(),
        )
        bundle = cls(pipeline, schema, info, reference)
        if schema.series and history_source is not None:
            bundle.history = bundle.series_builder().build_history(
                history_source
            )
        return bundle

    def verify_round_trip(
        self,
        directory: str | pathlib.Path,
        sample: pd.DataFrame,
        tolerance: float = PARITY_TOLERANCE,
    ) -> dict[str, Any]:
        """Reloads the saved bundle and checks it predicts like this one.

        Args:
            directory: Where the bundle was saved.
            sample: Raw rows to score with both copies.
            tolerance: Largest allowed absolute probability difference;
                labels and regression values must match to the same
                tolerance.

        Returns:
            The verification record (rows, tolerance, largest difference).

        Raises:
            ArtifactError: If the reloaded bundle disagrees with this one.
        """
        reloaded = ModelBundle.load(directory, trusted=True)
        before = np.asarray(self.predict(sample))
        after = np.asarray(reloaded.predict(sample))
        if before.dtype.kind in "fiu":
            worst = float(np.max(np.abs(before - after), initial=0.0))
        else:
            worst = float(np.sum(before != after))
        if self.schema.classes is not None:
            worst = max(
                worst,
                float(
                    np.max(
                        np.abs(
                            self.predict_proba(sample)
                            - reloaded.predict_proba(sample)
                        ),
                        initial=0.0,
                    )
                ),
            )
        if worst > tolerance:
            raise errors.ArtifactError(
                "the saved bundle does not reproduce this model's "
                f"predictions (largest difference {worst:.3g} > {tolerance:g})"
            )
        return {
            "rows": len(sample),
            "tolerance": tolerance,
            "largest_difference": worst,
        }

    def score(
        self,
        data: str | pathlib.Path | pd.DataFrame,
        mapping: mapping_module.ColumnMapping | None = None,
    ) -> dict[str, Any]:
        """Predicts and reports the schema diagnostics in one structured result.

        Args:
            data: DataFrame or path to a supported tabular dataset.
            mapping: Explicit column-name policy; exact names if omitted.

        Returns:
            Predictions, class probabilities for classifiers, the model
            version, whether the input satisfied the schema, and diagnostics
            (ignored columns, missing-value counts, unseen categories).

        Raises:
            SchemaError: If required columns are missing or malformed.
        """
        frame = self.read_frame(data)
        if mapping is not None:
            frame = mapping.apply(frame, [*self.features, self.schema.target])
        report = self.validate(frame)
        predictions = self.predict(frame)
        result: dict[str, Any] = {
            "predictions": np.asarray(predictions).tolist(),
            "model_version": self.info.model_version,
            "schema_valid": report.valid,
            "diagnostics": {
                "ignored_columns": list(report.extra),
                "missing_values": report.missing_value_counts,
                "unseen_categories": report.unseen_categories,
            },
        }
        if self.schema.classes is not None:
            result["classes"] = list(self.schema.classes)
            result["probabilities"] = self.predict_proba(frame).tolist()
        if self.novelty is not None:
            result["out_of_distribution"] = self.novelty_of(frame)[
                "out_of_distribution"
            ].tolist()
        return result

    def novelty_of(
        self,
        data: str | pathlib.Path | pd.DataFrame,
        mapping: mapping_module.ColumnMapping | None = None,
    ) -> pd.DataFrame:
        """Flags rows that look unlike the training data.

        Args:
            data: DataFrame or path to a supported tabular dataset.
            mapping: Explicit column-name policy; exact names if omitted.

        Returns:
            `novelty_score` and `out_of_distribution` per row.

        Raises:
            ArtifactError: If the bundle has no novelty detector.
        """
        if self.novelty is None:
            raise errors.ArtifactError("this bundle has no novelty detector")
        frame = self.prepare_frame(data, mapping)
        return self.novelty.score(frame)

    def calibrate(
        self,
        data: str | pathlib.Path | pd.DataFrame,
        target: Any,
        method: str = "sigmoid",
    ) -> dict[str, float]:
        """Fits an optional probability calibrator on labeled rows.

        The trained pipeline is frozen; only a monotone mapping from its
        probabilities to calibrated ones is learned. Use rows the model was
        not trained on. `predict_proba` then returns calibrated
        probabilities; `predict` is unchanged.

        Args:
            data: Labeled rows.
            target: Their true labels.
            method: `sigmoid` (Platt), `isotonic`, or `dirichlet` (a full
                map over all class probabilities; multiclass only).

        Returns:
            Brier score and top-label expected calibration error before and
            after, on the calibration rows (a diagnostic; the calibrator is
            fitted on the same rows).

        Raises:
            ArtifactError: For regression models.
            ConfigError: For an unknown method.
        """
        if (
            self.schema.classes is None
            or self.schema.task == tasks.TaskType.ORDINAL.value
        ):
            raise errors.ArtifactError("only classifiers can be calibrated")
        if method not in ("sigmoid", "isotonic", "dirichlet"):
            raise errors.ConfigError(
                f"unknown calibration method '{method}'; use sigmoid, "
                "isotonic or dirichlet"
            )
        if method == "dirichlet" and len(self.schema.classes) < 3:
            raise errors.ConfigError(
                "dirichlet calibration needs three or more classes"
            )
        frame = self.prepare_frame(data)
        labels = np.asarray(target)
        before = self.pipeline.predict_proba(frame)
        if method == "dirichlet":
            calibrator: Any = dirichlet.DirichletCalibrator(self.pipeline).fit(
                frame, labels
            )
        else:
            calibrator = calibration.CalibratedClassifierCV(
                frozen.FrozenEstimator(self.pipeline), method=method
            ).fit(frame, labels)
        after = calibrator.predict_proba(frame)
        self.calibrator = calibrator
        self.info = dataclasses.replace(
            self.info, calibration=f"{method} on {len(frame)} rows"
        )
        classes = self.schema.classes
        return {
            "brier_before": brier(before, labels, classes),
            "brier_after": brier(after, labels, classes),
            "ece_before": reliability.top_label_curve(before, labels, classes)[
                0
            ],
            "ece_after": reliability.top_label_curve(after, labels, classes)[0],
        }

    def series_builder(self) -> feature_graph.SeriesFeatureBuilder:
        """The builder for this model's series features, with its history."""
        return feature_graph.SeriesFeatureBuilder(
            [tasks.FeatureNode(**node) for node in self.schema.series],
            self.schema.group_column,
            self.schema.time_column,
            self.history,
        )

    def remember(self, data: str | pathlib.Path | pd.DataFrame) -> None:
        """Adds scored rows to the history later records will look back on.

        Call this after scoring when the rows are real, ordered observations
        (as the serving adapters do). Rows with the same entity and time as
        stored ones replace them. Nothing happens for models without series
        features.

        Args:
            data: The rows that were scored.
        """
        if not self.schema.series:
            return
        with self.history_lock:  # Read-modify-write of `history`.
            builder = self.series_builder()
            frame = self.read_frame(data)
            columns = builder.history_columns
            parts = [frame[columns]]
            if self.history is not None:
                parts.insert(0, self.history)
            combined = pd.concat(parts, ignore_index=True)
            key = [
                c
                for c in (self.schema.group_column, self.schema.time_column)
                if c
            ]
            combined = combined.drop_duplicates(subset=key, keep="last")
            self.history = builder.build_history(combined)

    def save(
        self,
        directory: str | pathlib.Path,
        signing_key: str | bytes | None = None,
    ) -> pathlib.Path:
        """Writes the bundle, signing it when a key is available.

        Args:
            directory: Destination directory (created if needed).
            signing_key: HMAC key; defaults to `$SCMLE_SIGNING_KEY`. An
                unsigned bundle can only be loaded with `trusted=True`.

        Returns:
            The directory path.
        """
        root = pathlib.Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        model_path = root / MODEL_FILE
        joblib.dump(self.pipeline, model_path)
        (root / SCHEMA_FILE).write_text(
            json.dumps(self.schema.to_dict(), indent=2),
            encoding="utf-8",
            newline="\n",
        )
        schema_path = root / SCHEMA_FILE
        files = {
            MODEL_FILE: file_sha256(model_path),
            SCHEMA_FILE: file_sha256(schema_path),
        }
        if self.reference is not None:
            reference_path = root / REFERENCE_FILE
            reference_path.write_text(
                json.dumps(self.reference.to_dict()),
                encoding="utf-8",
                newline="\n",
            )
            files[REFERENCE_FILE] = file_sha256(reference_path)
        if self.calibrator is not None:
            calibrator_path = root / CALIBRATOR_FILE
            joblib.dump(self.calibrator, calibrator_path)
            files[CALIBRATOR_FILE] = file_sha256(calibrator_path)
        elif (root / CALIBRATOR_FILE).exists():
            (root / CALIBRATOR_FILE).unlink()
        if self.conformal is not None:
            conformal_path = root / CONFORMAL_FILE
            conformal_path.write_text(
                json.dumps(self.conformal.to_dict()),
                encoding="utf-8",
                newline="\n",
            )
            files[CONFORMAL_FILE] = file_sha256(conformal_path)
        if self.novelty is not None:
            novelty_path = root / NOVELTY_FILE
            joblib.dump(self.novelty, novelty_path)
            files[NOVELTY_FILE] = file_sha256(novelty_path)
        if self.quantile is not None:
            quantile_path = root / QUANTILE_FILE
            joblib.dump(self.quantile, quantile_path)
            files[QUANTILE_FILE] = file_sha256(quantile_path)
        if self.cv_plus is not None:
            cv_plus_path = root / CV_PLUS_FILE
            joblib.dump(self.cv_plus, cv_plus_path)
            files[CV_PLUS_FILE] = file_sha256(cv_plus_path)
        if self.coverage_report is not None:
            coverage_path = root / COVERAGE_FILE
            coverage_path.write_text(
                json.dumps(self.coverage_report, sort_keys=True),
                encoding="utf-8",
                newline="\n",
            )
            files[COVERAGE_FILE] = file_sha256(coverage_path)
        if self.shift_weights is not None:
            shift_path = root / SHIFT_WEIGHTS_FILE
            joblib.dump(self.shift_weights, shift_path)
            files[SHIFT_WEIGHTS_FILE] = file_sha256(shift_path)
        if self.history is not None:
            history_path = root / HISTORY_FILE
            self.history.to_parquet(history_path, index=False)
            files[HISTORY_FILE] = file_sha256(history_path)
        elif (root / HISTORY_FILE).exists():
            (root / HISTORY_FILE).unlink()
        info = dataclasses.replace(self.info, format=BUNDLE_FORMAT, files=files)
        manifest_text = json.dumps(info.to_dict(), indent=2)
        (root / MANIFEST_FILE).write_text(
            manifest_text, encoding="utf-8", newline="\n"
        )
        signature_path = root / SIGNATURE_FILE
        key = resolve_key(signing_key)
        if key is not None:
            signature = sign_manifest(manifest_text.encode(), key)
            signature_path.write_text(
                json.dumps(
                    {"algorithm": SIGNATURE_ALGORITHM, "signature": signature}
                ),
                encoding="utf-8",
                newline="\n",
            )
        elif signature_path.exists():
            signature_path.unlink()
        return root

    @classmethod
    def rotate_signing_key(
        cls,
        directory: str | pathlib.Path,
        current_key: str | bytes,
        new_key: str | bytes,
    ) -> pathlib.Path:
        """Re-sign an existing bundle without reserializing its model.

        The current signature must verify before the new signature is written.
        Rotation changes only ``signature.json`` and uses an atomic replace, so
        a failed rotation leaves the previously valid signature in place.
        """
        root = pathlib.Path(directory)
        old = resolve_key(current_key)
        replacement = resolve_key(new_key)
        if old is None or replacement is None:
            raise errors.ArtifactError(
                "signing-key rotation requires non-empty current and new keys"
            )
        manifest_path = root / MANIFEST_FILE
        try:
            manifest_text = manifest_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise errors.ArtifactError(
                "cannot rotate an incomplete model bundle"
            ) from error
        if not verify_signature(root, old):
            raise errors.ArtifactError(
                "cannot rotate an unauthenticated model bundle"
            )
        signature_path = root / SIGNATURE_FILE
        fd, temporary_name = tempfile.mkstemp(
            dir=root, prefix=f".{SIGNATURE_FILE}.", text=True
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    {
                        "algorithm": SIGNATURE_ALGORITHM,
                        "signature": sign_manifest(
                            manifest_text.encode(), replacement
                        ),
                    },
                    handle,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, signature_path)
            durability.sync_directory(signature_path.parent)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary_name)
            raise
        return root

    @classmethod
    def load(
        cls,
        directory: str | pathlib.Path,
        trusted: bool = False,
        signing_key: str | bytes | None = None,
        allow_version_mismatch: bool = False,
        allow_legacy_format: bool = False,
    ) -> Self:
        """Loads and verifies a bundle.

        Bundles are pickle-based, and unpickling executes code, so loading is
        refused unless the bundle is authentic: either its HMAC signature
        verifies against `signing_key` / `$SCMLE_SIGNING_KEY`, or the caller
        vouches for its source with `trusted=True`. Checksums and signatures
        are verified *before* anything is unpickled. An invalid signature is
        always an error, even with `trusted=True`.

        Args:
            directory: Bundle directory.
            trusted: Load an unsigned bundle you vouch for.
            signing_key: HMAC key used to verify the signature.
            allow_version_mismatch: Load despite differing dependency
                versions.
            allow_legacy_format: Load a trusted format 1 bundle so it can be
                re-saved as the current authenticated format. This is refused
                unless `trusted=True`.

        Returns:
            The verified bundle.

        Raises:
            ArtifactError: If the bundle is unauthenticated, tampered with,
                incomplete, or incompatible.
        """
        root = pathlib.Path(directory)
        try:
            manifest = json.loads(
                (root / MANIFEST_FILE).read_text(encoding="utf-8")
            )
            schema = artifact_contracts.BundleSchema.from_dict(
                json.loads((root / SCHEMA_FILE).read_text(encoding="utf-8"))
            )
        except FileNotFoundError as error:
            raise errors.ArtifactError(
                f"not a model bundle (missing {error.filename})"
            ) from error
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise errors.ArtifactError(
                "invalid model bundle metadata"
            ) from error
        legacy = manifest.get("format") == 1
        if manifest.get("format") != BUNDLE_FORMAT and not (
            legacy and allow_legacy_format and trusted
        ):
            raise errors.ArtifactError(
                f"unsupported bundle format {manifest.get('format')}"
            )
        info = BundleInfo(**manifest)
        if SCHEMA_FILE not in info.files and not legacy:
            raise errors.ArtifactError(
                "bundle is missing a schema checksum; re-export it with "
                f"bundle format {BUNDLE_FORMAT}"
            )
        authentic = verify_signature(root, signing_key)
        if not authentic and not trusted:
            raise errors.ArtifactError(
                "refusing to load an unauthenticated bundle: it is pickle-"
                "based and loading executes code. Sign bundles with "
                f"${SIGNING_KEY_ENV}, or pass trusted=True if you trust its "
                "source."
            )
        model_path = root / MODEL_FILE
        if not model_path.exists():
            raise errors.ArtifactError(f"missing {MODEL_FILE}")
        if file_sha256(model_path) != info.files.get(MODEL_FILE):
            raise errors.ArtifactError(
                "checksum mismatch: model file was modified or corrupted"
            )
        schema_path = root / SCHEMA_FILE
        if not legacy and file_sha256(schema_path) != info.files.get(
            SCHEMA_FILE
        ):
            raise errors.ArtifactError(
                "checksum mismatch: schema file was modified or corrupted"
            )
        problems = info.compatibility_problems()
        if problems and not allow_version_mismatch:
            raise errors.ArtifactError(
                "bundle was built with different dependencies: "
                + "; ".join(problems),
                {
                    "mismatches": problems,
                    "environment_lock": info.environment_lock(),
                },
            )
        reference = None
        reference_path = root / REFERENCE_FILE
        if REFERENCE_FILE in info.files:
            if (
                not reference_path.exists()
                or file_sha256(reference_path) != info.files[REFERENCE_FILE]
            ):
                raise errors.ArtifactError(
                    "checksum mismatch: reference profile was modified"
                )
            reference = reference_module.ReferenceProfile.from_dict(
                json.loads(reference_path.read_text(encoding="utf-8"))
            )
        history = None
        if HISTORY_FILE in info.files:
            history_path = root / HISTORY_FILE
            if (
                not history_path.exists()
                or file_sha256(history_path) != info.files[HISTORY_FILE]
            ):
                raise errors.ArtifactError(
                    "checksum mismatch: series history was modified"
                )
            history = pd.read_parquet(history_path)
        calibrator = None
        if CALIBRATOR_FILE in info.files:
            calibrator_path = root / CALIBRATOR_FILE
            if (
                not calibrator_path.exists()
                or file_sha256(calibrator_path) != info.files[CALIBRATOR_FILE]
            ):
                raise errors.ArtifactError(
                    "checksum mismatch: calibrator was modified"
                )
            calibrator = joblib.load(calibrator_path)
        calibrated = None
        if CONFORMAL_FILE in info.files:
            conformal_path = root / CONFORMAL_FILE
            if (
                not conformal_path.exists()
                or file_sha256(conformal_path) != info.files[CONFORMAL_FILE]
            ):
                raise errors.ArtifactError(
                    "checksum mismatch: conformal scores were modified"
                )
            calibrated = conformal_module.ConformalCalibrator.from_dict(
                json.loads(conformal_path.read_text(encoding="utf-8"))
            )
        detector = None
        if NOVELTY_FILE in info.files:
            novelty_path = root / NOVELTY_FILE
            if (
                not novelty_path.exists()
                or file_sha256(novelty_path) != info.files[NOVELTY_FILE]
            ):
                raise errors.ArtifactError(
                    "checksum mismatch: novelty detector was modified"
                )
            detector = joblib.load(novelty_path)
        intervals = None
        if QUANTILE_FILE in info.files:
            quantile_path = root / QUANTILE_FILE
            if (
                not quantile_path.exists()
                or file_sha256(quantile_path) != info.files[QUANTILE_FILE]
            ):
                raise errors.ArtifactError(
                    "checksum mismatch: quantile models were modified"
                )
            intervals = joblib.load(quantile_path)
        cv_plus_models = None
        if CV_PLUS_FILE in info.files:
            cv_plus_path = root / CV_PLUS_FILE
            if (
                not cv_plus_path.exists()
                or file_sha256(cv_plus_path) != info.files[CV_PLUS_FILE]
            ):
                raise errors.ArtifactError(
                    "checksum mismatch: CV+ fold models were modified"
                )
            cv_plus_models = joblib.load(cv_plus_path)
        stored_coverage = None
        if COVERAGE_FILE in info.files:
            coverage_path = root / COVERAGE_FILE
            if (
                not coverage_path.exists()
                or file_sha256(coverage_path) != info.files[COVERAGE_FILE]
            ):
                raise errors.ArtifactError(
                    "checksum mismatch: coverage report was modified"
                )
            stored_coverage = json.loads(
                coverage_path.read_text(encoding="utf-8")
            )
        stored_weights = None
        if SHIFT_WEIGHTS_FILE in info.files:
            shift_path = root / SHIFT_WEIGHTS_FILE
            if (
                not shift_path.exists()
                or file_sha256(shift_path) != info.files[SHIFT_WEIGHTS_FILE]
            ):
                raise errors.ArtifactError(
                    "checksum mismatch: shift weights were modified"
                )
            stored_weights = joblib.load(shift_path)
        loaded = cls(
            joblib.load(model_path),
            schema,
            info,
            reference,
            history,
            calibrator,
            calibrated,
            detector,
            intervals,
        )
        loaded.cv_plus = cv_plus_models
        loaded.shift_weights = stored_weights
        loaded.coverage_report = stored_coverage
        return loaded

    @property
    def features(self) -> list[str]:
        """Feature columns expected at inference."""
        return self.schema.features

    @property
    def classes(self) -> tuple[Any, ...] | None:
        """Class labels, or None for regression."""
        return self.schema.classes

    def read_frame(
        self, data: str | pathlib.Path | pd.DataFrame
    ) -> pd.DataFrame:
        """Loads inference input from a path or copies a DataFrame."""
        if isinstance(data, (str, pathlib.Path)):
            return tabular.read_table(data)
        return interchange.to_pandas(data)

    def validate(
        self,
        data: str | pathlib.Path | pd.DataFrame,
        mapping: mapping_module.ColumnMapping | None = None,
    ) -> artifact_contracts.SchemaReport:
        """Checks input against the schema without predicting.

        Args:
            data: DataFrame or path to a supported tabular dataset.
            mapping: Explicit column-name policy; exact names if omitted.

        Returns:
            Missing/extra columns, missing-value counts and unseen categories.
        """
        frame = self.read_frame(data)
        if mapping is not None:
            frame = mapping.apply(frame, [*self.features, self.schema.target])
        return validation.InputValidator(self.schema).report(frame)

    def prepare_frame(
        self,
        data: str | pathlib.Path | pd.DataFrame,
        mapping: mapping_module.ColumnMapping | None = None,
    ) -> pd.DataFrame:
        """Validates input and returns the feature columns.

        Args:
            data: DataFrame or path to a supported tabular dataset.
            mapping: Explicit column-name policy; exact names if omitted.

        Raises:
            SchemaError: On missing columns or non-numeric values in numeric
                columns.
        """
        frame = self.read_frame(data)
        if mapping is not None:
            frame = mapping.apply(frame, [*self.features, self.schema.target])
        validation.InputValidator(self.schema).check(frame)
        if self.schema.series:
            frame = self.series_builder().apply(frame)
        wanted = [
            *self.schema.pipeline_columns,
            *(
                c
                for c in self.schema.optional_inputs
                if c in frame.columns and c not in self.schema.pipeline_columns
            ),
        ]
        return frame[wanted]

    def predict(
        self,
        data: str | pathlib.Path | pd.DataFrame,
        mapping: mapping_module.ColumnMapping | None = None,
    ) -> np.ndarray:
        """Predicts labels or values through the full fitted pipeline.

        Args:
            data: DataFrame or path to a supported tabular dataset.
            mapping: Explicit column-name policy; exact names if omitted.
        """
        predictions = self.pipeline.predict(self.prepare_frame(data, mapping))
        if self.schema.targets:
            return np.asarray(predictions)
        if self.schema.task == tasks.TaskType.ORDINAL.value:
            labels = np.asarray(self.schema.classes, dtype=object)
            return labels[np.asarray(predictions).astype(int)]
        return predictions

    def predict_frame(
        self,
        data: str | pathlib.Path | pd.DataFrame,
        mapping: mapping_module.ColumnMapping | None = None,
    ) -> pd.DataFrame:
        """Predictions of a multi-target model, one named column per target.

        Raises:
            ArtifactError: For single-target models.
        """
        if not self.schema.targets:
            raise errors.ArtifactError("this bundle has a single target")
        return pd.DataFrame(
            self.predict(data, mapping), columns=list(self.schema.targets)
        )

    def predict_proba(
        self,
        data: str | pathlib.Path | pd.DataFrame,
        mapping: mapping_module.ColumnMapping | None = None,
    ) -> np.ndarray:
        """Predicts class probabilities.

        Args:
            data: DataFrame or path to a supported tabular dataset.
            mapping: Explicit column-name policy; exact names if omitted.

        Raises:
            ArtifactError: For regression models or estimators without
                probabilities.
        """
        if self.schema.task in (
            "regression",
            tasks.TaskType.MULTIOUTPUT.value,
        ) or not hasattr(self.pipeline, "predict_proba"):
            raise errors.ArtifactError(
                "this model does not provide probabilities"
            )
        frame = self.prepare_frame(data, mapping)
        if self.calibrator is not None:
            return self.calibrator.predict_proba(frame)
        return self.pipeline.predict_proba(frame)

    def explain(
        self,
        data: str | pathlib.Path | pd.DataFrame,
        background: str | pathlib.Path | pd.DataFrame,
        mapping: mapping_module.ColumnMapping | None = None,
        class_label: Any = None,
    ) -> pd.DataFrame:
        """Explains individual predictions against a background sample.

        Args:
            data: Rows to explain (DataFrame or path).
            background: Rows describing typical input, for example a sample
                of the training data.
            mapping: Explicit column-name policy; exact names if omitted.
            class_label: Class whose probability is explained; the positive
                class when omitted for binary models.

        Returns:
            A `prediction` column (probability of the class, or the
            regression value) and one contribution column per feature.

        Raises:
            ArtifactError: For a multiclass model without `class_label`.
        """
        rows = self.prepare_frame(data, mapping)
        reference = self.prepare_frame(background, mapping)
        if self.schema.task == "regression":
            score = self.pipeline.predict
        else:
            classes = list(self.classes or ())
            if class_label is None:
                if len(classes) != BINARY_CLASSES:
                    raise errors.ArtifactError(
                        "a multiclass model needs class_label to explain"
                    )
                class_label = classes[1]
            column = classes.index(class_label)
            estimator = self.calibrator or self.pipeline

            def score(frame: pd.DataFrame) -> np.ndarray:
                return np.asarray(estimator.predict_proba(frame))[:, column]

        return local.OcclusionExplainer(score, self.schema.features).explain(
            rows, reference
        )

    def predict_interval(
        self,
        data: str | pathlib.Path | pd.DataFrame,
        coverage: float = conformal_module.DEFAULT_COVERAGE,
        mapping: mapping_module.ColumnMapping | None = None,
        method: str = "residual",
        shift: bool = False,
    ) -> pd.DataFrame:
        """Prediction intervals with a stated coverage (regression).

        Args:
            data: DataFrame or path to a supported tabular dataset.
            coverage: Share of new rows the interval should contain.
            mapping: Explicit column-name policy; exact names if omitted.
            method: `residual` for a margin of the same width everywhere, or
                `quantile` for conformalized quantile regression, whose
                width follows how noisy each row's neighbourhood is, or
                `cv_plus` for the CV+ interval from the stored fold models
                (present when the run had fewer than 1000 calibration rows
                held out once each).
            shift: Use the weighted conformal interval for covariate shift
                with the weights attached by `attach_shift_weights`
                (`residual` method only). The ratio is estimated, so the
                coverage is approximate and is reported that way.

        Returns:
            Columns `prediction`, `lower` and `upper`.

        Raises:
            ArtifactError: For classification models, unknown methods, or
                bundles saved without the needed calibration.
        """
        if shift and method != "residual":
            raise errors.ArtifactError(
                "shift-weighted intervals exist for the 'residual' method only"
            )
        if method == "quantile":
            if self.quantile is None:
                raise errors.ArtifactError(
                    "this bundle has no quantile calibration"
                )
            frame = self.prepare_frame(data, mapping)
            lower, upper = self.quantile.interval(frame, coverage)
            predictions = np.asarray(self.pipeline.predict(frame), dtype=float)
            return pd.DataFrame(
                {
                    "prediction": predictions,
                    "lower": np.minimum(lower, predictions),
                    "upper": np.maximum(upper, predictions),
                }
            )
        if method == "cv_plus":
            if self.cv_plus is None:
                raise errors.ArtifactError(
                    "this bundle has no CV+ fold models; the run stores them "
                    "for regression with fewer than "
                    f"{cv_plus_module.MIN_ROWS_FOR_SPLIT} calibration rows "
                    "held out once each (see the cv_plus.skipped event)"
                )
            frame = self.prepare_frame(data, mapping)
            lower, upper = self.cv_plus.interval(frame, coverage)
            predictions = np.asarray(self.pipeline.predict(frame), dtype=float)
            return pd.DataFrame(
                {"prediction": predictions, "lower": lower, "upper": upper}
            )
        if method != "residual":
            raise errors.ArtifactError(
                f"unknown interval method '{method}'; use 'residual', "
                "'quantile' or 'cv_plus'"
            )
        calibrator = self.require_conformal("regression")
        if shift:
            return self.shifted_interval(calibrator, data, coverage, mapping)
        predictions = np.asarray(self.predict(data, mapping), dtype=float)
        lower, upper = calibrator.interval(predictions, coverage)
        return pd.DataFrame(
            {"prediction": predictions, "lower": lower, "upper": upper}
        )

    def attach_shift_weights(
        self,
        calibration_rows: pd.DataFrame,
        target_rows: pd.DataFrame,
        columns: Sequence[str] | None = None,
        clip: float = shift_weights_module.DEFAULT_CLIP,
    ) -> shift_weights_module.StoredShiftWeights:
        """Estimates and stores likelihood ratios for weighted conformal.

        Args:
            calibration_rows: The rows the stored conformal scores came from,
                in score order (development rows in the order of the
                held-out positions of the split's folds).
            target_rows: Unlabeled rows of the population to predict for.
            columns: Numeric feature columns for the discriminator; all
                numeric schema features if omitted.
            clip: Upper bound on the ratio.

        Returns:
            The stored weights (also kept on the bundle and saved with it).

        Raises:
            ArtifactError: Without a regression conformal calibration, on a
                row count that differs from the stored scores, or on
                columns that are not numeric features of the schema.
            WeightRefusedError: When the effective sample size is below
                `MIN_EFFECTIVE_SIZE`; no weights are stored then.
        """
        calibrator = self.require_conformal("regression")
        if len(calibration_rows) != len(calibrator.scores):
            raise errors.ArtifactError(
                f"{len(calibration_rows)} calibration rows given but "
                f"{len(calibrator.scores)} conformal scores are stored"
            )
        chosen = tuple(
            columns
            if columns is not None
            else (
                c
                for c in self.features
                if pd.api.types.is_numeric_dtype(calibration_rows[c])
            )
        )
        if not chosen or any(
            c not in self.features
            or not pd.api.types.is_numeric_dtype(calibration_rows[c])
            for c in chosen
        ):
            raise errors.ArtifactError(
                "shift weights need numeric feature columns of the schema"
            )
        found = shift_weights_module.estimate_weights(
            calibration_rows[list(chosen)].to_numpy(float),
            target_rows[list(chosen)].to_numpy(float),
            clip,
        )
        self.shift_weights = shift_weights_module.StoredShiftWeights(
            found, chosen
        )
        return self.shift_weights

    def shifted_interval(
        self,
        calibrator: conformal_module.ConformalCalibrator,
        data: str | pathlib.Path | pd.DataFrame,
        coverage: float,
        mapping: mapping_module.ColumnMapping | None,
    ) -> pd.DataFrame:
        """Weighted conformal intervals with the stored shift weights."""
        if self.shift_weights is None:
            raise errors.ArtifactError(
                "this bundle has no shift weights; call attach_shift_weights"
            )
        frame = self.prepare_frame(data, mapping)
        predictions = np.asarray(self.pipeline.predict(frame), dtype=float)
        lower, upper = calibrator.weighted_interval(
            predictions,
            self.shift_weights.weights.calibration,
            self.shift_weights.test_weights(frame),
            coverage,
        )
        return pd.DataFrame(
            {"prediction": predictions, "lower": lower, "upper": upper}
        )

    def counterfactual(
        self,
        row: pd.DataFrame | dict[str, Any],
        desired: Any,
        actionable: list[str],
        background: str | pathlib.Path | pd.DataFrame,
        max_changes: int = 3,
    ) -> counterfactual_module.Counterfactual:
        """The smallest change to `row` that gives the wanted prediction.

        Only the columns in `actionable` are ever changed, and only to values
        seen in `background`.

        Args:
            row: One row (a dict or a one-row frame).
            desired: The wanted class label for classifiers, or a
                `(low, high)` range for regression.
            actionable: Columns the search may change.
            background: Typical rows, the source of candidate values.
            max_changes: Most columns it may change.

        Returns:
            The search result; `found` says whether the outcome was reached.

        Raises:
            ArtifactError: For unknown or non-feature actionable columns, an
                outcome this model cannot produce, or a model without
                probabilities when classifying.
        """
        features = [*self.schema.numeric, *self.schema.categorical]
        unknown = [c for c in actionable if c not in features]
        if unknown:
            raise errors.ArtifactError(
                f"not model features, so they cannot be changed: {unknown}"
            )
        frame = pd.DataFrame([row] if isinstance(row, dict) else row)
        if len(frame) != 1:
            raise errors.ArtifactError("give exactly one row")
        reference = self.read_frame(background)
        fixed = {c: frame.iloc[0][c] for c in frame.columns}
        if self.schema.classes is not None:
            if desired not in self.schema.classes:
                raise errors.ArtifactError(
                    f"{desired!r} is not one of {list(self.schema.classes)}"
                )
            column = list(self.schema.classes).index(desired)

            def progress(rows: pd.DataFrame) -> np.ndarray:
                return np.asarray(self.predict_proba(rows))[:, column]

            def reached(rows: pd.DataFrame) -> np.ndarray:
                return np.asarray(self.predict(rows)) == desired

        else:
            low, high = desired

            def progress(rows: pd.DataFrame) -> np.ndarray:
                values = np.asarray(self.predict(rows), dtype=float)
                return -(
                    np.maximum(low - values, 0) + np.maximum(values - high, 0)
                )

            def reached(rows: pd.DataFrame) -> np.ndarray:
                values = np.asarray(self.predict(rows), dtype=float)
                return (values >= low) & (values <= high)

        search = counterfactual_module.CounterfactualSearch(
            progress,
            reached,
            self.predict,
            reference,
            actionable,
            max_changes,
        )
        return search.search(fixed)

    def environment_lock(self, path: str | pathlib.Path | None = None) -> str:
        """The environment this model was built in, as pinned requirements.

        Args:
            path: Where to write it; omit to only return the text.

        Returns:
            Python version, platform and every installed distribution with
            its exact version, ready for `pip install -r`.
        """
        text = self.info.environment_lock()
        if path is not None:
            pathlib.Path(path).write_text(text, encoding="utf-8", newline="\n")
        return text

    def export_service(self, destination: str | pathlib.Path) -> pathlib.Path:
        """Writes a deployable service folder (works for every model).

        The folder holds this bundle, pinned requirements, the full build
        environment, a server script and a Dockerfile.

        Args:
            destination: Folder to create.

        Returns:
            The folder.
        """
        return packaging.write_service(self, destination)

    def export_onnx(
        self,
        path: str | pathlib.Path,
        check: pd.DataFrame | None = None,
        tolerance: float = onnx_export.DEFAULT_TOLERANCE,
    ) -> pathlib.Path:
        """Writes the pipeline as an ONNX graph (`onnx` extra).

        Args:
            path: Destination file.
            check: Rows to score with both the graph and the pipeline; the
                export fails if they disagree beyond `tolerance`.
            tolerance: Largest allowed probability / scaled value difference.

        Returns:
            The path written.

        Raises:
            ArtifactError: For models that cannot be exported faithfully, when
                a probability calibrator is attached, or on a failed check.
        """
        written = onnx_export.export(self, path)
        if check is not None:
            onnx_export.verify(self, written, check, tolerance)
        return written

    def predict_survival(
        self,
        data: str | pathlib.Path | pd.DataFrame,
        times: Any,
        mapping: mapping_module.ColumnMapping | None = None,
    ) -> np.ndarray:
        """Probability of no event yet at each time (survival tasks).

        Args:
            data: DataFrame or path to a supported tabular dataset.
            times: Times (in the duration column's unit) to read the curve at.
            mapping: Explicit column-name policy; exact names if omitted.

        Returns:
            Array of shape `(rows, len(times))`; each row never increases.

        Raises:
            ArtifactError: For models that are not survival models.
        """
        if self.schema.task != tasks.TaskType.SURVIVAL.value:
            raise errors.ArtifactError("this bundle is not a survival model")
        frame = self.prepare_frame(data, mapping)
        return np.asarray(
            self.pipeline.predict_survival(
                frame, np.asarray(times, dtype=float)
            )
        )

    def forecast(
        self,
        history: str | pathlib.Path | pd.DataFrame,
        steps: int | None = None,
        mapping: mapping_module.ColumnMapping | None = None,
    ) -> pd.DataFrame:
        """Forecasts the next values of every series in `history`.

        The history supplies each series' recent values (the lag and rolling
        features are computed from it), so the series need not have been
        seen in training.

        Args:
            history: Past rows with the value, time and (if used) entity
                columns.
            steps: How many steps ahead, at most the trained horizon.
            mapping: Explicit column-name policy; exact names if omitted.

        Returns:
            One row per series and step: the entity column (if any), `step`,
            the forecast time and `forecast`.

        Raises:
            ArtifactError: For models that are not forecasters or a request
                beyond the trained horizon.
        """
        if self.schema.task != tasks.TaskType.FORECAST.value:
            raise errors.ArtifactError("this bundle is not a forecaster")
        horizon = len(self.schema.targets)
        wanted = horizon if steps is None else steps
        if not 1 <= wanted <= horizon:
            raise errors.ArtifactError(
                f"this model forecasts 1 to {horizon} steps ahead"
            )
        frame = self.read_frame(history)
        if mapping is not None:
            frame = mapping.apply(frame, self.features)
        clock, entity = self.schema.time_column, self.schema.group_column
        time_name = clock or "time"
        prepared = self.prepare_frame(frame)
        stamps = pd.to_datetime(frame[clock], errors="coerce")
        numeric_clock = stamps.isna().any()
        order = frame[clock].to_numpy() if numeric_clock else stamps.to_numpy()
        positions = np.arange(len(frame))
        by_entity = (
            frame[entity].astype(str).to_numpy()
            if entity
            else np.zeros(len(frame), dtype=int)
        )
        latest = {}
        for position in positions[np.argsort(order, kind="stable")]:
            latest[by_entity[position]] = position
        chosen = np.array(sorted(latest.values()))
        predicted = np.asarray(self.pipeline.predict(prepared.iloc[chosen]))[
            :, :wanted
        ]
        distinct = np.unique(order)
        gaps = np.sort(np.diff(distinct))
        step_size = gaps[len(gaps) // 2] if len(gaps) else None
        rows = []
        for row, position in enumerate(chosen):
            for step in range(wanted):
                stamp = order[position]
                if step_size is not None:
                    stamp = stamp + step_size * (step + 1)
                record: dict[str, Any] = (
                    {}
                    if entity is None
                    else {entity: frame[entity].iloc[position]}
                )
                rows.append(
                    {
                        **record,
                        "step": step + 1,
                        time_name: stamp,
                        "forecast": float(predicted[row, step]),
                    }
                )
        return pd.DataFrame(rows)

    def rank(
        self,
        data: str | pathlib.Path | pd.DataFrame,
        query_column: str,
        mapping: mapping_module.ColumnMapping | None = None,
    ) -> pd.DataFrame:
        """Scores rows and orders them within each query (ranking tasks).

        Args:
            data: Rows to rank (DataFrame or supported tabular path).
            query_column: Column naming each row's query.
            mapping: Explicit column-name policy; exact names if omitted.

        Returns:
            The input rows with `score` and `rank` (1 = best within its
            query), sorted by query, then rank.

        Raises:
            ArtifactError: For models that are not ranking models, or when
                `query_column` is missing from the data.
        """
        if self.schema.task != tasks.TaskType.RANKING.value:
            raise errors.ArtifactError("this bundle is not a ranking model")
        frame = self.read_frame(data)
        if query_column not in frame.columns:
            raise errors.ArtifactError(
                f"query column '{query_column}' is not in the data"
            )
        ranked = frame.assign(score=self.predict(frame, mapping))
        ranked["rank"] = (
            ranked.groupby(query_column, sort=False)["score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        return ranked.sort_values(
            [query_column, "rank"], kind="stable"
        ).reset_index(drop=True)

    def rank_for_labeling(
        self,
        data: str | pathlib.Path | pd.DataFrame,
        count: int = 10,
        mapping: mapping_module.ColumnMapping | None = None,
    ) -> pd.DataFrame:
        """Picks the unlabeled rows a person should label next.

        Classification ranks by the smallest gap between the two most likely
        classes (the model cannot tell them apart). Regression ranks by the
        width of the quantile band (the model is least sure there).

        Args:
            data: Unlabeled rows (DataFrame or supported tabular path).
            count: How many rows to return.
            mapping: Explicit column-name policy; exact names if omitted.

        Returns:
            The `count` most informative rows: `position` in `data` and an
            `uncertainty` score in [0, 1] for classification (a width in
            target units for regression), most uncertain first.

        Raises:
            ArtifactError: For a regression bundle without quantile models.
        """
        frame = self.prepare_frame(data, mapping)
        if self.classes is not None:
            ordered = np.sort(
                np.asarray(self.pipeline.predict_proba(frame)), axis=1
            )
            top = ordered[:, -1]
            runner_up = ordered[:, -2] if ordered.shape[1] > 1 else 0.0
            uncertainty = 1.0 - (top - runner_up)
        else:
            if self.quantile is None:
                raise errors.ArtifactError(
                    "this bundle has no quantile models to rank by"
                )
            lower, upper = self.quantile.raw_band(frame)
            uncertainty = upper - lower
        order = np.argsort(-uncertainty, kind="stable")[:count]
        return pd.DataFrame(
            {"position": order, "uncertainty": uncertainty[order]}
        )

    def predict_set(
        self,
        data: str | pathlib.Path | pd.DataFrame,
        coverage: float = conformal_module.DEFAULT_COVERAGE,
        mapping: mapping_module.ColumnMapping | None = None,
        conditional: str | None = None,
    ) -> list[list[Any]]:
        """Prediction sets with a stated coverage (classification).

        Args:
            data: DataFrame or path to a supported tabular dataset.
            coverage: Share of new rows whose true label the set contains.
            mapping: Explicit column-name policy; exact names if omitted.
            conditional: `None` for a pooled margin (coverage on average
                over classes) or `class` for one margin per true class
                (coverage within each class). A class with too few
                calibration rows is kept in every set.

        Returns:
            For each row the labels that cannot be ruled out.

        Raises:
            ArtifactError: For regression models or bundles saved without
                conformal scores.
        """
        calibrator = self.require_conformal("classification")
        return calibrator.sets(
            self.predict_proba(data, mapping), coverage, conditional
        )

    def require_conformal(
        self, kind: str
    ) -> conformal_module.ConformalCalibrator:
        """Returns the calibrator of the right kind or explains its absence."""
        if self.conformal is None or self.conformal.kind != kind:
            raise errors.ArtifactError(
                f"this bundle has no {kind} conformal calibration; the run "
                "skips it when the time budget is spent or too few held-out "
                "rows exist (see the conformal.skipped event in events.jsonl)"
            )
        return self.conformal
