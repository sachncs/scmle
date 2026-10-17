"""Model bundle: a validated, portable inference pipeline.

SECURITY: bundles are serialized with joblib (pickle). Loading executes code,
so only load bundles from sources you trust. Checksums detect corruption or
tampering after saving, not malice at the source.
"""

import dataclasses
import hashlib
import hmac
import importlib
import json
import os
import pathlib
import platform
from typing import Any
from typing import Self

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn import calibration
from sklearn import frozen

import mml
from mml.contracts import artifacts as artifact_contracts
from mml.contracts import data as data_contracts
from mml.contracts import errors
from mml.contracts import tasks
from mml.data.readers import interchange
from mml.data.readers import tabular
from mml.evaluation import conformal as conformal_module
from mml.evaluation import quantile as quantile_module
from mml.evaluation import reliability
from mml.explainability import local
from mml.features import feature_graph
from mml.inference import mapping as mapping_module
from mml.inference import novelty as novelty_module
from mml.inference import validation
from mml.monitoring import reference as reference_module

BUNDLE_FORMAT = 1
MODEL_FILE = "pipeline.joblib"
SCHEMA_FILE = "schema.json"
REFERENCE_FILE = "reference.json"
CALIBRATOR_FILE = "calibrator.joblib"
CONFORMAL_FILE = "conformal.json"
QUANTILE_FILE = "quantile.joblib"
NOVELTY_FILE = "novelty.joblib"
HISTORY_FILE = "series_history.parquet"
PARITY_TOLERANCE = 1e-9
SIGNATURE_FILE = "signature.json"
SIGNING_KEY_ENV = "MAGICML_SIGNING_KEY"
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
    recorded = json.loads(signature_path.read_text()).get("signature", "")
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
        magicml_version: Framework version that built it.
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
    """

    model_name: str
    model_version: str
    metric: str
    format: int = BUNDLE_FORMAT
    magicml_version: str = mml.__version__
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
            classes=None
            if classes is None
            else tuple(np.asarray(classes).tolist()),
            extra_inputs=extra,
            series=tuple(n.model_dump(mode="json") for n in plan.series),
            series_columns=series_columns,
            graph_fingerprint=graph_fingerprint,
            group_column=group_column,
            time_column=time_column,
        )
        info = BundleInfo(
            model_name,
            model_version,
            metric,
            model_params=model_params or {},
            strategy=strategy,
            decision_threshold=decision_threshold,
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
            data: DataFrame or path to CSV/Parquet.
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
            "predictions": [
                item.item() if hasattr(item, "item") else item
                for item in predictions
            ],
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
            data: DataFrame or path to CSV/Parquet.
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
            method: `sigmoid` (Platt) or `isotonic`.

        Returns:
            Brier score and top-label expected calibration error before and
            after, on the calibration rows (a diagnostic; the calibrator is
            fitted on the same rows).

        Raises:
            ArtifactError: For regression models.
            ConfigError: For an unknown method.
        """
        if self.schema.classes is None:
            raise errors.ArtifactError("only classifiers can be calibrated")
        if method not in ("sigmoid", "isotonic"):
            raise errors.ConfigError(
                f"unknown calibration method '{method}'; use sigmoid or "
                "isotonic"
            )
        frame = self.prepare_frame(data)
        labels = np.asarray(target)
        before = self.pipeline.predict_proba(frame)
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
        builder = self.series_builder()
        frame = self.read_frame(data)
        columns = builder.history_columns
        parts = [frame[columns]]
        if self.history is not None:
            parts.insert(0, self.history)
        combined = pd.concat(parts, ignore_index=True)
        key = [
            c for c in (self.schema.group_column, self.schema.time_column) if c
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
            signing_key: HMAC key; defaults to `$MAGICML_SIGNING_KEY`. An
                unsigned bundle can only be loaded with `trusted=True`.

        Returns:
            The directory path.
        """
        root = pathlib.Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        model_path = root / MODEL_FILE
        joblib.dump(self.pipeline, model_path)
        (root / SCHEMA_FILE).write_text(
            json.dumps(self.schema.to_dict(), indent=2)
        )
        files = {MODEL_FILE: file_sha256(model_path)}
        if self.reference is not None:
            reference_path = root / REFERENCE_FILE
            reference_path.write_text(json.dumps(self.reference.to_dict()))
            files[REFERENCE_FILE] = file_sha256(reference_path)
        if self.calibrator is not None:
            calibrator_path = root / CALIBRATOR_FILE
            joblib.dump(self.calibrator, calibrator_path)
            files[CALIBRATOR_FILE] = file_sha256(calibrator_path)
        elif (root / CALIBRATOR_FILE).exists():
            (root / CALIBRATOR_FILE).unlink()
        if self.conformal is not None:
            conformal_path = root / CONFORMAL_FILE
            conformal_path.write_text(json.dumps(self.conformal.to_dict()))
            files[CONFORMAL_FILE] = file_sha256(conformal_path)
        if self.novelty is not None:
            novelty_path = root / NOVELTY_FILE
            joblib.dump(self.novelty, novelty_path)
            files[NOVELTY_FILE] = file_sha256(novelty_path)
        if self.quantile is not None:
            quantile_path = root / QUANTILE_FILE
            joblib.dump(self.quantile, quantile_path)
            files[QUANTILE_FILE] = file_sha256(quantile_path)
        if self.history is not None:
            history_path = root / HISTORY_FILE
            self.history.to_parquet(history_path, index=False)
            files[HISTORY_FILE] = file_sha256(history_path)
        elif (root / HISTORY_FILE).exists():
            (root / HISTORY_FILE).unlink()
        info = dataclasses.replace(self.info, files=files)
        manifest_text = json.dumps(info.to_dict(), indent=2)
        (root / MANIFEST_FILE).write_text(manifest_text)
        signature_path = root / SIGNATURE_FILE
        key = resolve_key(signing_key)
        if key is not None:
            signature = sign_manifest(manifest_text.encode(), key)
            signature_path.write_text(
                json.dumps(
                    {"algorithm": SIGNATURE_ALGORITHM, "signature": signature}
                )
            )
        elif signature_path.exists():
            signature_path.unlink()
        return root

    @classmethod
    def load(
        cls,
        directory: str | pathlib.Path,
        trusted: bool = False,
        signing_key: str | bytes | None = None,
        allow_version_mismatch: bool = False,
    ) -> Self:
        """Loads and verifies a bundle.

        Bundles are pickle-based, and unpickling executes code, so loading is
        refused unless the bundle is authentic: either its HMAC signature
        verifies against `signing_key` / `$MAGICML_SIGNING_KEY`, or the caller
        vouches for its source with `trusted=True`. Checksums and signatures
        are verified *before* anything is unpickled. An invalid signature is
        always an error, even with `trusted=True`.

        Args:
            directory: Bundle directory.
            trusted: Load an unsigned bundle you vouch for.
            signing_key: HMAC key used to verify the signature.
            allow_version_mismatch: Load despite differing dependency
                versions.

        Returns:
            The verified bundle.

        Raises:
            ArtifactError: If the bundle is unauthenticated, tampered with,
                incomplete, or incompatible.
        """
        root = pathlib.Path(directory)
        try:
            manifest = json.loads((root / MANIFEST_FILE).read_text())
            schema = artifact_contracts.BundleSchema.from_dict(
                json.loads((root / SCHEMA_FILE).read_text())
            )
        except FileNotFoundError as error:
            raise errors.ArtifactError(
                f"not a model bundle (missing {error.filename})"
            ) from error
        if manifest.get("format") != BUNDLE_FORMAT:
            raise errors.ArtifactError(
                f"unsupported bundle format {manifest.get('format')}"
            )
        info = BundleInfo(**manifest)
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
        problems = info.compatibility_problems()
        if problems and not allow_version_mismatch:
            raise errors.ArtifactError(
                "bundle was built with different dependencies: "
                + "; ".join(problems),
                {"mismatches": problems},
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
                json.loads(reference_path.read_text())
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
                json.loads(conformal_path.read_text())
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
        return cls(
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
            data: DataFrame or path to CSV/Parquet.
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
            data: DataFrame or path to CSV/Parquet.
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
        return frame[self.schema.pipeline_columns]

    def predict(
        self,
        data: str | pathlib.Path | pd.DataFrame,
        mapping: mapping_module.ColumnMapping | None = None,
    ) -> np.ndarray:
        """Predicts labels or values through the full fitted pipeline.

        Args:
            data: DataFrame or path to CSV/Parquet.
            mapping: Explicit column-name policy; exact names if omitted.
        """
        return self.pipeline.predict(self.prepare_frame(data, mapping))

    def predict_proba(
        self,
        data: str | pathlib.Path | pd.DataFrame,
        mapping: mapping_module.ColumnMapping | None = None,
    ) -> np.ndarray:
        """Predicts class probabilities.

        Args:
            data: DataFrame or path to CSV/Parquet.
            mapping: Explicit column-name policy; exact names if omitted.

        Raises:
            ArtifactError: For regression models or estimators without
                probabilities.
        """
        if self.schema.task == "regression" or not hasattr(
            self.pipeline, "predict_proba"
        ):
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
    ) -> pd.DataFrame:
        """Prediction intervals with a stated coverage (regression).

        Args:
            data: DataFrame or path to CSV/Parquet.
            coverage: Share of new rows the interval should contain.
            mapping: Explicit column-name policy; exact names if omitted.
            method: `residual` for a margin of the same width everywhere, or
                `quantile` for conformalized quantile regression, whose
                width follows how noisy each row's neighbourhood is.

        Returns:
            Columns `prediction`, `lower` and `upper`.

        Raises:
            ArtifactError: For classification models, unknown methods, or
                bundles saved without the needed calibration.
        """
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
        if method != "residual":
            raise errors.ArtifactError(
                f"unknown interval method '{method}'; use 'residual' or "
                "'quantile'"
            )
        calibrator = self.require_conformal("regression")
        predictions = np.asarray(self.predict(data, mapping), dtype=float)
        lower, upper = calibrator.interval(predictions, coverage)
        return pd.DataFrame(
            {"prediction": predictions, "lower": lower, "upper": upper}
        )

    def predict_survival(
        self,
        data: str | pathlib.Path | pd.DataFrame,
        times: Any,
        mapping: mapping_module.ColumnMapping | None = None,
    ) -> np.ndarray:
        """Probability of no event yet at each time (survival tasks).

        Args:
            data: DataFrame or path to CSV/Parquet.
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
            data: Unlabeled rows (DataFrame or path to CSV/Parquet).
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
    ) -> list[list[Any]]:
        """Prediction sets with a stated coverage (classification).

        Args:
            data: DataFrame or path to CSV/Parquet.
            coverage: Share of new rows whose true label the set contains.
            mapping: Explicit column-name policy; exact names if omitted.

        Returns:
            For each row the labels that cannot be ruled out.

        Raises:
            ArtifactError: For regression models or bundles saved without
                conformal scores.
        """
        calibrator = self.require_conformal("classification")
        return calibrator.sets(self.predict_proba(data, mapping), coverage)

    def require_conformal(
        self, kind: str
    ) -> conformal_module.ConformalCalibrator:
        """Returns the calibrator of the right kind or explains its absence."""
        if self.conformal is None or self.conformal.kind != kind:
            raise errors.ArtifactError(
                f"this bundle has no {kind} conformal calibration"
            )
        return self.conformal
