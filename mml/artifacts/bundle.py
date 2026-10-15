"""Model bundle: a validated, portable inference pipeline.

SECURITY: bundles are serialized with joblib (pickle). Loading executes code,
so only load bundles from sources you trust. Checksums detect corruption or
tampering after saving, not malice at the source.
"""

import dataclasses
import hashlib
import hmac
import json
import os
import pathlib
import platform
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

import mml
from mml.contracts import data as data_contracts
from mml.contracts import errors
from mml.data import readers
from mml.monitoring import reference as reference_module

BUNDLE_FORMAT = 1
MODEL_FILE = "pipeline.joblib"
SCHEMA_FILE = "schema.json"
REFERENCE_FILE = "reference.json"
SIGNATURE_FILE = "signature.json"
SIGNING_KEY_ENV = "MAGICML_SIGNING_KEY"
SIGNATURE_ALGORITHM = "hmac-sha256"
MANIFEST_FILE = "model_manifest.json"
MAX_KNOWN_CATEGORIES = 200
MAX_REPORTED_UNSEEN = 20


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
class BundleSchema:
    """Input contract of a trained model.

    Attributes:
        target: Name of the target column.
        task: Task type value.
        numeric: Numeric feature columns.
        categorical: Categorical feature columns.
        dtypes: Training dtype per feature.
        known_categories: Categories seen in training per categorical column.
        classes: Class labels for classification, else None.
    """

    target: str
    task: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    dtypes: dict[str, str]
    known_categories: dict[str, list[str]]
    classes: tuple[Any, ...] | None

    @property
    def features(self) -> list[str]:
        """All feature columns, numeric first."""
        return list(self.numeric + self.categorical)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        record = dataclasses.asdict(self)
        record["numeric"] = list(self.numeric)
        record["categorical"] = list(self.categorical)
        record["classes"] = None if self.classes is None else list(self.classes)
        return record

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> BundleSchema:
        """Rebuilds a schema from `to_dict` output."""
        classes = record["classes"]
        return cls(
            target=record["target"],
            task=record["task"],
            numeric=tuple(record["numeric"]),
            categorical=tuple(record["categorical"]),
            dtypes=record["dtypes"],
            known_categories=record["known_categories"],
            classes=None if classes is None else tuple(classes),
        )


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
        return problems


@dataclasses.dataclass(frozen=True, slots=True)
class SchemaReport:
    """Result of validating input data against a bundle schema.

    Attributes:
        missing: Required columns absent from the input.
        extra: Input columns the model ignores.
        missing_value_counts: Missing values per present feature.
        unseen_categories: Categories never seen in training.
    """

    missing: tuple[str, ...]
    extra: tuple[str, ...]
    missing_value_counts: dict[str, int]
    unseen_categories: dict[str, list[str]]

    @property
    def valid(self) -> bool:
        """Whether every required column is present."""
        return not self.missing


class ModelBundle:
    """A fitted pipeline plus the schema and metadata needed to serve it."""

    def __init__(
        self,
        pipeline: Any,
        schema: BundleSchema,
        info: BundleInfo,
        reference: reference_module.ReferenceProfile | None = None,
    ) -> None:
        """Initializes the bundle.

        Args:
            pipeline: Fitted preprocessing + estimator pipeline.
            schema: Input contract.
            info: Provenance metadata.
            reference: Training-time distributions for monitoring.
        """
        self.pipeline = pipeline
        self.schema = schema
        self.info = info
        self.reference = reference

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
    ) -> ModelBundle:
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

        Returns:
            The bundle.
        """
        known = {
            column: sorted(map(str, dev_features[column].dropna().unique()))[
                :MAX_KNOWN_CATEGORIES
            ]
            for column in plan.categorical
        }
        classes = getattr(pipeline, "classes_", None)
        schema = BundleSchema(
            target=target,
            task=task,
            numeric=plan.numeric,
            categorical=plan.categorical,
            dtypes={c: str(dev_features[c].dtype) for c in plan.features},
            known_categories=known,
            classes=None
            if classes is None
            else tuple(np.asarray(classes).tolist()),
        )
        info = BundleInfo(
            model_name,
            model_version,
            metric,
            model_params=model_params or {},
            strategy=strategy,
            decision_threshold=decision_threshold,
        )
        return cls(pipeline, schema, info, reference)

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
    ) -> ModelBundle:
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
            schema = BundleSchema.from_dict(
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
        return cls(joblib.load(model_path), schema, info, reference)

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
        if isinstance(data, pd.DataFrame):
            return data.copy()
        return readers.read_table(data)

    def validate(self, data: str | pathlib.Path | pd.DataFrame) -> SchemaReport:
        """Checks input against the schema without predicting.

        Args:
            data: DataFrame or path to CSV/Parquet.

        Returns:
            Missing/extra columns, missing-value counts and unseen categories.
        """
        frame = self.read_frame(data)
        wanted = self.features
        present = [c for c in wanted if c in frame.columns]
        unseen = {}
        for column in self.schema.categorical:
            if column in frame.columns:
                known = set(self.schema.known_categories[column])
                new = sorted(set(frame[column].dropna().astype(str)) - known)
                if new:
                    unseen[column] = new[:MAX_REPORTED_UNSEEN]
        return SchemaReport(
            missing=tuple(c for c in wanted if c not in frame.columns),
            extra=tuple(
                c
                for c in frame.columns
                if c not in wanted and c != self.schema.target
            ),
            missing_value_counts={
                c: int(frame[c].isna().sum())
                for c in present
                if frame[c].isna().any()
            },
            unseen_categories=unseen,
        )

    def prepare_frame(
        self, data: str | pathlib.Path | pd.DataFrame
    ) -> pd.DataFrame:
        """Validates input and returns the feature columns.

        Raises:
            SchemaError: On missing columns or non-numeric values in numeric
                columns.
        """
        frame = self.read_frame(data)
        report = self.validate(frame)
        if not report.valid:
            raise errors.SchemaError(
                f"input is missing required columns: {list(report.missing)}",
                {"missing": list(report.missing)},
            )
        bad = []
        for column in self.schema.numeric:
            coerced = pd.to_numeric(frame[column], errors="coerce")
            if (coerced.isna() & frame[column].notna()).any():
                bad.append(column)
        if bad:
            raise errors.SchemaError(
                f"non-numeric values in numeric columns: {bad}",
                {"columns": bad},
            )
        return frame[self.features]

    def predict(self, data: str | pathlib.Path | pd.DataFrame) -> np.ndarray:
        """Predicts labels or values through the full fitted pipeline."""
        return self.pipeline.predict(self.prepare_frame(data))

    def predict_proba(
        self, data: str | pathlib.Path | pd.DataFrame
    ) -> np.ndarray:
        """Predicts class probabilities.

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
        return self.pipeline.predict_proba(self.prepare_frame(data))
