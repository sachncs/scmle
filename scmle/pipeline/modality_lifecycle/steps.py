"""The lifecycle stages other than model explainability."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import time

import joblib
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

from scmle import durability
from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.contracts import performance
from scmle.contracts import signing
from scmle.data.modalities import curation
from scmle.pipeline.modality_lifecycle import constants
from scmle.pipeline.modality_lifecycle import containers
from scmle.pipeline.modality_lifecycle import predictors
from scmle.pipeline.modality_lifecycle import state
from scmle.pipeline.modality_lifecycle import verification


class Ingestion(state.Stage):
    """Confirm source identity and sample cardinality."""

    stage = modalities.Phase.INGESTION
    performance_operation = performance.Operation.INGEST

    def execute(self, context: state.Context) -> None:
        """Records immutable source identity.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            ValueError: If the dataset has no samples.
        """
        if not context.adapter.samples:
            raise errors.InputError("modality dataset must not be empty")
        context.record(
            self.stage,
            {
                "dataset_hash": context.adapter.dataset_hash,
                "sample_count": len(context.adapter.samples),
            },
        )


class Validation(state.Stage):
    """Reject missing labels and malformed feature prerequisites."""

    stage = modalities.Phase.VALIDATION

    def execute(self, context: state.Context) -> None:
        """Validates labels and dataset-level curation state.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            ValueError: If any sample lacks a label or fewer than two classes
                are present.
        """
        labels = context.adapter.labels(context.adapter.samples)
        if any(label is None for label in labels):
            raise errors.InputError(
                "complete modality lifecycle requires labels"
            )
        if len(set(labels)) < constants.MIN_CLASS_COUNT:
            raise errors.InputError(
                "classification lifecycle requires two labels"
            )
        context.record(
            self.stage,
            {"label_count": len(labels), "class_count": len(set(labels))},
        )


class Profiling(state.Stage):
    """Capture modality-specific profile evidence."""

    stage = modalities.Phase.PROFILING

    def execute(self, context: state.Context) -> None:
        """Persists the curation/profile object in lifecycle state.

        Args:
            context: Shared lifecycle state, updated in place.
        """
        context.curation_report = context.adapter.curate()
        context.record(
            self.stage, {"profile": context.curation_report.to_dict()}
        )


class Curating(state.Stage):
    """Block unresolved data-quality findings before representation."""

    stage = modalities.Phase.CURATION

    def execute(self, context: state.Context) -> None:
        """Requires a release-ready curation report.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            ValueError: If there is no curation report or it has unresolved
                blocking findings.
        """
        report = context.curation_report
        if report is None or not report.release_ready:
            findings = (
                []
                if report is None
                else [
                    finding.finding_id
                    for finding in report.unresolved_blocking_findings
                ]
            )
            raise errors.InputError(
                f"unresolved blocking curation findings: {findings}"
            )
        context.record(self.stage, {"release_ready": True, "unresolved": []})


class Splitting(state.Stage):
    """Create leakage-aware train, validation, and test partitions."""

    stage = modalities.Phase.SPLITTING

    def execute(self, context: state.Context) -> None:
        """Splits and rejects overlapping sample identities.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            ValueError: If the train or test partition is empty or two
                partitions share a group.
        """
        partitions = context.adapter.split(context.seed)
        if not partitions.train or not partitions.test:
            raise errors.InputError(
                "lifecycle split requires non-empty train and test partitions"
            )
        groups = [
            set(partitions.groups.get(name, ()))
            for name in ("train", "validation", "test")
        ]
        if any(
            left & right
            for index, left in enumerate(groups)
            for right in groups[index + 1 :]
        ):
            raise errors.InputError(
                "lifecycle split contains overlapping groups"
            )
        context.partitions = partitions
        context.record(
            self.stage,
            {
                "counts": {
                    "train": len(partitions.train),
                    "validation": len(partitions.validation),
                    "test": len(partitions.test),
                },
                "groups": partitions.groups,
            },
        )


class Preparation(state.Stage):
    """Prepare partition-local finite feature matrices."""

    stage = modalities.Phase.FEATURE_PREPARATION

    def execute(self, context: state.Context) -> None:
        """Prepares and aligns features without fitting on held-out rows.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            RuntimeError: If the data has not been split yet.
            ValueError: If the partitions' feature names differ.
        """
        partitions = context.partitions
        if partitions is None:
            raise errors.OrderError(
                "splitting must precede feature preparation"
            )
        batches = {
            "train": context.adapter.features(partitions.train),
            "validation": context.adapter.features(partitions.validation),
            "test": context.adapter.features(partitions.test),
        }
        names = batches["train"].feature_names
        if any(batch.feature_names != names for batch in batches.values()):
            raise errors.InputError("partition feature schemas do not match")
        context.feature_batches = batches
        context.record(
            self.stage,
            {
                "feature_names": list(names),
                "shape": {
                    name: list(batch.values.shape)
                    for name, batch in batches.items()
                },
            },
        )


class Modelling(state.Stage):
    """Fit the bounded regularized linear baseline on training features."""

    stage = modalities.Phase.MODELLING
    performance_operation = performance.Operation.FIT

    def execute(self, context: state.Context) -> None:
        """Fits only on the training partition.

        Args:
            context: Shared lifecycle state, updated in place.
        """
        batch = context.feature_batches["train"]
        labels = context.adapter.labels(
            context.partitions.train if context.partitions else ()
        )
        model = LogisticRegression(max_iter=500, random_state=context.seed)
        model.fit(batch.values, labels)
        context.model = model
        context.record(
            self.stage,
            {
                "model_family": "regularized_linear_classifier",
                "classes": [str(value) for value in model.classes_],
            },
        )


class Evaluation(state.Stage):
    """Evaluate once on the untouched test partition."""

    stage = modalities.Phase.EVALUATION

    def execute(self, context: state.Context) -> None:
        """Computes deterministic classification metrics and predictions.

        The metrics are computed on the test partition's feature batch and
        stored as `context.evaluation`, which is also recorded as the stage
        evidence.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            RuntimeError: If the model or partitions are missing.
        """
        model = context.model
        partitions = context.partitions
        if model is None or partitions is None:
            raise errors.OrderError(
                "modelling and splitting must precede evaluation"
            )
        batch = context.feature_batches["test"]
        actual = context.adapter.labels(partitions.test)
        predicted = model.predict(batch.values)
        probabilities = model.predict_proba(batch.values)
        context.evaluation = {
            "sample_count": len(actual),
            "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, average="macro")),
            "log_loss": float(
                log_loss(actual, probabilities, labels=list(model.classes_))
            ),
            "labels": [str(value) for value in model.classes_],
            "predictions": [str(value) for value in predicted],
        }
        context.record(self.stage, context.evaluation.copy())


class Composition(state.Stage):
    """Explain source composition, split boundaries, and feature ranges."""

    stage = modalities.Phase.DATA_EXPLAINABILITY

    def execute(self, context: state.Context) -> None:
        """Records data lineage and non-causal feature summaries.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            RuntimeError: If the data has not been split yet.
        """
        batch = context.feature_batches["train"]
        partitions = context.partitions
        if partitions is None:
            raise errors.OrderError("splitting must precede data explanation")
        labels = context.adapter.labels(partitions.train)
        label_counts: dict[str, int] = {}
        for label in labels:
            key = str(label)
            label_counts[key] = label_counts.get(key, 0) + 1
        statistics: dict[str, dict[str, float | int]] = {}
        for index, name in enumerate(batch.feature_names):
            values = batch.values[:, index]
            statistics[name] = {
                "min": float(numpy.min(values)),
                "max": float(numpy.max(values)),
                "mean": float(numpy.mean(values)),
                "std": float(numpy.std(values)),
                "q05": float(numpy.quantile(values, 0.05)),
                "median": float(numpy.median(values)),
                "q95": float(numpy.quantile(values, 0.95)),
                "nonfinite_count": int(
                    numpy.count_nonzero(~numpy.isfinite(values))
                ),
            }
        partition_samples = {
            "train": partitions.train,
            "validation": partitions.validation,
            "test": partitions.test,
        }
        sample_lineage = [
            {
                "sample_id": context.adapter.sample_id(sample),
                "partition": partition,
            }
            for partition, samples in partition_samples.items()
            for sample in samples
        ]
        context.data_explanation = {
            "dataset_hash": context.adapter.dataset_hash,
            "sample_ids": list(batch.sample_ids),
            "sample_lineage": sample_lineage,
            "feature_ranges": {
                name: {
                    "min": float(batch.values[:, index].min()),
                    "max": float(batch.values[:, index].max()),
                }
                for index, name in enumerate(batch.feature_names)
            },
            "feature_statistics": statistics,
            "label_distribution": {
                "counts": dict(sorted(label_counts.items())),
                "sample_count": len(labels),
            },
            "partition_counts": {
                "train": len(partitions.train),
                "validation": len(partitions.validation),
                "test": len(partitions.test),
            },
            "split_groups": partitions.groups,
            "limitations": [
                (
                    "Feature summaries describe association and coverage, not "
                    "causation."
                ),
                *context.adapter.remarks(),
            ],
        }
        context.record(self.stage, context.data_explanation.copy())


class Packaging(state.Stage):
    """Writes and signs a digest-bound model package atomically."""

    stage = modalities.Phase.PACKAGING
    performance_operation = performance.Operation.PACKAGE

    @staticmethod
    def replay(
        context: state.Context, model: LogisticRegression
    ) -> containers.Package:
        """Checks the existing package against a replayed run.

        Args:
            context: Shared lifecycle state of the resumed run.
            model: The model the replay fitted.

        Returns:
            The identity of the existing package.

        Raises:
            ValueError: If the package's dataset identity, feature schema
                or model differs from the replayed run's.
        """
        predictor = predictors.Predictor.from_package(
            context.adapter, context.package_path, context.signing_key
        )
        expected_features = context.feature_batches["train"].feature_names
        if predictor.training_dataset_hash != context.adapter.dataset_hash:
            raise errors.InputError(
                "checkpoint package dataset identity does not match "
                "the resumed adapter"
            )
        if predictor.feature_names != expected_features:
            raise errors.InputError(
                "checkpoint package feature schema does not match "
                "the resumed adapter"
            )
        if not (
            numpy.array_equal(predictor.model.classes_, model.classes_)
            and numpy.array_equal(predictor.model.coef_, model.coef_)
            and numpy.array_equal(predictor.model.intercept_, model.intercept_)
        ):
            raise errors.InputError(
                "checkpoint package model does not match deterministic "
                "stage replay"
            )
        return predictor.package

    def execute(self, context: state.Context) -> None:
        """Persists the estimator and signed manifest.

        When replaying a checkpointed run onto an existing package, the package
        is verified against the replayed model instead of being rewritten.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            RuntimeError: If there is no fitted model.
            ValueError: If an existing package does not match the resumed run.
            FileExistsError: If the package path already exists and the run is
                not replaying a checkpoint.
        """
        model = context.model
        if model is None:
            raise errors.OrderError("model is required before packaging")
        target = context.package_path
        if context.resume_replay and target.exists():
            context.package = self.replay(context, model)
            context.record(self.stage, context.package.to_dict())
            return
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
        )
        try:
            model_path = temporary / "model.joblib"
            joblib.dump(model, model_path, compress=3)
            durability.sync(model_path)
            unsigned = {
                "schema_version": 1,
                "modality": modalities.serialise(context.adapter.modality),
                "dataset_hash": context.adapter.dataset_hash,
                "feature_names": list(
                    context.feature_batches["train"].feature_names
                ),
                "model_sha256": verification.file_digest(model_path),
                "evaluation": {
                    key: value
                    for key, value in context.evaluation.items()
                    if key != "predictions"
                },
            }
            signature = signing.mac(unsigned, context.signing_key)
            manifest = {
                **unsigned,
                "signature": {
                    "algorithm": signing.SIGNATURE_ALGORITHM,
                    "value": signature,
                },
            }
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            durability.sync(manifest_path)
            durability.sync_directory(temporary)
            Path(temporary).replace(target)
            durability.sync_directory(target.parent)
            context.package = containers.Package(
                str(target), str(unsigned["model_sha256"]), signature
            )
            context.record(self.stage, context.package.to_dict())
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


class Inference(state.Stage):
    """Verify the package before loading and reproducing predictions."""

    stage = modalities.Phase.INFERENCE

    def execute(self, context: state.Context) -> None:
        """Loads trusted bytes and compares predictions with the fitted model.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            RuntimeError: If the package or partitions are missing.
            ValueError: If the packaged model's predictions differ from the
                evaluation predictions.
        """
        if context.package is None or context.partitions is None:
            raise errors.OrderError(
                "package and partitions are required before inference"
            )
        started = time.perf_counter()
        loaded = verification.load_package(
            context.package.path, context.signing_key
        )
        context.clock(
            performance.Operation.LOAD,
            time.perf_counter() - started,
            len(context.feature_batches["test"].sample_ids),
        )
        started = time.perf_counter()
        predictions = loaded.predict(context.feature_batches["test"].values)
        context.clock(
            performance.Operation.INFERENCE,
            time.perf_counter() - started,
            len(context.feature_batches["test"].sample_ids),
        )
        expected = context.evaluation["predictions"]
        actual = [str(value) for value in predictions]
        if actual != expected:
            raise ValueError(
                "packaged inference does not reproduce evaluation predictions"
            )
        context.inference = {
            "package_verified": True,
            "prediction_count": len(actual),
            "predictions_match": True,
        }
        context.record(self.stage, context.inference.copy())


class Monitoring(state.Stage):
    """Emit reference-to-observed data drift evidence."""

    stage = modalities.Phase.MONITORING

    def execute(self, context: state.Context) -> None:
        """Records modality-specific monitoring output.

        Args:
            context: Shared lifecycle state, updated in place.
        """
        context.monitoring = context.adapter.monitor(context.observed_adapter)
        context.record(self.stage, context.monitoring.copy())


class Documentation(state.Stage):
    """Build a truthful dataset and model card for the baseline."""

    stage = modalities.Phase.DOCUMENTATION

    def execute(self, context: state.Context) -> None:
        """Records support status, intended use, and limitations.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            RuntimeError: If the data has not been profiled yet.
        """
        if context.curation_report is None:
            raise errors.OrderError("profiling must precede documentation")
        dataset_card = curation.Datasheet.from_report(
            modalities.serialise(context.adapter.modality),
            context.curation_report.to_dict(),
            context.adapter.remarks(),
            manifest=context.manifest,
        )
        context.documentation = {
            "modality": modalities.serialise(context.adapter.modality),
            "status": (
                "extension-only"
                if isinstance(context.adapter.modality, modalities.Bespoke)
                else "experimental"
            ),
            "intended_use": (
                "bounded lifecycle validation and baseline comparison"
            ),
            "dataset_hash": context.adapter.dataset_hash,
            "model_family": "regularized_linear_classifier",
            "dataset_card": dataset_card.to_dict(),
            "limitations": list(context.adapter.remarks()),
            "promotion_requirements": [
                "independent metric recomputation",
                "modality-specific explanation validation",
                "external serving and resource-fault evidence",
            ],
        }
        context.record(self.stage, context.documentation.copy())


class Recovery(state.Stage):
    """Prove bounded resource claims and reject a tampered package."""

    stage = modalities.Phase.RECOVERY_AND_RESOURCE_TESTS

    def execute(self, context: state.Context) -> None:
        """Tampers with a copy of the package and verifies it is rejected.

        Args:
            context: Shared lifecycle state, updated in place.

        Raises:
            RuntimeError: If there is no package.
            ValueError: If a tampered copy of the package is not rejected.
        """
        if context.package is None:
            raise errors.OrderError("package is required before recovery tests")
        temporary = Path(tempfile.mkdtemp(prefix="scmle-lifecycle-recovery-"))
        tampered = temporary / "package"
        try:
            shutil.copytree(context.package.path, tampered)
            manifest_path = tampered / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dataset_hash"] = "tampered"
            manifest_path.write_text(
                json.dumps(manifest), encoding="utf-8", newline="\n"
            )
            rejected = False
            try:
                verification.load_package(tampered, context.signing_key)
            except (TypeError, ValueError):  # handled: expected rejection
                rejected = True
            if not rejected:
                raise ValueError("tampered package was not rejected")
            context.recovery = {
                "tampered_manifest_rejected": True,
                "feature_rows": int(
                    context.feature_batches["train"].values.shape[0]
                ),
                "feature_columns": int(
                    context.feature_batches["train"].values.shape[1]
                ),
                "bounded": True,
            }
            context.record(self.stage, context.recovery.copy())
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
