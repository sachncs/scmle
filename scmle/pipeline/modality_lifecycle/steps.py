"""The lifecycle stages other than model explainability.

Each stage names the stages whose products it reads (`requires`) and the
product it returns (`makes`); its run state comes from its phase
(`state.STATES`), which orders it on the engine. Its evidence is the same
record as before the engine, so signed checkpoints and reports keep their
digests.
"""

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
from scmle.pipeline import engine
from scmle.pipeline.modality_lifecycle import constants
from scmle.pipeline.modality_lifecycle import containers
from scmle.pipeline.modality_lifecycle import predictors
from scmle.pipeline.modality_lifecycle import state
from scmle.pipeline.modality_lifecycle import verification
from scmle.runtime import limits


class Ingestion(state.Stage):
    """Confirm source identity and sample cardinality."""

    stage = modalities.Phase.INGESTION
    makes = state.Intake
    operation = performance.Operation.INGEST

    def samples(self, board: state.Board) -> int:
        """How many samples ingestion read.

        Args:
            board: The run's board.

        Returns:
            The source samples.
        """
        return len(board.plan.adapter.samples)

    def execute(self, board: state.Board) -> state.Intake:
        """Records immutable source identity.

        Args:
            board: The run's inputs.

        Returns:
            The source identity.

        Raises:
            ValueError: If the dataset has no samples.
        """
        adapter = board.plan.adapter
        if not adapter.samples:
            raise errors.InputError("modality dataset must not be empty")
        return state.Intake(
            {
                "dataset_hash": adapter.digest,
                "sample_count": len(adapter.samples),
            }
        )


class Validation(state.Stage):
    """Reject missing labels and malformed feature prerequisites."""

    stage = modalities.Phase.VALIDATION
    requires = (Ingestion,)
    makes = state.Census

    def execute(self, board: state.Board) -> state.Census:
        """Validates labels and dataset-level curation state.

        Args:
            board: The run's inputs.

        Returns:
            The label and class counts.

        Raises:
            ValueError: If any sample lacks a label or fewer than two classes
                are present.
        """
        adapter = board.plan.adapter
        labels = adapter.labels(adapter.samples)
        if any(label is None for label in labels):
            raise errors.InputError(
                "complete modality lifecycle requires labels"
            )
        if len(set(labels)) < constants.MIN_CLASS_COUNT:
            raise errors.InputError(
                "classification lifecycle requires two labels"
            )
        return state.Census(
            {"label_count": len(labels), "class_count": len(set(labels))}
        )


class Profiling(state.Stage):
    """Capture modality-specific profile evidence."""

    stage = modalities.Phase.PROFILING
    requires = (Validation,)
    makes = state.Profile

    def execute(self, board: state.Board) -> state.Profile:
        """Profiles and curates the data.

        Args:
            board: The run's inputs.

        Returns:
            The curation report and its record.
        """
        report = board.plan.adapter.curate()
        return state.Profile({"profile": report.to_dict()}, report)


class Curating(state.Stage):
    """Block unresolved data-quality findings before representation."""

    stage = modalities.Phase.CURATION
    requires = (Profiling,)
    makes = state.Clearance

    def execute(self, board: state.Board) -> state.Clearance:
        """Requires a release-ready curation report.

        Args:
            board: The run's inputs and its profile.

        Returns:
            The release-ready verdict.

        Raises:
            ValueError: If the curation report has unresolved blocking
                findings.
        """
        report = board.get(state.Profile).report
        if report is None or not report.release:
            findings = (
                []
                if report is None
                else [finding.identifier for finding in report.blockers]
            )
            raise errors.InputError(
                f"unresolved blocking curation findings: {findings}"
            )
        return state.Clearance({"release_ready": True, "unresolved": []})


class Splitting(state.Stage):
    """Create leakage-aware train, validation, and test partitions."""

    stage = modalities.Phase.SPLITTING
    requires = (Curating,)
    makes = state.Partition

    def execute(self, board: state.Board) -> state.Partition:
        """Splits and rejects overlapping sample identities.

        Args:
            board: The run's inputs.

        Returns:
            The partitions.

        Raises:
            ValueError: If the train or test partition is empty or two
                partitions share a group.
        """
        partitions = board.plan.adapter.split(board.plan.seed)
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
        evidence: dict[str, object] = {
            "counts": {
                "train": len(partitions.train),
                "validation": len(partitions.validation),
                "test": len(partitions.test),
            },
            "groups": partitions.groups,
        }
        return state.Partition(evidence, partitions)


class Preparation(state.Stage):
    """Prepare partition-local finite feature matrices."""

    stage = modalities.Phase.FEATURE_PREPARATION
    requires = (Splitting,)
    makes = state.Features

    def execute(self, board: state.Board) -> state.Features:
        """Prepares and aligns features without fitting on held-out rows.

        Args:
            board: The run's inputs and partitions.

        Returns:
            The feature batch of each partition.

        Raises:
            ValueError: If the partitions' feature names differ.
        """
        adapter = board.plan.adapter
        partitions = board.get(state.Partition).partitions
        batches = {
            "train": adapter.features(partitions.train),
            "validation": adapter.features(partitions.validation),
            "test": adapter.features(partitions.test),
        }
        names = batches["train"].feature_names
        if any(batch.feature_names != names for batch in batches.values()):
            raise errors.InputError("partition feature schemas do not match")
        evidence: dict[str, object] = {
            "feature_names": list(names),
            "shape": {
                name: list(batch.values.shape)
                for name, batch in batches.items()
            },
        }
        return state.Features(evidence, batches)


class Modelling(state.Stage):
    """Fit the bounded regularized linear baseline on training features."""

    stage = modalities.Phase.MODELLING
    requires = (Splitting, Preparation)
    makes = state.Fit
    operation = performance.Operation.FIT

    def execute(self, board: state.Board) -> state.Fit:
        """Fits only on the training partition.

        Args:
            board: The run's inputs, partitions and features.

        Returns:
            The fitted classifier.
        """
        batch = board.get(state.Features).batches["train"]
        labels = board.plan.adapter.labels(
            board.get(state.Partition).partitions.train
        )
        model = LogisticRegression(max_iter=500, random_state=board.plan.seed)
        model.fit(batch.values, labels)
        evidence: dict[str, object] = {
            "model_family": "regularized_linear_classifier",
            "classes": [str(value) for value in model.classes_],
        }
        return state.Fit(evidence, model)


class Evaluation(state.Stage):
    """Evaluate once on the untouched test partition."""

    stage = modalities.Phase.EVALUATION
    requires = (Splitting, Preparation, Modelling)
    makes = state.Score

    def execute(self, board: state.Board) -> state.Score:
        """Computes deterministic classification metrics and predictions.

        Args:
            board: The run's inputs, partitions, features and model.

        Returns:
            The metrics on the test partition's feature batch, with its
            predictions.
        """
        model = board.get(state.Fit).model
        batch = board.get(state.Features).batches["test"]
        actual = board.plan.adapter.labels(
            board.get(state.Partition).partitions.test
        )
        predicted = model.predict(batch.values)
        probabilities = model.predict_proba(batch.values)
        return state.Score(
            {
                "samples": len(actual),
                "accuracy": float(accuracy_score(actual, predicted)),
                "macro_f1": float(f1_score(actual, predicted, average="macro")),
                "log_loss": float(
                    log_loss(actual, probabilities, labels=list(model.classes_))
                ),
                "labels": [str(value) for value in model.classes_],
                "predictions": [str(value) for value in predicted],
            }
        )


def statistics(values: numpy.ndarray) -> dict[str, float | int]:
    """Summarises one feature column.

    Args:
        values: The column.

    Returns:
        Its range, moments, quantiles and count of non-finite values.
    """
    return {
        "min": float(numpy.min(values)),
        "max": float(numpy.max(values)),
        "mean": float(numpy.mean(values)),
        "std": float(numpy.std(values)),
        "q05": float(numpy.quantile(values, 0.05)),
        "median": float(numpy.median(values)),
        "q95": float(numpy.quantile(values, 0.95)),
        "nonfinite_count": int(numpy.count_nonzero(~numpy.isfinite(values))),
    }


class Composition(state.Stage):
    """Explain source composition, split boundaries, and feature ranges."""

    stage = modalities.Phase.DATA_EXPLAINABILITY
    requires = (Splitting, Preparation, Evaluation)
    makes = state.Lineage

    def execute(self, board: state.Board) -> state.Lineage:
        """Records data lineage and non-causal feature summaries.

        Args:
            board: The run's inputs, partitions and features.

        Returns:
            The data explanation.
        """
        adapter = board.plan.adapter
        batch = board.get(state.Features).batches["train"]
        partitions = board.get(state.Partition).partitions
        labels = adapter.labels(partitions.train)
        label_counts: dict[str, int] = {}
        for label in labels:
            key = str(label)
            label_counts[key] = label_counts.get(key, 0) + 1
        partition_samples = {
            "train": partitions.train,
            "validation": partitions.validation,
            "test": partitions.test,
        }
        return state.Lineage(
            {
                "dataset_hash": adapter.digest,
                "sample_ids": list(batch.sample_ids),
                "sample_lineage": [
                    {
                        "sample_id": adapter.sample(sample),
                        "partition": partition,
                    }
                    for partition, samples in partition_samples.items()
                    for sample in samples
                ],
                "feature_ranges": {
                    name: {
                        "min": float(batch.values[:, index].min()),
                        "max": float(batch.values[:, index].max()),
                    }
                    for index, name in enumerate(batch.feature_names)
                },
                "feature_statistics": {
                    name: statistics(batch.values[:, index])
                    for index, name in enumerate(batch.feature_names)
                },
                "label_distribution": {
                    "counts": dict(sorted(label_counts.items())),
                    "sample_count": len(labels),
                },
                "partition_counts": {
                    name: len(samples)
                    for name, samples in partition_samples.items()
                },
                "split_groups": partitions.groups,
                "limitations": [
                    (
                        "Feature summaries describe association and coverage, "
                        "not causation."
                    ),
                    *adapter.remarks(),
                ],
            }
        )


class Packaging(state.Stage):
    """Writes and signs a digest-bound model package atomically."""

    stage = modalities.Phase.PACKAGING
    requires = (Preparation, Modelling, Evaluation)
    makes = state.Bundle
    operation = performance.Operation.PACKAGE

    @staticmethod
    def replay(
        board: state.Board, model: LogisticRegression
    ) -> containers.Package:
        """Checks the existing package against a replayed run.

        Args:
            board: The resumed run's inputs and features.
            model: The model the replay fitted.

        Returns:
            The identity of the existing package.

        Raises:
            ValueError: If the package's dataset identity, feature schema
                or model differs from the replayed run's.
        """
        plan = board.plan
        predictor = predictors.Predictor.load(
            plan.adapter, plan.package_target, plan.key
        )
        expected = board.get(state.Features).batches["train"].feature_names
        if predictor.dataset != plan.adapter.digest:
            raise errors.InputError(
                "checkpoint package dataset identity does not match "
                "the resumed adapter"
            )
        if predictor.names != expected:
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

    @staticmethod
    def write(board: state.Board, model: LogisticRegression) -> Path:
        """Writes the estimator and its signed manifest to a new folder.

        Args:
            board: The run's inputs, features and metrics.
            model: The fitted model.

        Returns:
            The temporary folder holding the package.
        """
        plan = board.plan
        target = plan.package_target
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
                "modality": modalities.serialise(plan.adapter.modality),
                "dataset_hash": plan.adapter.digest,
                "feature_names": list(
                    board.get(state.Features).batches["train"].feature_names
                ),
                "model_sha256": verification.digest(model_path),
                "evaluation": {
                    key: value
                    for key, value in board.get(state.Score).evidence.items()
                    if key != "predictions"
                },
            }
            manifest = {
                **unsigned,
                "signature": {
                    "algorithm": signing.SIGNATURE_ALGORITHM,
                    "value": signing.mac(unsigned, plan.key),
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
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return temporary

    def execute(self, board: state.Board) -> state.Bundle:
        """Persists the estimator and signed manifest.

        When replaying a checkpointed run onto an existing package, the package
        is verified against the replayed model instead of being rewritten.

        Args:
            board: The run's inputs, features, metrics and model.

        Returns:
            The package's identity.

        Raises:
            ValueError: If an existing package does not match the resumed run.
            FileExistsError: If the package path already exists and the run is
                not replaying a checkpoint.
        """
        model = board.get(state.Fit).model
        target = board.plan.package_target
        if board.replaying(self.stage) and target.exists():
            package = self.replay(board, model)
            return state.Bundle(package.to_dict(), package)
        if target.exists():
            raise FileExistsError(target)
        temporary = self.write(board, model)
        try:
            manifest = json.loads(
                (temporary / "manifest.json").read_text(encoding="utf-8")
            )
            temporary.replace(target)
            durability.sync_directory(target.parent)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        package = containers.Package(
            str(target),
            str(manifest["model_sha256"]),
            str(manifest["signature"]["value"]),
        )
        return state.Bundle(package.to_dict(), package)


class Inference(state.Stage):
    """Verify the package before loading and reproducing predictions."""

    stage = modalities.Phase.INFERENCE
    requires = (Preparation, Evaluation, Packaging)
    makes = state.Reproduction

    def execute(self, board: state.Board) -> state.Reproduction:
        """Loads trusted bytes and compares predictions with the fitted model.

        Args:
            board: The run's inputs, features, metrics and package.

        Returns:
            The inference check and its load and inference measurements.

        Raises:
            ValueError: If the packaged model's predictions differ from the
                evaluation predictions.
        """
        batch = board.get(state.Features).batches["test"]
        count = len(batch.sample_ids)
        started = time.perf_counter()
        loaded = verification.load(
            board.get(state.Bundle).package.path, board.plan.key
        )
        load = state.observe(
            performance.Operation.LOAD,
            engine.Lap(time.perf_counter() - started, limits.watermark()),
            count,
        )
        started = time.perf_counter()
        predictions = loaded.predict(batch.values)
        inference = state.observe(
            performance.Operation.INFERENCE,
            engine.Lap(time.perf_counter() - started, limits.watermark()),
            count,
        )
        actual = [str(value) for value in predictions]
        if actual != board.get(state.Score).evidence["predictions"]:
            raise ValueError(
                "packaged inference does not reproduce evaluation predictions"
            )
        evidence: dict[str, object] = {
            "package_verified": True,
            "prediction_count": len(actual),
            "predictions_match": True,
        }
        return state.Reproduction(evidence, (load, inference))


class Monitoring(state.Stage):
    """Emit reference-to-observed data drift evidence."""

    stage = modalities.Phase.MONITORING
    requires = (Ingestion,)
    makes = state.Drift

    def execute(self, board: state.Board) -> state.Drift:
        """Records modality-specific monitoring output.

        Args:
            board: The run's inputs.

        Returns:
            The monitoring output.
        """
        plan = board.plan
        return state.Drift(dict(plan.adapter.monitor(plan.observed)))


class Documentation(state.Stage):
    """Build a truthful dataset and model card for the baseline."""

    stage = modalities.Phase.DOCUMENTATION
    requires = (Profiling,)
    makes = state.Card

    def execute(self, board: state.Board) -> state.Card:
        """Records support status, intended use, and limitations.

        Args:
            board: The run's inputs and profile.

        Returns:
            The documentation.
        """
        plan = board.plan
        adapter = plan.adapter
        dataset_card = curation.Datasheet.draft(
            modalities.serialise(adapter.modality),
            board.get(state.Profile).report.to_dict(),
            adapter.remarks(),
            manifest=plan.manifest,
        )
        return state.Card(
            {
                "modality": modalities.serialise(adapter.modality),
                "status": (
                    "extension-only"
                    if isinstance(adapter.modality, modalities.Bespoke)
                    else "experimental"
                ),
                "intended_use": (
                    "bounded lifecycle validation and baseline comparison"
                ),
                "dataset_hash": adapter.digest,
                "model_family": "regularized_linear_classifier",
                "dataset_card": dataset_card.to_dict(),
                "limitations": list(adapter.remarks()),
                "promotion_requirements": [
                    "independent metric recomputation",
                    "modality-specific explanation validation",
                    "external serving and resource-fault evidence",
                ],
            }
        )


class Recovery(state.Stage):
    """Prove bounded resource claims and reject a tampered package."""

    stage = modalities.Phase.RECOVERY_AND_RESOURCE_TESTS
    requires = (Preparation, Packaging)
    makes = state.Resilience

    def execute(self, board: state.Board) -> state.Resilience:
        """Tampers with a copy of the package and verifies it is rejected.

        Args:
            board: The run's inputs, features and package.

        Returns:
            The recovery and resource evidence.

        Raises:
            ValueError: If a tampered copy of the package is not rejected.
        """
        temporary = Path(tempfile.mkdtemp(prefix="scmle-lifecycle-recovery-"))
        tampered = temporary / "package"
        try:
            shutil.copytree(board.get(state.Bundle).package.path, tampered)
            manifest_path = tampered / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dataset_hash"] = "tampered"
            manifest_path.write_text(
                json.dumps(manifest), encoding="utf-8", newline="\n"
            )
            rejected = False
            try:
                verification.load(tampered, board.plan.key)
            except (TypeError, ValueError):  # handled: expected rejection
                rejected = True
            if not rejected:
                raise ValueError("tampered package was not rejected")
            values = board.get(state.Features).batches["train"].values
            return state.Resilience(
                {
                    "tampered_manifest_rejected": True,
                    "feature_rows": int(values.shape[0]),
                    "feature_columns": int(values.shape[1]),
                    "bounded": True,
                }
            )
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
