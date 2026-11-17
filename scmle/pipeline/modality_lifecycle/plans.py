"""The fixed inputs of one lifecycle run and what they imply."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import shutil
from typing import Any
from typing import cast
from typing import Literal
from typing import TYPE_CHECKING

from scmle import durability
from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.contracts import performance
from scmle.governance import attestation
from scmle.pipeline.modality_lifecycle import checkpoints
from scmle.pipeline.modality_lifecycle import containers
from scmle.pipeline.modality_lifecycle import curators
from scmle.pipeline.modality_lifecycle import predictors
from scmle.pipeline.modality_lifecycle import reporting
from scmle.pipeline.modality_lifecycle import state

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence


def typecheck(
    require_independent_ground_truth: object,
    cancellation_check: object,
    resume: object,
) -> None:
    """Rejects lifecycle run options of the wrong type.

    Args:
        require_independent_ground_truth: Candidate for the ground-truth flag.
        cancellation_check: Candidate for the cancellation callable.
        resume: Candidate for the resume flag.

    Raises:
        TypeError: If a flag is not a bool or the cancellation check is not
            callable.
    """
    if not isinstance(require_independent_ground_truth, bool):
        raise errors.KindError("require_independent_ground_truth must be bool")
    if cancellation_check is not None and not callable(cancellation_check):
        raise errors.KindError("cancellation_check must be callable")
    if not isinstance(resume, bool):
        raise errors.KindError("resume must be bool")


def coherent(
    resume: bool, checkpoint_target: Path | None, report_target: Path | None
) -> None:
    """Rejects inconsistent checkpoint and report paths.

    Args:
        resume: Whether the run resumes from a checkpoint.
        checkpoint_target: The checkpoint file, if any.
        report_target: The report file, if any.

    Raises:
        ValueError: If `resume` is set without a checkpoint path, a
            checkpoint path comes without a report path, or the two paths
            are the same.
    """
    if resume and checkpoint_target is None:
        raise errors.InputError("resume requires checkpoint_path")
    if checkpoint_target is not None and report_target is None:
        raise errors.InputError(
            "checkpoint_path requires report_path so completion can be verified"
        )
    if checkpoint_target is not None and checkpoint_target == report_target:
        raise errors.InputError("checkpoint_path and report_path must differ")


@dataclasses.dataclass(frozen=True, slots=True)
class Plan:
    """The fixed inputs of one lifecycle run and what they imply.

    Attributes:
        adapter: Adapter for the reference dataset.
        observed: Adapter over the data to monitor.
        package_target: Directory the model package is written to.
        report_target: File the signed report is written to, if any.
        checkpoint_target: File where signed checkpoints are kept, if any.
        signing_key: Key that signs the package, report and checkpoints.
        seed: Random seed of the run.
        require_independent_ground_truth: Whether the run demands
            independently authored explanation ground truth.
        manifest: Modality manifest to check against the adapter, if any.
        cancellation_check: Callable polled at stage boundaries, if any.
        package_preexisting: Whether the package directory existed before
            the run.
        report_preexisting: Whether the report file existed before the run.
    """

    adapter: curators.Preparer
    observed: curators.Preparer
    package_target: Path
    report_target: Path | None
    checkpoint_target: Path | None
    signing_key: str | bytes
    seed: int
    require_independent_ground_truth: bool
    manifest: data.Passport | None
    cancellation_check: Callable[[], bool] | None
    package_preexisting: bool
    report_preexisting: bool

    @property
    def manifest_digest(self) -> str | None:
        """The digest of the manifest, or None without one."""
        if self.manifest is None:
            return None
        return attestation.checksum(self.manifest.to_dict())

    def poll(self) -> None:
        """Stops before publication and removes this run's artifacts.

        Does nothing when no check was given or it reports no cancellation.
        On cancellation it deletes the package directory and the report file
        unless they existed before the run.

        Raises:
            LifecycleCancelledError: If `cancellation_check` reports
                cancellation.
        """
        if self.cancellation_check is None or not self.cancellation_check():
            return
        self.discard()
        if self.report_target is not None and not self.report_preexisting:
            self.report_target.unlink(missing_ok=True)
        raise containers.LifecycleCancelledError(
            "modality lifecycle cancelled at a stage boundary"
        )

    def discard(self) -> None:
        """Deletes the package directory unless it existed before the run."""
        if not self.package_preexisting:
            shutil.rmtree(self.package_target, ignore_errors=True)

    def conform(self) -> None:
        """Checks that the monitored data has the reference modality.

        Raises:
            ValueError: If the observed adapter's modality differs.
        """
        if self.observed.modality != self.adapter.modality:
            raise errors.InputError(
                "observed adapter modality must match reference modality"
            )

    def audit(self) -> None:
        """Checks the manifest, if any, against the adapter.

        Raises:
            ValueError: If the manifest's modality, content hash or sample
                count differs from the adapter's.
        """
        manifest = self.manifest
        if manifest is None:
            return
        if manifest.modality != modalities.serialise(self.adapter.modality):
            raise errors.InputError("manifest modality must match adapter")
        if manifest.content_hash != self.adapter.dataset_hash:
            raise errors.InputError("manifest content_hash must match adapter")
        if manifest.sample_count != len(self.adapter.samples):
            raise errors.InputError("manifest sample_count must match adapter")

    def expect(self, *, resume: bool) -> None:
        """Checks that the checkpoint file exists exactly when resuming.

        Args:
            resume: Whether the run resumes from the checkpoint.

        Raises:
            FileExistsError: If the checkpoint exists and `resume` is False.
            FileNotFoundError: If `resume` is True and it does not exist.
        """
        target = self.checkpoint_target
        if target is None:
            return
        if target.exists() and not resume:
            raise FileExistsError(target)
        if resume and not target.exists():
            raise FileNotFoundError(target)

    def identity(self) -> dict[str, object]:
        """Describes the run as a checkpoint must record it.

        Returns:
            The identity fields a checkpoint of this run has to match.
        """
        return {
            "modality": modalities.serialise(self.adapter.modality),
            "dataset_hash": self.adapter.dataset_hash,
            "observed_dataset_hash": self.observed.dataset_hash,
            "package_path": str(self.package_target),
            "report_path": str(self.report_target),
            "seed": self.seed,
            "require_independent_ground_truth": (
                self.require_independent_ground_truth
            ),
            "manifest_digest": self.manifest_digest,
        }

    def reload(self, stage_names: Sequence[str]) -> checkpoints.Checkpoint:
        """Reads the checkpoint to resume from and checks that it fits.

        Args:
            stage_names: Values of the lifecycle stages, in order.

        Returns:
            The verified checkpoint.

        Raises:
            ValueError: If the checkpoint identity or stage order does not
                match this run.
        """
        checkpoint = checkpoints.Checkpoint.read_verified(
            cast("Path", self.checkpoint_target), self.signing_key
        )
        actual_identity = {
            "modality": checkpoint.modality,
            "dataset_hash": checkpoint.dataset_hash,
            "observed_dataset_hash": checkpoint.observed_dataset_hash,
            "package_path": checkpoint.package_path,
            "report_path": checkpoint.report_path,
            "seed": checkpoint.seed,
            "require_independent_ground_truth": (
                checkpoint.require_independent_ground_truth
            ),
            "manifest_digest": checkpoint.manifest_digest,
        }
        if actual_identity != self.identity():
            raise errors.InputError(
                "lifecycle checkpoint identity does not match the "
                "requested resume"
            )
        if (
            checkpoint.completed_stages
            != stage_names[: len(checkpoint.completed_stages)]
        ):
            raise errors.InputError(
                "lifecycle checkpoint stage order does not match "
                "the current lifecycle"
            )
        return checkpoint

    def verify(self) -> reporting.Record:
        """Verifies the report of a run whose checkpoint is completed.

        Returns:
            The previously written report, after checking it and its package.

        Raises:
            ValueError: If the report is missing or does not match the run.
        """
        report_target = self.report_target
        if report_target is None or not report_target.is_file():
            raise errors.InputError(
                "completed lifecycle checkpoint has no report"
            )
        completed_report = reporting.Record.read_verified(
            report_target, self.signing_key
        )
        if (
            completed_report.modality
            != modalities.serialise(self.adapter.modality)
            or completed_report.dataset_hash != self.adapter.dataset_hash
            or Path(completed_report.package.path) != self.package_target
        ):
            raise errors.InputError(
                "completed lifecycle report does not match the requested resume"
            )
        predictors.Predictor.from_package(
            self.adapter,
            completed_report.package.path,
            self.signing_key,
            lifecycle_report_path=report_target,
        )
        return completed_report

    def persist(
        self,
        completed_stages: Sequence[str],
        stage_digests: Sequence[str],
        status: Literal["running", "completed"],
    ) -> None:
        """Atomically persists a signed stage boundary.

        Does nothing when the run keeps no checkpoint.

        Args:
            completed_stages: Values of the stages finished so far.
            stage_digests: Digest of each finished stage's evidence.
            status: "running" or "completed".
        """
        if self.checkpoint_target is None:
            return
        record = checkpoints.Checkpoint(
            modality=modalities.serialise(self.adapter.modality),
            dataset_hash=self.adapter.dataset_hash,
            observed_dataset_hash=self.observed.dataset_hash,
            package_path=str(self.package_target),
            report_path=str(self.report_target),
            seed=self.seed,
            completed_stages=tuple(completed_stages),
            stage_digests=tuple(stage_digests),
            require_independent_ground_truth=(
                self.require_independent_ground_truth
            ),
            manifest_digest=self.manifest_digest,
            status=status,
        ).sign(self.signing_key)
        durability.write_atomically(
            self.checkpoint_target,
            (
                json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n"
            ).encode("utf-8"),
        )


def exported(value: Any) -> dict[str, object] | None:
    """Converts an optional performance record to JSON-native values.

    Args:
        value: A budget, baseline, result or comparison, or None.

    Returns:
        The record's mapping, or None without a record.
    """
    return value.to_dict() if value is not None else None


def benchmark(
    plan: Plan,
    context: state.Context,
    performance_budget: performance.Budget | None,
    performance_baseline: performance.Baseline | None,
) -> dict[str, object]:
    """Measures a run against its budget and baseline.

    Args:
        plan: The run's fixed inputs.
        context: Shared lifecycle state with the observations.
        performance_budget: Limits the measured operations must meet, if
            any.
        performance_baseline: Earlier run the measurements are compared
            with, if any.

    Returns:
        The performance evidence of the report.

    Raises:
        ValueError: If the budget or the baseline is violated; the
            package this run created is removed first.
    """
    observations = tuple(
        context.performance_observations[operation]
        for operation in performance.Operation
        if operation in context.performance_observations
    )
    performance_run = performance.Run(
        run_id=f"{plan.adapter.dataset_hash}:{plan.seed}",
        observations=observations,
    )
    performance_result = (
        performance_budget.evaluate(observations)
        if performance_budget is not None
        else None
    )
    performance_comparison = (
        performance_baseline.evaluate(performance_run)
        if performance_baseline is not None
        else None
    )
    evidence: dict[str, object] = {
        "run": performance_run.to_dict(),
        "observations": [observation.to_dict() for observation in observations],
        "budget": exported(performance_budget),
        "result": exported(performance_result),
        "baseline": exported(performance_baseline),
        "comparison": exported(performance_comparison),
    }
    if performance_result is not None and not performance_result.passed:
        plan.discard()
        raise errors.InputError(
            "performance budget failed: "
            + "; ".join(performance_result.violations)
        )
    if performance_comparison is not None and not performance_comparison.passed:
        plan.discard()
        raise errors.InputError(
            "performance baseline failed: "
            + "; ".join(performance_comparison.violations)
        )
    return evidence


def prepare(
    adapter: curators.Preparer,
    package_path: str | Path,
    signing_key: str | bytes,
    seed: int,
    observed_adapter: curators.Preparer | None,
    report_path: str | Path | None,
    manifest: data.Passport | None,
    require_independent_ground_truth: bool,
    cancellation_check: Callable[[], bool] | None,
    checkpoint_path: str | Path | None,
    resume: bool,
) -> Plan:
    """Checks the options of a lifecycle run and fixes its inputs.

    Args:
        adapter: Adapter for the reference dataset.
        package_path: Directory the model package is written to.
        signing_key: Key that signs the package, report and checkpoints.
        seed: Random seed for the split and the model.
        observed_adapter: Adapter over the data to monitor, or None to
            monitor the reference dataset.
        report_path: File the signed report is written to, if any.
        manifest: Modality manifest to check against the adapter, if any.
        require_independent_ground_truth: Whether to fail unless the
            independent explanation ground-truth suite passes.
        cancellation_check: Callable polled at stage boundaries, if any.
        checkpoint_path: File where signed stage checkpoints are kept, if
            any.
        resume: Whether to resume from the checkpoint.

    Returns:
        The run's plan, recording whether the package and report existed.

    Raises:
        TypeError: If `require_independent_ground_truth` or `resume` is not
            a bool, or `cancellation_check` is not callable.
        ValueError: If `resume` is set without a checkpoint path, a
            checkpoint path comes without a report path, or the two paths
            are the same.
    """
    typecheck(require_independent_ground_truth, cancellation_check, resume)
    report_target = Path(report_path) if report_path is not None else None
    checkpoint_target = (
        Path(checkpoint_path) if checkpoint_path is not None else None
    )
    coherent(resume, checkpoint_target, report_target)
    package_target = Path(package_path)
    return Plan(
        adapter=adapter,
        observed=observed_adapter or adapter,
        package_target=package_target,
        report_target=report_target,
        checkpoint_target=checkpoint_target,
        signing_key=signing_key,
        seed=seed,
        require_independent_ground_truth=require_independent_ground_truth,
        manifest=manifest,
        cancellation_check=cancellation_check,
        package_preexisting=package_target.exists(),
        report_preexisting=(
            report_target.exists() if report_target is not None else False
        ),
    )


def publish(
    plan: Plan,
    context: state.Context,
    performance_budget: performance.Budget | None,
    performance_baseline: performance.Baseline | None,
) -> reporting.Record:
    """Signs the report of a run whose stages all completed and writes it.

    Args:
        plan: The run's fixed inputs.
        context: Shared lifecycle state after the last stage.
        performance_budget: Limits the measured operations must meet, if
            any.
        performance_baseline: Earlier run the measurements are compared
            with, if any.

    Returns:
        The signed lifecycle report, also written to the report file if the
        run has one.

    Raises:
        RuntimeError: If packaging produced no package.
    """
    records = tuple(
        modalities.Proof(
            stage=stage,
            evidence_links=("modality_lifecycle.json",),
            evidence=context.stage_records[stage],
            notes=(f"completed {stage.value}",),
        )
        for stage in modalities.REQUIRED_LIFECYCLE_STAGES
    )
    package = context.package
    if package is None:
        raise RuntimeError("packaging stage did not produce a package")
    performance_evidence = benchmark(
        plan, context, performance_budget, performance_baseline
    )
    report = reporting.Record(
        modality=modalities.serialise(plan.adapter.modality),
        dataset_hash=plan.adapter.dataset_hash,
        stage_evidence=records,
        evaluation=context.evaluation,
        performance=performance_evidence,
        package=package,
        inference=context.inference,
        monitoring=context.monitoring,
        documentation=context.documentation,
        recovery=context.recovery,
    ).sign(plan.signing_key)
    if plan.report_target is not None:
        durability.write_atomically(
            plan.report_target,
            (
                json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
            ).encode("utf-8"),
        )
        plan.poll()
    return report
