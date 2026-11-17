"""The evaluation policy record and the registry that holds one per task."""

from collections.abc import Iterable
import json
from typing import Any
from typing import Self

import pydantic

from scmle.contracts import errors
from scmle.contracts import records
from scmle.contracts import tasks
from scmle.contracts.evaluation import parts
from scmle.contracts.evaluation import support


class Criteria(records.Record):
    """Complete task-aware evaluation and release declaration."""

    task: tasks.Task
    primary_metric: str
    metrics: tuple[parts.Declaration, ...]
    uncertainty: parts.Uncertainty
    calibration: parts.Calibration
    fairness: parts.Fairness
    small_groups: parts.Subgroups
    explanations: parts.Explanations
    release: parts.Gates

    @pydantic.model_validator(mode="after")
    def check(self) -> Self:
        """Validates task support and all cross-field policy references.

        Returns:
            The policy, unchanged.

        Raises:
            UnsupportedError: If a metric, uncertainty,
                calibration, fairness, explanation or diagnostic choice is not
                supported for the task.
            InputError: If the primary metric is blank, not among the metrics
                or not required, metrics are missing or repeated, a built-in
                metric has the wrong direction, or a release gate is
                inconsistent with the declarations.
        """
        parts.sanitize(self.primary_metric, "policy.primary_metric")
        if not self.metrics:
            raise errors.InputError("evaluation policy requires metrics")
        metric_names = [metric.name for metric in self.metrics]
        if len(metric_names) != len(set(metric_names)):
            raise errors.InputError("evaluation metrics must be unique")
        if self.primary_metric not in metric_names:
            raise errors.InputError("primary_metric must be one of metrics")
        primary = next(
            metric
            for metric in self.metrics
            if metric.name == self.primary_metric
        )
        if not primary.required:
            raise errors.InputError("primary_metric must be required")
        supported_metrics = support.SUPPORTED_METRICS[self.task]
        for metric in self.metrics:
            if metric.custom:
                continue
            if metric.name not in supported_metrics:
                raise support.UnsupportedError(
                    f"metric '{metric.name}' is unsupported for task "
                    f"'{self.task.value}'"
                )
            expected = support.METRIC_DIRECTIONS[metric.name]
            if expected is not metric.direction:
                raise errors.InputError(
                    f"metric '{metric.name}' has direction "
                    f"{expected.value}, not {metric.direction.value}"
                )
        self.enforce()
        self.scope()
        self.audit()
        return self

    def enforce(self) -> None:
        """Rejects uncertainty, calibration and explanation choices.

        Raises:
            UnsupportedError: If the uncertainty method, the calibration
                method or the explanation level is not supported for the
                task.
        """
        for kind, choice, supported in (
            (
                "uncertainty method",
                self.uncertainty.method,
                support.SUPPORTED_UNCERTAINTY,
            ),
            (
                "calibration method",
                self.calibration.method,
                support.SUPPORTED_CALIBRATION,
            ),
            (
                "explanation level",
                self.explanations.availability,
                support.SUPPORTED_EXPLANATIONS,
            ),
        ):
            if choice not in supported[self.task]:
                raise support.UnsupportedError(
                    f"{kind} '{choice.value}' is unsupported for task "
                    f"'{self.task.value}'"
                )

    def scope(self) -> None:
        """Rejects fairness methods and metrics outside the task boundary.

        Raises:
            UnsupportedError: If the method or a metric is not
                supported.
        """
        method = self.fairness.method
        if method not in support.SUPPORTED_FAIRNESS[self.task]:
            raise support.UnsupportedError(
                f"fairness method '{method.value}' is unsupported for task "
                f"'{self.task.value}'"
            )
        supported_metrics = support.FAIRNESS_METRICS.get(self.task, frozenset())
        for metric in self.fairness.metrics:
            if metric not in supported_metrics:
                raise support.UnsupportedError(
                    f"fairness metric '{metric}' is unsupported for task "
                    f"'{self.task.value}'"
                )

    def gatekeep(self) -> None:
        """Validates release metric gates against the declared metrics.

        Raises:
            UnsupportedError: If a gate names an undeclared
                metric that the task does not support.
            InputError: If a gate names an undeclared supported metric or has
                a direction that differs from the declaration.
        """
        declared_metrics = {metric.name: metric for metric in self.metrics}
        for gate in self.release.metric_gates:
            declaration = declared_metrics.get(gate.metric)
            if declaration is None:
                if gate.metric not in support.SUPPORTED_METRICS[self.task]:
                    raise support.UnsupportedError(
                        f"metric '{gate.metric}' is unsupported for task "
                        f"'{self.task.value}'"
                    )
                raise errors.InputError(
                    f"release metric gate '{gate.metric}' is undeclared"
                )
            if declaration.direction is not gate.direction:
                raise errors.InputError(
                    f"release metric gate '{gate.metric}' has a mismatched "
                    "direction"
                )

    def audit(self) -> None:
        """Validates release gates against declared metrics and diagnostics.

        Raises:
            UnsupportedError: If a metric gate names an
                undeclared metric that the task does not support, or a
                diagnostic gate names a diagnostic the task does not support.
            InputError: If a metric gate names an undeclared supported metric
                or has a direction that differs from the declaration, a
                fairness gate names a metric absent from the fairness
                declaration, or fairness gates exist without a fairness
                method or a fairness method has no gates.
        """
        self.gatekeep()
        for fairness_gate in self.release.fairness_gates:
            if fairness_gate.metric not in self.fairness.metrics:
                raise errors.InputError(
                    f"release fairness gate '{fairness_gate.metric}' is "
                    "undeclared"
                )
        supported_diagnostics = support.SUPPORTED_DIAGNOSTICS[self.task]
        for diagnostic_gate in self.release.diagnostic_gates:
            if diagnostic_gate.name not in supported_diagnostics:
                raise support.UnsupportedError(
                    f"diagnostic '{diagnostic_gate.name}' is unsupported for "
                    f"task '{self.task.value}'"
                )
        declared = self.fairness.method is not support.Technique.NONE
        if not declared and self.release.fairness_gates:
            raise errors.InputError(
                "fairness release gates need a fairness declaration"
            )
        if declared and not self.release.fairness_gates:
            raise errors.InputError(
                "fairness declaration needs release fairness gates"
            )

    def canonize(self) -> str:
        """Returns stable JSON suitable for hashing and evidence storage."""
        return json.dumps(
            self.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


class Registry:
    """Explicit registry that fails closed for undeclared task policies."""

    def __init__(self, policies: Iterable[Criteria] = ()) -> None:
        """Creates a registry from complete policy declarations.

        Args:
            policies: The policies to register.
        """
        self.records: dict[tasks.Task, Criteria] = {}
        for policy in policies:
            self.register(policy)

    def register(self, policy: Criteria) -> None:
        """Registers a policy without silently replacing another policy.

        The policy is stored under its task.

        Args:
            policy: The policy to register.

        Raises:
            ConfigError: If the task already has a policy.
        """
        if policy.task in self.records:
            raise errors.ConfigError(
                f"evaluation policy already registered for task "
                f"'{policy.task.value}'"
            )
        self.records[policy.task] = policy

    def require(self, task: tasks.Task) -> Criteria:
        """Returns a task policy or rejects an undeclared task explicitly.

        Args:
            task: The task.

        Returns:
            The task's policy.

        Raises:
            TypeError: If `task` is not a `Task`.
            UndeclaredError: If no policy is declared for the
                task.
        """
        if not isinstance(task, tasks.Task):
            raise errors.KindError("task must be Task")
        policy = self.records.get(task)
        if policy is None:
            raise support.UndeclaredError(
                f"no evaluation policy declared for task '{task.value}'"
            )
        return policy

    def list_policies(self) -> tuple[Criteria, ...]:
        """Returns policies in deterministic task order."""
        return tuple(
            self.records[task]
            for task in sorted(self.records, key=lambda item: item.value)
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns a deterministic JSON-compatible registry record."""
        return {
            "schema_version": Criteria.latest(),
            "policies": [
                policy.model_dump(mode="json")
                for policy in self.list_policies()
            ],
        }

    def canonize(self) -> str:
        """Returns stable JSON for the complete policy registry."""
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
