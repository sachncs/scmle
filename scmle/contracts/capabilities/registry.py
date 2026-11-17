"""The registry that resolves complete capabilities before training."""

from collections.abc import Collection
from collections.abc import Iterable
import json
from typing import NoReturn

from scmle.contracts import errors
from scmle.contracts.capabilities import axes
from scmle.contracts.capabilities import declarations

CAPABILITY_SCHEMA_VERSION = 1
"""Schema version for serialized capability-registry evidence."""


class Registry:
    """Registry that resolves complete capabilities before training.

    The registry has no implicit defaults. Callers must register an explicit
    declaration for every combination they intend to train, which prevents a
    model adapter's existence from being mistaken for end-to-end support.

    Attributes:
        records: Registered capabilities keyed by their complete coordinate.
        model_families: Registered family declarations keyed by family name.
        runtime_environment: The bound runtime environment, or None until
            :meth:`bind` is called.
    """

    def __init__(
        self,
        capabilities: Iterable[declarations.Capability] = (),
        model_families: Iterable[declarations.Declaration] = (),
    ) -> None:
        """Creates a registry from complete capability declarations.

        Args:
            capabilities: The capabilities to register.
            model_families: The model-family declarations to register.
        """
        self.records: dict[
            declarations.Coordinate, declarations.Capability
        ] = {}
        self.model_families: dict[str, declarations.Declaration] = {}
        self.runtime_environment: declarations.Environment | None = None
        for capability in capabilities:
            self.register(capability)
        for declaration in model_families:
            self.enroll(declaration)

    @classmethod
    def seed(
        cls,
        capabilities: Iterable[declarations.Capability] = (),
    ) -> "Registry":
        """Creates a registry with the explicit transformer/JEPA catalog.

        The catalog is metadata, not support promotion. Callers still need a
        complete capability declaration for every task and contract dimension
        before :meth:`require` can succeed.

        Args:
            capabilities: The capabilities to register.

        Returns:
            The registry, with the default model-family declarations.
        """
        return cls(capabilities, declarations.DEFAULT_MODEL_FAMILY_DECLARATIONS)

    def register(self, capability: declarations.Capability) -> None:
        """Registers a capability without silently replacing an existing one.

        Args:
            capability: Complete capability declaration.

        Raises:
            TypeError: If `capability` is not a `Capability`.
            errors.ConfigError: If the key is already registered.
        """
        if not isinstance(capability, declarations.Capability):
            raise errors.KindError("capability must be Capability")
        if capability.key in self.records:
            raise errors.ConfigError(
                f"capability already registered: {capability.key.to_dict()}"
            )
        self.records[capability.key] = capability

    def enroll(
        self,
        declaration: declarations.Declaration,
    ) -> None:
        """Registers a model-family status without replacing another one.

        Args:
            declaration: The family declaration.

        Raises:
            ConfigError: If the family is already registered.
            TypeError: If `declaration` is not a `Declaration`.
        """
        if not isinstance(declaration, declarations.Declaration):
            raise errors.KindError("declaration must be ModelFamilyDeclaration")
        if declaration.model_family in self.model_families:
            raise errors.ConfigError(
                f"model family already registered: {declaration.model_family}"
            )
        self.model_families[declaration.model_family] = declaration

    def find(
        self,
        model_family: str,
    ) -> declarations.Declaration | None:
        """Returns family metadata or ``None`` when it is not catalogued.

        Args:
            model_family: The family name.

        Returns:
            The family declaration, or None.

        Raises:
            TypeError: If `model_family` is not a string.
        """
        if not isinstance(model_family, str):
            raise errors.KindError("model_family must be a string")
        return self.model_families.get(model_family)

    def bind(self, environment: declarations.Environment) -> None:
        """Binds observed facts used when a caller omits an environment.

        Args:
            environment: The observed runtime environment.

        Raises:
            TypeError: If `environment` is not a `Environment`.
        """
        if not isinstance(environment, declarations.Environment):
            raise errors.KindError("environment must be Environment")
        self.runtime_environment = environment

    def diagnose(
        self,
        key: declarations.Coordinate,
        allowed_statuses: Collection[axes.Status] | None = None,
        *,
        environment: declarations.Environment | None = None,
        row_count: int | None = None,
    ) -> tuple[declarations.Reason, ...]:
        """Returns every pre-training rejection for a capability request.

        Diagnostics are collected rather than short-circuited so a caller can
        fix one configuration or environment and see the remaining blockers
        in the same response. An empty tuple is an affirmative decision.

        Args:
            key: Complete capability request.
            allowed_statuses: Statuses the caller accepts; only ``supported``
                if None.
            environment: Observed extras and resources; the bound runtime
                environment if None.
            row_count: Dataset size used to check a declared row limit.

        Returns:
            The diagnostics. A key with no declaration yields a missing
            declaration entry plus, when its model family is catalogued, an
            entry for an unsupported modality or a disallowed family status.
            A declared key yields an entry for a disallowed status and the
            resource diagnostics. Resource checks run only when an
            environment is available, except that the row limit is checked
            whenever a row count is given or the declaration has a limit.

        Raises:
            TypeError: If `key`, `environment` or a status has the wrong type.
        """
        if not isinstance(key, declarations.Coordinate):
            raise errors.KindError("key must be Coordinate")
        if environment is not None and not isinstance(
            environment, declarations.Environment
        ):
            raise errors.KindError("environment must be Environment")
        if environment is None:
            environment = self.runtime_environment
        statuses = (
            frozenset({axes.Status.SUPPORTED})
            if allowed_statuses is None
            else frozenset(allowed_statuses)
        )
        if any(not isinstance(status, axes.Status) for status in statuses):
            raise errors.KindError("allowed_statuses must contain Status")

        capability = self.records.get(key)
        if capability is None:
            return self.reject(key, statuses)

        diagnostics: list[declarations.Reason] = []
        if capability.status not in statuses:
            diagnostics.append(
                declarations.Reason(
                    code="status_not_allowed",
                    message=(
                        f"registered as {capability.status.value}; accepted "
                        "statuses do not include it"
                    ),
                    expected=sorted(status.value for status in statuses),
                    observed=capability.status.value,
                )
            )
        if environment is not None:
            diagnostics.extend(self.audit(capability, environment, row_count))
        elif row_count is not None or capability.resources.max_rows is not None:
            diagnostics.extend(self.audit(capability, None, row_count))
        return tuple(diagnostics)

    def reject(
        self, key: declarations.Coordinate, statuses: frozenset[axes.Status]
    ) -> tuple[declarations.Reason, ...]:
        """Returns the rejections for a request that has no declaration.

        Args:
            key: Complete capability request.
            statuses: Statuses the caller accepts.

        Returns:
            A missing declaration entry plus, when the model family is
            catalogued, an entry for an unsupported modality or a disallowed
            family status.
        """
        family = self.find(key.model_family)
        family_status = None if family is None else family.status.value
        family_diagnostics = [
            declarations.Reason(
                code="missing_declaration",
                message=("no declaration exists for this complete capability"),
                expected=key.to_dict(),
                observed={"model_family_status": family_status},
            )
        ]
        if family is not None and not family.accommodates(key.modality):
            family_diagnostics.append(
                declarations.Reason(
                    code="model_family_modality_unsupported",
                    message=(
                        "model family is not declared for the requested "
                        "modality"
                    ),
                    expected=[item.value for item in family.modalities],
                    observed=key.modality.value,
                )
            )
        if family is not None and family.status not in statuses:
            family_diagnostics.append(
                declarations.Reason(
                    code="model_family_status_not_allowed",
                    message=(
                        f"model family is registered as "
                        f"{family.status.value}; accepted statuses do "
                        "not include it"
                    ),
                    expected=sorted(status.value for status in statuses),
                    observed=family.status.value,
                )
            )
        return tuple(family_diagnostics)

    def audit(
        self,
        capability: declarations.Capability,
        environment: declarations.Environment | None,
        row_count: int | None = None,
    ) -> tuple[declarations.Reason, ...]:
        """Checks extras and resource assumptions without starting work.

        Args:
            capability: The declared capability.
            environment: Observed extras and resources; the bound runtime
                environment if None.
            row_count: Dataset size used to check a declared row limit.

        Returns:
            A diagnostic for each missing extra and unmet numeric, GPU or
            network assumption, which are checked only when an environment is
            available. If the capability declares a row limit, a diagnostic
            is also returned for an exceeded limit or an unknown row count.

        Raises:
            TypeError: If `capability` or `environment` has the wrong type.
        """
        if not isinstance(capability, declarations.Capability):
            raise errors.KindError("capability must be Capability")
        if environment is not None and not isinstance(
            environment, declarations.Environment
        ):
            raise errors.KindError("environment must be Environment")
        if environment is None:
            environment = self.runtime_environment
        diagnostics: list[declarations.Reason] = []
        if environment is not None:
            missing_extras = sorted(
                set(capability.required_extras)
                - set(environment.available_extras)
            )
            if missing_extras:
                diagnostics.append(
                    declarations.Reason(
                        code="missing_extra",
                        message=(
                            "required optional dependency extra is unavailable"
                        ),
                        expected=missing_extras,
                        observed=sorted(environment.available_extras),
                    )
                )
            diagnostics.extend(self.quantify(capability, environment))
        if capability.resources.max_rows is not None:
            if row_count is None:
                diagnostics.append(
                    declarations.Reason(
                        code="unknown_row_count",
                        message=(
                            "row count is required for this capability limit"
                        ),
                        expected=capability.resources.max_rows,
                    )
                )
            elif row_count > capability.resources.max_rows:
                diagnostics.append(
                    declarations.Reason(
                        code="row_limit_exceeded",
                        message=(
                            "requested dataset exceeds capability row limit"
                        ),
                        expected=capability.resources.max_rows,
                        observed=row_count,
                    )
                )
        return tuple(diagnostics)

    def quantify(
        self,
        capability: declarations.Capability,
        environment: declarations.Environment,
    ) -> tuple[declarations.Reason, ...]:
        """Checks CPU, memory, disk, GPU, and network assumptions.

        Args:
            capability: The declared capability.
            environment: Observed resources.

        Returns:
            A diagnostic for each assumption that is unobserved or not met.
        """
        requirements = capability.resources
        diagnostics: list[declarations.Reason] = []
        numeric_values = (
            ("cpu_cores", requirements.cpu_cores, environment.cpu_cores),
            ("memory_gb", requirements.memory_gb, environment.memory_gb),
            ("disk_gb", requirements.disk_gb, environment.disk_gb),
        )
        for name, expected, observed in numeric_values:
            if observed is None:
                diagnostics.append(
                    declarations.Reason(
                        code=f"unknown_{name}",
                        message=f"{name} is required to verify resources",
                        expected=expected,
                    )
                )
            elif observed < expected:
                diagnostics.append(
                    declarations.Reason(
                        code=f"insufficient_{name}",
                        message=f"runtime {name} is below the declared minimum",
                        expected=expected,
                        observed=observed,
                    )
                )
        boolean_values = (
            (
                "gpu_available",
                requirements.gpu_required,
                environment.gpu_available,
                "GPU",
            ),
            (
                "network_available",
                requirements.network_required,
                environment.network_available,
                "network",
            ),
        )
        for name, required, observed, label in boolean_values:
            if not required:
                continue
            if observed is None:
                diagnostics.append(
                    declarations.Reason(
                        code=f"unknown_{name}",
                        message=(
                            f"{label} availability is required to verify "
                            "resources"
                        ),
                        expected=True,
                    )
                )
            elif not observed:
                diagnostics.append(
                    declarations.Reason(
                        code=f"missing_{name}",
                        message=f"declared {label} resource is unavailable",
                        expected=True,
                        observed=False,
                    )
                )
        return tuple(diagnostics)

    def require(
        self,
        key: declarations.Coordinate,
        allowed_statuses: Collection[axes.Status] | None = None,
        *,
        environment: declarations.Environment | None = None,
        row_count: int | None = None,
    ) -> declarations.Capability:
        """Requires a capability to be eligible before training begins.

        By default only ``supported`` declarations cross the boundary.
        Experimental or extension-only declarations may be used only when the
        caller explicitly includes their status in ``allowed_statuses``. A
        request with any diagnostic is refused with
        `UnsupportedError` (raised by `refuse`) whose
        message is the first diagnostic. `TypeError` from `diagnose` for
        arguments of the wrong type propagates unchanged.

        Args:
            key: Complete capability request.
            allowed_statuses: Explicit statuses accepted by the caller.
            environment: Observed optional extras and runtime resources.
            row_count: Dataset size used to check a declared row limit.

        Returns:
            The registered capability declaration.

        Raises:
            ConfigError: Defensive guard that cannot trigger in practice,
                since an unregistered key always produces a diagnostic.
        """
        capability = self.records.get(key)
        diagnostics = self.diagnose(
            key,
            allowed_statuses,
            environment=environment,
            row_count=row_count,
        )
        if diagnostics:
            self.refuse(
                key,
                diagnostics[0].message,
                diagnostics,
            )
        if capability is None:
            raise errors.ConfigError("capability resolution failed")
        return capability

    def catalogue(self) -> tuple[declarations.Capability, ...]:
        """Returns declarations in deterministic key order."""
        return tuple(
            self.records[key]
            for key in sorted(
                self.records,
                key=lambda item: tuple(item.to_dict().values()),
            )
        )

    def list_model_families(self) -> tuple[declarations.Declaration, ...]:
        """Returns model-family declarations in deterministic order."""
        return tuple(
            self.model_families[name] for name in sorted(self.model_families)
        )

    def to_dict(self) -> dict[str, object]:
        """Returns a deterministic JSON-compatible registry record."""
        return {
            "schema_version": CAPABILITY_SCHEMA_VERSION,
            "model_families": [
                declaration.to_dict()
                for declaration in self.list_model_families()
            ],
            "capabilities": [
                capability.to_dict() for capability in self.catalogue()
            ],
        }

    def canonize(self) -> str:
        """Returns stable JSON suitable for hashing and release evidence."""
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def to_json(self) -> str:
        """Returns the canonical serialized registry."""
        return self.canonize()

    def refuse(
        self,
        key: declarations.Coordinate,
        reason: str,
        diagnostics: Iterable[declarations.Reason] = (),
    ) -> NoReturn:
        """Raises a structured pre-training rejection for ``key``.

        Args:
            key: The capability request that was refused.
            reason: Why it was refused.
            diagnostics: The diagnostics behind the refusal.

        Raises:
            UnsupportedError: Always, with details naming the key, the
                reason, the diagnostics and the registered keys.
        """
        details = {
            "status": "unsupported",
            "reason": reason,
            "key": key.to_dict(),
            "diagnostics": [diagnostic.to_dict() for diagnostic in diagnostics],
            "registered_keys": [
                item.key.to_dict() for item in self.catalogue()
            ],
        }
        raise declarations.UnsupportedError(
            f"unsupported capability before training: {reason}; "
            f"key={key.to_dict()}",
            details,
        )
