"""Registered capability declarations and their resource checks."""

from collections.abc import Iterable

from scmle.contracts import errors
from scmle.contracts.capabilities import declarations


class Catalog:
    """Registered declarations and the resource checks a declaration makes.

    The catalog has no implicit defaults: it holds exactly the capability
    and model-family declarations registered with it, and checks a declared
    capability's extras and resource assumptions against an environment.

    Attributes:
        records: Registered capabilities keyed by their complete coordinate.
        families: Registered family declarations keyed by family name.
        environment: The bound runtime environment, or None until
            :meth:`bind` is called.
    """

    def __init__(
        self,
        capabilities: Iterable[declarations.Capability] = (),
        families: Iterable[declarations.Declaration] = (),
    ) -> None:
        """Creates a registry from complete capability declarations.

        Args:
            capabilities: The capabilities to register.
            families: The model-family declarations to register.
        """
        self.records: dict[
            declarations.Coordinate, declarations.Capability
        ] = {}
        self.families: dict[str, declarations.Declaration] = {}
        self.environment: declarations.Environment | None = None
        for capability in capabilities:
            self.register(capability)
        for declaration in families:
            self.enroll(declaration)

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
        if declaration.family in self.families:
            raise errors.ConfigError(
                f"model family already registered: {declaration.family}"
            )
        self.families[declaration.family] = declaration

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
        return self.families.get(model_family)

    def bind(self, environment: declarations.Environment) -> None:
        """Binds observed facts used when a caller omits an environment.

        Args:
            environment: The observed runtime environment.

        Raises:
            TypeError: If `environment` is not a `Environment`.
        """
        if not isinstance(environment, declarations.Environment):
            raise errors.KindError("environment must be Environment")
        self.environment = environment

    def catalogue(self) -> tuple[declarations.Capability, ...]:
        """Returns declarations in deterministic key order."""
        return tuple(
            self.records[key]
            for key in sorted(
                self.records,
                key=lambda item: tuple(item.to_dict().values()),
            )
        )

    def inventory(self) -> tuple[declarations.Declaration, ...]:
        """Returns model-family declarations in deterministic order."""
        return tuple(self.families[name] for name in sorted(self.families))

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
            environment = self.environment
        diagnostics: list[declarations.Reason] = []
        if environment is not None:
            missing_extras = sorted(
                set(capability.extras) - set(environment.extras)
            )
            if missing_extras:
                diagnostics.append(
                    declarations.Reason(
                        code="missing_extra",
                        message=(
                            "required optional dependency extra is unavailable"
                        ),
                        expected=missing_extras,
                        observed=sorted(environment.extras),
                    )
                )
            diagnostics.extend(self.quantify(capability, environment))
        if capability.resources.rows is not None:
            if row_count is None:
                diagnostics.append(
                    declarations.Reason(
                        code="unknown_row_count",
                        message=(
                            "row count is required for this capability limit"
                        ),
                        expected=capability.resources.rows,
                    )
                )
            elif row_count > capability.resources.rows:
                diagnostics.append(
                    declarations.Reason(
                        code="row_limit_exceeded",
                        message=(
                            "requested dataset exceeds capability row limit"
                        ),
                        expected=capability.resources.rows,
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
            ("cpu_cores", requirements.cores, environment.cores),
            ("memory_gb", requirements.memory, environment.memory),
            ("disk_gb", requirements.disk, environment.disk),
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
                requirements.required,
                environment.available,
                "GPU",
            ),
            (
                "network_available",
                requirements.network,
                environment.online,
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
