"""The built-in capability registry.

Every registered kind declares its own capability records
(`kinds.Kind.declare`), and `enrol` enrols them all; the extension
declaration is here too. The planning stage checks every candidate against
this registry.
"""

from scmle.contracts import capabilities
from scmle.pipeline import kinds


def enrol() -> capabilities.Registry:
    """Builds the explicit capability registry of every registered kind.

    Modalities, task/model paths and contract dimensions no kind declares
    are rejected before fitting rather than being inferred from an
    adapter's existence.

    Returns:
        A registry of the records every kind in `kinds.KINDS` declares,
        each with its status, failure modes, evidence links and
        limitations.
    """
    return capabilities.Registry.seed(
        record for kind in kinds.KINDS.values() for record in kind.declare()
    )


def extension_capability(
    key: capabilities.Coordinate,
) -> capabilities.Capability:
    """Describes an explicitly supplied adapter outside the beta guarantee.

    User-registered and test adapters are allowed to exercise the extension
    boundary, but they must never inherit the supported status of a built-in
    family. The plan records this distinction before the adapter is fitted.

    Args:
        key: The capability key of the supplied adapter.

    Returns:
        A capability with extension-only status, no required extras and the
        limitations that the built-in guarantee does not cover it.
    """
    return capabilities.Capability(
        key=key,
        status=capabilities.Status.EXTENSION_ONLY,
        required_extras=(),
        resources=capabilities.Resources(memory_gb=1.0),
        failure_modes=(
            "adapter conformance or dependency failure",
            "resource limits exceeded",
            "missing end-to-end lifecycle evidence",
        ),
        evidence_links=("explicit adapter registration",),
        limitations=(
            "not covered by the built-in production-beta guarantee",
            "model-specific evaluation and recovery evidence is caller-owned",
        ),
    )
