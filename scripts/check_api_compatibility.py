#!/usr/bin/env python3
"""Check the public Python API against its versioned compatibility snapshot."""

import argparse
import enum
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import scmle
from scmle.core import retry as core_retry
from scmle.data.store import objectstore as store_objectstore
from scmle.data.store import processing as store_processing
from scmle.data.store import quality as store_quality
from scmle.data.store import snapshot as store_snapshot
from scmle.deploy import contract as deploy_contract
from scmle.deploy import control as deploy_control
from scmle.deploy import gate as deploy_gate
from scmle.deploy import livetests as deploy_livetests
from scmle.deploy import observation as deploy_observation
from scmle.deploy import record as deploy_record
from scmle.governance import release_state as governance_release_state
from scmle.operations import classify as operations_classify
from scmle.operations import continuous as operations_continuous
from scmle.operations import notify as operations_notify
from scmle.operations import recovery as operations_recovery
from scmle.operations import signals as operations_signals
from scmle.operations import spool as operations_spool
from scmle.runners import base as runners_base
from scmle.runners import checkpoint as runners_checkpoint
from scmle.runners import local as runners_local
from scmle.runners import reproducibility as runners_reproducibility
from scmle.runners import roles as runners_roles
from scmle.security import access as security_access
from scmle.security import audit as security_audit
from scmle.security import tokens as security_tokens
import scmle.trail as trail
from scmle.workflow import definition as workflow_definition
from scmle.workflow import executor as workflow_executor
from scmle.workflow import schedule as workflow_schedule

# Extension surfaces: documented contracts below the top-level API. Their
# signatures are pinned too, without changing any `scmle.__all__` entry.
EXTENSION_SURFACES = {
    "scmle.deploy.contract": (
        deploy_contract,
        (
            "Architecture",
            "BundleRef",
            "DeploymentEnvironment",
            "DeploymentSpec",
            "HealthSpec",
            "ImageRef",
            "ProviderAdapter",
            "Resources",
            "RuntimeReport",
            "SecretRef",
            "SecretSource",
            "conformance_problems",
        ),
    ),
    "scmle.deploy.control": (
        deploy_control,
        (
            "ConfirmPolicy",
            "ControlPlane",
            "RuntimeUnconfirmed",
            "Unsupported",
            "confirm",
            "expecting",
        ),
    ),
    "scmle.governance.release_state": (
        governance_release_state,
        (
            "Finding",
            "Outcome",
            "ReconcileReport",
            "ReleaseRegistry",
            "ReleaseState",
            "RuntimeDisagrees",
        ),
    ),
    "scmle.deploy.gate": (
        deploy_gate,
        (
            "Candidate",
            "GateDecision",
            "ImageScan",
            "admit",
            "certify",
            "model_digest_of",
        ),
    ),
    "scmle.deploy.record": (
        deploy_record,
        ("DeploymentRecord", "read_record", "write_record"),
    ),
    "scmle.deploy.livetests": (
        deploy_livetests,
        ("CheckResult", "Response", "Target", "http_fetch", "run_checks"),
    ),
    "scmle.deploy.observation": (
        deploy_observation,
        ("Observation", "Traffic"),
    ),
    "scmle.security.tokens": (
        security_tokens,
        (
            "KeySet",
            "Principal",
            "RevocationList",
            "TokenError",
            "TokenVerifier",
            "VerificationKey",
            "sign_hs256",
        ),
    ),
    "scmle.security.access": (
        security_access,
        (
            "AccessDenied",
            "Gateway",
            "RateLimiter",
            "gateway_from_environment",
            "require_authentication_for",
        ),
    ),
    "scmle.security.audit": (security_audit, ("AuditLog",)),
    "scmle.core.retry": (
        core_retry,
        (
            "RetryPolicy",
            "RetryExhaustedError",
            "TransientError",
            "call_with_retries",
        ),
    ),
    "scmle.data.store.objectstore": (
        store_objectstore,
        (
            "LocalObjectStore",
            "ObjectMeta",
            "ObjectStore",
            "RetryingStore",
            "StoreError",
        ),
    ),
    "scmle.data.store.snapshot": (
        store_snapshot,
        ("DataContract", "ReadLog", "SnapshotReader", "SnapshotRecord"),
    ),
    "scmle.data.store.quality": (
        store_quality,
        (
            "GateReport",
            "Policy",
            "QualityGate",
            "QualityGateError",
            "Rule",
            "Stage",
        ),
    ),
    "scmle.runners.base": (
        runners_base,
        (
            "CostBudget",
            "CostRate",
            "JobSpec",
            "Resources",
            "RunManifest",
            "RunState",
            "Runner",
        ),
    ),
    "scmle.runners.local": (runners_local, ("LocalRunner",)),
    "scmle.runners.roles": (
        runners_roles,
        ("Role", "RoleStore", "check_distinct", "check_separation"),
    ),
    "scmle.runners.checkpoint": (runners_checkpoint, ("CheckpointStore",)),
    "scmle.runners.reproducibility": (
        runners_reproducibility,
        ("Tolerances", "compare"),
    ),
    "scmle.workflow.definition": (
        workflow_definition,
        (
            "CatchUp",
            "Schedule",
            "Step",
            "StepKind",
            "Workflow",
            "standard_workflow",
        ),
    ),
    "scmle.workflow.executor": (
        workflow_executor,
        ("ApprovalError", "LocalExecutor", "RunState", "StepContext"),
    ),
    "scmle.workflow.schedule": (workflow_schedule, ("Scheduler", "TickReport")),
    "scmle.data.store.processing": (
        store_processing,
        ("Checkpoint", "IncrementalProcessor", "RunResult"),
    ),
    "scmle.operations.signals": (
        operations_signals,
        (
            "Category",
            "Signal",
            "SignalMonitor",
            "SignalThresholds",
            "Status",
            "WindowPolicy",
        ),
    ),
    "scmle.operations.classify": (
        operations_classify,
        ("Classification", "Verdict", "classify"),
    ),
    "scmle.operations.continuous": (
        operations_continuous,
        (
            "ContinuousEvaluation",
            "MetricSpec",
            "Replacement",
            "RetrainDecision",
            "RetrainPolicy",
            "TrainingHistory",
            "post_promotion_check",
        ),
    ),
    "scmle.operations.recovery": (
        operations_recovery,
        ("Fault", "Measurement", "Objective", "Result", "Runner"),
    ),
    "scmle.operations.spool": (
        operations_spool,
        ("FlushResult", "MetricsSpool"),
    ),
    "scmle.trail": (
        trail,
        ("Finding", "Report", "build", "render", "seal", "verify"),
    ),
    "scmle.operations.notify": (
        operations_notify,
        (
            "Delivery",
            "Notifier",
            "Request",
            "WebhookSender",
            "sign",
            "verify_signature",
        ),
    ),
}


def stable_value(value: Any) -> Any:
    """Convert a default value into deterministic JSON-native evidence.

    Args:
        value: A parameter default; may be `inspect.Signature.empty`.

    Returns:
        `"<empty>"` for no default; an `{"enum", "value"}` record for an
        enumeration member; a list for a list or tuple (sorted when all items
        are enumeration members); a sorted list for a set; `"<factory>"` for
        a dataclass factory; the value itself for a string, number, boolean
        or None; otherwise its `str`.
    """
    if value is inspect.Signature.empty:
        return "<empty>"
    if isinstance(value, enum.Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": stable_value(value.value),
        }
    if isinstance(value, (list, tuple)):
        values = [stable_value(item) for item in value]
        if all(isinstance(item, enum.Enum) for item in value):
            return sorted(
                values,
                key=lambda item: json.dumps(item, sort_keys=True),
            )
        return values
    if isinstance(value, (set, frozenset)):
        values = [stable_value(item) for item in value]
        return sorted(values, key=lambda item: json.dumps(item, sort_keys=True))
    if str(value) == "<factory>":
        return "<factory>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def stable_annotation(annotation: Any) -> str:
    """Return an annotation string stable across supported Python builds.

    Args:
        annotation: A parameter or return annotation.

    Returns:
        The annotation's text with the platform-specific `pathlib` module
        names replaced by `pathlib.Path`.
    """
    rendered = str(annotation)
    # Python 3.13 exposes the concrete platform-specific pathlib module in
    # ``str(Path)`` while other supported interpreters render ``pathlib``.
    # The private module spelling is not part of the public API contract.
    return rendered.replace("pathlib._local.Path", "pathlib.Path").replace(
        "pathlib._windows.Path", "pathlib.Path"
    )


def signature_record(value: Any) -> dict[str, Any]:
    """Return a deterministic signature record for one public object.

    An enumeration is recorded by its members: `inspect.signature` of an
    `Enum` class differs between Python versions (3.11 and 3.12 describe the
    functional API differently), which made the same code look changed.

    Args:
        value: A public class or callable.

    Returns:
        For an enumeration, its `members` as (name, value) pairs; otherwise
        the `parameters` (name, kind, annotation, default) and the
        `return_annotation`.
    """
    if inspect.isclass(value) and issubclass(value, enum.Enum):
        return {
            "members": [
                [member.name, stable_value(member.value)] for member in value
            ]
        }
    signature = inspect.signature(value)
    parameters = [
        {
            "name": parameter.name,
            "kind": parameter.kind.name,
            "annotation": stable_annotation(parameter.annotation),
            "default": stable_value(parameter.default),
        }
        for parameter in signature.parameters.values()
    ]
    return {
        "parameters": parameters,
        "return_annotation": stable_annotation(signature.return_annotation),
    }


def signature_digest(value: Any) -> str:
    """Return the digest of a canonical public signature record.

    Args:
        value: A public class or callable.

    Returns:
        The hex SHA-256 of the record's canonical JSON.
    """
    content = json.dumps(
        signature_record(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def current_snapshot() -> dict[str, Any]:
    """Return the current top-level API in canonical JSON form.

    Returns:
        The schema version, the release, the digest of each name in
        `scmle.__all__` and the digest of each name of the extension
        surfaces.
    """
    exports: dict[str, Any] = {}
    for name in sorted(scmle.__all__):
        value = getattr(scmle, name)
        if name == "__version__":
            exports[name] = {"kind": "value", "value": value}
        else:
            exports[name] = {
                "kind": "class" if inspect.isclass(value) else "callable",
                "signature_sha256": signature_digest(value),
            }
    extensions: dict[str, Any] = {}
    for module_name, (module, names) in EXTENSION_SURFACES.items():
        extensions[module_name] = {
            name: {
                "kind": "class"
                if inspect.isclass(getattr(module, name))
                else "callable",
                "signature_sha256": signature_digest(getattr(module, name)),
            }
            for name in sorted(names)
        }
    return {
        "schema_version": 1,
        "release": scmle.__version__,
        "exports": exports,
        "extension_surfaces": extensions,
    }


def differences(
    expected: dict[str, Any], observed: dict[str, Any]
) -> list[str]:
    """Return human-readable compatibility differences.

    Args:
        expected: The committed snapshot.
        observed: The snapshot of the current API.

    Returns:
        One message per difference (a changed schema version or release, or
        an export or extension that was added, removed or changed); empty
        when compatible.
    """
    if expected.get("schema_version") != observed.get("schema_version"):
        return ["compatibility snapshot schema version changed"]
    differences_found: list[str] = []
    if expected.get("release") != observed.get("release"):
        differences_found.append(
            f"release changed: {expected.get('release')} -> "
            f"{observed.get('release')}"
        )
    expected_exports = expected.get("exports", {})
    observed_exports = observed.get("exports", {})
    for name in sorted(set(expected_exports) | set(observed_exports)):
        if name not in observed_exports:
            differences_found.append(f"export removed: {name}")
        elif name not in expected_exports:
            differences_found.append(f"export added: {name}")
        elif expected_exports[name] != observed_exports[name]:
            differences_found.append(f"export contract changed: {name}")
    expected_extensions = expected.get("extension_surfaces", {})
    observed_extensions = observed.get("extension_surfaces", {})
    for module_name in sorted(
        set(expected_extensions) | set(observed_extensions)
    ):
        before = expected_extensions.get(module_name, {})
        after = observed_extensions.get(module_name, {})
        for name in sorted(set(before) | set(after)):
            label = f"{module_name}.{name}"
            if name not in after:
                differences_found.append(f"extension removed: {label}")
            elif name not in before:
                differences_found.append(f"extension added: {label}")
            elif before[name] != after[name]:
                differences_found.append(f"extension contract changed: {label}")
    return differences_found


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse compatibility-check arguments.

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        The namespace with the snapshot path and the `write` flag.
    """
    parser = argparse.ArgumentParser(
        description="Check the public API compatibility snapshot."
    )
    parser.add_argument("snapshot", type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite the snapshot from the current API (a deliberate "
        "contract change: add a changelog entry and a migration note)",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Return zero only when the current API matches the snapshot.

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        0 after `--write` or when the API matches; 1 when it differs; 2 when
        the snapshot cannot be read.
    """
    options = parse_arguments(arguments)
    if options.write:
        options.snapshot.write_text(
            json.dumps(current_snapshot(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"wrote {options.snapshot}")
        return 0
    try:
        expected = json.loads(options.snapshot.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"cannot read compatibility snapshot: {error}")
        return 2
    found = differences(expected, current_snapshot())
    if found:
        for difference in found:
            print(difference)
        return 1
    print(
        json.dumps(
            {
                "compatible": True,
                "release": scmle.__version__,
                "exports": len(scmle.__all__),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
