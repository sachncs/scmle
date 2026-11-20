#!/usr/bin/env python3
"""Check the public Python API against its versioned compatibility snapshot."""

import argparse
import enum
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

import scmle
from scmle.assurance.detectors import minhash
from scmle.assurance.ledger.confirm import thresholdout
from scmle.data.modalities.audio import wav
from scmle.data.store import processing
from scmle.data.store import quality
from scmle.data.store import snapshot
from scmle.deploy import contract
from scmle.deploy import control
from scmle.deploy import gate
from scmle.deploy import livetests
from scmle.deploy import observation
from scmle.deploy import record
from scmle.deploy import release_state
from scmle.operations import classify
from scmle.operations import continuous
from scmle.operations import notify
from scmle.operations import recovery
from scmle.operations import signals
from scmle.operations import spool
from scmle.pipeline import kinds
from scmle.pipeline.kinds import audio
from scmle.pipeline.kinds import document
from scmle.pipeline.kinds import geospatial
from scmle.pipeline.kinds import graph
from scmle.pipeline.kinds import image
from scmle.pipeline.kinds import media
from scmle.pipeline.kinds import raster
from scmle.pipeline.kinds import rows
from scmle.pipeline.kinds import video
from scmle.runners import base
from scmle.runners import checkpoint
from scmle.runners import local
from scmle.runners import reproducibility
from scmle.runners import roles
from scmle.runtime import retry
from scmle.security import access
from scmle.security import audit
from scmle.security import tokens
from scmle.storage import objectstore
from scmle.storage import tenancy
import scmle.trail.builder
import scmle.trail.chain
import scmle.trail.show
import scmle.trail.trace
from scmle.workflow import definition
from scmle.workflow import executor
from scmle.workflow import schedule

# Extension surfaces: documented contracts below the top-level API. Their
# signatures are pinned too.
EXTENSION_SURFACES = {
    "scmle.deploy.contract": (
        contract,
        (
            "Architecture",
            "Bundle",
            "Environment",
            "Deployment",
            "Probes",
            "Image",
            "Provider",
            "Resources",
            "Runtime",
            "Secret",
            "Source",
            "audit",
        ),
    ),
    "scmle.deploy.control": (
        control,
        (
            "Polling",
            "Platform",
            "UnconfirmedError",
            "Unsupported",
            "confirm",
            "expecting",
        ),
    ),
    "scmle.deploy.release_state": (
        release_state,
        (
            "Finding",
            "Outcome",
            "Reconciliation",
            "Registry",
            "State",
            "DisagreementError",
        ),
    ),
    "scmle.deploy.gate": (
        gate,
        (
            "Candidate",
            "Decision",
            "Scan",
            "admit",
            "certify",
            "fingerprint",
        ),
    ),
    "scmle.deploy.record": (
        record,
        ("Record", "read", "write"),
    ),
    "scmle.deploy.livetests": (
        livetests,
        ("Check", "Response", "Target", "fetch", "check"),
    ),
    "scmle.deploy.observation": (
        observation,
        ("Observation", "Traffic"),
    ),
    "scmle.security.tokens": (
        tokens,
        (
            "Keystore",
            "Principal",
            "Revocations",
            "TokenError",
            "Verifier",
            "Key",
            "sign",
        ),
    ),
    "scmle.security.access": (
        access,
        (
            "DeniedError",
            "Gateway",
            "Limiter",
            "assemble",
            "insist",
        ),
    ),
    "scmle.security.audit": (audit, ("Log",)),
    "scmle.runtime.retry": (
        retry,
        (
            "Retries",
            "RetryError",
            "TransientError",
            "call",
        ),
    ),
    "scmle.storage.objectstore": (
        objectstore,
        (
            "Disk",
            "Metadata",
            "Depot",
            "Retrier",
            "StoreError",
            "EscapeError",
        ),
    ),
    "scmle.storage.tenancy": (
        tenancy,
        ("Tenancy", "TenantError", "confine", "directory", "qualify"),
    ),
    "scmle.assurance.ledger.confirm.thresholdout": (
        thresholdout,
        ("Answer", "Session", "Terms", "Thresholdout", "minimum"),
    ),
    "scmle.assurance.detectors.minhash": (
        minhash,
        ("Bands", "flags", "pairs", "similarity"),
    ),
    "scmle.data.store.snapshot": (
        snapshot,
        ("Contract", "Log", "Reader", "Record"),
    ),
    "scmle.data.store.quality": (
        quality,
        (
            "Assessment",
            "Severity",
            "Gate",
            "GateError",
            "Check",
            "Stage",
        ),
    ),
    "scmle.runners.base": (
        base,
        (
            "Budget",
            "Rate",
            "Job",
            "Resources",
            "Manifest",
            "Progress",
            "Runner",
            "Exit",
        ),
    ),
    "scmle.runners.local": (local, ("Runner", "Process", "Container")),
    "scmle.runners.roles": (
        roles,
        ("Role", "Store", "check", "separate"),
    ),
    "scmle.runners.checkpoint": (checkpoint, ("Store",)),
    "scmle.runners.reproducibility": (
        reproducibility,
        ("Tolerances", "compare"),
    ),
    "scmle.workflow.definition": (
        definition,
        (
            "Backfill",
            "Kind",
            "Schedule",
            "Step",
            "Workflow",
            "scaffold",
        ),
    ),
    "scmle.workflow.executor": (
        executor,
        ("ApprovalError", "Context", "Executor", "Progress"),
    ),
    "scmle.workflow.schedule": (schedule, ("Scheduler", "Tick")),
    "scmle.pipeline.kinds": (kinds, ("Kind", "Tabular", "pick")),
    "scmle.pipeline.kinds.rows": (rows, ("Cohorts", "Rows", "unite")),
    "scmle.pipeline.kinds.media": (media, ("Description", "Media")),
    "scmle.pipeline.kinds.audio": (audio, ("Audio",)),
    "scmle.pipeline.kinds.document": (document, ("Document",)),
    "scmle.pipeline.kinds.geospatial": (geospatial, ("Geospatial",)),
    "scmle.pipeline.kinds.graph": (graph, ("Graph",)),
    "scmle.pipeline.kinds.image": (image, ("Image",)),
    "scmle.pipeline.kinds.raster": (raster, ("Raster",)),
    "scmle.pipeline.kinds.video": (video, ("Video",)),
    "scmle.data.modalities.audio.wav": (wav, ("pcm",)),
    "scmle.data.store.processing": (
        processing,
        ("Checkpoint", "Processor", "Processing"),
    ),
    "scmle.operations.signals": (
        signals,
        (
            "Category",
            "Monitor",
            "Signal",
            "Status",
            "Thresholds",
            "Window",
        ),
    ),
    "scmle.operations.classify": (
        classify,
        ("Classification", "Verdict", "classify"),
    ),
    "scmle.operations.continuous": (
        continuous,
        (
            "Decision",
            "Evaluation",
            "History",
            "Indicator",
            "Policy",
            "Replacement",
            "reassess",
        ),
    ),
    "scmle.operations.recovery": (
        recovery,
        ("Fault", "Measurement", "Objective", "Outcome", "Runner"),
    ),
    "scmle.operations.spool": (
        spool,
        ("Flush", "Spool"),
    ),
    "scmle.trail.builder": (scmle.trail.builder, ("build",)),
    "scmle.trail.chain": (
        scmle.trail.chain,
        ("Verification", "seal", "verify"),
    ),
    "scmle.trail.show": (scmle.trail.show, ("render",)),
    "scmle.trail.trace": (scmle.trail.trace, ("Finding",)),
    "scmle.operations.notify": (
        notify,
        (
            "Delivery",
            "Notifier",
            "Request",
            "Sender",
            "sign",
            "verify",
        ),
    ),
}


# The names the package root used to re-export, and the user-facing API
# added since, pinned by the module that defines them. The package root
# itself exports nothing.
ROOT_SURFACES = {
    "scmle.api.clustering": (
        "Segmentation",
        "cluster",
        "evaluate",
    ),
    "scmle.api.erasure": (
        "Deletion",
        "Forgetting",
        "Receipt",
        "RetiredError",
        "Retirement",
        "Subject",
    ),
    "scmle.api.non_tabular": (
        "Modeler",
        "Predictor",
        "Scope",
        "UnsupportedError",
    ),
    "scmle.api.predict": (
        "load",
        "predict",
    ),
    "scmle.api.project": (
        "Curator",
        "Modality",
        "Project",
    ),
    "scmle.artifacts.results": ("Run",),
    "scmle.api.sequential": ("Estimation",),
    "scmle.artifacts.bundle": ("Bundle",),
    "scmle.data.catalog": ("Catalog",),
    "scmle.evaluation.longitudinal": (
        "Config",
        "Estimand",
        "Estimation",
        "Estimator",
        "Period",
    ),
    "scmle.evaluation.metrics": ("register",),
    # Constants (FAMILIES, ALPHA, PAIRS, TIMED, ENTITY_AWARE and the
    # calibration constants) have no signature, so only callables are
    # pinned here.
    "scmle.models.forecast": (
        "Boosted",
        "Forecaster",
        "Panel",
        "Seasonal",
        "Smoothing",
    ),
    "scmle.evaluation.forecasting": (
        "Backtest",
        "Score",
        "backtest",
        "contenders",
        "origins",
        "run",
        "spacing",
    ),
    "scmle.data.splitting.splitters.temporal": ("Succession",),
    "scmle.data.splitting.splitters.spatial": ("Tiling",),
    "scmle.evaluation.calibration": (
        "Check",
        "Claim",
        "Group",
        "Interval",
        "Platt",
        "area",
        "bernoulli",
        "bootstrap",
        "chance",
        "crossfit",
        "deviation",
        "effects",
        "gain",
        "horizon",
        "meier",
        "multilabel",
        "ordinal",
        "partition",
        "probable",
        "quartiles",
        "ranking",
        "spread",
        "survival",
        "uplift",
        "vector",
    ),
    "scmle.evaluation.parameter_estimation": (
        "Config",
        "Estimand",
        "Estimate",
        "Estimation",
        "Prior",
        "infer",
    ),
    "scmle.evaluation.sequential": (
        "Config",
        "Diagnostics",
        "Estimand",
        "Estimate",
        "Estimation",
        "Estimator",
    ),
    "scmle.serving.modality": (
        "Decoder",
        "ModalityHttpServing",
    ),
    "scmle.models.custom": (
        "register",
        "unregister",
    ),
    "scmle.pipeline.modality_lifecycle": (
        "AbortedError",
        "Cancellation",
        "Checkpoint",
        "Curation",
        "Curator",
        "Extension",
        "Predictor",
    ),
}


def stable_collection(value: list | tuple | set | frozenset) -> list[Any]:
    """Convert a collection default into a deterministic list.

    Args:
        value: A list, tuple, set or frozenset default.

    Returns:
        The items converted with `stable_value`; sorted when the collection
        is a set or when all its items are enumeration members, otherwise in
        the collection's own order.
    """
    values = [stable_value(item) for item in value]
    if isinstance(value, (set, frozenset)) or all(
        isinstance(item, enum.Enum) for item in value
    ):
        return sorted(values, key=lambda item: json.dumps(item, sort_keys=True))
    return values


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
    if isinstance(value, (list, tuple, set, frozenset)):
        return stable_collection(value)
    if str(value) == "<factory>":
        return "<factory>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def stable_annotation(annotation: Any) -> str:
    """Returns an annotation string stable across supported Python builds.

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
    """Returns a deterministic signature record for one public object.

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
    """Returns the digest of a canonical public signature record.

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
    """Returns the current API in canonical JSON form.

    Returns:
        The schema version, the release and, for every pinned module, the
        digest of each pinned name. The package root exports nothing, so
        every pinned name is read from the module that defines it.
    """
    modules: dict[str, tuple[Any, tuple[str, ...]]] = {}
    for module_name, (module, names) in EXTENSION_SURFACES.items():
        modules[module_name] = (module, tuple(names))
    for module_name, names in ROOT_SURFACES.items():
        module, known = modules.get(
            module_name, (importlib.import_module(module_name), ())
        )
        modules[module_name] = (module, tuple(sorted({*known, *names})))
    surfaces: dict[str, Any] = {}
    for module_name, (module, names) in modules.items():
        surfaces[module_name] = {
            name: {
                "kind": "class"
                if inspect.isclass(getattr(module, name))
                else "callable",
                "signature_sha256": signature_digest(getattr(module, name)),
            }
            for name in sorted(names)
        }
    return {
        "schema_version": 2,
        "release": scmle.__version__,
        "surfaces": surfaces,
    }


def differences(
    expected: dict[str, Any], observed: dict[str, Any]
) -> list[str]:
    """Returns human-readable compatibility differences.

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
    expected_extensions = expected.get("surfaces", {})
    observed_extensions = observed.get("surfaces", {})
    for module_name in sorted(
        set(expected_extensions) | set(observed_extensions)
    ):
        before = expected_extensions.get(module_name, {})
        after = observed_extensions.get(module_name, {})
        for name in sorted(set(before) | set(after)):
            label = f"{module_name}.{name}"
            if name not in after:
                differences_found.append(f"surface removed: {label}")
            elif name not in before:
                differences_found.append(f"surface added: {label}")
            elif before[name] != after[name]:
                differences_found.append(f"surface contract changed: {label}")
    return differences_found


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parses compatibility-check arguments.

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
    """Returns zero only when the current API matches the snapshot.

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
                "surfaces": len(current_snapshot()["surfaces"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
