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
from scmle.deploy import contract as deploy_contract

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
}


def stable_value(value: Any) -> Any:
    """Convert a default value into deterministic JSON-native evidence."""
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
    """Return an annotation string stable across supported Python builds."""
    rendered = str(annotation)
    # Python 3.13 exposes the concrete platform-specific pathlib module in
    # ``str(Path)`` while other supported interpreters render ``pathlib``.
    # The private module spelling is not part of the public API contract.
    return rendered.replace("pathlib._local.Path", "pathlib.Path").replace(
        "pathlib._windows.Path", "pathlib.Path"
    )


def signature_record(value: Any) -> dict[str, Any]:
    """Return a deterministic signature record for one public object."""
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
    """Return the digest of a canonical public signature record."""
    content = json.dumps(
        signature_record(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def current_snapshot() -> dict[str, Any]:
    """Return the current top-level API in canonical JSON form."""
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
    """Return human-readable compatibility differences."""
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
    """Parse compatibility-check arguments."""
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
    """Return zero only when the current API matches the snapshot."""
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
