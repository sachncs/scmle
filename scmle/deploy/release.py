"""Reads a release manifest and its evidence files into a gate `Candidate`.

The release workflow writes one JSON manifest next to the evidence it
collected and calls ``scmle deploy gate``. The manifest names the release
(environment, provider, commit, image digest, model, approver) and points,
relative to its own folder, at the files the gate needs: the registry, the
model approval, the image scan report, the live-check results, the platform
observation and, for production, the staging record. A file that is named and
missing is an error; a file that is not named is evidence not supplied, which
the gate refuses on.
"""

from collections.abc import Mapping
import json
import pathlib
from typing import Any

from scmle.contracts import errors
from scmle.deploy import gate
from scmle.deploy import livetests
from scmle.deploy import observation
from scmle.deploy import provenance
from scmle.deploy import vuln_gate
import scmle.deploy.record
from scmle.governance import registry

REQUIRED = (
    "environment",
    "provider",
    "source_commit",
    "registry",
    "model",
    "version",
    "container_digest",
    "config_digest",
    "requested_by",
    "approver",
    "approved_at",
)


def read(path: pathlib.Path) -> Any:
    """Reads JSON or raises an `ArtifactError` naming the file.

    Args:
        path: JSON file to read.

    Returns:
        The parsed JSON value.

    Raises:
        ArtifactError: If the file cannot be read or parsed.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise errors.ArtifactError(f"cannot read {path}") from error


def revive(record: Mapping[str, Any]) -> registry.Sanction:
    """Rebuilds a promotion approval from its JSON record.

    Args:
        record: The approval's fields, in the written layout (`content`,
            `approved`) or version 1 (`content_id`, `approved_at`);
            `evidence` defaults to empty.

    Returns:
        The approval.

    Raises:
        ArtifactError: If the fields do not make a valid approval.
    """
    try:
        return registry.Sanction.load(record)
    except (TypeError, ValueError) as error:
        raise errors.ArtifactError(f"bad approval record: {error}") from error


def rebuild(record: Mapping[str, Any]) -> observation.Observation:
    """Rebuilds an observation from `Observation.to_dict`.

    Args:
        record: A record produced by `Observation.to_dict`, in the written
            layout (`digests.image`, `observed`) or version 1
            (`image_digest`, `observed_at`).

    Returns:
        The observation.
    """
    if "digests" in record or "observed" in record:
        record = observation.Observation.KEYS.downgrade(dict(record))
    return observation.Observation(
        provider=record["provider"],
        service=record["service"],
        revision=record["revision"],
        image_digest=record["image_digest"],
        ready=bool(record["ready"]),
        traffic=tuple(observation.Traffic(**t) for t in record["traffic"]),
        observed=record["observed_at"],
        revisions=tuple((a, b) for a, b in record.get("revisions", [])),
    )


def triage(
    report: pathlib.Path,
    container_digest: str,
    policy: pathlib.Path | None = None,
) -> gate.Scan:
    """Applies the vulnerability policy to a scanner report.

    A report that cannot be parsed, an empty scan or any blocking finding is
    a failed scan, never a pass. A report file that cannot be opened for
    hashing raises `OSError` instead.

    Args:
        report: Scanner report file.
        container_digest: Digest of the scanned image.
        policy: Policy file; the default policy if None.

    Returns:
        The scan result, carrying the digest of the report file.
    """
    digest = provenance.file_digest(report)
    try:
        verdict = vuln_gate.evaluate(
            vuln_gate.deserialize(report),
            vuln_gate.deserialize(policy)
            if policy
            else vuln_gate.DEFAULT_POLICY,
        )
    except vuln_gate.GateError:
        # handled: unparseable report becomes failed ImageScan, documented
        return gate.Scan(
            digest=container_digest,
            passed=False,
            report=digest,
        )
    return gate.Scan(container_digest, bool(verdict["passed"]), digest)


def locate(
    manifest: Mapping[str, Any], base: pathlib.Path, name: str
) -> pathlib.Path | None:
    """Resolves an optional evidence file named in a release manifest.

    Args:
        manifest: The release manifest.
        base: The manifest's folder.
        name: Manifest key.

    Returns:
        The path relative to the manifest's folder, or None if the manifest
        does not name the key.
    """
    value = manifest.get(name)
    return None if value is None else base / str(value)


def gather(manifest_path: str | pathlib.Path) -> gate.Candidate:
    """Reads a manifest and the evidence it names.

    Args:
        manifest_path: Release manifest; evidence files are named relative to
            its folder.

    Returns:
        The candidate. Evidence that the manifest does not name is left out
        (None or empty) for the gate to refuse.

    Raises:
        ArtifactError: If the manifest or a named file cannot be read, or the
            registry has no such version.
    """
    path = pathlib.Path(manifest_path)
    manifest = read(path)
    if not isinstance(manifest, dict):
        raise errors.ArtifactError(f"{path} is not a release manifest")
    missing = [name for name in REQUIRED if name not in manifest]
    if missing:
        raise errors.ArtifactError("manifest lacks " + ", ".join(missing))
    base = path.parent

    store = registry.Registry(base / manifest["registry"])
    version = store.resolve(manifest["model"], int(manifest["version"]))
    approval_path = locate(manifest, base, "approval")
    scan_path = locate(manifest, base, "scan_report")
    checks_path = locate(manifest, base, "checks")
    seen_path = locate(manifest, base, "observation")
    staging_path = locate(manifest, base, "staging_record")
    container = manifest["container_digest"]
    return gate.Candidate(
        environment=manifest["environment"],
        provider=manifest["provider"],
        commit=manifest["source_commit"],
        model=version,
        reported=gate.fingerprint(store, version),
        approval=(
            None if approval_path is None else revive(read(approval_path))
        ),
        container=container,
        digest=manifest["config_digest"],
        requester=manifest["requested_by"],
        approver=manifest["approver"],
        approved=manifest["approved_at"],
        scan=(
            None
            if scan_path is None
            else triage(
                scan_path, container, locate(manifest, base, "scan_policy")
            )
        ),
        checks=(
            ()
            if checks_path is None
            else tuple(
                livetests.Check.from_dict(item) for item in read(checks_path)
            )
        ),
        observation=(None if seen_path is None else rebuild(read(seen_path))),
        record=(
            None
            if staging_path is None
            else scmle.deploy.record.read(staging_path)
        ),
    )
