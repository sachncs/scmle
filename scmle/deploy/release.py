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


def read_json(path: pathlib.Path) -> Any:
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


def approval_from_dict(record: Mapping[str, Any]) -> registry.PromotionApproval:
    """Rebuilds a promotion approval from its JSON record.

    Args:
        record: The approval's fields; `evidence` defaults to empty.

    Returns:
        The approval.

    Raises:
        ArtifactError: If the fields do not make a valid approval.
    """
    values = dict(record)
    values["evidence"] = tuple(values.get("evidence", ()))
    try:
        return registry.PromotionApproval(**values)
    except (TypeError, ValueError) as error:
        raise errors.ArtifactError(f"bad approval record: {error}") from error


def observation_from_dict(record: Mapping[str, Any]) -> observation.Observation:
    """Rebuilds an observation from `Observation.to_dict`.

    Args:
        record: A record produced by `Observation.to_dict`.

    Returns:
        The observation.
    """
    return observation.Observation(
        provider=record["provider"],
        service=record["service"],
        revision=record["revision"],
        image_digest=record["image_digest"],
        ready=bool(record["ready"]),
        traffic=tuple(observation.Traffic(**t) for t in record["traffic"]),
        observed_at=record["observed_at"],
        revisions=tuple((a, b) for a, b in record.get("revisions", [])),
    )


def scan_result(
    report: pathlib.Path,
    container_digest: str,
    policy: pathlib.Path | None = None,
) -> gate.ImageScan:
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
            vuln_gate.load_json(report),
            vuln_gate.load_json(policy) if policy else vuln_gate.DEFAULT_POLICY,
        )
    except vuln_gate.GateError:
        return gate.ImageScan(container_digest, False, digest)
    return gate.ImageScan(container_digest, bool(verdict["passed"]), digest)


def load_candidate(manifest_path: str | pathlib.Path) -> gate.Candidate:
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
    manifest = read_json(path)
    if not isinstance(manifest, dict):
        raise errors.ArtifactError(f"{path} is not a release manifest")
    missing = [name for name in REQUIRED if name not in manifest]
    if missing:
        raise errors.ArtifactError("manifest lacks " + ", ".join(missing))
    base = path.parent

    def file(name: str) -> pathlib.Path | None:
        """Resolves an optional evidence file named in the manifest.

        Args:
            name: Manifest key.

        Returns:
            The path relative to the manifest's folder, or None if the manifest
            does not name the key.
        """
        value = manifest.get(name)
        return None if value is None else base / str(value)

    store = registry.ModelRegistryStore(base / manifest["registry"])
    version = store.resolve(manifest["model"], int(manifest["version"]))
    approval_path = file("approval")
    scan_path = file("scan_report")
    checks_path = file("checks")
    seen_path = file("observation")
    staging_path = file("staging_record")
    container = manifest["container_digest"]
    return gate.Candidate(
        environment=manifest["environment"],
        provider=manifest["provider"],
        source_commit=manifest["source_commit"],
        model=version,
        model_digest=gate.model_digest_of(store, version),
        approval=(
            None
            if approval_path is None
            else approval_from_dict(read_json(approval_path))
        ),
        container_digest=container,
        config_digest=manifest["config_digest"],
        requested_by=manifest["requested_by"],
        approver=manifest["approver"],
        approved_at=manifest["approved_at"],
        scan=(
            None
            if scan_path is None
            else scan_result(scan_path, container, file("scan_policy"))
        ),
        checks=(
            ()
            if checks_path is None
            else tuple(
                livetests.CheckResult.from_dict(item)
                for item in read_json(checks_path)
            )
        ),
        observation=(
            None
            if seen_path is None
            else observation_from_dict(read_json(seen_path))
        ),
        staging_record=(
            None
            if staging_path is None
            else scmle.deploy.record.read_record(staging_path)
        ),
    )
