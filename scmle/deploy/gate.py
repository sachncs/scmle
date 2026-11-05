"""The promotion gate: a pure function that allows or refuses a release.

Two questions are asked in two steps, because the evidence for the second
exists only after a deploy:

`admit`
    May this build be deployed to this environment? The model must be signed
    and approved by an independent reviewer for exactly this bundle, the
    image scan must have passed for exactly this image, the approver must be
    someone other than the requester, and, for production, a signed staging
    record must show that the same commit, model and image passed every live
    check in staging.
`certify`
    After the deploy, did it pass? The platform must report the candidate's
    image digest, every live check must be present, passed and bound to this
    commit, image and environment, and only then a signed deployment record
    is produced.

Both return every reason they refuse, not the first. Neither calls anything:
the live checks, the runtime observation and the scan result are inputs, so
the same code runs on recorded cloud responses in tests and on live ones in
the release workflow. Missing evidence is a refusal, never a pass.
"""

from collections.abc import Sequence
import dataclasses
from typing import Any

from scmle.contracts import errors
from scmle.contracts import events
from scmle.deploy import contract
from scmle.deploy import livetests
from scmle.deploy import observation as observation_module
from scmle.deploy import provenance
from scmle.deploy import record as record_module
from scmle.governance import registry

PRODUCTION = "production"
STAGING = "staging"


@dataclasses.dataclass(frozen=True, slots=True)
class ImageScan:
    """The result of scanning one image."""

    container_digest: str
    passed: bool
    report_digest: str


@dataclasses.dataclass(frozen=True, slots=True)
class Candidate:
    """Everything the gate looks at for one release to one environment.

    Attributes:
        environment: ``staging`` or ``production``.
        provider: ``aws`` or ``gcp``.
        source_commit: Commit the image was built from.
        model: The registered version being released.
        model_digest: Digest the service reports for this model; compute it
            with `model_digest_of`.
        approval: The independent promotion review of this exact bundle.
        container_digest: Digest of the image being deployed.
        config_digest: Digest of the deployment configuration.
        requested_by: Who started the release.
        approver: Who approved it in the protected environment.
        approved_at: UTC time of that approval.
        scan: The image scan result.
        checks: Live check results (needed by `certify`).
        observation: What the platform reports as running (for `certify`).
        staging_record: The signed staging record (needed for production).
    """

    environment: str
    provider: str
    source_commit: str
    model: registry.VersionRecord
    model_digest: str
    approval: registry.PromotionApproval | None
    container_digest: str
    config_digest: str
    requested_by: str
    approver: str
    approved_at: str
    scan: ImageScan | None = None
    checks: Sequence[livetests.CheckResult] = ()
    observation: observation_module.Observation | None = None
    staging_record: record_module.DeploymentRecord | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class GateDecision:
    """The outcome: allowed or not, every reason, and the record if any."""

    allowed: bool
    reasons: tuple[str, ...]
    record: record_module.DeploymentRecord | None = None

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-ready record."""
        return {
            "allowed": self.allowed,
            "reasons": list(self.reasons),
            "record": None if self.record is None else self.record.to_dict(),
        }


def model_digest_of(
    store: registry.ModelRegistryStore, version: registry.VersionRecord
) -> str:
    """The digest a service running this registered bundle reports."""
    return provenance.bundle_digest(store.root / version.path)


def model_problems(candidate: Candidate, key: str | bytes) -> list[str]:
    """Why the model may not be released, empty if it may."""
    problems = []
    if not candidate.model.signed:
        problems.append("the model artifact is unsigned")
    approval = candidate.approval
    if approval is None:
        problems.append("the model has no approval")
    else:
        try:
            approval.validate_target(
                candidate.model.model,
                candidate.model.version,
                candidate.model.content_id,
                key,
            )
        except (TypeError, ValueError) as error:
            problems.append(f"the model approval is not valid: {error}")
    try:
        contract.require_digest(candidate.model_digest, "model_digest")
    except errors.ConfigError as error:
        problems.append(str(error))
    return problems


def admit_problems(candidate: Candidate, key: str | bytes) -> list[str]:
    """Reasons the build may not be deployed to the environment."""
    problems = model_problems(candidate, key)
    if candidate.environment not in record_module.ENVIRONMENTS:
        problems.append(f"unknown environment {candidate.environment!r}")
    if record_module.COMMIT_PATTERN.fullmatch(candidate.source_commit) is None:
        problems.append("the source commit is not a full commit hash")
    scan = candidate.scan
    if scan is None:
        problems.append("the image has no scan result")
    elif scan.container_digest != candidate.container_digest:
        problems.append("the scan result is for a different image")
    elif not scan.passed:
        problems.append("the image scan did not pass")
    if not candidate.approver.strip():
        problems.append("no approver is recorded")
    elif candidate.approver == candidate.requested_by:
        problems.append("the approver is the person who requested the release")
    if candidate.environment == PRODUCTION:
        problems.extend(staging_problems(candidate, key))
    return problems


def staging_problems(candidate: Candidate, key: str | bytes) -> list[str]:
    """Reasons the staging record does not support a production release."""
    staged = candidate.staging_record
    if staged is None:
        return ["production needs a staging deployment record"]
    problems = []
    if not staged.verify(key):
        problems.append("the staging record's signature is not valid")
    if staged.environment != STAGING:
        problems.append("the staging record is not for staging")
    expected = {
        "source_commit": candidate.source_commit,
        "model_digest": candidate.model_digest,
        "container_digest": candidate.container_digest,
        "provider": candidate.provider,
    }
    for name, value in expected.items():
        if getattr(staged, name) != value:
            problems.append(f"the staging record has a different {name}")
    shown = dict(staged.evidence)
    missing = [n for n in livetests.CHECK_NAMES if n not in shown]
    if missing:
        problems.append("staging evidence is missing: " + ", ".join(missing))
    return problems


def check_problems(candidate: Candidate) -> list[str]:
    """Reasons the live checks do not support the release."""
    problems = []
    results = {item.name: item for item in candidate.checks}
    for name in livetests.CHECK_NAMES:
        result = results.get(name)
        if result is None:
            problems.append(f"live check {name} is missing")
            continue
        if not result.passed:
            detail = "; ".join(result.problems)
            problems.append(f"live check {name} failed: {detail}")
        bound = (
            result.environment == candidate.environment
            and result.source_commit == candidate.source_commit
            and result.container_digest == candidate.container_digest
            and result.model_digest == candidate.model_digest
        )
        if not bound:
            problems.append(f"live check {name} was run for a different build")
    return problems


def runtime_problems(candidate: Candidate) -> list[str]:
    """Reasons the platform does not show the candidate running."""
    seen = candidate.observation
    if seen is None:
        return ["the platform's running state was not read"]
    problems = []
    if seen.provider != candidate.provider:
        problems.append("the observation is from another provider")
    if seen.image_digest != candidate.container_digest:
        problems.append(
            f"the platform runs {seen.image_digest}, "
            f"not {candidate.container_digest}"
        )
    if not seen.ready:
        problems.append("the platform does not report the service ready")
    return problems


def admit(candidate: Candidate, key: str | bytes) -> GateDecision:
    """Decides whether the build may be deployed.

    Args:
        candidate: The release; `checks` and `observation` are not used.
        key: The HMAC key that verifies approvals and records.

    Returns:
        The decision with every reason for a refusal.
    """
    problems = admit_problems(candidate, key)
    return GateDecision(not problems, tuple(problems))


def certify(
    candidate: Candidate, key: str | bytes, now: str | None = None
) -> GateDecision:
    """Decides whether the deploy passed, and signs its record if so.

    Args:
        candidate: The release with checks and the platform observation.
        key: The HMAC key that verifies approvals and signs the record.
        now: UTC time for the record (default: the current time).

    Returns:
        The decision; when allowed, it carries the signed record.
    """
    problems = admit_problems(candidate, key)
    problems.extend(check_problems(candidate))
    problems.extend(runtime_problems(candidate))
    seen = candidate.observation
    if problems or seen is None:
        return GateDecision(False, tuple(problems))
    results = {item.name: item.evidence_digest for item in candidate.checks}
    unsigned = record_module.DeploymentRecord(
        source_commit=candidate.source_commit,
        model=candidate.model.model,
        model_version=candidate.model.version,
        model_digest=candidate.model_digest,
        container_digest=candidate.container_digest,
        config_digest=candidate.config_digest,
        environment=candidate.environment,
        provider=candidate.provider,
        requested_by=candidate.requested_by,
        approver=candidate.approver,
        approved_at=candidate.approved_at,
        evidence=tuple(sorted(results.items())),
        revision=seen.revision,
        recorded_at=now or events.utc_timestamp(),
    )
    return GateDecision(True, (), unsigned.signed(key))
