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
from scmle.deploy import provenance
import scmle.deploy.observation
import scmle.deploy.record
from scmle.governance import registry

PRODUCTION = "production"
STAGING = "staging"


@dataclasses.dataclass(frozen=True, slots=True)
class Scan:
    """The result of scanning one image.

    Attributes:
        container_digest: Digest of the image that was scanned.
        passed: Whether the scan passed.
        report_digest: Digest of the scan report.
    """

    digest: str
    passed: bool
    report: str


@dataclasses.dataclass(frozen=True, slots=True)
class Candidate:
    """Everything the gate looks at for one release to one environment.

    Attributes:
        environment: ``staging`` or ``production``.
        provider: ``aws`` or ``gcp``.
        commit: Commit the image was built from.
        model: The registered version being released.
        reported: Digest the service reports for this model; compute it
            with `fingerprint`.
        approval: The independent promotion review of this exact bundle.
        container: Digest of the image being deployed.
        digest: Digest of the deployment configuration.
        requester: Who started the release.
        approver: Who approved it in the protected environment.
        approved: UTC time of that approval.
        scan: The image scan result.
        checks: Live check results (needed by `certify`).
        observation: What the platform reports as running (for `certify`).
        record: The signed staging record (needed for production).
    """

    environment: str
    provider: str
    commit: str
    model: registry.Release
    reported: str
    approval: registry.Sanction | None
    container: str
    digest: str
    requester: str
    approver: str
    approved: str
    scan: Scan | None = None
    checks: Sequence[livetests.Check] = ()
    observation: scmle.deploy.observation.Observation | None = None
    record: scmle.deploy.record.Record | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Decision:
    """The outcome: allowed or not, every reason, and the record if any.

    Attributes:
        allowed: Whether the release may proceed.
        reasons: Every reason for a refusal; empty when allowed.
        record: The signed deployment record, present only when `certify`
            allowed the release.
    """

    allowed: bool
    reasons: tuple[str, ...]
    record: scmle.deploy.record.Record | None = None

    def to_dict(self) -> dict[str, Any]:
        """Builds a JSON-ready record.

        Returns:
            A dict with `allowed`, `reasons` and `record` (a dict or None).
        """
        return {
            "allowed": self.allowed,
            "reasons": list(self.reasons),
            "record": None if self.record is None else self.record.to_dict(),
        }


def fingerprint(store: registry.Registry, version: registry.Release) -> str:
    """Computes the digest a service running this registered bundle reports.

    Args:
        store: The model registry holding the bundle.
        version: The registered version.

    Returns:
        The bundle digest of the version's directory in the registry.
    """
    return provenance.digest(store.root / version.path)


def veto(candidate: Candidate, key: str | bytes) -> list[str]:
    """Lists why the model may not be released, empty if it may.

    Args:
        candidate: The release.
        key: The HMAC key that verifies the approval.

    Returns:
        One message per problem with the model's signature, approval or digest.
    """
    problems = []
    if not candidate.model.signed:
        problems.append("the model artifact is unsigned")
    approval = candidate.approval
    if approval is None:
        problems.append("the model has no approval")
    else:
        try:
            approval.validate(
                candidate.model.model,
                candidate.model.version,
                candidate.model.content_id,
                key,
            )
        except (TypeError, ValueError) as error:
            # handled: returned as failed finding/problem/refusal with error
            # text
            problems.append(f"the model approval is not valid: {error}")
    try:
        contract.require(candidate.reported, "model_digest")
    except errors.ConfigError as error:
        # handled: returned as failed finding/problem/refusal with error text
        problems.append(str(error))
    return problems


def objections(candidate: Candidate, key: str | bytes) -> list[str]:
    """Lists the reasons the build may not be deployed to the environment.

    Args:
        candidate: The release.
        key: The HMAC key that verifies approvals and records.

    Returns:
        One message per problem; empty if the build may be deployed.
    """
    problems = veto(candidate, key)
    if candidate.environment not in scmle.deploy.record.ENVIRONMENTS:
        problems.append(f"unknown environment {candidate.environment!r}")
    if scmle.deploy.record.COMMIT_PATTERN.fullmatch(candidate.commit) is None:
        problems.append("the source commit is not a full commit hash")
    scan = candidate.scan
    if scan is None:
        problems.append("the image has no scan result")
    elif scan.digest != candidate.container:
        problems.append("the scan result is for a different image")
    elif not scan.passed:
        problems.append("the image scan did not pass")
    if not candidate.approver.strip():
        problems.append("no approver is recorded")
    elif candidate.approver == candidate.requester:
        problems.append("the approver is the person who requested the release")
    if candidate.environment == PRODUCTION:
        problems.extend(audit(candidate, key))
    return problems


def audit(candidate: Candidate, key: str | bytes) -> list[str]:
    """Lists the reasons the staging record does not support production.

    Args:
        candidate: The release; its `staging_record` is examined.
        key: The HMAC key that verifies the record.

    Returns:
        One message per problem; empty if the record supports the release.
    """
    staged = candidate.record
    if staged is None:
        return ["production needs a staging deployment record"]
    problems = []
    if not staged.verify(key):
        problems.append("the staging record's signature is not valid")
    if staged.environment != STAGING:
        problems.append("the staging record is not for staging")
    expected = {
        "commit": (staged.commit, candidate.commit),
        "model digest": (staged.digests["model"], candidate.reported),
        "container digest": (
            staged.digests["container"],
            candidate.container,
        ),
        "provider": (staged.provider, candidate.provider),
    }
    for name, (recorded, value) in expected.items():
        if recorded != value:
            problems.append(f"the staging record has a different {name}")
    shown = dict(staged.evidence)
    missing = [n for n in livetests.CHECK_NAMES if n not in shown]
    if missing:
        problems.append("staging evidence is missing: " + ", ".join(missing))
    return problems


def vet(candidate: Candidate) -> list[str]:
    """Lists the reasons the live checks do not support the release.

    Args:
        candidate: The release; its `checks` are examined.

    Returns:
        One message per missing, failed or mis-bound check; empty if all
        required checks passed for this build.
    """
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
            and result.commit == candidate.commit
            and result.digests["container"] == candidate.container
            and result.digests["model"] == candidate.reported
        )
        if not bound:
            problems.append(f"live check {name} was run for a different build")
    return problems


def absences(candidate: Candidate) -> list[str]:
    """Lists the reasons the platform does not show the candidate running.

    Args:
        candidate: The release; its `observation` is examined.

    Returns:
        One message per problem; empty if the platform reports the candidate's
        image, from the same provider, ready.
    """
    seen = candidate.observation
    if seen is None:
        return ["the platform's running state was not read"]
    problems = []
    if seen.provider != candidate.provider:
        problems.append("the observation is from another provider")
    if seen.image_digest != candidate.container:
        problems.append(
            f"the platform runs {seen.image_digest}, not {candidate.container}"
        )
    if not seen.ready:
        problems.append("the platform does not report the service ready")
    return problems


def admit(candidate: Candidate, key: str | bytes) -> Decision:
    """Decides whether the build may be deployed.

    Args:
        candidate: The release; `checks` and `observation` are not used.
        key: The HMAC key that verifies approvals and records.

    Returns:
        The decision with every reason for a refusal.
    """
    problems = objections(candidate, key)
    return Decision(not problems, tuple(problems))


def certify(
    candidate: Candidate, key: str | bytes, now: str | None = None
) -> Decision:
    """Decides whether the deploy passed, and signs its record if so.

    All admission, live-check and runtime problems are collected first.

    Args:
        candidate: The release with checks and the platform observation.
        key: The HMAC key that verifies approvals and signs the record.
        now: UTC time for the record (default: the current time).

    Returns:
        The decision; when allowed, it carries the signed record.
    """
    problems = objections(candidate, key)
    problems.extend(vet(candidate))
    problems.extend(absences(candidate))
    seen = candidate.observation
    if problems or seen is None:
        return Decision(allowed=False, reasons=tuple(problems))
    results = {item.name: item.digests["evidence"] for item in candidate.checks}
    unsigned = scmle.deploy.record.Record(
        commit=candidate.commit,
        model=candidate.model.model,
        version=candidate.model.version,
        digests={
            "model": candidate.reported,
            "container": candidate.container,
            "config": candidate.digest,
        },
        environment=candidate.environment,
        provider=candidate.provider,
        requester=candidate.requester,
        approver=candidate.approver,
        approved=candidate.approved,
        evidence=tuple(sorted(results.items())),
        revision=seen.revision,
        recorded=now or events.stamp(),
    )
    return Decision(allowed=True, reasons=(), record=unsigned.signed(key))
