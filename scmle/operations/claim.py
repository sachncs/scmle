"""The production claim and the evidence it needs.

The claim is one fixed sentence, `CLAIM`, and it is also written in
``docs/RELEASING.md`` (`claim_text_problems` checks the two are the same).
`check_claim` decides whether it may be made. It reads, for each of AWS and
GCP, the signed records in ``<root>/<provider>/`` and refuses unless every
one is present, authentic, from staging, for the same commit, passing, and
covered by a reviewer's signature that is not the author's:

``acceptance.json``
    The six machine-checkable acceptance criteria of F14, each passed.
``deployment.json``
    The signed staging `DeploymentRecord` of the release.
``recovery.json``
    A recovery report for the provider: every fault in the catalogue ran in
    this provider's staging environment and passed within the objectives.
``observability.json``
    An injected service failure classified as infrastructure and a controlled
    drift event classified as data quality or mixed, with alerts delivered,
    telemetry retained and operator actions recorded.
``continuous-evaluation.json``
    A controlled degradation that produced a proposal, a replacement that
    passed the gate and the approvals, and a post-promotion check.
``signoff.json``
    The reviewer's signed statement, listing the SHA-256 of each file above.

Records are signed with an HMAC key the pipeline holds; the sign-off is
signed with the reviewer's own key, which the pipeline must not hold. A
missing directory, a missing or unreadable file, a bad signature, a local or
production environment, a mismatched commit or a record that says anything
but ``passed`` is a refusal, and every refusal is listed.
"""

from collections.abc import Mapping
import dataclasses
import hashlib
import hmac
import json
import pathlib
from typing import Any

from scmle.contracts import errors
from scmle.contracts import signing
from scmle.deploy import record as record_module
from scmle.operations import recovery

CLAIM = (
    "SCMLE is production-qualified on AWS and GCP: every P0 acceptance "
    "criterion passed in staging on both providers, with evidence signed by "
    "a reviewer other than the author."
)
CLAIM_START = "<!-- claim:start -->"
CLAIM_END = "<!-- claim:end -->"
PROVIDERS = ("aws", "gcp")
ENVIRONMENT = "staging"
KINDS = (
    "acceptance",
    "deployment",
    "recovery",
    "observability",
    "continuous-evaluation",
)
SIGNOFF = "signoff"
ACCEPTANCE = (
    "local_demo_with_rollback_drill",
    "bundle_parity_across_environments",
    "provision_deploy_test_destroy",
    "gated_release_with_evidence_chain",
    "registry_and_runtime_reconcile",
    "recovery_within_rto_and_rpo",
)


@dataclasses.dataclass(frozen=True, slots=True)
class ClaimVerdict:
    """Whether the claim may be made, and why not.

    Attributes:
        allowed: True only when no provider has a problem.
        problems: Every reason, prefixed with the provider.
        providers: Problems per provider.
    """

    allowed: bool
    problems: tuple[str, ...]
    providers: Mapping[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready record."""
        return {
            "claim": CLAIM,
            "allowed": self.allowed,
            "problems": list(self.problems),
            "providers": {k: list(v) for k, v in self.providers.items()},
        }


def envelope(
    kind: str,
    provider: str,
    commit: str,
    recorded_at: str,
    body: Mapping[str, Any],
    key: str | bytes,
    environment: str = ENVIRONMENT,
) -> dict[str, Any]:
    """A signed record, as the pipeline or the reviewer writes it."""
    unsigned = {
        "kind": kind,
        "provider": provider,
        "environment": environment,
        "commit": commit,
        "recorded_at": recorded_at,
        "body": json.loads(json.dumps(body)),
    }
    return {**unsigned, "signature": signing.sign_payload(unsigned, key)}


def read_signed(
    path: pathlib.Path,
    kind: str,
    provider: str,
    key: str | bytes,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Reads and authenticates one record.

    Returns:
        The record (None if it cannot be trusted at all) and its problems.
    """
    name = path.name
    if not path.is_file():
        return None, [f"{name} is missing"]
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None, [f"{name} is not readable JSON"]
    if not isinstance(loaded, dict) or "signature" not in loaded:
        return None, [f"{name} is not a signed record"]
    unsigned = {k: v for k, v in loaded.items() if k != "signature"}
    try:
        expected = signing.sign_payload(unsigned, key)
    except (TypeError, ValueError):
        return None, [f"{name} cannot be verified"]
    if not hmac.compare_digest(str(loaded["signature"]), expected):
        return None, [f"{name}: the signature does not verify"]
    problems = []
    if loaded.get("kind") != kind:
        problems.append(f"{name}: kind is {loaded.get('kind')!r}, not {kind!r}")
    if loaded.get("provider") != provider:
        problems.append(f"{name}: it is for {loaded.get('provider')!r}")
    if loaded.get("environment") != ENVIRONMENT:
        problems.append(
            f"{name}: environment is {loaded.get('environment')!r}, "
            f"not {ENVIRONMENT!r}"
        )
    if (
        record_module.COMMIT_PATTERN.fullmatch(str(loaded.get("commit")))
        is None
    ):
        problems.append(f"{name}: the commit is not a full hash")
    if not isinstance(loaded.get("body"), dict):
        problems.append(f"{name}: there is no body")
        return None, problems
    return loaded, problems


def acceptance_problems(body: Mapping[str, Any]) -> list[str]:
    """The six acceptance criteria must all be present and passed."""
    items = body.get("items", {})
    found = []
    for name in ACCEPTANCE:
        item = items.get(name) if isinstance(items, dict) else None
        if not isinstance(item, dict) or item.get("status") != "passed":
            found.append(f"acceptance {name} did not pass")
        elif not item.get("evidence"):
            found.append(f"acceptance {name} cites no evidence")
    return found


def deployment_problems(
    body: Mapping[str, Any], provider: str, commit: str, key: str | bytes
) -> list[str]:
    """The staging deployment record must verify and match."""
    try:
        deployed = record_module.DeploymentRecord.from_dict(body["record"])
    except (KeyError, TypeError, ValueError, AttributeError, errors.ScmleError):
        return ["deployment.json holds no valid deployment record"]
    found = []
    if not deployed.verify(key):
        found.append("the deployment record's signature does not verify")
    if deployed.environment != ENVIRONMENT:
        found.append("the deployment record is not for staging")
    if deployed.provider != provider:
        found.append("the deployment record is for another provider")
    if deployed.source_commit != commit:
        found.append("the deployment record is for another commit")
    if deployed.requested_by == deployed.approver:
        found.append("the deployment's approver is its requester")
    return found


def recovery_problems(body: Mapping[str, Any], provider: str) -> list[str]:
    """Every catalogued fault ran in staging on the provider and passed."""
    found = []
    results = {
        r.get("fault_id"): r
        for r in body.get("results", [])
        if isinstance(r, dict)
    }
    for fault in recovery.CATALOGUE:
        result = results.get(fault.id)
        if result is None:
            found.append(f"recovery: {fault.id} has no result")
            continue
        if result.get("status") != recovery.PASSED:
            found.append(f"recovery: {fault.id} is {result.get('status')!r}")
        if not str(result.get("environment", "")).startswith(
            f"{provider}:{ENVIRONMENT}"
        ):
            found.append(
                f"recovery: {fault.id} did not run in {provider} staging"
            )
        for measured in result.get("measurements", []):
            component = measured.get("component")
            goal = recovery.OBJECTIVE_OF.get(component)
            rto = measured.get("rto_seconds")
            lost = measured.get("rpo_lost")
            if goal is None:
                found.append(f"recovery: unknown component {component!r}")
            elif rto is None or rto > goal.rto_seconds:
                found.append(f"recovery: {fault.id}/{component} RTO misses")
            elif lost is None or lost > goal.rpo_lost:
                found.append(f"recovery: {fault.id}/{component} RPO misses")
    measured_components = {
        m.get("component")
        for r in results.values()
        for m in r.get("measurements", [])
    }
    for goal in recovery.OBJECTIVES:
        if goal.component not in measured_components:
            found.append(f"recovery: {goal.component} was never measured")
    return found


def observability_problems(body: Mapping[str, Any]) -> list[str]:
    """Failure and drift were classified, alerted, retained and answered."""
    found = []
    failure = body.get("injected_failure", {})
    drift = body.get("controlled_drift", {})
    if failure.get("verdict") != "infrastructure":
        found.append(
            "the injected failure was not classified as infrastructure"
        )
    if drift.get("verdict") not in {"data_quality", "mixed"}:
        found.append(
            "the drift event was not classified as data quality or mixed"
        )
    for name in ("alerts_delivered", "telemetry_retained"):
        if body.get(name) is not True:
            found.append(f"observability: {name} is not true")
    actions = body.get("operator_actions", [])
    for needed in ("incident_acknowledged", "incident_resolved"):
        if needed not in actions:
            found.append(f"observability: no {needed} operator action")
    return found


def continuous_problems(body: Mapping[str, Any]) -> list[str]:
    """A degradation produced a proposal; replacement passed the gate."""
    found = []
    if body.get("proposal") not in {"recommend", "trigger"}:
        found.append("continuous evaluation: no retraining proposal")
    if not body.get("escalation_id"):
        found.append("continuous evaluation: no escalation was opened")
    if body.get("gate_allowed") is not True:
        found.append("continuous evaluation: the gate did not admit the model")
    if body.get("replacement_confirmed") is not True:
        found.append("continuous evaluation: the runtime did not confirm")
    if "overall" not in body.get("post_promotion_check", {}):
        found.append("continuous evaluation: no post-promotion check")
    return found


def signoff_problems(
    folder: pathlib.Path,
    provider: str,
    commit: str,
    reviewer_key: str | bytes,
    requested_by: str | None,
) -> list[str]:
    """The reviewer's signature, independence and coverage."""
    signed, found = read_signed(
        folder / f"{SIGNOFF}.json", SIGNOFF, provider, reviewer_key
    )
    if signed is None:
        return found
    body = signed["body"]
    if signed.get("commit") != commit:
        found.append("signoff.json is for another commit")
    reviewer = str(body.get("reviewer", "")).strip()
    if not reviewer:
        found.append("the sign-off names no reviewer")
    if body.get("independent") is not True:
        found.append("the reviewer is not recorded as independent")
    if requested_by is not None and reviewer == requested_by:
        found.append("the reviewer is the person who requested the release")
    covers = body.get("covers", {})
    for kind in KINDS:
        path = folder / f"{kind}.json"
        if not path.is_file():
            continue
        digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if covers.get(kind) != digest:
            found.append(f"the sign-off does not cover the current {kind}.json")
    return found


def check_provider(
    folder: pathlib.Path,
    provider: str,
    key: str | bytes,
    reviewer_key: str | bytes,
) -> list[str]:
    """Every problem with one provider's evidence."""
    if not folder.is_dir():
        return [f"the evidence folder {folder} does not exist"]
    problems: list[str] = []
    bodies: dict[str, dict[str, Any]] = {}
    commits = set()
    for kind in KINDS:
        signed, found = read_signed(
            folder / f"{kind}.json", kind, provider, key
        )
        problems.extend(found)
        if signed is not None:
            bodies[kind] = signed["body"]
            commits.add(signed["commit"])
    if len(commits) > 1:
        problems.append("the records are for different commits")
    commit = next(iter(commits)) if len(commits) == 1 else ""
    checks = {
        "acceptance": lambda b: acceptance_problems(b),
        "deployment": lambda b: deployment_problems(b, provider, commit, key),
        "recovery": lambda b: recovery_problems(b, provider),
        "observability": observability_problems,
        "continuous-evaluation": continuous_problems,
    }
    for kind, check in checks.items():
        if kind in bodies:
            problems.extend(check(bodies[kind]))
    requested_by = None
    if "deployment" in bodies:
        requested_by = (
            bodies["deployment"].get("record", {}).get("requested_by")
        )
    problems.extend(
        signoff_problems(folder, provider, commit, reviewer_key, requested_by)
    )
    return problems


def check_claim(
    root: str | pathlib.Path,
    key: str | bytes,
    reviewer_key: str | bytes,
) -> ClaimVerdict:
    """Decides whether the production claim may be made.

    Args:
        root: ``plan/evidence/phase-17`` (it holds ``aws/`` and ``gcp/``).
        key: The key the pipeline signs its records with.
        reviewer_key: The reviewer's key; the pipeline must not hold it.

    Returns:
        The verdict; ``allowed`` only if both providers have no problem.
    """
    if not key or not reviewer_key:
        raise ValueError("both keys are required")
    if key == reviewer_key:
        raise ValueError("the reviewer's key must differ from the pipeline's")
    base = pathlib.Path(root)
    per = {
        provider: tuple(
            check_provider(base / provider, provider, key, reviewer_key)
        )
        for provider in PROVIDERS
    }
    flat = tuple(f"{p}: {m}" for p, ms in per.items() for m in ms)
    return ClaimVerdict(not flat, flat, per)


def claim_text_problems(path: str | pathlib.Path) -> list[str]:
    """Whether a document holds the fixed claim between its markers."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    if CLAIM_START not in text or CLAIM_END not in text:
        return [f"{path} has no claim markers"]
    between = text.split(CLAIM_START, 1)[1].split(CLAIM_END, 1)[0]
    flat = " ".join(between.replace(">", " ").split())
    return [] if flat == CLAIM else [f"{path} does not hold the fixed claim"]
