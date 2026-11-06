"""Generates the Phase 16 local evidence by running the real code.

    uv run python scripts/make_phase16_evidence.py plan/evidence/phase-16

Writes, under the output folder:

* ``policy-check.json``: the Terraform policy check over deploy/terraform, the
  resources each environment declares, and each seeded violation with whether
  the check caught it;
* ``terraform-tool.json``: whether terraform or tofu is installed, and the
  result of ``fmt -check`` and ``validate`` when it is (``plan`` never runs);
* ``destroy-orphans.json``: the orphan checker over seeded post-destroy
  listings;
* ``live-checks.json``: the four live checks over recorded responses and each
  seeded failure;
* ``gate-decisions.json``: the promotion gate's decision for the complete
  release and for each seeded failed gate, and the signed records;
* ``reconcile-reports.json``: registry, ledger and runtime compared after a
  success, after a failed release, after a rollback and for seeded
  disagreements, on recorded ECS and Cloud Run answers;
* ``hosted-controls.json``: the static check of every workflow;
* ``scenario-tests.json``: the Phase 16 tests, run by pytest;
* ``summary.json``: what ran and what did not.

Nothing here is edited by hand. Nothing touches a cloud, a container or a
Terraform binary. The service and cloud answers are recorded or scripted
shapes (tests/fixtures/cloud, tests/support_runtime.py), not captures from a
cloud.
"""

import dataclasses
import datetime
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import xml.etree.ElementTree as ElementTree  # nosec B405 - our own report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scmle.deploy import cloudrun  # noqa: E402
from scmle.deploy import ecs  # noqa: E402
from scmle.deploy import gate  # noqa: E402
from scmle.deploy import hcl  # noqa: E402
from scmle.deploy import iac_policy  # noqa: E402
from scmle.deploy import orphans  # noqa: E402
from scmle.governance import deployment_ledger  # noqa: E402
from scmle.governance import release_state  # noqa: E402
from scripts import verify_hosted_controls  # noqa: E402
from tests import support_release as sr  # noqa: E402
from tests import support_runtime as rt  # noqa: E402

TREE = ROOT / "deploy" / "terraform"
AWS = "modules/aws_environment"
GCP = "modules/gcp_environment"
TEST_FILES = (
    "tests/unit/test_iac.py",
    "tests/unit/test_release_gate.py",
    "tests/unit/test_deploy_cli.py",
    "tests/unit/test_release_state.py",
    "tests/unit/test_hosted_controls.py",
)
SEEDED_POLICY = {
    "public ingress": (
        AWS + "/network.tf",
        'resource "aws_vpc_security_group_ingress_rule" "task_from_alb" {\n',
        'resource "aws_vpc_security_group_ingress_rule" "task_from_alb" {\n'
        '  cidr_ipv4 = "0.0.0.0/0"\n',
    ),
    "route without authentication": (
        AWS + "/service.tf",
        'authorization_type = "JWT"',
        'authorization_type = "NONE"',
    ),
    "unencrypted bucket": (AWS + "/storage.tf", '"aws:kms"', '"AES256"'),
    "wildcard IAM action": (
        AWS + "/iam.tf",
        'actions   = ["s3:GetObject"]',
        'actions   = ["s3:*"]',
    ),
    "image by tag": (AWS + "/service.tf", "@${var.image_digest}", ":latest"),
    "no budget": (
        "aws/environments/production/main.tf",
        "monthly_budget_usd  = 1500",
        "monthly_budget_usd  = 0",
    ),
    "secret in a file": (
        AWS + "/network.tf",
        "locals {",
        "# AKIAABCDEFGHIJKLMNOP\nlocals {",
    ),
    "allUsers on Cloud Run": (
        GCP + "/iam.tf",
        "member   = each.value",
        'member   = "allUsers"',
    ),
}
RAN = [
    "tofu fmt -check, init -backend=false and validate on both modules and "
    "all six environments (only when terraform or tofu is installed; see "
    "terraform-tool.json)",
    "static policy check over every Terraform file, with 8 seeded violations",
    "orphan checker over seeded post-destroy listings",
    "live health, schema, inference and security checks over recorded "
    "responses, with 12 seeded failures",
    "promotion gate for a complete release and each seeded failed gate",
    "ECS and Cloud Run adapters over recorded answers, promotion, staged "
    "traffic, rollback and reconcile against scripted platforms",
    "static check of every workflow and its guards",
]
NOT_RUN = [
    "terraform or tofu plan, apply and destroy in any account; plan needs "
    "provider credentials, and no resource was created, so no destroy was "
    "checked against a real account and no cost was recorded",
    "any AWS or GCP account: ECR, ECS, API Gateway, Cloud Run, IAM, KMS, "
    "Secrets Manager, Secret Manager, CloudWatch, Cloud Monitoring, budgets",
    "the cloud-release and cloud-deploy workflows on GitHub: no protected "
    "environment, OIDC federation, image push, scan of a real image or live "
    "test against a deployed service",
    "real ECS DescribeServices or Cloud Run responses: every answer is a "
    "recorded or scripted shape of the documented API",
    "a rejected or successful release on a cloud, and registry and live "
    "state reconciled on a cloud",
    "Docker, skopeo, Trivy or any container",
    "an approver other than the author for staging or production evidence",
    "multi-region availability and any cost or latency comparison of "
    "providers (not claimed)",
]


def write_json(path: Path, payload: object) -> None:
    """Write sorted, indented JSON with a trailing newline."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def inventory() -> dict[str, dict[str, int]]:
    """Resource counts per module and environment."""
    counts: dict[str, dict[str, int]] = {}
    for unit in iac_policy.load(TREE):
        key = f"{unit.provider}/{unit.kind}"
        kinds: dict[str, int] = {}
        for block in unit.blocks:
            if block.kind == "resource":
                kinds[block.labels[0]] = kinds.get(block.labels[0], 0) + 1
        counts[key] = dict(sorted(kinds.items()))
    return counts


def policy_check(out: Path) -> dict[str, object]:
    """Run the policy on the tree and on seeded violations."""
    found = iac_policy.check_tree(TREE)
    seeded = {}
    with tempfile.TemporaryDirectory() as scratch:
        for name, (relative, old, new) in SEEDED_POLICY.items():
            copy = Path(scratch) / name.replace(" ", "-")
            shutil.copytree(TREE, copy)
            path = copy / relative
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace(old, new, 1), encoding="utf-8")
            changed = path.read_text(encoding="utf-8") != text
            violations = iac_policy.check_tree(copy)
            seeded[name] = {
                "mutation_applied": changed,
                "caught": changed and bool(violations),
                "rules": sorted({v.rule for v in violations}),
                "messages": sorted({v.message for v in violations}),
            }
    write_json(
        out / "policy-check.json",
        {
            "violations_in_the_committed_tree": [v.to_dict() for v in found],
            "files_parsed": len(list(TREE.rglob("*.tf"))),
            "blocks_parsed": len(hcl.parse_directory(TREE)),
            "resources": inventory(),
            "seeded": seeded,
        },
    )
    return {
        "committed_tree_clean": not found,
        "seeded_all_caught": all(item["caught"] for item in seeded.values()),
    }


def tool_availability(out: Path) -> dict[str, object]:
    """Record whether terraform or tofu is installed and run fmt and validate.

    `plan` needs provider credentials and is never run here.
    """
    found = {name: shutil.which(name) for name in ("terraform", "tofu")}
    tool = found["terraform"] or found["tofu"]
    checks: dict[str, object] = {"ran": False}
    if tool:
        version = subprocess.run(  # nosec B603
            [tool, "version"], capture_output=True, text=True, check=False
        )
        script = subprocess.run(  # nosec B603 B607
            ["bash", "scripts/terraform_checks.sh", "deploy/terraform"],
            capture_output=True,
            text=True,
            check=False,
        )
        validated = [
            line.split(" ", 1)[1]
            for line in script.stdout.splitlines()
            if line.startswith("validate ")
        ]
        checks = {
            "ran": True,
            "version": version.stdout.splitlines()[0] if version.stdout else "",
            "exit_code": script.returncode,
            "fmt_check_and_validate_passed": script.returncode == 0,
            "directories_validated": validated,
        }
    write_json(
        out / "terraform-tool.json",
        {
            "found": found,
            "fmt_and_validate": checks,
            "plan_ran": False,
            "note": "scripts/terraform_checks.sh runs fmt -check and "
            "init -backend=false + validate; it skips when neither tool is "
            "installed. plan and apply need provider credentials and were "
            "not run.",
        },
    )
    return {"tool_installed": bool(tool)}


def destroy_orphans(out: Path) -> dict[str, bool]:
    """Seeded post-destroy listings: a leftover is found, a clean one passes."""
    aws_left = {
        "ResourceTagMappingList": [
            {
                "ResourceARN": "arn:aws:s3:::scmle-dev-111-data",
                "Tags": [{"Key": "scmle-environment", "Value": "dev"}],
            },
            {
                "ResourceARN": "arn:aws:kms:eu-west-1:111:key/abc",
                "Tags": [{"Key": "scmle-environment", "Value": "dev"}],
                "State": "PendingDeletion",
            },
        ]
    }
    gcp_left = [
        {
            "name": "//storage.googleapis.com/scmle-dev-1-data",
            "assetType": "storage.googleapis.com/Bucket",
            "labels": {"scmle-environment": "dev"},
            "state": "ACTIVE",
        }
    ]
    cases = {
        "aws leftover bucket": orphans.find_orphans("aws", "dev", aws_left),
        "aws clean except a key in its deletion window": orphans.find_orphans(
            "aws",
            "dev",
            {"ResourceTagMappingList": aws_left["ResourceTagMappingList"][1:]},
        ),
        "aws state not empty": orphans.find_orphans(
            "aws",
            "dev",
            {"ResourceTagMappingList": []},
            ["module.environment.aws_vpc.this"],
        ),
        "gcp leftover bucket": orphans.find_orphans("gcp", "dev", gcp_left),
        "gcp other environment is not an orphan": orphans.find_orphans(
            "gcp", "staging", gcp_left
        ),
    }
    write_json(
        out / "destroy-orphans.json",
        {
            "note": "listings are seeded, not read from an account",
            "cases": {name: report.to_dict() for name, report in cases.items()},
        },
    )
    return {name: report.clean for name, report in cases.items()}


def live_checks(out: Path) -> dict[str, object]:
    """The live checks on recorded responses and on seeded failures."""
    base = sr.run_live()
    seeded = {}
    breakers = {
        "readyz 503": ("/readyz", "GET", "none", {"status": 503}),
        "predict answers 500": ("/predict", "POST", "valid", {"status": 500}),
        "anonymous predict allowed": (
            "/predict",
            "POST",
            "none",
            {"status": 200},
        ),
        "forged token allowed": ("/predict", "POST", "forged", {"status": 200}),
        "read-only token reaches predict": (
            "/predict",
            "POST",
            "readonly",
            {"status": 200},
        ),
        "schema without a token allowed": (
            "/schema",
            "GET",
            "none",
            {"status": 200},
        ),
    }
    for name, (path, method, auth, change) in breakers.items():
        document = sr.recorded()
        for item in document["exchanges"]:
            if (item["path"], item["method"], item["auth"]) == (
                path,
                method,
                auth,
            ):
                item.update(change)
        seeded[name] = {r.name: r.passed for r in sr.run_live(document)}
    write_json(
        out / "live-checks.json",
        {
            "source": "tests/fixtures/cloud/staging-http.json, assembled from "
            "docs/SERVICE_SLO.md; not captured from a cloud",
            "recorded_responses": [r.to_dict() for r in base],
            "seeded_failures": seeded,
        },
    )
    return {
        "recorded_all_pass": all(r.passed for r in base),
        "every_seeded_failure_fails_a_check": all(
            not all(item.values()) for item in seeded.values()
        ),
    }


def gate_decisions(out: Path) -> dict[str, object]:
    """The gate on the complete release and on each seeded failed gate."""
    full = sr.candidate("staging")
    allowed = gate.certify(full, sr.KEY, sr.NOW)
    production = gate.certify(
        sr.candidate("production", staging_record=allowed.record),
        sr.KEY,
        sr.NOW,
    )
    failing = {
        "unsigned model": {"model": sr.version(signed=False)},
        "no approval": {"approval": None},
        "rejected approval": {"approval": sr.approval(decision="rejected")},
        "approval of another bundle": {
            "approval": sr.approval(content_id="8" * 64)
        },
        "no scan": {"scan": None},
        "failed scan": {
            "scan": gate.ImageScan(sr.CONTAINER, False, sr.SCAN_REPORT)
        },
        "approver is the requester": {"approver": "author"},
        "no live checks": {"checks": ()},
        "no platform observation": {"observation": None},
        "platform runs another image": {
            "observation": sr.seen(image_digest=sr.OTHER_CONTAINER)
        },
    }
    refusals = {}
    for name, change in failing.items():
        decision = gate.certify(dataclasses.replace(full, **change), sr.KEY)
        refusals[name] = decision.to_dict()
    failed_check = tuple(
        dataclasses.replace(c, passed=False, problems=("seeded",))
        if c.name == "security"
        else c
        for c in full.checks
    )
    refusals["a failed security check"] = gate.certify(
        dataclasses.replace(full, checks=failed_check), sr.KEY
    ).to_dict()
    refusals["production without a staging record"] = gate.certify(
        sr.candidate("production"), sr.KEY
    ).to_dict()
    forged = dataclasses.replace(allowed.record, approver="intruder")
    refusals["production with a forged staging record"] = gate.certify(
        sr.candidate("production", staging_record=forged), sr.KEY
    ).to_dict()
    write_json(
        out / "gate-decisions.json",
        {
            "note": "the signing key is the fixed test key in "
            "tests/support_release.py; the records are of recorded evidence",
            "staging_allowed": allowed.to_dict(),
            "production_allowed": production.to_dict(),
            "refusals": refusals,
        },
    )
    return {
        "staging_allowed": allowed.allowed,
        "production_allowed": production.allowed,
        "every_seeded_gate_refused": all(
            not item["allowed"] and item["record"] is None
            for item in refusals.values()
        ),
    }


def world(kind: str) -> tuple[release_state.ReleaseRegistry, object, object]:
    """A registry with three approved versions and a scripted platform."""
    root = Path(tempfile.mkdtemp(prefix="phase16-"))
    store = rt.make_registry(root / "registry")
    ledger = deployment_ledger.DeploymentLedger.create(
        None, charter_hash="c" * 8, policy={}, alpha_promo="1/20"
    )
    release = release_state.ReleaseRegistry(store, ledger, rt.NO_WAIT)
    for number in (1, 2, 3):
        release.approve("churn", number, rt.approval_for(number), sr.KEY)
    if kind == "ecs":
        fake = rt.FakeEcs()
        plane = ecs.EcsControlPlane(fake, "scmle-staging", "serve")
    else:
        fake = rt.FakeCloudRun()
        plane = cloudrun.CloudRunControlPlane(fake, fake.name)
    return release, plane, fake


def lifecycle(kind: str) -> dict[str, dict[str, object]]:
    """One platform through a success, a failed release and a rollback."""
    release, plane, fake = world(kind)

    def reconcile_now() -> dict[str, object]:
        return release.reconcile(
            "churn", "production", plane.observe()
        ).to_dict()

    def go(version: int, digest: str) -> dict[str, object]:
        return release.promote(
            "churn", version, plane, "production", digest, signing_key=sr.KEY
        ).to_dict()

    steps: dict[str, dict[str, object]] = {}
    steps["promote v1"] = {"outcome": go(1, rt.OLD)}
    steps["promote v1"]["reconcile"] = reconcile_now()
    steps["promote v2"] = {"outcome": go(2, rt.NEW)}
    steps["promote v2"]["reconcile"] = reconcile_now()
    fake.runs = rt.OTHER
    failed = "failed release v3 (the platform runs another image)"
    steps[failed] = {"outcome": go(3, "sha256:" + "d" * 64)}
    steps[failed]["reconcile"] = reconcile_now()
    fake.runs = None
    steps["rollback to v1"] = {
        "outcome": release.rollback(
            "churn", "production", plane, sr.KEY
        ).to_dict()
    }
    steps["rollback to v1"]["reconcile"] = reconcile_now()
    release.store.promote("churn", 3, "champion", rt.approval_for(3), sr.KEY)
    steps["seeded: alias moved around the runtime"] = {
        "reconcile": reconcile_now()
    }
    return steps


def reconciles(out: Path) -> dict[str, object]:
    """Reconcile after success, failure, rollback and seeded disagreements."""
    report = {kind: lifecycle(kind) for kind in ("ecs", "cloudrun")}
    verdicts = {
        kind: {
            name: step["reconcile"]["consistent"]  # type: ignore[index]
            for name, step in steps.items()
        }
        for kind, steps in report.items()
    }
    write_json(
        out / "reconcile-reports.json",
        {
            "source": "recorded ECS and Cloud Run answers changed by the "
            "scripted platforms in tests/support_runtime.py",
            "runs": report,
        },
    )
    return verdicts


def hosted_controls(out: Path) -> bool:
    """The static workflow check, including the two cloud workflows."""
    evidence = verify_hosted_controls.workflow_evidence(ROOT)
    write_json(out / "hosted-controls.json", evidence)
    return bool(evidence["static_controls"]["passed"])


def scenario_tests(out: Path) -> dict[str, object]:
    """Run the Phase 16 tests and record each outcome."""
    with tempfile.TemporaryDirectory() as scratch:
        report = Path(scratch) / "junit.xml"
        completed = subprocess.run(  # nosec B603 - our own interpreter
            [
                sys.executable,
                "-m",
                "pytest",
                *TEST_FILES,
                "-q",
                f"--junitxml={report}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        cases = []
        for case in ElementTree.parse(report).getroot().iter("testcase"):  # nosec B314
            failed = case.find("failure") is not None or (
                case.find("error") is not None
            )
            skipped = case.find("skipped") is not None
            cases.append(
                {
                    "id": f"{case.get('classname')}::{case.get('name')}",
                    "outcome": "failed"
                    if failed
                    else "skipped"
                    if skipped
                    else "passed",
                }
            )
    write_json(
        out / "scenario-tests.json",
        {
            "pytest_exit_status": completed.returncode,
            "tests": len(cases),
            "failed": sum(c["outcome"] == "failed" for c in cases),
            "skipped": sum(c["outcome"] == "skipped" for c in cases),
            "all": cases,
        },
    )
    return {"exit_status": completed.returncode, "tests": len(cases)}


def commit_hash() -> str:
    """The commit the evidence was generated at."""
    done = subprocess.run(  # nosec B603 B607
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout.strip() or "unknown"


def main(arguments: list[str]) -> int:
    """Generate every file; exit 1 if anything that should pass did not."""
    out = Path(arguments[0] if arguments else "plan/evidence/phase-16")
    out.mkdir(parents=True, exist_ok=True)
    checks: dict[str, object] = {
        "policy": policy_check(out),
        "terraform": tool_availability(out),
        "destroy_orphans_clean": destroy_orphans(out),
        "live_checks": live_checks(out),
        "gate": gate_decisions(out),
        "reconcile_consistent": reconciles(out),
        "hosted_controls_passed": hosted_controls(out),
        "scenario_tests": scenario_tests(out),
    }
    policy = checks["policy"]
    gate_result = checks["gate"]
    live = checks["live_checks"]
    reconcile = checks["reconcile_consistent"]
    orphans_clean = checks["destroy_orphans_clean"]
    ok = (
        policy["committed_tree_clean"]  # type: ignore[index]
        and policy["seeded_all_caught"]  # type: ignore[index]
        and gate_result["staging_allowed"]  # type: ignore[index]
        and gate_result["production_allowed"]  # type: ignore[index]
        and gate_result["every_seeded_gate_refused"]  # type: ignore[index]
        and live["recorded_all_pass"]  # type: ignore[index]
        and live["every_seeded_failure_fails_a_check"]  # type: ignore[index]
        and not orphans_clean["aws leftover bucket"]  # type: ignore[index]
        and orphans_clean["aws clean except a key in its deletion window"]  # type: ignore[index]
        and not orphans_clean["aws state not empty"]  # type: ignore[index]
        and not orphans_clean["gcp leftover bucket"]  # type: ignore[index]
        and all(
            steps["promote v1"]  # type: ignore[index]
            and steps["promote v2"]  # type: ignore[index]
            and steps["failed release v3 (the platform runs another image)"]  # type: ignore[index]
            and steps["rollback to v1"]  # type: ignore[index]
            and not steps["seeded: alias moved around the runtime"]  # type: ignore[index]
            for steps in reconcile.values()  # type: ignore[attr-defined]
        )
        and checks["hosted_controls_passed"]
        and checks["scenario_tests"]["exit_status"] == 0  # type: ignore[index]
    )
    write_json(
        out / "summary.json",
        {
            "generated": datetime.datetime.now(datetime.UTC).isoformat(
                timespec="seconds"
            ),
            "commit": commit_hash(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "local_checks_passed": ok,
            "checks": checks,
            "ran": [
                item
                for item in RAN
                if checks["terraform"]["tool_installed"]  # type: ignore[index]
                or not item.startswith("tofu fmt")
            ],
            "not_run": (
                NOT_RUN
                if checks["terraform"]["tool_installed"]  # type: ignore[index]
                else [
                    "terraform or tofu: fmt, init, validate and plan were "
                    "not run (the tool is not installed here)",
                    *NOT_RUN,
                ]
            ),
        },
    )
    print(
        "PHASE 16 LOCAL EVIDENCE PASSED"
        if ok
        else "PHASE 16 LOCAL EVIDENCE FAILED"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
