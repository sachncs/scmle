"""Adapters, release states, promotion, rollback and reconcile.

The ECS and Cloud Run answers are recorded JSON shaped by the documented
responses (tests/fixtures/cloud) and the fakes in tests/support_runtime.py
change them as the real services would be expected to. Nothing here called
AWS or Google Cloud.
"""

import copy
import json

import pytest

from scmle import cli
from scmle.contracts import errors
from scmle.deploy import cloudrun
from scmle.deploy import control
from scmle.deploy import ecs
from scmle.deploy import observation
from scmle.governance import deployment_ledger
from scmle.governance import registry
from scmle.governance import release_state as rs
from tests import support_release as sr
from tests import support_runtime as rt

MODEL = "churn"
PRODUCTION = "production"


def world(tmp_path, kind="ecs", **options):
    store = rt.make_registry(tmp_path / "registry")
    ledger = deployment_ledger.DeploymentLedger.create(
        None, charter_hash="c" * 8, policy={}, alpha_promo="1/20"
    )
    release = rs.ReleaseRegistry(store, ledger, rt.NO_WAIT)
    for number in (1, 2, 3):
        release.approve(MODEL, number, rt.approval_for(number), sr.KEY)
    if kind == "ecs":
        fake = rt.FakeEcs(**options)
        plane = ecs.EcsControlPlane(fake, "scmle-staging", "serve")
    else:
        fake = rt.FakeCloudRun(**options)
        plane = cloudrun.CloudRunControlPlane(fake, fake.name)
    return release, plane, fake


def promote(release, plane, version, digest, **keywords):
    return release.promote(
        MODEL,
        version,
        plane,
        keywords.pop("environment", PRODUCTION),
        digest,
        signing_key=sr.KEY,
        **keywords,
    )


def snapshot(release):
    root = release.store.root
    return {
        "registry": (root / "registry.json").read_text(),
        "state": (root / rs.STATE_FILE).read_text(),
        "aliases": release.store.aliases(MODEL),
        "ledger_rows": len(release.ledger.rows),
    }


KINDS = ["ecs", "cloudrun"]


# ---- adapters over the recorded answers --------------------------------


def test_ecs_answers_become_an_observation():
    seen = ecs.observe_services(
        rt.load("ecs-describe-services"),
        rt.load("ecs-describe-tasks"),
        None,
        sr.NOW,
    )
    assert seen.provider == "aws" and seen.service == "serve"
    assert seen.revision == "scmle-staging-serve:12"
    assert seen.image_digest == rt.OLD and seen.ready
    assert [(t.revision, t.percent) for t in seen.traffic] == [
        ("scmle-staging-serve:12", 100)
    ]


def test_ecs_falls_back_to_the_pinned_task_definition_image():
    seen = ecs.observe_services(
        rt.load("ecs-describe-services"),
        None,
        rt.load("ecs-describe-task-definition"),
        sr.NOW,
    )
    assert seen.image_digest == rt.OLD


def test_ecs_without_a_digest_is_an_error_not_a_guess():
    with pytest.raises(errors.ArtifactError, match="digest"):
        ecs.observe_services(rt.load("ecs-describe-services"), None, None)


def test_ecs_tasks_of_one_revision_that_disagree_are_an_error():
    tasks = rt.load("ecs-describe-tasks")
    tasks["tasks"][1]["containers"][0]["imageDigest"] = rt.OTHER
    with pytest.raises(errors.ArtifactError, match="different image digests"):
        ecs.observe_services(rt.load("ecs-describe-services"), tasks)


@pytest.mark.parametrize("missing", ["services", "primary"])
def test_ecs_answers_without_a_service_or_a_primary_are_errors(missing):
    document = rt.load("ecs-describe-services")
    if missing == "services":
        document["services"] = []
        document["failures"] = [{"reason": "MISSING"}]
    else:
        document["services"][0]["deployments"][0]["status"] = "ACTIVE"
    with pytest.raises(errors.ArtifactError):
        ecs.observe_services(document, rt.load("ecs-describe-tasks"))


def test_a_rolling_ecs_deployment_is_not_ready_and_splits_traffic():
    fake = rt.FakeEcs(never_settles=True)
    plane = ecs.EcsControlPlane(fake, "c", "serve")
    plane.deploy(rt.NEW)
    seen = plane.observe()
    assert not seen.ready
    assert [t.percent for t in seen.traffic] == [50, 50]
    assert seen.digest_of("scmle-staging-serve:13") == rt.NEW


def test_cloud_run_answers_become_an_observation():
    seen = cloudrun.observe_service(
        rt.load("cloudrun-service"), rt.load("cloudrun-revisions"), sr.NOW
    )
    assert seen.provider == "gcp" and seen.ready
    assert seen.revision == "scmle-staging-serve-00011-abc"
    assert seen.image_digest == rt.OLD
    assert seen.share_of(seen.revision) == 100


def test_cloud_run_reports_what_is_served_not_what_was_requested():
    service = rt.load("cloudrun-service")
    service["traffic"] = [
        {
            "type": cloudrun.REVISION_TARGET,
            "revision": "scmle-staging-serve-00012-new",
            "percent": 100,
        }
    ]
    seen = cloudrun.observe_service(service, rt.load("cloudrun-revisions"))
    assert seen.revision == "scmle-staging-serve-00011-abc"


def test_cloud_run_latest_target_resolves_to_the_latest_ready_revision():
    service = rt.load("cloudrun-service")
    service["trafficStatuses"] = [
        {"type": cloudrun.LATEST_TARGET, "percent": 100}
    ]
    seen = cloudrun.observe_service(service, rt.load("cloudrun-revisions"))
    assert seen.revision == "scmle-staging-serve-00011-abc"


def test_cloud_run_revision_not_pinned_by_digest_is_an_error():
    revisions = rt.load("cloudrun-revisions")
    revisions["scmle-staging-serve-00011-abc"]["containers"][0]["image"] = (
        "europe-west1-docker.pkg.dev/p/r/serve:latest"
    )
    with pytest.raises(errors.ArtifactError, match="not pinned"):
        cloudrun.observe_service(rt.load("cloudrun-service"), revisions)


def test_cloud_run_unready_service_is_not_ready():
    service = rt.load("cloudrun-service")
    service["terminalCondition"]["state"] = "CONDITION_FAILED"
    assert not cloudrun.observe_service(
        service, rt.load("cloudrun-revisions")
    ).ready


def test_an_observation_refuses_a_tag_and_a_broken_split():
    with pytest.raises(errors.ConfigError):
        sr.seen(image_digest="repo:latest")
    with pytest.raises(errors.ConfigError):
        sr.seen(traffic=(observation.Traffic("a", 60),))


# ---- release states ----------------------------------------------------


def test_every_registered_version_starts_as_a_candidate(tmp_path):
    store = rt.make_registry(tmp_path / "registry")
    release = rs.ReleaseRegistry(store)
    assert set(release.states(MODEL).values()) == {rs.ReleaseState.CANDIDATE}


def test_states_are_distinct_and_the_hand_moves_are_limited(tmp_path):
    store = rt.make_registry(tmp_path / "registry")
    release = rs.ReleaseRegistry(store)
    assert len({s.value for s in rs.ReleaseState}) == 5
    release.transition(MODEL, 1, rs.ReleaseState.CHALLENGER, "compare")
    assert release.state_of(MODEL, 1) is rs.ReleaseState.CHALLENGER
    for forbidden in (rs.ReleaseState.PRODUCTION, rs.ReleaseState.APPROVED):
        with pytest.raises(errors.ArtifactError, match="cannot be set by hand"):
            release.transition(MODEL, 2, forbidden, "shortcut")
    release.transition(MODEL, 1, rs.ReleaseState.RETIRED, "withdrawn")
    with pytest.raises(errors.ArtifactError, match="not allowed"):
        release.transition(MODEL, 1, rs.ReleaseState.CHALLENGER, "again")


def test_approval_needs_a_signed_bundle_and_a_valid_independent_review(
    tmp_path,
):
    store = rt.make_registry(tmp_path / "registry")
    release = rs.ReleaseRegistry(store)
    bad = {
        "rejected": rt.approval_for(1, decision="rejected"),
        "not independent": rt.approval_for(1, independent=False),
        "another bundle": sr.approval(version=1, content_id="8" * 64),
        "unsigned approval": rt.approval_for(1).__class__(
            **{
                **rt.approval_for(1).unsigned_dict(),
                "decision": "approved",
                "evidence": ("t",),
            }
        ),
    }
    for name, approval in bad.items():
        with pytest.raises(errors.ArtifactError):
            release.approve(MODEL, 1, approval, sr.KEY)
        assert release.state_of(MODEL, 1) is rs.ReleaseState.CANDIDATE, name
    with pytest.raises(errors.ArtifactError):
        release.approve(MODEL, 1, rt.approval_for(1), "wrong-key")
    release.approve(MODEL, 1, rt.approval_for(1), sr.KEY)
    assert release.state_of(MODEL, 1) is rs.ReleaseState.APPROVED


def test_an_unsigned_bundle_cannot_be_approved(tmp_path):
    store = rt.make_registry(tmp_path / "registry", signed=False)
    with pytest.raises(errors.ArtifactError, match="unsigned"):
        rs.ReleaseRegistry(store).approve(MODEL, 1, rt.approval_for(1), sr.KEY)


# ---- promotion ----------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_a_confirmed_promotion_changes_state_alias_and_ledger(tmp_path, kind):
    release, plane, ignored = world(tmp_path, kind)
    first = promote(release, plane, 1, rt.OLD)
    assert first.done and first.restored is None
    second = promote(release, plane, 2, rt.NEW)
    assert second.done and second.version == 2
    assert release.state_of(MODEL, 2) is rs.ReleaseState.PRODUCTION
    assert release.state_of(MODEL, 1) is rs.ReleaseState.RETIRED
    assert release.state_of(MODEL, 3) is rs.ReleaseState.APPROVED
    assert release.store.aliases(MODEL)[registry.CHAMPION] == 2
    recorded = release.environment(MODEL, PRODUCTION)
    assert recorded["version"] == 2 and recorded["image_digest"] == rt.NEW
    assert recorded["known_good"][-1]["version"] == 1
    events = [r["body"]["event"] for r in release.ledger.rows[1:]]
    assert events == ["promoted", "promoted"]
    release.ledger.verify()
    seen = plane.observe()
    assert release.reconcile(MODEL, PRODUCTION, seen).consistent
    assert release.deployed_version(MODEL, PRODUCTION, seen) == 2


@pytest.mark.parametrize("kind", KINDS)
def test_a_staging_release_does_not_make_a_version_production(tmp_path, kind):
    release, plane, ignored = world(tmp_path, kind)
    outcome = promote(release, plane, 1, rt.OLD, environment="staging")
    assert outcome.done
    assert release.state_of(MODEL, 1) is rs.ReleaseState.APPROVED
    assert registry.CHAMPION not in release.store.aliases(MODEL)
    assert release.environment(MODEL, "staging")["version"] == 1
    assert release.reconcile(MODEL, "staging", plane.observe()).consistent


@pytest.mark.parametrize("kind", KINDS)
def test_a_staged_rollout_passes_through_each_share_only_on_cloud_run(
    tmp_path, kind
):
    release, plane, ignored = world(tmp_path, kind)
    assert promote(release, plane, 1, rt.OLD).done
    seen_shares = []

    def watch(seen, percent):
        seen_shares.append((percent, seen.share_of(seen.revision)))
        return []

    outcome = promote(
        release, plane, 2, rt.NEW, steps=(10, 50, 100), health=watch
    )
    if kind == "ecs":
        assert not outcome.done
        assert "all traffic to one revision" in outcome.reasons[0]
        assert outcome.restored is True
    else:
        assert outcome.done
        assert [p for p, ignored in seen_shares] == [10, 50, 100]
        assert release.state_of(MODEL, 2) is rs.ReleaseState.PRODUCTION


def test_each_unconfirmed_or_refused_promotion_changes_nothing(tmp_path):
    release, plane, ignored = world(tmp_path, "cloudrun")
    assert promote(release, plane, 1, rt.OLD).done
    release.transition(MODEL, 3, rs.ReleaseState.RETIRED, "withdrawn")
    before = snapshot(release)
    # not approved (retired), then a bad digest request
    refused = promote(release, plane, 3, rt.NEW)
    assert not refused.done and "retired, not approved" in refused.reasons[0]
    with pytest.raises(errors.ConfigError):
        promote(release, plane, 2, "repo:latest")
    with pytest.raises(errors.ConfigError):
        promote(release, plane, 2, rt.NEW, steps=(50,))
    after = snapshot(release)
    assert {k: v for k, v in after.items() if k != "ledger_rows"} == {
        k: v for k, v in before.items() if k != "ledger_rows"
    }


@pytest.mark.parametrize("kind", KINDS)
def test_a_platform_that_never_confirms_leaves_the_registry_unchanged(
    tmp_path, kind
):
    release, plane, fake = world(tmp_path, kind)
    assert promote(release, plane, 1, rt.OLD).done
    before = snapshot(release)
    fake.never_settles = True
    outcome = promote(release, plane, 2, rt.NEW)
    assert not outcome.done
    assert "did not confirm" in outcome.reasons[0]
    after = snapshot(release)
    assert after["registry"] == before["registry"]
    assert after["state"] == before["state"]
    assert after["aliases"] == before["aliases"]
    assert release.state_of(MODEL, 2) is rs.ReleaseState.APPROVED
    assert release.ledger.rows[-1]["body"]["event"] == "rejected"
    assert release.environment(MODEL, PRODUCTION)["version"] == 1


@pytest.mark.parametrize("kind", KINDS)
def test_a_platform_that_runs_another_image_is_not_confirmed(tmp_path, kind):
    release, plane, fake = world(tmp_path, kind)
    assert promote(release, plane, 1, rt.OLD).done
    before = snapshot(release)
    fake.runs = rt.OTHER
    outcome = promote(release, plane, 2, rt.NEW)
    assert not outcome.done and rt.OTHER in outcome.reasons[0]
    after = snapshot(release)
    assert (after["registry"], after["state"], after["aliases"]) == (
        before["registry"],
        before["state"],
        before["aliases"],
    )


def test_a_failing_health_check_stops_a_staged_rollout_and_restores(tmp_path):
    release, plane, ignored = world(tmp_path, "cloudrun")
    assert promote(release, plane, 1, rt.OLD).done
    before = snapshot(release)
    outcome = promote(
        release,
        plane,
        2,
        rt.NEW,
        steps=(10, 100),
        health=lambda seen, percent: ["error rate above the bar"],
    )
    assert not outcome.done
    assert "stopped the rollout at 10%" in outcome.reasons[0]
    assert outcome.restored is True
    seen = plane.observe()
    assert seen.image_digest == rt.OLD and seen.share_of(seen.revision) == 100
    after = snapshot(release)
    assert (after["registry"], after["state"], after["aliases"]) == (
        before["registry"],
        before["state"],
        before["aliases"],
    )
    assert release.reconcile(MODEL, PRODUCTION, seen).consistent


def test_a_failed_restore_is_reported_and_reconcile_flags_the_runtime(tmp_path):
    release, plane, fake = world(tmp_path, "ecs")
    assert promote(release, plane, 1, rt.OLD).done
    fake.never_settles = True
    outcome = promote(release, plane, 2, rt.NEW)
    assert not outcome.done and outcome.restored is False
    report = release.reconcile(MODEL, PRODUCTION, plane.observe())
    assert not report.consistent


def test_the_registry_refusing_the_alias_restores_the_platform(tmp_path):
    release, plane, ignored = world(tmp_path, "cloudrun")
    assert promote(release, plane, 1, rt.OLD).done
    release.store.attach_guard(RefusingGuard(), "scope")
    before = snapshot(release)
    outcome = promote(release, plane, 2, rt.NEW)
    assert not outcome.done and outcome.restored is True
    after = snapshot(release)
    assert after["state"] == before["state"]
    assert after["aliases"] == before["aliases"]
    assert plane.observe().image_digest == rt.OLD


class RefusingGuard:
    def require(self, action, detail, scope):
        raise errors.BlockedError("an attestation is required")


# ---- rollback -----------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_rollback_restores_the_previous_known_good_revision(tmp_path, kind):
    release, plane, ignored = world(tmp_path, kind)
    assert promote(release, plane, 1, rt.OLD).done
    first = release.environment(MODEL, PRODUCTION)
    assert promote(release, plane, 2, rt.NEW).done
    outcome = release.rollback(MODEL, PRODUCTION, plane, sr.KEY)
    assert outcome.done and outcome.version == 1
    assert outcome.revision == first["revision"]
    assert release.state_of(MODEL, 1) is rs.ReleaseState.PRODUCTION
    assert release.state_of(MODEL, 2) is rs.ReleaseState.RETIRED
    assert release.store.aliases(MODEL)[registry.CHAMPION] == 1
    seen = plane.observe()
    assert seen.image_digest == rt.OLD
    assert release.reconcile(MODEL, PRODUCTION, seen).consistent
    assert release.ledger.rows[-1]["body"]["event"] == "rolled_back"


def test_rollback_without_a_previous_revision_does_nothing(tmp_path):
    release, plane, ignored = world(tmp_path, "cloudrun")
    assert promote(release, plane, 1, rt.OLD).done
    before = snapshot(release)
    outcome = release.rollback(MODEL, PRODUCTION, plane, sr.KEY)
    assert not outcome.done and "no previous" in outcome.reasons[0]
    assert snapshot(release) == before


def test_an_unconfirmed_rollback_changes_nothing(tmp_path):
    release, plane, fake = world(tmp_path, "cloudrun")
    assert promote(release, plane, 1, rt.OLD).done
    assert promote(release, plane, 2, rt.NEW).done
    before = snapshot(release)
    fake.never_settles = True
    outcome = release.rollback(MODEL, PRODUCTION, plane, sr.KEY)
    assert not outcome.done
    after = snapshot(release)
    assert (after["registry"], after["state"], after["aliases"]) == (
        before["registry"],
        before["state"],
        before["aliases"],
    )


# ---- reconcile ----------------------------------------------------------


@pytest.fixture
def released(tmp_path):
    release, plane, ignored = world(tmp_path, "cloudrun")
    assert promote(release, plane, 1, rt.OLD).done
    assert promote(release, plane, 2, rt.NEW).done
    return release, plane.observe()


def kinds(release, seen):
    return {f.kind for f in release.reconcile(MODEL, PRODUCTION, seen).findings}


def test_reconcile_is_clean_after_a_success(released):
    release, seen = released
    assert kinds(release, seen) == set()


def test_reconcile_detects_a_seeded_revision_and_digest_disagreement(released):
    release = released[0]
    other = sr.seen(
        provider="gcp",
        revision="scmle-staging-serve-00099-zzz",
        image_digest=rt.OTHER,
        traffic=(observation.Traffic("scmle-staging-serve-00099-zzz", 100),),
    )
    found = kinds(release, other)
    assert {"revision_mismatch", "digest_mismatch", "traffic_split"} <= found
    with pytest.raises(rs.RuntimeDisagrees) as raised:
        release.deployed_version(MODEL, PRODUCTION, other)
    assert raised.value.details["findings"]


def test_reconcile_detects_partial_traffic_and_unreadiness(released):
    release, seen = released
    split = sr.seen(
        provider="gcp",
        revision=seen.revision,
        image_digest=seen.image_digest,
        traffic=(
            observation.Traffic(seen.revision, 90),
            observation.Traffic("other-rev", 10),
        ),
        ready=False,
    )
    assert {"traffic_split", "not_ready"} <= kinds(release, split)


def test_an_alias_moved_around_the_runtime_is_never_deployed(released):
    release, seen = released
    release.store.promote(
        MODEL, 3, registry.CHAMPION, rt.approval_for(3), sr.KEY
    )
    assert release.store.aliases(MODEL)[registry.CHAMPION] == 3
    assert "alias_not_deployed" in kinds(release, seen)
    with pytest.raises(rs.RuntimeDisagrees):
        release.deployed_version(MODEL, PRODUCTION, seen)


def test_reconcile_detects_a_state_that_was_edited(released):
    release, seen = released
    path = release.store.root / rs.STATE_FILE
    state = json.loads(path.read_text())
    state["models"][MODEL]["versions"]["3"] = {
        "state": "production",
        "history": [],
    }
    path.write_text(json.dumps(state))
    assert "state_mismatch" in kinds(release, seen)


def test_reconcile_detects_a_ledger_that_disagrees_or_is_empty(released):
    release, seen = released
    release.ledger.commit(
        deployment_ledger.DEPLOYMENT,
        {
            "event": "promoted",
            "model": MODEL,
            "environment": PRODUCTION,
            "version": 3,
            "revision": "x",
            "image_digest": rt.OTHER,
        },
    )
    assert "ledger_disagrees" in kinds(release, seen)
    fresh = deployment_ledger.DeploymentLedger.create(
        None, charter_hash="c" * 8, policy={}, alpha_promo="1/20"
    )
    release.ledger = fresh
    assert "ledger_missing" in kinds(release, seen)


def test_the_runtime_running_something_unrecorded_is_a_finding(tmp_path):
    release, plane, ignored = world(tmp_path, "ecs")
    report = release.reconcile(MODEL, PRODUCTION, plane.observe())
    assert [f.kind for f in report.findings] == ["no_registry_record"]
    with pytest.raises(rs.RuntimeDisagrees):
        release.deployed_version(MODEL, PRODUCTION, plane.observe())


@pytest.mark.parametrize("kind", KINDS)
def test_registry_and_runtime_reconcile_after_a_success_and_a_failure(
    tmp_path, kind
):
    release, plane, fake = world(tmp_path, kind)
    assert promote(release, plane, 1, rt.OLD).done
    assert release.reconcile(MODEL, PRODUCTION, plane.observe()).consistent
    fake.runs = rt.OTHER
    assert not promote(release, plane, 2, rt.NEW).done
    after = release.reconcile(MODEL, PRODUCTION, plane.observe())
    assert after.consistent, [f.to_dict() for f in after.findings]
    assert plane.observe().image_digest == rt.OLD


# ---- the command line ----------------------------------------------------


def write(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def test_observe_and_reconcile_commands(tmp_path, capsys):
    release, plane, fake = world(tmp_path, "cloudrun")
    assert promote(release, plane, 1, rt.OLD).done
    seen = plane.observe()
    observed = tmp_path / "observed.json"
    code = cli.main(
        [
            "deploy",
            "observe",
            "--provider",
            "gcp",
            "--input",
            write(tmp_path / "service.json", fake.service),
            "--revisions",
            write(tmp_path / "revisions.json", fake.revisions),
            "--output",
            str(observed),
        ]
    )
    assert code == cli.EXIT_OK
    capsys.readouterr()
    base = [
        "deploy",
        "reconcile",
        str(release.store.root),
        "--model",
        MODEL,
        "--environment",
        PRODUCTION,
        "--observation",
        str(observed),
    ]
    assert cli.main(base) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["consistent"] is True
    document = json.loads(observed.read_text())
    document["image_digest"] = rt.OTHER
    document["revisions"] = [[document["revision"], rt.OTHER]]
    observed.write_text(json.dumps(document))
    assert cli.main(base) == cli.EXIT_FAILED
    out = json.loads(capsys.readouterr().out)
    assert "digest_mismatch" in {f["kind"] for f in out["findings"]}
    assert seen.revision == document["revision"]


def test_observe_ecs_command_and_missing_inputs(tmp_path, capsys):
    base = [
        "deploy",
        "observe",
        "--provider",
        "aws",
        "--input",
        str(rt.FIXTURES / "ecs-describe-services.json"),
        "--output",
        str(tmp_path / "o.json"),
    ]
    assert cli.main(base) == cli.EXIT_FAILED
    code = cli.main(
        [
            *base,
            "--tasks",
            str(rt.FIXTURES / "ecs-describe-tasks.json"),
        ]
    )
    assert code == cli.EXIT_OK
    saved = json.loads((tmp_path / "o.json").read_text())
    assert saved["image_digest"] == rt.OLD
    code = cli.main(
        [
            "deploy",
            "observe",
            "--provider",
            "gcp",
            "--input",
            str(rt.FIXTURES / "cloudrun-service.json"),
            "--output",
            str(tmp_path / "g.json"),
        ]
    )
    assert code == cli.EXIT_FAILED


def test_confirm_waits_and_then_gives_up():
    waits = []
    policy = control.ConfirmPolicy(attempts=3, delay=2, sleep=waits.append)
    fake = rt.FakeEcs(never_settles=True)
    plane = ecs.EcsControlPlane(fake, "c", "serve")
    plane.deploy(rt.NEW)
    with pytest.raises(control.RuntimeUnconfirmed):
        control.confirm(
            plane,
            control.expecting("scmle-staging-serve:13", rt.NEW, 100),
            policy,
        )
    assert waits == [2, 2]
    with pytest.raises(errors.ConfigError):
        control.ConfirmPolicy(attempts=0)


def test_the_ecs_plane_refuses_a_traffic_split_and_an_unpinned_deploy():
    plane = ecs.EcsControlPlane(rt.FakeEcs(), "c", "serve")
    with pytest.raises(control.Unsupported):
        plane.set_traffic({"a:1": 50, "a:2": 50})
    with pytest.raises(errors.ConfigError):
        plane.deploy("repo:latest")
    assert copy.deepcopy(plane.deploy(rt.NEW)) == "scmle-staging-serve:13"
