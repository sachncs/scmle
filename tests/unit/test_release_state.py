"""Release states, promotion, rollback and reconcile.

The control plane is the in-memory `MemoryPlane` in tests/support_runtime.py,
which can settle after a number of reads, never settle, run another image or
refuse a traffic split.
"""

import json

import pytest

from scmle import cli
from scmle.contracts import errors
from scmle.deploy import control
from scmle.deploy import observation
from scmle.governance import deployment_ledger
from scmle.governance import registry
from scmle.governance import release_state
from tests import support_release
from tests import support_runtime

MODEL = "churn"
PRODUCTION = "production"


def world(tmp_path, split=True, **options):
    store = support_runtime.make_registry(tmp_path / "registry")
    ledger = deployment_ledger.DeploymentLedger.create(
        None, charter_hash="c" * 8, policy={}, alpha_promo="1/20"
    )
    release = release_state.ReleaseRegistry(
        store, ledger, support_runtime.NO_WAIT
    )
    for number in (1, 2, 3):
        release.approve(
            MODEL,
            number,
            support_runtime.approval_for(number),
            support_release.KEY,
        )
    plane = support_runtime.MemoryPlane(split=split, **options)
    return release, plane, plane


def promote(release, plane, version, digest, **keywords):
    return release.promote(
        MODEL,
        version,
        plane,
        keywords.pop("environment", PRODUCTION),
        digest,
        signing_key=support_release.KEY,
        **keywords,
    )


def snapshot(release):
    root = release.store.root
    return {
        "registry": (root / "registry.json").read_text(),
        "state": (root / release_state.STATE_FILE).read_text(),
        "aliases": release.store.aliases(MODEL),
        "ledger_rows": len(release.ledger.rows),
    }


SPLITS = [True, False]


# ---- observations ------------------------------------------------------


def test_a_plane_that_has_not_settled_is_not_ready():
    plane = support_runtime.MemoryPlane(never_settles=True)
    assert plane.observe().ready
    plane.deploy(support_runtime.NEW)
    seen = plane.observe()
    assert not seen.ready
    assert seen.digest_of("serve-2") == support_runtime.NEW


def test_an_observation_refuses_a_tag_and_a_broken_split():
    with pytest.raises(errors.ConfigError):
        support_release.seen(image_digest="repo:latest")
    with pytest.raises(errors.ConfigError):
        support_release.seen(traffic=(observation.Traffic("a", 60),))


# ---- release states ----------------------------------------------------


def test_every_registered_version_starts_as_a_candidate(tmp_path):
    store = support_runtime.make_registry(tmp_path / "registry")
    release = release_state.ReleaseRegistry(store)
    assert set(release.states(MODEL).values()) == {
        release_state.ReleaseState.CANDIDATE
    }


def test_states_are_distinct_and_the_hand_moves_are_limited(tmp_path):
    store = support_runtime.make_registry(tmp_path / "registry")
    release = release_state.ReleaseRegistry(store)
    assert len({s.value for s in release_state.ReleaseState}) == 5
    release.transition(
        MODEL, 1, release_state.ReleaseState.CHALLENGER, "compare"
    )
    assert release.state_of(MODEL, 1) is release_state.ReleaseState.CHALLENGER
    for forbidden in (
        release_state.ReleaseState.PRODUCTION,
        release_state.ReleaseState.APPROVED,
    ):
        with pytest.raises(errors.ArtifactError, match="cannot be set by hand"):
            release.transition(MODEL, 2, forbidden, "shortcut")
    release.transition(
        MODEL, 1, release_state.ReleaseState.RETIRED, "withdrawn"
    )
    with pytest.raises(errors.ArtifactError, match="not allowed"):
        release.transition(
            MODEL, 1, release_state.ReleaseState.CHALLENGER, "again"
        )


def test_approval_needs_a_signed_bundle_and_a_valid_independent_review(
    tmp_path,
):
    store = support_runtime.make_registry(tmp_path / "registry")
    release = release_state.ReleaseRegistry(store)
    bad = {
        "rejected": support_runtime.approval_for(1, decision="rejected"),
        "not independent": support_runtime.approval_for(1, independent=False),
        "another bundle": support_release.approval(
            version=1, content_id="8" * 64
        ),
        "unsigned approval": support_runtime.approval_for(1).__class__(
            **{
                **support_runtime.approval_for(1).unsigned_dict(),
                "decision": "approved",
                "evidence": ("t",),
            }
        ),
    }
    for name, approval in bad.items():
        with pytest.raises(errors.ArtifactError):
            release.approve(MODEL, 1, approval, support_release.KEY)
        assert (
            release.state_of(MODEL, 1) is release_state.ReleaseState.CANDIDATE
        ), name
    with pytest.raises(errors.ArtifactError):
        release.approve(MODEL, 1, support_runtime.approval_for(1), "wrong-key")
    release.approve(
        MODEL, 1, support_runtime.approval_for(1), support_release.KEY
    )
    assert release.state_of(MODEL, 1) is release_state.ReleaseState.APPROVED


def test_an_unsigned_bundle_cannot_be_approved(tmp_path):
    store = support_runtime.make_registry(tmp_path / "registry", signed=False)
    with pytest.raises(errors.ArtifactError, match="unsigned"):
        release_state.ReleaseRegistry(store).approve(
            MODEL, 1, support_runtime.approval_for(1), support_release.KEY
        )


# ---- promotion ----------------------------------------------------------


@pytest.mark.parametrize("split", SPLITS)
def test_a_confirmed_promotion_changes_state_alias_and_ledger(tmp_path, split):
    release, plane, ignored = world(tmp_path, split)
    first = promote(release, plane, 1, support_runtime.OLD)
    assert first.done and first.restored is None
    second = promote(release, plane, 2, support_runtime.NEW)
    assert second.done and second.version == 2
    assert release.state_of(MODEL, 2) is release_state.ReleaseState.PRODUCTION
    assert release.state_of(MODEL, 1) is release_state.ReleaseState.RETIRED
    assert release.state_of(MODEL, 3) is release_state.ReleaseState.APPROVED
    assert release.store.aliases(MODEL)[registry.CHAMPION] == 2
    recorded = release.environment(MODEL, PRODUCTION)
    assert (
        recorded["version"] == 2
        and recorded["image_digest"] == support_runtime.NEW
    )
    assert recorded["known_good"][-1]["version"] == 1
    events = [r["body"]["event"] for r in release.ledger.rows[1:]]
    assert events == ["promoted", "promoted"]
    release.ledger.verify()
    seen = plane.observe()
    assert release.reconcile(MODEL, PRODUCTION, seen).consistent
    assert release.deployed_version(MODEL, PRODUCTION, seen) == 2


@pytest.mark.parametrize("split", SPLITS)
def test_a_staging_release_does_not_make_a_version_production(tmp_path, split):
    release, plane, ignored = world(tmp_path, split)
    outcome = promote(
        release, plane, 1, support_runtime.OLD, environment="staging"
    )
    assert outcome.done
    assert release.state_of(MODEL, 1) is release_state.ReleaseState.APPROVED
    assert registry.CHAMPION not in release.store.aliases(MODEL)
    assert release.environment(MODEL, "staging")["version"] == 1
    assert release.reconcile(MODEL, "staging", plane.observe()).consistent


@pytest.mark.parametrize("split", SPLITS)
def test_a_staged_rollout_passes_through_each_share_only_where_traffic_splits(
    tmp_path, split
):
    release, plane, ignored = world(tmp_path, split)
    assert promote(release, plane, 1, support_runtime.OLD).done
    seen_shares = []

    def watch(seen, percent):
        seen_shares.append((percent, seen.share_of(seen.revision)))
        return []

    outcome = promote(
        release,
        plane,
        2,
        support_runtime.NEW,
        steps=(10, 50, 100),
        health=watch,
    )
    if not split:
        assert not outcome.done
        assert "cannot split traffic" in outcome.reasons[0]
        assert outcome.restored is True
    else:
        assert outcome.done
        assert [p for p, ignored in seen_shares] == [10, 50, 100]
        assert (
            release.state_of(MODEL, 2) is release_state.ReleaseState.PRODUCTION
        )


def test_each_unconfirmed_or_refused_promotion_changes_nothing(tmp_path):
    release, plane, ignored = world(tmp_path)
    assert promote(release, plane, 1, support_runtime.OLD).done
    release.transition(
        MODEL, 3, release_state.ReleaseState.RETIRED, "withdrawn"
    )
    before = snapshot(release)
    # not approved (retired), then a bad digest request
    refused = promote(release, plane, 3, support_runtime.NEW)
    assert not refused.done and "retired, not approved" in refused.reasons[0]
    with pytest.raises(errors.ConfigError):
        promote(release, plane, 2, "repo:latest")
    with pytest.raises(errors.ConfigError):
        promote(release, plane, 2, support_runtime.NEW, steps=(50,))
    after = snapshot(release)
    assert {k: v for k, v in after.items() if k != "ledger_rows"} == {
        k: v for k, v in before.items() if k != "ledger_rows"
    }


@pytest.mark.parametrize("split", SPLITS)
def test_a_platform_that_never_confirms_leaves_the_registry_unchanged(
    tmp_path, split
):
    release, plane, fake = world(tmp_path, split)
    assert promote(release, plane, 1, support_runtime.OLD).done
    before = snapshot(release)
    fake.never_settles = True
    outcome = promote(release, plane, 2, support_runtime.NEW)
    assert not outcome.done
    assert "did not confirm" in outcome.reasons[0]
    after = snapshot(release)
    assert after["registry"] == before["registry"]
    assert after["state"] == before["state"]
    assert after["aliases"] == before["aliases"]
    assert release.state_of(MODEL, 2) is release_state.ReleaseState.APPROVED
    assert release.ledger.rows[-1]["body"]["event"] == "rejected"
    assert release.environment(MODEL, PRODUCTION)["version"] == 1


@pytest.mark.parametrize("split", SPLITS)
def test_a_platform_that_runs_another_image_is_not_confirmed(tmp_path, split):
    release, plane, fake = world(tmp_path, split)
    assert promote(release, plane, 1, support_runtime.OLD).done
    before = snapshot(release)
    fake.runs = support_runtime.OTHER
    outcome = promote(release, plane, 2, support_runtime.NEW)
    assert not outcome.done and support_runtime.OTHER in outcome.reasons[0]
    after = snapshot(release)
    assert (after["registry"], after["state"], after["aliases"]) == (
        before["registry"],
        before["state"],
        before["aliases"],
    )


def test_a_failing_health_check_stops_a_staged_rollout_and_restores(tmp_path):
    release, plane, ignored = world(tmp_path)
    assert promote(release, plane, 1, support_runtime.OLD).done
    before = snapshot(release)
    outcome = promote(
        release,
        plane,
        2,
        support_runtime.NEW,
        steps=(10, 100),
        health=lambda seen, percent: ["error rate above the bar"],
    )
    assert not outcome.done
    assert "stopped the rollout at 10%" in outcome.reasons[0]
    assert outcome.restored is True
    seen = plane.observe()
    assert (
        seen.image_digest == support_runtime.OLD
        and seen.share_of(seen.revision) == 100
    )
    after = snapshot(release)
    assert (after["registry"], after["state"], after["aliases"]) == (
        before["registry"],
        before["state"],
        before["aliases"],
    )
    assert release.reconcile(MODEL, PRODUCTION, seen).consistent


def test_a_failed_restore_is_reported_and_reconcile_flags_the_runtime(tmp_path):
    release, plane, fake = world(tmp_path, split=False)
    assert promote(release, plane, 1, support_runtime.OLD).done
    fake.never_settles = True
    outcome = promote(release, plane, 2, support_runtime.NEW)
    assert not outcome.done and outcome.restored is False
    report = release.reconcile(MODEL, PRODUCTION, plane.observe())
    assert not report.consistent


def test_the_registry_refusing_the_alias_restores_the_platform(tmp_path):
    release, plane, ignored = world(tmp_path)
    assert promote(release, plane, 1, support_runtime.OLD).done
    release.store.attach_guard(RefusingGuard(), "scope")
    before = snapshot(release)
    outcome = promote(release, plane, 2, support_runtime.NEW)
    assert not outcome.done and outcome.restored is True
    after = snapshot(release)
    assert after["state"] == before["state"]
    assert after["aliases"] == before["aliases"]
    assert plane.observe().image_digest == support_runtime.OLD


class RefusingGuard:
    def require(self, action, detail, scope):
        raise errors.BlockedError("an attestation is required")


# ---- rollback -----------------------------------------------------------


@pytest.mark.parametrize("split", SPLITS)
def test_rollback_restores_the_previous_known_good_revision(tmp_path, split):
    release, plane, ignored = world(tmp_path, split)
    assert promote(release, plane, 1, support_runtime.OLD).done
    first = release.environment(MODEL, PRODUCTION)
    assert promote(release, plane, 2, support_runtime.NEW).done
    outcome = release.rollback(MODEL, PRODUCTION, plane, support_release.KEY)
    assert outcome.done and outcome.version == 1
    assert outcome.revision == first["revision"]
    assert release.state_of(MODEL, 1) is release_state.ReleaseState.PRODUCTION
    assert release.state_of(MODEL, 2) is release_state.ReleaseState.RETIRED
    assert release.store.aliases(MODEL)[registry.CHAMPION] == 1
    seen = plane.observe()
    assert seen.image_digest == support_runtime.OLD
    assert release.reconcile(MODEL, PRODUCTION, seen).consistent
    assert release.ledger.rows[-1]["body"]["event"] == "rolled_back"


def test_rollback_without_a_previous_revision_does_nothing(tmp_path):
    release, plane, ignored = world(tmp_path)
    assert promote(release, plane, 1, support_runtime.OLD).done
    before = snapshot(release)
    outcome = release.rollback(MODEL, PRODUCTION, plane, support_release.KEY)
    assert not outcome.done and "no previous" in outcome.reasons[0]
    assert snapshot(release) == before


def test_an_unconfirmed_rollback_changes_nothing(tmp_path):
    release, plane, fake = world(tmp_path)
    assert promote(release, plane, 1, support_runtime.OLD).done
    assert promote(release, plane, 2, support_runtime.NEW).done
    before = snapshot(release)
    fake.never_settles = True
    outcome = release.rollback(MODEL, PRODUCTION, plane, support_release.KEY)
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
    release, plane, ignored = world(tmp_path)
    assert promote(release, plane, 1, support_runtime.OLD).done
    assert promote(release, plane, 2, support_runtime.NEW).done
    return release, plane.observe()


def kinds(release, seen):
    return {f.kind for f in release.reconcile(MODEL, PRODUCTION, seen).findings}


def test_reconcile_is_clean_after_a_success(released):
    release, seen = released
    assert kinds(release, seen) == set()


def test_reconcile_detects_a_seeded_revision_and_digest_disagreement(released):
    release = released[0]
    other = support_release.seen(
        provider="memory",
        revision="scmle-staging-serve-00099-zzz",
        image_digest=support_runtime.OTHER,
        traffic=(observation.Traffic("scmle-staging-serve-00099-zzz", 100),),
    )
    found = kinds(release, other)
    assert {"revision_mismatch", "digest_mismatch", "traffic_split"} <= found
    with pytest.raises(release_state.RuntimeDisagrees) as raised:
        release.deployed_version(MODEL, PRODUCTION, other)
    assert raised.value.details["findings"]


def test_reconcile_detects_partial_traffic_and_unreadiness(released):
    release, seen = released
    split = support_release.seen(
        provider="memory",
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
        MODEL,
        3,
        registry.CHAMPION,
        support_runtime.approval_for(3),
        support_release.KEY,
    )
    assert release.store.aliases(MODEL)[registry.CHAMPION] == 3
    assert "alias_not_deployed" in kinds(release, seen)
    with pytest.raises(release_state.RuntimeDisagrees):
        release.deployed_version(MODEL, PRODUCTION, seen)


def test_reconcile_detects_a_state_that_was_edited(released):
    release, seen = released
    path = release.store.root / release_state.STATE_FILE
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
            "image_digest": support_runtime.OTHER,
        },
    )
    assert "ledger_disagrees" in kinds(release, seen)
    fresh = deployment_ledger.DeploymentLedger.create(
        None, charter_hash="c" * 8, policy={}, alpha_promo="1/20"
    )
    release.ledger = fresh
    assert "ledger_missing" in kinds(release, seen)


def test_the_runtime_running_something_unrecorded_is_a_finding(tmp_path):
    release, plane, ignored = world(tmp_path, split=False)
    report = release.reconcile(MODEL, PRODUCTION, plane.observe())
    assert [f.kind for f in report.findings] == ["no_registry_record"]
    with pytest.raises(release_state.RuntimeDisagrees):
        release.deployed_version(MODEL, PRODUCTION, plane.observe())


@pytest.mark.parametrize("split", SPLITS)
def test_registry_and_runtime_reconcile_after_a_success_and_a_failure(
    tmp_path, split
):
    release, plane, fake = world(tmp_path, split)
    assert promote(release, plane, 1, support_runtime.OLD).done
    assert release.reconcile(MODEL, PRODUCTION, plane.observe()).consistent
    fake.runs = support_runtime.OTHER
    assert not promote(release, plane, 2, support_runtime.NEW).done
    after = release.reconcile(MODEL, PRODUCTION, plane.observe())
    assert after.consistent, [f.to_dict() for f in after.findings]
    assert plane.observe().image_digest == support_runtime.OLD


# ---- the command line ----------------------------------------------------


def test_reconcile_command(tmp_path, capsys):
    release, plane, ignored = world(tmp_path)
    assert promote(release, plane, 1, support_runtime.OLD).done
    seen = plane.observe()
    observed = tmp_path / "observed.json"
    observed.write_text(json.dumps(seen.to_dict()), encoding="utf-8")
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
    document["image_digest"] = support_runtime.OTHER
    document["revisions"] = [[document["revision"], support_runtime.OTHER]]
    observed.write_text(json.dumps(document))
    assert cli.main(base) == cli.EXIT_FAILED
    out = json.loads(capsys.readouterr().out)
    assert "digest_mismatch" in {f["kind"] for f in out["findings"]}
    assert seen.revision == document["revision"]


def test_confirm_waits_and_then_gives_up():
    waits = []
    policy = control.ConfirmPolicy(attempts=3, delay=2, sleep=waits.append)
    plane = support_runtime.MemoryPlane(never_settles=True)
    revision = plane.deploy(support_runtime.NEW)
    with pytest.raises(control.RuntimeUnconfirmed):
        control.confirm(
            plane, control.expecting(revision, support_runtime.NEW, 100), policy
        )
    assert waits == [2, 2]
    with pytest.raises(errors.ConfigError):
        control.ConfirmPolicy(attempts=0)


def test_a_plane_that_cannot_split_traffic_refuses_a_split():
    plane = support_runtime.MemoryPlane(split=False)
    with pytest.raises(control.Unsupported):
        plane.set_traffic({"serve-1": 50, "serve-2": 50})
    assert plane.deploy(support_runtime.NEW) == "serve-2"
