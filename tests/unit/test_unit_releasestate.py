"""Behaviour of each module of the release state, against a memory plane."""

import json

import pytest

from scmle.contracts import errors
from scmle.deploy import control
from scmle.deploy import observation
from scmle.deploy.release_state import approval
from scmle.deploy.release_state import promotion
from scmle.deploy.release_state import reconcile
from scmle.deploy.release_state import records
from scmle.deploy.release_state import rollback
from scmle.deploy.release_state import rollout
from scmle.deploy.release_state import status
from scmle.governance import deployment_ledger
from scmle.governance import registry
from tests import support_release
from tests import support_runtime

MODEL = "churn"


def ledger():
    return deployment_ledger.Ledger.create(
        None, charter_hash="c" * 8, policy={}, alpha_promo="1/20"
    )


@pytest.fixture
def release(tmp_path):
    store = support_runtime.make_registry(tmp_path / "registry")
    made = reconcile.Registry(store, ledger(), support_runtime.NO_WAIT)
    for number in (1, 2):
        made.approve(
            MODEL, number, support_runtime.sanction(number), support_release.KEY
        )
    return made


@pytest.fixture
def plane():
    return support_runtime.MemoryPlane()


def ship(
    release,
    plane,
    version,
    digest=support_runtime.NEW,
    environment="production",
    **kw,
):
    return release.promote(
        MODEL,
        version,
        plane,
        environment,
        digest,
        signing_key=support_release.KEY,
        **kw,
    )


def events(release):
    return [row["body"]["event"] for row in release.ledger.rows[1:]]


# ---- status ----------------------------------------------------------------


def test_status_constants_are_the_documented_values():
    assert status.STATE_FILE == "release_state.json"
    assert status.STATE_FORMAT == 1
    assert status.PRODUCTION_ENVIRONMENT == "production"
    assert status.FULL_ROLLOUT == 100
    assert {
        status.State.CHALLENGER,
        status.State.RETIRED,
    } == status.MANUAL


def test_only_promotion_and_rollback_paths_lead_into_production():
    into = [
        s for s, targets in status.ALLOWED.items() if "production" in targets
    ]
    assert sorted(into) == ["approved", "retired"]
    assert status.ALLOWED[status.State.PRODUCTION] == {"retired"}
    assert status.ALLOWED[status.State.CHALLENGER] == {
        "approved",
        "retired",
    }


def test_halt_names_the_share_and_every_reason():
    with pytest.raises(
        control.UnconfirmedError,
        match="health stopped the rollout at 25%: slow; errors",
    ):
        status.halt(25, ["slow", "errors"])


def test_predecessor_needs_a_previous_revision():
    assert status.predecessor({"revision": "r"}) == {"revision": "r"}
    with pytest.raises(errors.ConfigError, match="needs a previous revision"):
        status.predecessor(None)


def test_outcomes_and_reconciliations_dump_their_fields():
    outcome = status.Outcome(True, (), version=2, revision="r2")
    assert outcome.to_dict() == {
        "done": True,
        "reasons": [],
        "restored": None,
        "version": 2,
        "revision": "r2",
    }
    report = status.Reconciliation(
        "m", "prod", (status.Finding("not_ready", "x"),)
    )
    assert not report.consistent
    assert report.to_dict() == {
        "model": "m",
        "environment": "prod",
        "consistent": False,
        "findings": [{"kind": "not_ready", "detail": "x"}],
    }
    assert status.Reconciliation("m", "p", ()).consistent
    assert status.DisagreementError.category == "runtime"


# ---- records ---------------------------------------------------------------


def test_a_new_state_file_reads_as_empty(tmp_path):
    store = support_runtime.make_registry(tmp_path)
    keeper = records.Records(store)
    assert keeper.read() == {"format": 1, "models": {}}
    assert keeper.ledger is None
    assert keeper.policy == control.Polling()
    assert keeper.lookup(MODEL, 1) is status.State.CANDIDATE
    assert keeper.environment(MODEL, "production") is None


def test_a_state_file_of_another_format_is_refused(tmp_path):
    store = support_runtime.make_registry(tmp_path)
    (tmp_path / "release_state.json").write_text('{"format": 2}')
    with pytest.raises(errors.ArtifactError, match="unsupported release state"):
        records.Records(store).read()


def test_lookup_refuses_an_unregistered_version(tmp_path):
    store = support_runtime.make_registry(tmp_path)
    with pytest.raises(errors.ArtifactError):
        records.Records(store).lookup(MODEL, 9)


def test_move_records_where_a_version_came_from(tmp_path):
    keeper = records.Records(support_runtime.make_registry(tmp_path))
    state = keeper.read()
    entry = keeper.entry(state, MODEL)
    assert entry == {"versions": {}, "environments": {}}
    assert keeper.entry(state, MODEL) is entry
    keeper.move(
        entry,
        2,
        status.State.CHALLENGER,
        "why",
        status.State.CANDIDATE,
    )
    record = entry["versions"]["2"]
    assert record["state"] == "challenger"
    assert len(record["history"]) == 1
    step = record["history"][0]
    assert (step["from"], step["to"], step["reason"]) == (
        "candidate",
        "challenger",
        "why",
    )
    keeper.write(state)
    assert keeper.states(MODEL) == {
        1: status.State.CANDIDATE,
        2: status.State.CHALLENGER,
        3: status.State.CANDIDATE,
    }


def test_settle_returns_done_only_without_reasons(release, plane):
    done = release.settle(
        "promoted", MODEL, 2, "prod", plane, "r", support_runtime.NEW
    )
    assert done == status.Outcome(True, (), None, 2, "r")
    refused = release.settle(
        "rejected",
        MODEL,
        2,
        "prod",
        plane,
        None,
        support_runtime.NEW,
        ["no"],
        False,
    )
    assert refused == status.Outcome(False, ("no",), False)
    body = release.ledger.rows[-1]["body"]
    assert body == {
        "event": "rejected",
        "model": MODEL,
        "version": 2,
        "environment": "prod",
        "provider": "memory",
        "revision": None,
        "image_digest": support_runtime.NEW,
        "reasons": ["no"],
        "restored": False,
    }


def test_log_without_a_ledger_still_audits(tmp_path, plane):
    keeper = records.Records(support_runtime.make_registry(tmp_path))
    keeper.log(
        keeper.body(MODEL, 1, "p", plane, "x", None, support_runtime.NEW, [])
    )
    audit = (tmp_path / "audit.jsonl").read_text()
    assert "release_event" in audit


# ---- approval --------------------------------------------------------------


def test_production_and_approval_cannot_be_set_by_hand(release):
    for target in (
        status.State.PRODUCTION,
        status.State.APPROVED,
    ):
        with pytest.raises(errors.ArtifactError, match="cannot be set by hand"):
            release.transition(MODEL, 3, target, "x")


def test_a_hand_transition_follows_the_allowed_moves(release):
    release.transition(MODEL, 3, status.State.CHALLENGER, "compare")
    assert release.lookup(MODEL, 3) is status.State.CHALLENGER
    release.transition(MODEL, 3, status.State.RETIRED, "drop")
    with pytest.raises(
        errors.ArtifactError, match="churn v3: retired -> challenger"
    ):
        release.transition(MODEL, 3, status.State.CHALLENGER, "again")


def test_an_approval_is_stored_with_the_version(release):
    stored = release.read()["models"][MODEL]["versions"]["1"]
    assert stored["state"] == "approved"
    assert stored["approval"]["reviewer"] == "reviewer-1"
    assert release.refusal(MODEL, 1, support_release.KEY) == []


def test_a_version_cannot_be_approved_twice(release):
    with pytest.raises(errors.ArtifactError, match="churn v1 is approved"):
        release.approve(
            MODEL, 1, support_runtime.sanction(1), support_release.KEY
        )


def test_an_approval_for_another_bundle_is_refused(release):
    with pytest.raises(errors.ArtifactError):
        release.approve(
            MODEL, 3, support_runtime.sanction(2), support_release.KEY
        )
    assert release.lookup(MODEL, 3) is status.State.CANDIDATE


def test_an_unsigned_bundle_cannot_be_approved(tmp_path):
    store = support_runtime.make_registry(tmp_path, signed=False)
    gate = approval.Approval(store)
    with pytest.raises(errors.ArtifactError, match="unsigned bundle"):
        gate.approve(MODEL, 1, support_runtime.sanction(1), support_release.KEY)
    assert gate.refusal(MODEL, 1, support_release.KEY) == [
        "churn v1 is candidate, not approved",
        "the bundle is unsigned",
        "the version has no stored approval",
    ]


def test_a_stored_approval_under_another_key_is_not_valid(release):
    reasons = release.refusal(MODEL, 1, "another-key")
    assert len(reasons) == 1
    assert reasons[0].startswith("the approval is not valid: ")


# ---- rollout ---------------------------------------------------------------


def test_restore_reports_nothing_true_or_false(release, plane):
    assert release.restore(plane, None) is None
    back = {"revision": "serve-1", "image_digest": support_runtime.OLD}
    assert release.restore(plane, back) is True
    wrong = {"revision": "serve-1", "image_digest": support_runtime.NEW}
    assert release.restore(plane, wrong) is False


def test_present_checks_the_revision_and_its_digest(plane):
    check = rollout.Rollout.present("serve-1", support_runtime.OLD)
    seen = plane.observe()
    assert check(seen) == []
    assert rollout.Rollout.present("serve-1", support_runtime.NEW)(seen) == [
        f"revision serve-1 runs {support_runtime.OLD}"
    ]
    assert rollout.Rollout.present("serve-9", support_runtime.OLD)(seen) == [
        "revision serve-9 is not reported"
    ]


def test_shift_splits_with_the_previous_revision_below_full(release, plane):
    revision = plane.deploy(support_runtime.NEW)
    asked = []
    original = plane.route

    def record(allocation):
        asked.append(dict(allocation))
        original(allocation)

    plane.route = record
    release.shift(
        plane,
        revision,
        support_runtime.NEW,
        (10, 100),
        None,
        {"revision": "serve-1"},
    )
    assert asked == [{"serve-2": 10, "serve-1": 90}, {"serve-2": 100}]


def test_shift_stops_when_health_objects(release, plane):
    revision = plane.deploy(support_runtime.NEW)
    seen = []

    def health(observed, percent):
        seen.append(percent)
        return ["too many errors"] if percent == 50 else []

    with pytest.raises(control.UnconfirmedError, match="at 50%"):
        release.shift(
            plane,
            revision,
            support_runtime.NEW,
            (10, 50, 100),
            health,
            {"revision": "serve-1"},
        )
    assert seen == [10, 50]


# ---- promotion -------------------------------------------------------------


def test_a_promotion_moves_the_registry_only_after_confirmation(release, plane):
    outcome = ship(release, plane, 1)
    assert outcome == status.Outcome(True, (), None, 1, "serve-2")
    assert release.lookup(MODEL, 1) is status.State.PRODUCTION
    assert release.store.aliases(MODEL)[registry.CHAMPION] == 1
    record = release.environment(MODEL, "production")
    assert record["revision"] == "serve-2"
    assert record["image_digest"] == support_runtime.NEW
    assert record["known_good"] == []
    assert events(release) == ["promoted"]


def test_a_second_promotion_retires_the_first_and_keeps_it_known_good(
    release, plane
):
    ship(release, plane, 1)
    other = "sha256:" + "7" * 64
    outcome = ship(release, plane, 2, other, steps=(25, 100))
    assert outcome.done
    assert outcome.revision == "serve-3"
    assert release.lookup(MODEL, 1) is status.State.RETIRED
    assert release.lookup(MODEL, 2) is status.State.PRODUCTION
    assert release.environment(MODEL, "production")["known_good"] == [
        {
            "version": 1,
            "revision": "serve-2",
            "image_digest": support_runtime.NEW,
        }
    ]


@pytest.mark.parametrize("steps", [(), (50,), (50, 25, 100), (50, 50, 100)])
def test_promotion_steps_must_increase_and_end_in_full(release, plane, steps):
    with pytest.raises(errors.ConfigError, match="increase and end in 100"):
        ship(release, plane, 1, steps=steps)
    assert plane.calls == []


def test_an_unapproved_version_is_refused_without_asking_the_platform(
    release, plane
):
    outcome = ship(release, plane, 3)
    assert not outcome.done
    assert outcome.reasons == (
        "churn v3 is candidate, not approved",
        "the version has no stored approval",
    )
    assert outcome.restored is None
    assert plane.calls == []
    assert events(release) == ["rejected"]


def test_an_unconfirmed_promotion_restores_and_changes_nothing(release):
    plane = support_runtime.MemoryPlane(runs=support_runtime.OTHER)
    ship(release, support_runtime.MemoryPlane(), 1)
    before = release.read()
    outcome = ship(release, plane, 2)
    assert not outcome.done
    assert outcome.restored is False
    assert release.read() == before
    assert events(release) == ["promoted", "rejected"]


def test_a_staged_first_promotion_without_a_previous_revision_fails(
    release, plane
):
    outcome = ship(release, plane, 1, steps=(50, 100))
    assert not outcome.done
    assert outcome.reasons == ("a staged rollout needs a previous revision",)
    assert outcome.restored is None


def test_a_promotion_outside_production_leaves_the_alias(release, plane):
    outcome = ship(release, plane, 1, environment="staging")
    assert outcome.done
    assert registry.CHAMPION not in release.store.aliases(MODEL)
    assert release.lookup(MODEL, 1) is status.State.APPROVED


def test_history_appends_the_previous_record_last():
    entry = {"environments": {"p": {"known_good": [{"version": 1}]}}}
    previous = {"version": 2, "revision": "r", "image_digest": "d", "x": 1}
    made = promotion.Promotion.history(entry, "p", previous)
    assert made == [
        {"version": 1},
        {"version": 2, "revision": "r", "image_digest": "d"},
    ]
    assert promotion.Promotion.history(entry, "q", None) == []


# ---- rollback --------------------------------------------------------------


def test_a_rollback_without_a_known_good_revision_does_nothing(release, plane):
    outcome = release.rollback(MODEL, "production", plane, support_release.KEY)
    assert outcome == status.Outcome(
        False, ("there is no previous known-good revision",)
    )
    ship(release, plane, 1)
    assert not release.rollback(
        MODEL, "production", plane, support_release.KEY
    ).done
    assert plane.calls.count("set_traffic") == 1


def test_a_confirmed_rollback_restores_the_previous_version(release, plane):
    ship(release, plane, 1)
    ship(release, plane, 2, "sha256:" + "7" * 64)
    outcome = release.rollback(MODEL, "production", plane, support_release.KEY)
    assert outcome == status.Outcome(True, (), None, 1, "serve-2")
    assert release.lookup(MODEL, 1) is status.State.PRODUCTION
    assert release.lookup(MODEL, 2) is status.State.RETIRED
    assert release.store.aliases(MODEL)[registry.CHAMPION] == 1
    record = release.environment(MODEL, "production")
    assert record["version"] == 1
    assert record["known_good"] == []
    assert events(release)[-1] == "rolled_back"


def test_an_unconfirmed_rollback_puts_the_serving_revision_back(release, plane):
    ship(release, plane, 1)
    ship(release, plane, 2, "sha256:" + "7" * 64)
    plane.digests["serve-2"] = support_runtime.OTHER
    outcome = release.rollback(MODEL, "production", plane, support_release.KEY)
    assert not outcome.done
    assert outcome.restored is True
    assert release.lookup(MODEL, 2) is status.State.PRODUCTION
    assert events(release)[-1] == "rollback_rejected"


def test_rollback_is_part_of_the_release_registry():
    assert issubclass(reconcile.Registry, rollback.Rollback)
    assert issubclass(reconcile.Registry, promotion.Promotion)


# ---- reconcile -------------------------------------------------------------


def test_live_returns_the_confirmed_version(release, plane):
    ship(release, plane, 1)
    seen = plane.observe()
    assert release.live(MODEL, "production", seen) == 1
    assert release.reconcile(MODEL, "production", seen).consistent
    assert release.recall(MODEL, "production")["version"] == 1
    assert release.recall(MODEL, "staging") is None


def test_live_without_a_record_disagrees(release, plane):
    seen = plane.observe()
    with pytest.raises(
        status.DisagreementError, match="not confirmed deployed"
    ):
        release.live(MODEL, "production", seen)
    report = release.reconcile(MODEL, "production", seen)
    assert [f.kind for f in report.findings] == ["no_registry_record"]
    assert report.findings[0].detail == (
        f"the platform runs {support_runtime.OLD} but the registry records no deployment "
        "of churn in production"
    )


def test_every_drift_is_a_finding(release, plane):
    ship(release, plane, 1)
    seen = observation.Observation(
        provider="memory",
        service="serve",
        revision="serve-1",
        image_digest=support_runtime.OLD,
        ready=False,
        traffic=(
            observation.Traffic("serve-1", 60),
            observation.Traffic("serve-2", 40),
        ),
        observed=support_release.NOW,
        revisions=(
            ("serve-1", support_runtime.OLD),
            ("serve-2", support_runtime.NEW),
        ),
    )
    report = release.reconcile(MODEL, "production", seen)
    assert [f.kind for f in report.findings] == [
        "revision_mismatch",
        "digest_mismatch",
        "traffic_split",
        "not_ready",
    ]
    assert report.findings[0].detail == (
        "registry says serve-2, the platform serves serve-1"
    )
    assert (
        report.findings[2].detail
        == "the recorded revision holds 40% of traffic"
    )


def test_a_state_file_out_of_step_is_a_state_mismatch(release, plane):
    ship(release, plane, 1)
    state = release.read()
    versions = state["models"][MODEL]["versions"]
    versions["1"]["state"] = "retired"
    versions["2"]["state"] = "production"
    release.write(state)
    report = release.reconcile(MODEL, "production", plane.observe())
    assert [f.detail for f in report.findings] == [
        "v1 is deployed but its state is retired",
        "v2 is in production state but the environment records v1",
    ]


def test_an_alias_and_a_ledger_out_of_step_are_findings(tmp_path, plane):
    store = support_runtime.make_registry(tmp_path)
    bare = reconcile.Registry(store, None, support_runtime.NO_WAIT)
    bare.approve(MODEL, 1, support_runtime.sanction(1), support_release.KEY)
    bare.promote(
        MODEL,
        1,
        plane,
        "production",
        support_runtime.NEW,
        signing_key=support_release.KEY,
    )
    seen = plane.observe()
    assert bare.recall(MODEL, "production") is None
    assert bare.reconcile(MODEL, "production", seen).consistent
    bare.ledger = ledger()
    report = bare.reconcile(MODEL, "production", seen)
    assert [f.kind for f in report.findings] == ["ledger_missing"]
    state = bare.read()
    state["models"][MODEL]["environments"] = {}
    bare.write(state)
    report = bare.reconcile(MODEL, "production", seen)
    kinds = [f.kind for f in report.findings]
    assert kinds == ["no_registry_record", "alias_not_deployed"]
    assert report.findings[1].detail == (
        "the champion alias points at v1, which the runtime has not "
        "confirmed (confirmed: None)"
    )


def test_a_ledger_naming_another_revision_disagrees(release, plane):
    ship(release, plane, 1)
    state = release.read()
    state["models"][MODEL]["environments"]["production"]["image_digest"] = (
        support_runtime.OLD
    )
    release.write(state)
    report = release.reconcile(MODEL, "production", plane.observe())
    assert "ledger_disagrees" in [f.kind for f in report.findings]
    assert report.to_dict()["consistent"] is False
    assert json.dumps(report.to_dict())
