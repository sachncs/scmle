"""R5 records (governance, deploy, monitoring, runners) across key versions.

Each fixture in `tests/fixtures/records/r5` was written by the writer that
existed before the phase 5 key vocabulary (version 1). These tests check that
every such record still loads and migrates, that signed, sealed and chained
ones still verify on their stored bytes, that identity digests are unchanged,
and that every migration is undone by its revert (round trip).
"""

import dataclasses
import json
import pathlib
import shutil

import hypothesis
from hypothesis import strategies
import pytest

from scmle.contracts import errors
from scmle.contracts import observability
from scmle.contracts import records
from scmle.data import reference
from scmle.deploy import contract
from scmle.deploy import livetests
import scmle.deploy.record
from scmle.governance import attestation
from scmle.governance import deployment_ledger
from scmle.governance import keyring
from scmle.governance import review
import scmle.governance.loop
from scmle.governance.loop import terms
from scmle.monitoring import incidents
from scmle.monitoring import slo
from scmle.runners.base import ledger

FIXTURES = pathlib.Path(__file__).parents[1] / "fixtures" / "records" / "r5"
TEXT = strategies.text(
    strategies.characters(min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=12,
)


def stored(name):
    """A version-1 fixture."""
    return json.loads((FIXTURES / f"{name}.json").read_text("utf-8"))


def revert(migrations, layout, version=records.ORIGIN):
    """A current layout, without its version key, moved down to `version`."""
    content = {k: v for k, v in layout.items() if k != records.KEY}
    return records.downgrade(migrations, content, version)


def lift(migrations, layout):
    """A version-1 layout moved up to the current version."""
    return records.upgrade(migrations, json.loads(json.dumps(layout)), 1)


# ---- the deployment spec: an identity digest anchored at format 1 ------


def test_a_format_one_spec_loads_and_keeps_its_digest():
    fixture = stored("deployment_spec")
    spec = contract.Deployment.from_dict(fixture["record"])
    assert spec.bundle.version == "v3"
    assert spec.resources.requests == contract.Quota(0.5, 256)
    assert spec.resources.limits == contract.Quota(1.0, 512)
    assert (spec.health.period, spec.health.timeout) == (5.0, 1.0)
    assert spec.digest() == fixture["digest"]
    written = spec.to_dict()
    assert written["format"] == contract.SPEC_FORMAT == 2
    assert written["resources"]["limits"] == {"cpu": 1.0, "memory": 512}
    again = contract.Deployment.from_dict(written)
    assert again == spec and again.digest() == fixture["digest"]


def test_a_spec_format_newer_than_the_build_is_refused():
    record = stored("deployment_spec")["record"]
    with pytest.raises(errors.ConfigError, match="format"):
        contract.Deployment.from_dict({**record, "format": 3})


@hypothesis.given(memory=strategies.integers(1, 4096), path=TEXT)
def test_the_spec_migration_round_trips(memory, path):
    layout = stored("deployment_spec")["record"]
    del layout["format"]
    layout["resources"]["memory_limit_mb"] = memory
    layout["health"]["liveness_path"] = "/" + path
    migrations = contract.Deployment.migrations
    assert revert(migrations, lift(migrations, layout)) == layout


# ---- the live check -----------------------------------------------------


def test_a_version_one_check_loads_and_writes_version_two():
    record = stored("live_check")["record"]
    check = livetests.Check.from_dict(record)
    assert check.commit == record["source_commit"]
    assert check.digests == {
        "evidence": record["evidence_digest"],
        "container": record["container_digest"],
        "model": record["model_digest"],
    }
    written = check.to_dict()
    assert written[records.KEY] == 2
    assert livetests.Check.from_dict(written) == check
    assert revert(livetests.Check.migrations, written) == record


def test_a_check_missing_a_digest_is_refused():
    record = stored("live_check")["record"]
    del record["model_digest"]
    with pytest.raises(errors.ConfigError, match="missing"):
        livetests.Check.from_dict(record)


# ---- the signed deployment record --------------------------------------


def test_a_version_one_deployment_record_still_verifies():
    fixture = stored("deploy_record")
    record = scmle.deploy.record.Record.from_dict(fixture["record"])
    assert record.schema_version == 1
    assert record.commit == fixture["record"]["source_commit"]
    assert record.digests["config"] == fixture["record"]["config_digest"]
    assert record.verify(fixture["key"])
    assert record.digest() == fixture["digest"]
    assert record.to_dict() == fixture["record"]
    forged = dataclasses.replace(record, approver="intruder")
    assert not forged.verify(fixture["key"])


def test_a_deployment_record_signed_now_is_version_two():
    fixture = stored("deploy_record")
    old = scmle.deploy.record.Record.from_dict(fixture["record"])
    fresh = dataclasses.replace(
        old, schema_version=scmle.deploy.record.VERSION
    ).signed(fixture["key"])
    written = fresh.to_dict()
    assert written[records.KEY] == 2 and "format" not in written
    again = scmle.deploy.record.Record.from_dict(written)
    assert again == fresh and again.verify(fixture["key"])
    assert not dataclasses.replace(again, schema_version=1).verify(
        fixture["key"]
    )


@hypothesis.given(approver=TEXT, revision=TEXT)
def test_the_deployment_record_migration_round_trips(approver, revision):
    layout = stored("deploy_record")["record"]
    del layout["format"]
    layout.update(approver=approver, revision=revision)
    migrations = scmle.deploy.record.MIGRATIONS
    assert revert(migrations, lift(migrations, layout)) == layout


# ---- the release attestation -------------------------------------------


def test_a_version_one_attestation_still_verifies():
    fixture = stored("release_attestation")
    loaded = attestation.Attestation.from_dict(fixture["record"])
    assert loaded.schema_version == 1
    assert loaded.run == "run-1"
    assert loaded.to_dict() == fixture["record"]
    assert loaded.verify(fixture["provenance"], fixture["sbom"], fixture["key"])
    assert not loaded.verify(fixture["provenance"], fixture["sbom"], "other")
    tampered = json.loads(json.dumps(fixture["record"]))
    tampered["subjects"]["sbom_sha256"] = "0" * 64
    assert not attestation.Attestation.from_dict(tampered).verify(
        fixture["provenance"], fixture["sbom"], fixture["key"]
    )


def test_an_attestation_made_now_is_version_two():
    fixture = stored("release_attestation")
    fresh = attestation.Attestation.from_records(
        fixture["provenance"], fixture["sbom"], fixture["key"]
    )
    written = fresh.to_dict()
    assert written[records.KEY] == 2
    assert set(written["digests"]) == {"manifest", "provenance", "sbom"}
    assert attestation.Attestation.from_dict(written).verify(
        fixture["provenance"], fixture["sbom"], fixture["key"]
    )


@hypothesis.given(run=TEXT, signed=strategies.booleans())
def test_the_attestation_migration_round_trips(run, signed):
    layout = stored("release_attestation")["record"]
    del layout[records.KEY]
    layout.update(run_id=run, signature_present=signed)
    migrations = attestation.Attestation.migrations
    assert revert(migrations, lift(migrations, layout)) == layout


# ---- the keyring and its signature envelopes ----------------------------


def test_a_version_one_keyring_loads_and_its_signatures_verify():
    fixture = stored("keyring")
    ring = keyring.Keyring.from_dict(fixture["record"])
    retired = ring.get("k0")
    assert (retired.changed, retired.successor) == (
        "2026-10-01T00:00:00+00:00",
        "k1",
    )
    wrapper = keyring.Wrapper.from_dict(fixture["wrapper"])
    assert wrapper.issued == fixture["wrapper"]["signed_at"]
    assert ring.verify(
        fixture["payload"],
        wrapper,
        {"k1": fixture["secret"]},
        "2026-10-03T00:00:00+00:00",
    )
    written = ring.to_dict()
    assert written[records.KEY] == 2
    assert keyring.Keyring.from_dict(written).to_json() == ring.to_json()
    assert revert(keyring.Keyring.migrations, {"keys": written["keys"]}) == {
        "keys": fixture["record"]["keys"]
    }


def test_a_keyring_newer_than_the_build_is_refused():
    record = stored("keyring")["record"]
    with pytest.raises(ValueError, match="schema version"):
        keyring.Keyring.from_dict({**record, records.KEY: 3})


@hypothesis.given(owner=TEXT, reason=TEXT)
def test_the_keyring_migration_round_trips(owner, reason):
    layout = {"keys": stored("keyring")["record"]["keys"]}
    layout["keys"][0].update(owner=owner, reason=reason)
    migrations = keyring.Keyring.migrations
    assert revert(migrations, lift(migrations, layout)) == layout
    wrapper = stored("keyring")["wrapper"]
    steps = keyring.Wrapper.migrations
    assert revert(steps, lift(steps, wrapper)) == wrapper


# ---- the release review -------------------------------------------------


def test_a_version_one_review_still_verifies():
    fixture = stored("release_review")
    loaded = review.Signoff.from_dict(fixture["record"])
    assert loaded.schema_version == 1
    assert loaded.digests["artifacts"] == {"wheel": "sha256:w"}
    assert loaded.claims[0].criterion == "metric.auc"
    assert loaded.to_dict() == fixture["record"]
    assert loaded.verify(fixture["key"])
    assert loaded.accepted("abc1234", ("metric",), fixture["key"])
    changed = dataclasses.replace(loaded, reviewer="someone@example.com")
    assert not changed.verify(fixture["key"])


def test_a_review_signed_now_is_version_two():
    fixture = stored("release_review")
    old = review.Signoff.from_dict(fixture["record"])
    fresh = dataclasses.replace(old, schema_version=2, signature=None).sign(
        fixture["key"]
    )
    written = fresh.to_dict()
    assert written[records.KEY] == 2 and written["claims"][0]["uri"]
    again = review.Signoff.from_dict(written)
    assert again == fresh and again.verify(fixture["key"])


@hypothesis.given(release=TEXT, notes=TEXT)
def test_the_review_migration_round_trips(release, notes):
    layout = stored("release_review")["record"]
    del layout[records.KEY]
    layout["release_id"] = release
    layout["claims"][0]["notes"] = notes
    migrations = review.Signoff.migrations
    assert revert(migrations, lift(migrations, layout)) == layout


# ---- the deployment ledger: chained, signed rows and the loop policy ---


def copy_ledger(tmp_path):
    """The version-1 deployment ledger, in a deployment directory."""
    directory = tmp_path / "deploy"
    directory.mkdir()
    shutil.copy(
        FIXTURES / "deployment_ledger.jsonl",
        directory / deployment_ledger.FILE_NAME,
    )
    return directory


def test_a_version_one_ledger_verifies_and_its_replies_still_verify(tmp_path):
    fixture = stored("deployment_ledger")
    directory = copy_ledger(tmp_path)
    loaded = deployment_ledger.Ledger.load(
        directory / deployment_ledger.FILE_NAME
    )
    reply = deployment_ledger.Attestation.from_dict(
        loaded.attestations["esc-1"]
    )
    assert reply.schema_version == 1 and reply.escalation == "esc-1"
    assert reply.digests["question"] == fixture["reply"]["question_hash"]
    assert reply.to_dict() == fixture["reply"]
    assert reply.verify(fixture["key"])
    envelope = deployment_ledger.Envelope.from_dict(loaded.envelopes["e1"])
    assert envelope.identifier == "e1" and envelope.attestor == "P. Owner"
    assert envelope.to_dict() == fixture["envelope"]
    assert envelope.verify(fixture["key"])
    forged = dataclasses.replace(envelope, limits={"train_challenger": 9})
    assert not forged.verify(fixture["key"])


def test_a_tampered_version_one_ledger_row_is_refused(tmp_path):
    directory = copy_ledger(tmp_path)
    path = directory / deployment_ledger.FILE_NAME
    text = path.read_text("utf-8")
    path.write_text(text.replace('"P. Owner"', '"Q. Owner"', 1), "utf-8")
    with pytest.raises(deployment_ledger.MisuseError):
        deployment_ledger.Ledger.load(path)


def test_a_version_one_ledger_resumes_with_its_amended_policy(tmp_path):
    fixture = stored("deployment_ledger")
    directory = copy_ledger(tmp_path)
    loop = scmle.governance.loop.Loop.resume(directory)
    original = terms.Preregistration.from_dict(fixture["policy"])
    assert original.target == 0.6 and original.critical == ("deny",)
    assert loop.policy == dataclasses.replace(
        original,
        target=0.5,
        rollback={**original.rollback, "delta": 0.02},
    )
    assert loop.policy.promotion == {"alpha": "1/20", "delta": 0.01}
    resumed = scmle.governance.loop.Loop.resume(directory)
    assert resumed.policy == loop.policy


def test_a_new_amendment_on_a_version_one_ledger_carries_its_version(tmp_path):
    directory = copy_ledger(tmp_path)
    loop = scmle.governance.loop.Loop.resume(directory)
    loop.amend(target=0.4, promotion={"delta": 0.05})
    body = loop.ledger.rows[-1]["body"]["policy_amendment"]
    assert body[records.KEY] == 2
    again = scmle.governance.loop.Loop.resume(directory)
    assert again.policy.target == 0.4
    assert again.policy.promotion == {"alpha": "1/20", "delta": 0.05}
    assert again.policy.rollback["delta"] == 0.02


@hypothesis.given(
    target=strategies.floats(0.01, 0.99), scope=TEXT, timeout=TEXT
)
def test_the_policy_and_ledger_row_migrations_round_trip(
    target, scope, timeout
):
    fixture = stored("deployment_ledger")
    policy = {**fixture["policy"], "r_star": target, "scope": scope}
    assert revert(terms.MIGRATIONS, lift(terms.MIGRATIONS, policy)) == policy
    reply = {**fixture["reply"], "statement": timeout}
    steps = deployment_ledger.Attestation.migrations
    assert revert(steps, lift(steps, reply)) == reply
    envelope = {**fixture["envelope"], "statement": timeout}
    steps = deployment_ledger.Envelope.migrations
    assert revert(steps, lift(steps, envelope)) == envelope


# ---- the incident journal ------------------------------------------------


def evaluation():
    """The breached evaluation that opened the fixture incident."""
    target = slo.Slo(
        name="latency",
        metric="p95_latency_ms",
        kind=observability.Indicator.LATENCY,
        maximum=250.0,
        window=7,
        samples=1,
        owner="ml-on-call",
        runbook="https://runbooks.example/latency",
        route="on-call",
    )
    sample = observability.Sample(
        name="p95_latency_ms",
        kind=observability.Indicator.LATENCY,
        value=400.0,
        unit="ms",
        timestamp="2026-10-20T12:00:00+00:00",
    )
    return slo.evaluate(
        target, (sample,), observed_at="2026-10-20T12:00:00+00:00"
    )


def test_a_version_one_journal_verifies_and_continues_at_version_two(
    tmp_path,
):
    path = tmp_path / "incidents.jsonl"
    shutil.copy(FIXTURES / "incidents.jsonl", path)
    flow = incidents.Incidents(path)
    (current,) = flow.records()
    assert current.status is incidents.Phase.ACKNOWLEDGED
    assert (current.run, current.acknowledger) == ("run-42", "operator")
    reopened = flow.open(
        evaluation(),
        opened_at="2026-10-20T12:05:00+00:00",
        run_id="run-43",
        rollback_target="other",
    )
    assert reopened.identifier == current.identifier
    resolved = flow.resolve(
        current.identifier,
        actor="operator",
        resolved_at="2026-10-20T12:03:00+00:00",
        resolution="rolled back",
        evidence_digest="sha256:fix",
    )
    events = flow.read()
    assert [e["incident"].get(records.KEY) for e in events] == [None, None, 2]
    assert flow.records() == (resolved,)
    assert (
        revert(incidents.Incident.migrations, events[-1]["incident"])[
            "evidence_digest"
        ]
        == "sha256:fix"
    )


def test_a_tampered_version_one_journal_fails_closed(tmp_path):
    path = tmp_path / "incidents.jsonl"
    text = (FIXTURES / "incidents.jsonl").read_text("utf-8")
    path.write_text(text.replace('"run-42"', '"run-99"', 1), "utf-8")
    with pytest.raises(ValueError, match="digest"):
        incidents.Incidents(path).records()


@hypothesis.given(owner=TEXT, reason=TEXT)
def test_the_incident_migration_round_trips(owner, reason):
    line = (FIXTURES / "incidents.jsonl").read_text("utf-8").splitlines()[0]
    layout = {**json.loads(line)["incident"], "owner": owner, "reason": reason}
    migrations = incidents.Incident.migrations
    assert revert(migrations, lift(migrations, layout)) == layout
    rebuilt = incidents.Incident.from_dict(layout)
    assert incidents.Incident.from_dict(rebuilt.to_dict()) == rebuilt


# ---- the bundle's reference profile and the run manifest ----------------


def test_a_version_one_baseline_loads():
    record = stored("baseline")["record"]
    profile = reference.Baseline.from_dict(record)
    assert (profile.name, profile.metrics) == ("score", {"roc_auc": 0.8})
    written = profile.to_dict()
    assert written[records.KEY] == 3
    assert revert(reference.Baseline.migrations, written) == record


def test_a_version_one_run_manifest_loads():
    record = stored("run_manifest")["record"]
    manifest = ledger.Manifest.from_dict(record)
    assert manifest.identifier == "run-1" and manifest.cost == 0.1
    assert manifest.digests == {
        "spec": record["spec_digest"],
        "project": record["project_digest"],
        "result": record["result_sha256"],
    }
    written = manifest.to_dict()
    assert written["format"] == ledger.MANIFEST_FORMAT == 2
    assert ledger.Manifest.from_dict(written) == manifest
    content = {k: v for k, v in written.items() if k != "format"}
    restored = revert(ledger.Manifest.migrations, content)
    assert {**restored, "format": 1} == record


def test_a_run_manifest_newer_than_the_build_is_refused():
    record = stored("run_manifest")["record"]
    with pytest.raises(ValueError, match="newer"):
        ledger.Manifest.from_dict({**record, "format": 3})


@hypothesis.given(runner=TEXT, name=TEXT)
def test_the_manifest_and_baseline_migrations_round_trip(runner, name):
    layout = {**stored("run_manifest")["record"], "runner": runner}
    del layout["format"]
    steps = ledger.Manifest.migrations
    assert revert(steps, lift(steps, layout)) == layout
    profile = {**stored("baseline")["record"], "output_name": name}
    steps = reference.Baseline.migrations
    assert revert(steps, lift(steps, profile)) == profile


def test_every_r5_kind_registers_one_invertible_step():
    kinds = (
        contract.Deployment,
        livetests.Check,
        scmle.deploy.record.Record,
        attestation.Attestation,
        deployment_ledger.Attestation,
        deployment_ledger.Envelope,
        keyring.Keyring,
        keyring.Wrapper,
        terms.Preregistration,
        review.Signoff,
        incidents.Incident,
        reference.Baseline,
        ledger.Manifest,
    )
    for kind in kinds:
        # the baseline gained a second step: each reference writes `missing`
        assert len(kind.migrations) == 1 + (kind is reference.Baseline), kind
        step = kind.migrations[0]
        assert (step.change is None) == (step.revert is None), kind
