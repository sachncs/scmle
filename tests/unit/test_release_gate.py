"""Live checks and the promotion gate against recorded responses.

The HTTP exchanges in tests/fixtures/cloud/staging-http.json are shaped by
the documented service contract; none was captured from a cloud.
"""

import dataclasses

import pytest

from scmle.contracts import errors
from scmle.deploy import gate
from scmle.deploy import livetests
from scmle.deploy import record as record_module
from tests import support_release as sr


def named(results):
    return {r.name: r for r in results}


def test_the_recorded_staging_responses_pass_every_live_check():
    results = sr.run_live()
    assert [r.name for r in results] == list(livetests.CHECK_NAMES)
    assert all(r.passed for r in results), [r.problems for r in results]
    assert all(r.source_commit == sr.COMMIT for r in results)
    assert all(r.evidence_digest.startswith("sha256:") for r in results)


def test_the_evidence_digest_is_repeatable_and_tracks_the_exchange():
    first = named(sr.run_live())
    again = named(sr.run_live())
    assert {k: v.evidence_digest for k, v in first.items()} == {
        k: v.evidence_digest for k, v in again.items()
    }
    document = sr.recorded()
    for item in document["exchanges"]:
        if item["path"] == "/livez":
            item["body"] = {"status": "alive", "uptime": 1}
    changed = named(sr.run_live(document))
    assert changed["health"].evidence_digest != first["health"].evidence_digest
    assert changed["schema"].evidence_digest == first["schema"].evidence_digest


def mutate(path, method, auth, **fields):
    document = sr.recorded()
    for item in document["exchanges"]:
        if (item["path"], item["method"], item["auth"]) == (path, method, auth):
            for key, value in fields.items():
                if key == "body_update":
                    item["body"].update(value)
                else:
                    item[key] = value
    return document


SEEDED = {
    "health: not ready": (
        mutate(
            "/readyz", "GET", "none", status=503, body_update={"ready": False}
        ),
        "health",
    ),
    "health: other model": (
        mutate(
            "/health",
            "GET",
            "valid",
            body_update={"artifact_digest": "sha256:" + "1" * 64},
        ),
        "health",
    ),
    "schema: no digest": (
        mutate("/schema", "GET", "valid", status=503),
        "schema",
    ),
    "schema: other model": (
        mutate(
            "/schema",
            "GET",
            "valid",
            body_update={"artifact_digest": "sha256:" + "1" * 64},
        ),
        "schema",
    ),
    "inference: other model": (
        mutate(
            "/predict",
            "POST",
            "valid",
            body_update={"artifact_digest": "sha256:" + "1" * 64},
        ),
        "inference",
    ),
    "inference: error": (
        mutate("/predict", "POST", "valid", status=500),
        "inference",
    ),
    "security: anonymous predict allowed": (
        mutate("/predict", "POST", "none", status=200),
        "security",
    ),
    "security: forged token allowed": (
        mutate("/predict", "POST", "forged", status=200),
        "security",
    ),
    "security: anonymous schema allowed": (
        mutate("/schema", "GET", "none", status=200),
        "security",
    ),
    "security: traceback leaked": (
        mutate(
            "/predict",
            "POST",
            "none",
            body={"error": "Traceback (most recent call last)"},
        ),
        "security",
    ),
    "security: read-only token reaches predict": (
        mutate("/predict", "POST", "readonly", status=200),
        "security",
    ),
    "security: no challenge header": (
        mutate(
            "/predict",
            "POST",
            "none",
            headers={"content-type": "application/json"},
        ),
        "security",
    ),
}


@pytest.mark.parametrize("name", sorted(SEEDED))
def test_each_seeded_live_failure_fails_its_check_only(name):
    document, failing = SEEDED[name]
    results = named(sr.run_live(document))
    assert not results[failing].passed
    assert results[failing].problems
    assert all(r.passed for k, r in results.items() if k != failing)


def test_shifted_probabilities_fail_inference():
    document = sr.recorded()
    for item in document["exchanges"]:
        if item["path"] == "/predict" and item["auth"] == "valid":
            item["body"]["probabilities"][0][1] += 0.01
    result = named(sr.run_live(document))["inference"]
    assert not result.passed and "record 0" in result.problems[0]


def test_a_plain_http_url_fails_the_security_check_unless_loopback():
    assert not named(sr.run_live(base_url="http://staging.example.test"))[
        "security"
    ].passed
    assert named(sr.run_live(base_url="http://localhost:8000"))[
        "security"
    ].passed


def test_check_results_round_trip_and_refuse_unknown_fields():
    result = sr.run_live()[0]
    assert livetests.CheckResult.from_dict(result.to_dict()) == result
    with pytest.raises(errors.ConfigError):
        livetests.CheckResult.from_dict({**result.to_dict(), "extra": 1})


def test_http_fetch_refuses_a_relative_url():
    with pytest.raises(errors.ConfigError):
        livetests.http_fetch("staging.example.test")


# ---- the gate ----------------------------------------------------------


def test_a_complete_staging_release_is_admitted_and_certified_with_a_record():
    candidate = sr.candidate("staging")
    assert gate.admit(candidate, sr.KEY).allowed
    decision = gate.certify(candidate, sr.KEY, sr.NOW)
    assert decision.allowed and decision.reasons == ()
    record = decision.record
    assert record.verify(sr.KEY)
    assert (record.source_commit, record.container_digest) == (
        sr.COMMIT,
        sr.CONTAINER,
    )
    assert record.model_digest == candidate.model_digest
    assert record.config_digest == sr.CONFIG
    assert record.approver == "release-manager"
    assert set(dict(record.evidence)) == set(livetests.CHECK_NAMES)
    assert record.revision == "serve:12"


def test_a_record_round_trips_and_detects_tampering(tmp_path):
    record = sr.staging_record()
    path = tmp_path / "staging.json"
    record_module.write_record(path, record)
    again = record_module.read_record(path)
    assert again == record and again.verify(sr.KEY)
    assert not again.verify("another-key")
    forged = dataclasses.replace(again, approver="someone-else")
    assert not forged.verify(sr.KEY)
    assert again.digest() == record.digest() != forged.digest()


def test_a_record_refuses_malformed_fields():
    record = sr.staging_record()
    with pytest.raises(errors.ConfigError):
        dataclasses.replace(record, source_commit="abc123")
    with pytest.raises(errors.ConfigError):
        dataclasses.replace(record, container_digest="repo:latest")
    with pytest.raises(errors.ConfigError):
        dataclasses.replace(record, environment="local")
    with pytest.raises(errors.ConfigError):
        record_module.DeploymentRecord.from_dict(
            {**record.to_dict(), "format": 9}
        )


def refused(candidate, expected, certify=False):
    step = gate.certify if certify else gate.admit
    decision = step(candidate, sr.KEY)
    assert not decision.allowed
    assert decision.record is None
    assert any(expected in reason for reason in decision.reasons), (
        decision.reasons
    )
    return decision


MODEL_FAILURES = {
    "unsigned": ({"model": sr.version(signed=False)}, "unsigned"),
    "unapproved": ({"approval": None}, "has no approval"),
    "rejected": (
        {"approval": sr.approval(decision="rejected")},
        "rejected by the reviewer",
    ),
    "not independent": (
        {"approval": sr.approval(independent=False)},
        "independent review",
    ),
    "other bundle": (
        {"approval": sr.approval(content_id="8" * 64)},
        "does not match bundle identity",
    ),
    "other version": ({"approval": sr.approval(version=2)}, "does not match"),
    "forged": (
        {"approval": dataclasses.replace(sr.approval(), signature="00" * 32)},
        "valid signature",
    ),
}


@pytest.mark.parametrize("name", sorted(MODEL_FAILURES))
@pytest.mark.parametrize("environment", ["staging", "production"])
def test_a_bad_model_is_refused_at_every_step(name, environment):
    changes, expected = MODEL_FAILURES[name]
    staging = (
        {"staging_record": sr.staging_record()}
        if environment == "production"
        else {}
    )
    candidate = sr.candidate(environment, **changes, **staging)
    refused(candidate, expected)
    refused(candidate, expected, certify=True)


SCAN_FAILURES = {
    "no scan": ({"scan": None}, "no scan result"),
    "failed scan": (
        {"scan": gate.ImageScan(sr.CONTAINER, False, sr.SCAN_REPORT)},
        "did not pass",
    ),
    "scan of another image": (
        {"scan": gate.ImageScan(sr.OTHER_CONTAINER, True, sr.SCAN_REPORT)},
        "different image",
    ),
    "no approver": ({"approver": ""}, "no approver"),
    "self approval": ({"approver": "author"}, "requested the release"),
    "short commit": ({"source_commit": "abc"}, "full commit hash"),
    "unknown environment": ({"environment": "prod"}, "unknown environment"),
    "bad model digest": ({"model_digest": "latest"}, "model_digest"),
}


@pytest.mark.parametrize("name", sorted(SCAN_FAILURES))
def test_missing_or_failed_pre_deploy_evidence_is_refused(name):
    changes, expected = SCAN_FAILURES[name]
    base = sr.candidate("staging")
    refused(dataclasses.replace(base, **changes), expected)


def test_missing_evidence_is_never_a_pass():
    full = sr.candidate("staging")
    for name in livetests.CHECK_NAMES:
        fewer = tuple(c for c in full.checks if c.name != name)
        refused(
            dataclasses.replace(full, checks=fewer),
            f"live check {name} is missing",
            certify=True,
        )
    refused(dataclasses.replace(full, checks=()), "is missing", certify=True)
    refused(
        dataclasses.replace(full, observation=None),
        "was not read",
        certify=True,
    )


@pytest.mark.parametrize("name", livetests.CHECK_NAMES)
def test_a_failed_live_check_blocks_the_record(name):
    full = sr.candidate("staging")
    failed = tuple(
        dataclasses.replace(c, passed=False, problems=("seeded",))
        if c.name == name
        else c
        for c in full.checks
    )
    decision = refused(
        dataclasses.replace(full, checks=failed),
        f"live check {name} failed: seeded",
        certify=True,
    )
    assert len(decision.reasons) == 1


def test_evidence_from_another_build_is_refused():
    full = sr.candidate("staging")
    for field, value in (
        ("source_commit", "0" * 40),
        ("container_digest", sr.OTHER_CONTAINER),
        ("environment", "production"),
        ("model_digest", "sha256:" + "5" * 64),
    ):
        stale = tuple(
            dataclasses.replace(c, **{field: value}) for c in full.checks
        )
        refused(
            dataclasses.replace(full, checks=stale),
            "different build",
            certify=True,
        )


def test_the_platform_must_show_the_candidate_running():
    full = sr.candidate("staging")
    refused(
        dataclasses.replace(
            full, observation=sr.seen(image_digest=sr.OTHER_CONTAINER)
        ),
        "not " + sr.CONTAINER,
        certify=True,
    )
    refused(
        dataclasses.replace(full, observation=sr.seen(ready=False)),
        "not report the service ready",
        certify=True,
    )
    refused(
        dataclasses.replace(full, observation=sr.seen(provider="gcp")),
        "another provider",
        certify=True,
    )


def test_production_needs_a_matching_signed_staging_record():
    staged = sr.staging_record()
    good = sr.candidate("production", staging_record=staged)
    assert gate.certify(good, sr.KEY, sr.NOW).allowed
    refused(sr.candidate("production"), "needs a staging deployment record")
    forged = dataclasses.replace(staged, approver="intruder")
    refused(
        sr.candidate("production", staging_record=forged),
        "signature is not valid",
    )
    for field, value, expected in (
        ("container_digest", sr.OTHER_CONTAINER, "different container_digest"),
        ("source_commit", "0" * 40, "different source_commit"),
        ("model_digest", "sha256:" + "2" * 64, "different model_digest"),
    ):
        other = dataclasses.replace(staged, **{field: value}).signed(sr.KEY)
        refused(sr.candidate("production", staging_record=other), expected)
    thin = dataclasses.replace(staged, evidence=staged.evidence[:2]).signed(
        sr.KEY
    )
    refused(
        sr.candidate("production", staging_record=thin),
        "staging evidence is missing",
    )
    again = dataclasses.replace(staged, environment="production").signed(sr.KEY)
    refused(sr.candidate("production", staging_record=again), "not for staging")


def test_a_staging_record_signed_with_another_key_is_refused():
    other = dataclasses.replace(sr.staging_record()).signed("some-other-key")
    refused(
        sr.candidate("production", staging_record=other),
        "signature is not valid",
    )


def test_every_reason_is_reported_not_only_the_first():
    decision = gate.admit(
        sr.candidate(
            "staging",
            model=sr.version(signed=False),
            approval=None,
            scan=None,
            approver="author",
        ),
        sr.KEY,
    )
    assert len(decision.reasons) >= 4
    assert (
        gate.admit(sr.candidate("staging"), sr.KEY).to_dict()["allowed"] is True
    )


def test_an_unusable_signing_key_never_certifies():
    decision = gate.certify(sr.candidate("staging"), "", sr.NOW)
    assert not decision.allowed and decision.record is None
