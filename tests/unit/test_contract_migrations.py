"""Version-1 contract records still load, verify and keep their identities.

Each fixture in `tests/fixtures/records/contracts` was written by the
contract code before the record keys were renamed (schema version 1). Every
kind must load it through its migration, round-trip at the current version,
and step back to the exact version-1 layout. Signatures are checked on the
stored layout, and identity digests are computed on the version-1 layout, so
neither changes when a key is renamed.
"""

import json
import pathlib

import hypothesis
from hypothesis import strategies
import pytest

from scmle.assurance.ledger.ledger import chain
from scmle.contracts import artifacts
from scmle.contracts import data
from scmle.contracts import evidence
from scmle.contracts import experiments
from scmle.contracts import lineage
from scmle.contracts import performance
from scmle.contracts import qualification
from scmle.contracts import records
from scmle.contracts import reliability
from scmle.contracts import serving
from scmle.governance import attestation

FIXTURES = pathlib.Path(__file__).parents[1] / "fixtures" / "records"
VERSION_ONE = FIXTURES / "contracts"
KEY = "fixture-signing-key"
STANDALONE = {
    "interface": artifacts.Interface,
    "dataset_manifest": data.Manifest,
    "passport": data.Passport,
    "experiment_plan": experiments.Plan,
    "performance_run": performance.Run,
    "qualification_check": qualification.Check,
    "qualification_claim": qualification.Claim,
    "qualification_certificate": qualification.Certificate,
}
CONTAINERS = {
    "evidence_index": evidence.Index,
    "lineage_graph": lineage.Graph,
    "reliability_registry": reliability.Registry,
}
TEXT = strategies.text(min_size=1, max_size=12).filter(str.strip)


def stored(name):
    """The parsed version-1 fixture."""
    return json.loads((VERSION_ONE / f"{name}.json").read_text("utf-8"))


def test_every_fixture_names_a_kind():
    names = {path.stem for path in VERSION_ONE.glob("*.json")}
    assert names == set(STANDALONE) | set(CONTAINERS)


@pytest.mark.parametrize("name", sorted(STANDALONE))
def test_a_version_one_record_loads_and_steps_back_exactly(name):
    kind = STANDALONE[name]
    old = stored(name)
    record = kind.from_dict(old)
    current = record.to_dict()
    assert current[records.KEY] == records.ORIGIN + 1
    assert kind.from_dict(current) == record
    layout = dict(current)
    del layout[records.KEY]
    restored = records.downgrade(kind.migrations, layout)
    expected = {k: v for k, v in old.items() if k != records.KEY}
    assert json.loads(json.dumps(restored)) == expected


@pytest.mark.parametrize("name", sorted(STANDALONE))
def test_a_record_newer_than_the_build_is_refused(name):
    current = STANDALONE[name].from_dict(stored(name)).to_dict()
    with pytest.raises(ValueError, match="newer than this build"):
        STANDALONE[name].from_dict({**current, records.KEY: 3})


@pytest.mark.parametrize(
    "kind",
    [
        artifacts.Interface,
        data.Origin,
        data.Ancestry,
        data.Confidentiality,
        data.Finding,
        data.Manifest,
        data.Passport,
        evidence.Record,
        experiments.Plan,
        lineage.Node,
        lineage.Edge,
        performance.Observation,
        performance.Run,
        qualification.Check,
        qualification.Claim,
        qualification.Certificate,
        reliability.Evidence,
        reliability.Plan,
    ],
)
def test_each_kind_registers_one_invertible_step(kind):
    assert len(kind.migrations) == 1
    step = kind.migrations[0]
    assert len(set(step.renames.values())) == len(step.renames)
    assert (step.change is None) == (step.revert is None)


@hypothesis.given(
    identifier=TEXT,
    statement=TEXT,
    checks=strategies.lists(TEXT, min_size=1, max_size=3),
    effect=strategies.floats(-10, 10),
)
def test_a_claim_round_trips_through_both_versions(
    identifier, statement, checks, effect
):
    claim = qualification.Claim(
        identifier=identifier,
        statement=statement,
        checks=tuple(checks),
        adjusted_p_value=None,
        effect=effect,
        tier=qualification.Tier.UNVERIFIED,
        assumptions=(),
    )
    current = claim.to_dict()
    del current[records.KEY]
    old = records.downgrade(qualification.Claim.migrations, current)
    assert "claim_id" in old and "check_ids" in old
    assert qualification.Claim.from_dict(old) == claim
    assert qualification.Claim.from_dict(claim.to_dict()) == claim


@hypothesis.given(
    identifier=TEXT,
    rows=strategies.integers(0, 10**6),
    columns=strategies.integers(0, 100),
    schema=TEXT,
    content=TEXT,
    owner=TEXT,
    pii=strategies.booleans(),
)
def test_a_manifest_round_trips_through_both_versions(
    identifier, rows, columns, schema, content, owner, pii
):
    manifest = data.Manifest(
        identifier=identifier,
        digests={"schema": schema, "data": content},
        rows=rows,
        columns=columns,
        target="y",
        source=data.Origin(owner=owner, acquisition="export"),
        lineage=data.Ancestry(snapshot="raw", digests={"part": content}),
        privacy=data.Confidentiality(identifying=pii),
    )
    old = manifest.anchor()
    assert old["schema_hash"] == schema and old["data_hash"] == content
    assert old["privacy"]["contains_pii"] is pii
    assert data.Manifest.from_dict(old) == manifest
    assert data.Manifest.from_dict(manifest.to_dict()) == manifest


@hypothesis.given(
    elapsed=strategies.floats(0, 1e6),
    memory=strategies.none() | strategies.floats(0, 1e3),
    samples=strategies.integers(0, 10**6),
)
def test_a_performance_run_round_trips_through_both_versions(
    elapsed, memory, samples
):
    run = performance.Run(
        identifier="run",
        observations=(
            performance.Observation(
                operation=performance.Operation.FIT,
                elapsed=elapsed,
                memory=memory,
                samples=samples,
                machine="m",
                observed="2026-10-20T12:00:00+00:00",
                software="s",
            ),
        ),
    )
    current = run.to_dict()
    del current[records.KEY]
    old = records.downgrade(performance.Run.migrations, current)
    assert old["run_id"] == "run"
    assert old["observations"][0]["elapsed_seconds"] == elapsed
    assert performance.Run.from_dict(old) == run


def test_a_version_one_signed_evidence_index_still_verifies():
    old = stored("evidence_index")
    index = evidence.Index.from_dict(old)
    assert index.version == records.ORIGIN
    assert index.verify(KEY)
    assert not index.verify("another-key")
    assert index.ordered()[0].criterion == "accuracy"
    assert index.to_dict() == old


def test_a_new_evidence_index_is_signed_at_the_current_version():
    index = evidence.Index.from_dict(stored("evidence_index"))
    fresh = evidence.Index(index.ordered())
    fresh.sign(KEY)
    record = fresh.to_dict()
    assert record[records.KEY] == evidence.VERSION
    assert "criterion" in record["records"][0]
    assert evidence.Index.from_dict(record).verify(KEY)


def test_a_tampered_version_one_index_fails_verification():
    old = stored("evidence_index")
    old["records"][0]["claim"] = "a different claim"
    assert not evidence.Index.from_dict(old).verify(KEY)


def test_the_lineage_digest_is_anchored_at_version_one():
    pinned = (VERSION_ONE / "lineage_graph.digest").read_text("utf-8").strip()
    graph = lineage.Graph.from_dict(stored("lineage_graph"))
    assert graph.digest() == pinned
    current = graph.to_dict()
    assert current[records.KEY] == lineage.Graph.schema_version
    assert "identifier" in current["nodes"][0]
    assert "source" in current["edges"][0]
    assert lineage.Graph.from_dict(current).digest() == pinned


def test_the_serving_schema_digest_is_anchored_at_version_one():
    old = stored("interface")
    schema = artifacts.Interface.from_dict(old)
    assert schema.digests == {"graph": "f" * 64}
    assert serving.digest(schema.anchor()) == serving.digest(old)


def test_the_passport_checksum_is_anchored_at_version_one():
    old = stored("passport")
    passport = data.Passport.from_dict(old)
    assert passport.digests == {"content": "b" * 64}
    assert passport.findings[0].identifier == "quality.missingness"
    assert attestation.checksum(passport.anchor()) == attestation.checksum(old)


def test_a_version_one_reliability_registry_migrates_its_evidence():
    registry = reliability.Registry.from_dict(stored("reliability_registry"))
    plan = registry.plans[reliability.Contingency.PROCESS_DEATH]
    assert plan.verification is reliability.Verification.PASSED
    assert all(item.digests["artifact"] for item in plan.evidence)
    record = registry.to_dict()
    assert record[records.KEY] == reliability.VERSION
    assert reliability.Registry.from_dict(record).to_dict() == record


def test_a_version_one_ledger_check_row_is_read_by_the_chain():
    replay = chain.Replay(None)
    replay.diagnose(
        {"source": "fixture", "check": stored("qualification_check")}
    )
    assert replay.diagnostics[0].identifier == "assoc.age"
    assert replay.diagnostics[0].subset is qualification.Subset.EXPLORATION
