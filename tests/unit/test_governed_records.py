"""The charter, custody and ledger records at schema version 2 (phase 5, R1).

Each fixture in `tests/fixtures/records/v1/custody` was written by the
writers that existed before the record-key vocabulary (the version 1
layout). They must still load, migrate both ways without loss, and keep
every hash, seal, signature and chain that names them; `trail_sample` holds
a whole version 1 run (charter, split manifest and hypothesis ledger).
"""

import copy
import json
import pathlib

import hypothesis
from hypothesis import strategies
import pytest

from scmle.assurance.charter import contracts
from scmle.assurance.charter import feasibility
from scmle.assurance.charter import sealing
from scmle.assurance.custody import manifest
from scmle.assurance.ledger import budget
from scmle.assurance.ledger import controls
from scmle.assurance.ledger import usage
from scmle.assurance.ledger.ledger import journal
import scmle.assurance.ledger.ledger.records
from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.contracts import records

FIXTURES = pathlib.Path(__file__).parents[1] / "fixtures"
VERSION_ONE = FIXTURES / "records" / "v1" / "custody"
TRAIL = FIXTURES / "trail_sample" / "trail" / "retained" / "run"
KEY = b"r1-fixture-key"
KINDS = {
    "budget": budget.Decision,
    "control": controls.Control,
    "feasibility": feasibility.Feasibility,
    "test": scmle.assurance.ledger.ledger.records.Test,
}


def stored(name):
    """The decoded version-1 fixture `name`."""
    return json.loads((VERSION_ONE / f"{name}.json").read_text("utf-8"))


def current(record):
    """A written record without its version key."""
    layout = dict(record)
    assert layout.pop(records.KEY) == records.ORIGIN + 1
    return layout


@pytest.mark.parametrize("name", sorted(KINDS))
def test_a_version_one_record_loads_writes_version_two_and_reverts(name):
    kind = KINDS[name]
    old = stored(name)
    record = kind.from_dict(old)
    written = record.to_dict()
    assert kind.from_dict(written) == record
    assert kind.from_dict(json.loads(json.dumps(written))) == record
    assert records.downgrade(kind.migrations, current(written)) == old


@pytest.mark.parametrize("name", sorted(KINDS))
def test_a_newer_version_is_refused(name):
    newer = {**KINDS[name].from_dict(stored(name)).to_dict()}
    newer[records.KEY] = records.ORIGIN + 2
    with pytest.raises(ValueError, match="newer than this build"):
        KINDS[name].from_dict(newer)


def charters():
    """The two version-1 charter wrappers: a rich one and the trail's."""
    return (
        stored("charter"),
        json.loads((TRAIL / "analysis_charter.json").read_text("utf-8")),
    )


@pytest.mark.parametrize("wrapper", charters(), ids=("rich", "trail"))
def test_a_version_one_charter_keeps_its_hash(wrapper):
    charter = contracts.Charter.from_dict(wrapper["charter"])
    assert charter.anchor() == wrapper["charter"]
    assert charter.digest == wrapper["charter_hash"]
    again = contracts.Charter.from_dict(charter.to_dict())
    assert again == charter
    assert again.digest == wrapper["charter_hash"]


@pytest.mark.parametrize("wrapper", charters(), ids=("rich", "trail"))
def test_the_charter_migration_is_invertible(wrapper):
    step = contracts.Charter.migrations[0]
    old = copy.deepcopy(wrapper["charter"])
    new = step.upgrade(copy.deepcopy(old))
    assert step.downgrade(copy.deepcopy(new)) == old
    assert step.upgrade(step.downgrade(copy.deepcopy(new))) == new


def test_the_rich_charter_moves_every_renamed_key():
    charter = contracts.Charter.from_dict(stored("charter")["charter"])
    written = charter.to_dict()
    assert {"population", "gates", "triggers", "quota", "costs"} <= set(written)
    assert written["estimand"]["causal"]["sensitivity"]["multiple"] == 2.5
    assert written["decision"]["rollout"]["guardrails"][0]["direction"] == (
        "lower"
    )
    assert written["decision"]["rollout"]["guardrails"][1]["direction"] == (
        "higher"
    )
    assert written["ingest"]["tolerances"] == {
        "key": 0.001,
        "null": 0.002,
        "link": 0.05,
    }
    assert written["ingest"]["reconcile"]["tables"][0]["column"] == "order_id"
    assert written["budget"]["tolerances"][0]["detector"] == "leakage"
    assert written["uncertainty"]["confirmatory"] is True
    assert charter.format_version == stored("charter")["format"]


def test_a_signed_version_one_charter_still_verifies(tmp_path):
    path = tmp_path / "analysis_charter.json"
    path.write_text(json.dumps(stored("charter")), encoding="utf-8")
    verified = sealing.read(path, KEY, require_signature=True)
    assert verified.signed
    assert verified.digest == stored("charter")["charter_hash"]
    rewritten = sealing.inscribe(tmp_path / "again.json", verified.charter, KEY)
    again = sealing.read(rewritten, KEY, require_signature=True)
    assert again.digest == verified.digest
    wrapper = json.loads(rewritten.read_text("utf-8"))
    assert wrapper["signature"] == stored("charter")["signature"]
    assert wrapper["charter"][records.KEY] == records.ORIGIN + 1


def test_an_edited_version_one_charter_is_refused(tmp_path):
    wrapper = stored("charter")
    wrapper["charter"]["k_max"] = 9
    path = tmp_path / "analysis_charter.json"
    path.write_text(json.dumps(wrapper), encoding="utf-8")
    with pytest.raises(errors.ArtifactError, match="hash mismatch"):
        sealing.read(path, KEY)


def test_a_spent_version_one_confirmation_set_is_refused_by_new_code(
    tmp_path,
):
    """A use recorded under the v1 charter hash refuses a second opening."""
    wrapper = stored("charter")
    rows = stored("manifest")["confirmation_ids"]
    first = usage.key(wrapper["charter_hash"], "exploration-data", rows)
    assert usage.Register(tmp_path, first, "run-v1").record() is None
    charter = contracts.Charter.from_dict(wrapper["charter"])
    sealed = manifest.Manifest.from_dict(stored("manifest"))
    again = usage.key(
        charter.digest, "exploration-data", sealed.sets.confirmation
    )
    assert again == first
    refusal = usage.Register(tmp_path, again, "run-v2").record()
    assert refusal is not None
    assert "run-v1" in refusal


@pytest.mark.parametrize(
    "path",
    [VERSION_ONE / "manifest.json", TRAIL / "split_manifest.json"],
    ids=("rich", "trail"),
)
def test_a_sealed_version_one_manifest_still_verifies(path):
    old = json.loads(path.read_text("utf-8"))
    sealed = manifest.Manifest.from_dict(old)
    assert sealed.sha256 == old["sha256"]
    written = sealed.to_dict()
    assert written["sha256"] == old["sha256"]
    assert manifest.Manifest.from_dict(written) == sealed
    body = current({k: v for k, v in written.items() if k != "sha256"})
    assert records.downgrade(manifest.Manifest.migrations, body) == {
        k: v for k, v in old.items() if k != "sha256"
    }
    assert sealed.digests.charter == old["charter_hash"]


@pytest.mark.parametrize("stamped", [False, True])
def test_an_edited_manifest_is_refused(stamped):
    # Enforces website/src/pages/docs/recovery.astro:11.
    record = stored("manifest")
    if stamped:
        record = manifest.Manifest.from_dict(record).to_dict()
        record["sets"]["confirmation"] = [3]
    else:
        record["confirmation_ids"] = [3]
    with pytest.raises(errors.ArtifactError, match="hash mismatch"):
        manifest.Manifest.from_dict(record)


def test_a_version_one_ledger_still_verifies_and_replays():
    ledger = journal.Ledger.load(TRAIL / "hypothesis_ledger.jsonl")
    specs = {h: hypothesis.spec for h, hypothesis in ledger.hypotheses.items()}
    for hypothesis_id, spec in specs.items():
        assert spec.anchor()["code_hash"] == spec.digests.code
        assert (
            scmle.assurance.ledger.ledger.records.Test.from_dict(spec.anchor())
            == specs[hypothesis_id]
        )


def test_the_survivor_fingerprint_uses_the_version_one_layout():
    spec = scmle.assurance.ledger.ledger.records.Test.from_dict(stored("test"))
    assert spec.anchor() == stored("test")
    expected = scmle.assurance.ledger.ledger.records.digest(
        {"ids": ["h000001"], "specs": [stored("test")]}
    )
    assert (
        scmle.assurance.ledger.ledger.records.fingerprint(
            ("h000001",), {"h000001": spec}
        )
        == expected
    )


fractions = strategies.fractions(
    min_value=0, max_value=1, max_denominator=1000
).filter(lambda value: 0 < value < 1)


@hypothesis.given(promotion=fractions, rollback=fractions)
def test_decision_budget_round_trips(promotion, rollback):
    record = budget.Decision(promotion=promotion, rollback=rollback)
    written = record.to_dict()
    assert budget.Decision.from_dict(written) == record
    old = records.downgrade(record.migrations, current(written))
    assert budget.Decision.from_dict(old) == record


@hypothesis.given(
    subset=strategies.sampled_from(list(qualification.Subset)),
    column=strategies.none() | strategies.text(min_size=1, max_size=5),
)
def test_test_round_trips(subset, column):
    spec = scmle.assurance.ledger.ledger.records.Test.from_dict(stored("test"))
    record = scmle.assurance.ledger.ledger.records.Test(
        name=spec.name,
        family=spec.family,
        statistic=spec.statistic,
        null=spec.null,
        direction=spec.direction,
        subset=subset,
        column=column,
        digests=spec.digests,
    )
    assert (
        scmle.assurance.ledger.ledger.records.Test.from_dict(record.to_dict())
        == record
    )
    assert (
        scmle.assurance.ledger.ledger.records.Test.from_dict(record.anchor())
        == record
    )


@hypothesis.given(
    counts=strategies.lists(
        strategies.integers(min_value=0, max_value=50), min_size=20, max_size=30
    ),
    observed=strategies.integers(min_value=0, max_value=60),
)
def test_control_round_trips(counts, observed):
    record = controls.summarize(
        "permuted_target", "global", observed, tuple(counts)
    )
    written = record.to_dict()
    assert controls.Control.from_dict(written) == record
    old = records.downgrade(record.migrations, current(written))
    assert controls.Control.from_dict(old) == record


@hypothesis.given(
    rows=strategies.integers(min_value=0, max_value=5000),
    effect=strategies.floats(min_value=0.05, max_value=1.0),
)
def test_feasibility_round_trips(rows, effect):
    record = feasibility.assess(
        n_confirmation=rows, min_effect=effect, alpha=0.05, k_max=3, power=0.8
    )
    written = record.to_dict()
    restored = feasibility.Feasibility.from_dict(written)
    assert restored.to_dict() == written
    old = records.downgrade(record.migrations, current(written))
    assert feasibility.Feasibility.from_dict(old).to_dict() == written


@hypothesis.given(
    ids=strategies.lists(
        strategies.integers(min_value=0, max_value=200),
        unique=True,
        min_size=4,
        max_size=20,
    ),
    embargo=strategies.integers(min_value=0, max_value=5),
)
def test_manifest_round_trips(ids, embargo):
    quarter = len(ids) // 4
    sealed = manifest.Manifest(
        strategy="random",
        seed=1,
        embargo=embargo,
        digests=manifest.Digests(charter="c" * 64, data="d" * 64),
        sets=manifest.Sets(
            exploration=tuple(ids[: 2 * quarter]),
            confirmation=tuple(ids[2 * quarter : 3 * quarter]),
            embargo=(),
            final=tuple(ids[3 * quarter :]),
        ),
        folds=(),
        isolation=(),
        scope="confirmatory",
    )
    written = sealed.to_dict()
    assert manifest.Manifest.from_dict(written) == sealed
    old = records.downgrade(
        sealed.migrations,
        current({k: v for k, v in written.items() if k != "sha256"}),
    )
    assert manifest.Manifest.from_dict({**old, "sha256": sealed.sha256}) == (
        sealed
    )
