"""Single-word keys of the write-only contract records and policy parts.

Each record writes the approved vocabulary (plan/stage_b/
phase5_vocabulary_rest.md, F3 and F8) while its Python fields keep their
names. Its `migrations` (or `spelling`) moves the written layout back to the
version-1 keys and up again, so a version-1 sample still reads where a
reader exists (`Trial.load`, `Criteria`, `Evidence`), and identity
digests are computed on the version-1 anchor.
"""

import datetime
import hashlib
import json
import pathlib

import pytest

from scmle.contracts import capabilities
from scmle.contracts import data
from scmle.contracts import evaluation
from scmle.contracts import experiments
from scmle.contracts import explanations
from scmle.contracts import modalities
from scmle.contracts import observability
from scmle.contracts import performance
from scmle.contracts import records
from scmle.contracts import tasks
from scmle.contracts.capabilities import axes

FIXTURES = pathlib.Path(__file__).parents[1] / "fixtures"
STAMP = "2026-10-20T10:00:00+00:00"


def resources():
    return capabilities.Resources(
        cpu_cores=2, memory_gb=4.0, disk_gb=1.0, max_rows=100, notes=("n",)
    )


def coordinate():
    return capabilities.Coordinate(
        modality=axes.Modality.TABULAR,
        task=tasks.Task.BINARY,
        model_family="tree",
        evaluation=axes.Evaluation.CALIBRATED,
        uncertainty=axes.Uncertainty.CALIBRATED_PROBABILITY,
        fairness=axes.Fairness.GROUP_REPORT,
        explainability=axes.Explainability.FULL,
        inference=axes.Inference.LIBRARY,
    )


def finding(identifier):
    return data.Finding(
        identifier=identifier,
        rule="rule.v1",
        severity=data.FindingSeverity.ERROR,
        confidence=0.9,
        scope={"columns": ["a"]},
        evidence={"message": "m"},
        remediation="review",
        owner="owner",
        reviewer="reviewer",
        disposition=data.Disposition.OPEN,
        blocking=True,
        timestamp=STAMP,
    )


def manifest():
    return data.DatasetManifest(
        "ds_1",
        {"schema": "schema", "data": "data"},
        10,
        2,
        "label",
        snapshot="snapshot-1",
        task="binary_classification",
        source=data.Origin(owner="ml-platform"),
    )


def curation():
    return data.Curation(
        manifest=manifest(),
        source_rows=12,
        profiled_rows=10,
        read_seconds=0.25,
        streamed=False,
        cleaning_notes=("note",),
        quality_findings=(finding("quality.a"),),
        risk_findings=(finding("risk.b"),),
        feature_columns={"numeric": ("amount",)},
        dropped_columns={"email": "sensitive"},
    )


def proof(stage):
    return modalities.Proof(stage=stage, evidence_links=("tests/x.py",))


def observation(operation, elapsed=1.0):
    return performance.Observation(
        operation=operation,
        elapsed=elapsed,
        memory=1.0,
        samples=10,
        machine="m",
        observed=STAMP,
        software="s",
    )


def run(identifier, elapsed=1.0):
    return performance.Run(
        identifier, (observation(performance.Operation.INGEST, elapsed),)
    )


def baseline():
    return performance.Baseline((run("r1"), run("r2")))


def records_by_kind():
    """One instance of each write-only record, its old and its new keys."""
    declaration = capabilities.Declaration(
        model_family="tree",
        status=axes.Status.EXPERIMENTAL,
        modalities=(axes.Modality.TABULAR,),
        required_extras=("core",),
        resources=resources(),
        failure_modes=("f",),
        evidence_links=("l",),
    )
    capability = capabilities.Capability(
        key=coordinate(),
        status=axes.Status.SUPPORTED,
        required_extras=("core",),
        resources=resources(),
        failure_modes=("f",),
        evidence_links=("l",),
    )
    portrait = data.Portrait(
        manifest=manifest(),
        column_statistics={},
        quality_findings=("q",),
        risk_findings=("r",),
        target_summary={},
        feature_plan=data.Blueprint(
            ("x",), (), extra_inputs=("raw",), high_cardinality=("c",)
        ),
        dataset_statistics={"shape": {}},
        quality_evidence=({"seed": 1},),
    )
    declared = modalities.Declaration(
        modality=modalities.Modality.TABULAR,
        status=modalities.Status.EXPERIMENTAL,
        stage_evidence=(proof(modalities.Phase.INGESTION),),
        required_extras=("core",),
        resources=resources(),
        failure_modes=("f",),
    )
    event = observability.Inference(
        request_id="req",
        run_id="run",
        model_fingerprint="model",
        outcome="ok",
        latency_ms=2.0,
        timestamp=STAMP,
        input_fingerprint="payload",
        redacted_fields=("email",),
    )
    policy = observability.Requirements(
        retention_days=30,
        required_signals=(observability.Indicator.LATENCY,),
        redacted_fields=("email",),
        alert_thresholds={"latency": 1.0},
    )
    budget = performance.Budget(
        {performance.Operation.INGEST: 2.0}, max_peak_memory_gb=1.0
    )
    return {
        "declaration": (
            declaration,
            {"model_family", "required_extras", "failure_modes"},
            {"family", "extras", "failures", "links"},
        ),
        "coordinate": (coordinate(), {"model_family"}, {"family"}),
        "resources": (
            resources(),
            {"cpu_cores", "memory_gb", "max_rows"},
            {"cpu", "memory", "disk", "gpu", "network", "rows"},
        ),
        "capability": (
            capability,
            {"required_extras", "evidence_links"},
            {"extras", "failures", "links"},
        ),
        "blueprint": (
            portrait.feature_plan,
            {"extra_inputs", "high_cardinality"},
            {"extras", "cardinal"},
        ),
        "portrait": (
            portrait,
            {"quality_findings", "risk_findings", "feature_plan", "dataset"},
            {"findings", "plan", "statistics", "evidence"},
        ),
        "curation": (
            curation(),
            {"source_rows", "profiled_rows", "unresolved_blocking_findings"},
            {"rows", "read", "notes", "findings", "ready", "blockers"},
        ),
        "datasheet": (
            data.Datasheet.distill(curation()),
            {"dataset_id", "intended_use", "out_of_scope", "generated_at"},
            {"dataset", "snapshot", "use", "excluded", "review", "generated"},
        ),
        "proof": (
            proof(modalities.Phase.INGESTION),
            {"evidence_links"},
            {"links"},
        ),
        "modality": (
            declared,
            {"stage_evidence", "missing_stages"},
            {"evidence", "extras", "failures", "missing"},
        ),
        "inference": (
            event,
            {"request_id", "model_fingerprint", "latency_ms"},
            {"request", "run", "latency", "error", "redacted", "digests"},
        ),
        "requirements": (
            policy,
            {"retention_days", "alert_thresholds"},
            {"retention", "signals", "redacted", "thresholds"},
        ),
        "comparison": (
            baseline().evaluate(run("now")),
            {"baseline_run_count", "current_run_id"},
            {"baselines", "run", "regressions"},
        ),
        "baseline": (
            baseline(),
            {"reference_runs", "max_relative_regression"},
            {"runs", "regression"},
        ),
        "budget": (budget, {"max_seconds"}, {"seconds", "memory"}),
    }


KINDS = sorted(records_by_kind())


def multiword(layout):
    """Every key with an underscore, at any depth of mappings."""
    found = set()
    for key, value in layout.items():
        if "_" in key and key != records.KEY:
            found.add(key)
        if isinstance(value, dict):
            found |= multiword(value)
    return found


@pytest.mark.parametrize("kind", KINDS)
def test_a_record_writes_the_new_keys_and_moves_to_version_one(kind):
    record, old, new = records_by_kind()[kind]
    written = record.to_dict()
    assert new <= set(written)
    assert not old & set(written)
    stored = {k: v for k, v in written.items() if k != records.KEY}
    first = records.downgrade(type(record).migrations, stored)
    assert old <= set(first)
    assert records.upgrade(type(record).migrations, first, 1) == stored


@pytest.mark.parametrize(
    "kind", ["coordinate", "resources", "proof", "requirements", "budget"]
)
def test_a_flat_record_writes_no_multi_word_key(kind):
    record = records_by_kind()[kind][0]
    assert multiword(record.to_dict()) == set()


def test_the_versioned_records_moved_to_version_two():
    registry = capabilities.Registry.seed()
    assert capabilities.CAPABILITY_SCHEMA_VERSION == 2
    assert registry.to_dict()["schema_version"] == 2
    assert curation().to_dict()["schema_version"] == 2
    assert data.Datasheet.distill(curation()).to_dict()["schema_version"] == 2
    families = registry.to_dict()["model_families"]
    assert {"family", "extras", "failures", "links"} <= set(families[0])
    first = records.downgrade(
        capabilities.registry.MIGRATIONS,
        {k: v for k, v in registry.to_dict().items() if k != records.KEY},
    )
    assert first["model_families"][0]["model_family"] == families[0]["family"]
    assert "cpu_cores" in first["model_families"][0]["resources"]


def test_the_grouped_keys_hold_the_right_values():
    written = curation().to_dict()
    assert written["rows"] == {"source": 12, "profiled": 10}
    assert [f["identifier"] for f in written["findings"]["quality"]] == [
        "quality.a"
    ]
    assert written["blockers"] == ["quality.a", "risk.b"]
    event = records_by_kind()["inference"][0].to_dict()
    assert event["digests"] == {"model": "model", "payload": "payload"}


def trial():
    return experiments.Trial(
        trial_id="trial_001",
        model_name="logistic",
        status=experiments.Termination.FAILED,
        metric="roc_auc",
        score_mean=0.7,
        score_std=0.1,
        fold_scores=[0.6, 0.8],
        fit_seconds=1.5,
        error="boom",
        failure={"type": "ValueError"},
        observed_access=[{"rows": 3}],
        evaluation_config={"folds": 2},
        train_scores=[0.9, 0.9],
    )


def test_a_trial_writes_the_new_keys_and_reads_both_versions():
    written = trial().to_dict()
    assert {
        "identifier",
        "model",
        "score",
        "deviation",
        "folds",
        "fit",
        "access",
        "evaluation",
        "training",
    } <= set(written)
    assert multiword(written) == set()
    assert experiments.Trial.load(written) == trial()
    first = records.downgrade(experiments.Trial.migrations, written)
    assert first["trial_id"] == "trial_001"
    assert first["score_mean"] == 0.7
    assert experiments.Trial.load(first) == trial()


def test_a_trial_mixing_old_and_new_keys_is_refused():
    written = trial().to_dict()
    with pytest.raises(ValueError, match="overwrite"):
        experiments.Trial.load({**written, "trial_id": "trial_002"})


def policy():
    return evaluation.Criteria.model_validate_json(
        (FIXTURES / "records" / "v1" / "evaluation_policy.json").read_text()
    )


def test_a_policy_writes_the_new_keys_at_every_depth():
    written = policy().model_dump(mode="json")
    assert written["schema_version"] == 2
    assert {"metric", "small"} <= set(written)
    assert set(written["small"]) == {"warning", "blocking"}
    assert {"confidence", "coverage", "rows"} <= set(written["uncertainty"])
    assert {"metrics", "diagnostics", "fairness", "evidence"} == set(
        written["release"]
    )
    assert set(written["explanations"]["stability"]) == {
        "required",
        "threshold",
    }
    assert multiword(written) == set()
    assert evaluation.Criteria.model_validate(written) == policy()


def test_a_policy_keeps_its_field_names_in_code():
    loaded = policy()
    assert loaded.primary_metric == loaded.model_dump()["metric"]
    assert loaded.small_groups.warning_threshold >= 1
    assert loaded.release.metric_gates


def test_a_policy_digest_is_anchored_at_version_one():
    loaded = policy()
    stored = json.loads(
        (FIXTURES / "records" / "v1" / "evaluation_policy.json").read_text()
    )
    assert records.anchor(loaded) == stored
    pinned = json.loads((FIXTURES / "record_digests.json").read_text())
    digest = hashlib.sha256(loaded.canonize().encode()).hexdigest()
    assert digest == pinned["evaluation_policy"]["sha256"]


def evidence():
    return explanations.Evidence(
        model_fingerprint="model",
        bundle_digest="bundle",
        input_fingerprint="payload",
        dataset_snapshot=explanations.Snapshot(
            identifier="d", version="1", digest="sha"
        ),
        feature_graph=explanations.Lineage(version=1, fingerprint="g"),
        method=explanations.Method(name="permutation", parameters={}),
        output={"features": []},
        timestamp=datetime.datetime(2026, 10, 20, tzinfo=datetime.UTC),
        software_version="0.1",
        uncertainty={"method": "none"},
        stability={"score": 1.0},
        limitations=("associational",),
        causal_warning="not causal",
        privacy_decision=explanations.Clearance(
            decision="redacted", reason="r", redacted_fields=("email",)
        ),
        evidence_links=(explanations.Link(relation="r", uri="artifact://x"),),
    )


def test_explanation_evidence_writes_the_new_keys_and_reads_version_one():
    written = evidence().model_dump(mode="json")
    assert written["schema_version"] == 2
    assert written["digests"] == {
        "model": "model",
        "bundle": "bundle",
        "payload": "payload",
    }
    assert written["clearance"]["redacted"] == ["email"]
    assert {"snapshot", "lineage", "software", "caveat", "links"} <= set(
        written
    )
    assert multiword(written) == set()
    assert explanations.Evidence.model_validate(written) == evidence()
    first = records.anchor(evidence())
    assert first["schema_version"] == 1
    assert first["privacy_decision"]["redacted_fields"] == ["email"]
    assert explanations.Evidence.model_validate(first) == evidence()


def test_explanation_evidence_digest_is_anchored_at_version_one():
    stored = (
        FIXTURES / "records" / "v1" / "explanation_evidence.json"
    ).read_text()
    loaded = explanations.Evidence.model_validate_json(stored)
    pinned = json.loads((FIXTURES / "record_digests.json").read_text())
    digest = hashlib.sha256(loaded.canonize().encode()).hexdigest()
    assert digest == pinned["explanation_evidence"]["sha256"]
    assert "model_fingerprint" in loaded.canonize()
