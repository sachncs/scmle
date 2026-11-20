"""Tests for dataset provenance and curation evidence contracts."""

import dataclasses
import json

import pytest

from scmle.contracts import data


def make_finding(
    identifier: str = "quality.missingness",
    severity: data.Severity = data.Severity.WARNING,
) -> data.Finding:
    """Builds a representative structured curation finding."""
    return data.Finding(
        identifier=identifier,
        rule="missingness.v2",
        severity=severity,
        confidence=0.97,
        scope={"columns": ["income"], "rows": 12},
        evidence={"missing_fraction": 0.12, "sample": "rows:0-11"},
        remediation="Review the source collection and impute explicitly.",
        owner="data-platform",
        reviewer="ml-governance",
        disposition=data.Disposition.OPEN,
        blocking=severity
        in {
            data.Severity.ERROR,
            data.Severity.CRITICAL,
        },
        timestamp="2026-10-20T10:00:00+00:00",
    )


def test_curation_finding_has_stable_json_contract_and_round_trips():
    finding = make_finding()

    record = finding.to_dict()

    assert set(record) == {
        "identifier",
        "rule",
        "severity",
        "confidence",
        "scope",
        "evidence",
        "remediation",
        "owner",
        "reviewer",
        "disposition",
        "blocking",
        "timestamp",
    }
    assert json.loads(json.dumps(record)) == record
    assert data.Finding.from_dict(record) == finding


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan")])
def test_curation_finding_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError, match="confidence"):
        make_finding().__class__(
            identifier="quality.invalid",
            rule="rule.v1",
            severity=data.Severity.INFO,
            confidence=confidence,
            scope={},
            evidence={},
            remediation="Investigate the source.",
            owner="data-platform",
            reviewer=None,
            disposition=data.Disposition.OPEN,
            blocking=False,
        )


def test_observation_conversion_has_deterministic_identity_and_evidence():
    first = data.Finding.glean("column 'income' has missing values", "quality")
    second = data.Finding.glean("column 'income' has missing values", "quality")

    assert first.identifier == second.identifier
    assert first.identifier.startswith("quality.")
    assert first.rule == second.rule == "pipeline-observation.v1"
    assert (
        first.evidence
        == second.evidence
        == {"message": "column 'income' has missing values"}
    )


def test_dataset_manifest_carries_source_lineage_privacy_and_findings():
    finding = make_finding()
    manifest = data.Manifest(
        identifier="ds_12345678",
        digests={"schema": "schema-hash", "data": "data-hash"},
        rows=100,
        columns=4,
        target="label",
        uri="s3://warehouse/events/",
        snapshot="snapshot-2026-10-20",
        modality="tabular",
        task="binary",
        schema_version=2,
        source=data.Origin(
            owner="analytics",
            acquisition="warehouse-export",
            captured="2026-10-20T09:00:00+00:00",
            timezone="UTC",
            sampling="production customers",
            units={"income": "USD"},
            definitions={"label": "conversion within 30 days"},
            query="select * from events",
            license="internal",
            access="restricted",
        ),
        lineage=data.Ancestry(
            snapshot="warehouse:events:2026-10-20",
            parents=("raw-events",),
            transformations=("deduplicate-events.v3",),
            labelling="labels/conversion.py",
            generator="labels-4.1",
            annotator="annotators-2.0",
            adjudication="complete",
            partitions=("date=2026-10-20",),
            digests={"part-000": "abc123"},
            evolution="compatible-additive",
        ),
        privacy=data.Confidentiality(
            classification="confidential",
            identifying=True,
            sensitive=("email",),
            consent="service agreement",
            retention="90d",
            deletion="delete-on-request",
            access="approved-data-users",
            redaction=True,
        ),
        findings=(finding,),
    )

    record = manifest.to_dict()

    assert record["snapshot"] == "snapshot-2026-10-20"
    assert record["source"]["units"] == {"income": "USD"}
    assert record["lineage"]["partitions"] == ["date=2026-10-20"]
    assert record["privacy"]["sensitive"] == ["email"]
    assert record["findings"] == [finding.to_dict()]
    assert json.loads(json.dumps(record)) == record
    assert data.Manifest.from_dict(record) == manifest


def test_dataset_profile_serializes_typed_quality_evidence():
    """Quality evidence is JSON-native without changing existing fields."""
    profile = data.Portrait(
        manifest=data.Manifest(
            "ds_1", {"schema": "schema", "data": "data"}, 10, 2, "y"
        ),
        statistics={},
        quality=("label noise: review",),
        risk_findings=(),
        summary={"task": "binary"},
        plan=data.Blueprint(("x",), ()),
        evidence=(
            {
                "method": "out_of_fold_confident_disagreement",
                "thresholds": {"minimum_rows": 50},
                "sample_bounds": {"rows_checked": 100},
                "seed": 3,
                "metrics": {"suspect_rate": 0.08},
                "limitations": ["candidate review is required"],
            },
        ),
    )

    record = profile.to_dict()

    assert record["findings"]["quality"] == ["label noise: review"]
    assert record["evidence"][0]["seed"] == 3
    assert json.loads(json.dumps(record)) == record


def test_dataset_profile_rejects_non_json_quality_evidence():
    """Quality evidence must not silently stringify arbitrary objects."""
    with pytest.raises(TypeError, match="JSON-native"):
        data.Portrait(
            manifest=None,
            statistics={},
            quality=(),
            risk_findings=(),
            summary={},
            plan=data.Blueprint(("x",), ()),
            evidence=({"metrics": {"value": object()}},),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("identifier", "", "identifier"),
        ("digests", {"schema": "", "data": "data"}, "digests.schema"),
        ("digests", {"schema": "schema", "data": ""}, "digests.data"),
        ("target", "", "target"),
        ("rows", -1, "rows"),
        ("columns", -1, "columns"),
        ("schema_version", 0, "schema_version"),
    ],
)
def test_dataset_manifest_rejects_invalid_identity(field, value, message):
    """Dataset identity cannot be persisted with incomplete metadata."""
    values = {
        "identifier": "dataset",
        "digests": {"schema": "schema", "data": "data"},
        "rows": 1,
        "columns": 1,
        "target": "label",
        "schema_version": 1,
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        data.Manifest(**values)


def test_dataset_manifest_rejects_untyped_evidence_components():
    """Lineage, privacy, and findings remain typed contract components."""
    with pytest.raises(TypeError, match="source"):
        data.Manifest(
            "dataset",
            {"schema": "schema", "data": "data"},
            1,
            1,
            "label",
            source={},
        )
    with pytest.raises(TypeError, match="findings"):
        data.Manifest(
            "dataset",
            {"schema": "schema", "data": "data"},
            1,
            1,
            "label",
            findings=(object(),),
        )


def test_curation_report_serializes_structured_quality_and_risk_findings():
    quality = make_finding()
    risk = make_finding(
        identifier="risk.target-leakage",
        severity=data.Severity.ERROR,
    )
    report = data.Curation(
        manifest=data.Manifest(
            "ds_1", {"schema": "schema", "data": "data"}, 10, 2, "label"
        ),
        source_rows=10,
        rows=10,
        read=0.25,
        streamed=False,
        notes=("normalised headers",),
        quality=(quality,),
        risk_findings=(risk,),
        features={"numeric": ("amount",)},
        dropped={"email": "sensitive"},
    )

    record = report.to_dict()

    assert record["findings"] == {
        "quality": [quality.to_dict()],
        "risk": [risk.to_dict()],
    }
    assert record["rows"] == {"source": 10, "profiled": 10}
    assert record["schema_version"] == 2
    assert json.loads(json.dumps(record)) == record


def test_curation_report_is_not_release_ready_with_open_blocking_findings():
    """Open blocking observations must stop the lifecycle before training."""
    risk = make_finding(
        identifier="risk.target-leakage",
        severity=data.Severity.ERROR,
    )
    report = data.Curation(
        manifest=data.Manifest(
            "ds_1", {"schema": "schema", "data": "data"}, 10, 2, "label"
        ),
        source_rows=10,
        rows=10,
        read=0.25,
        streamed=False,
        notes=(),
        quality=(),
        risk_findings=(risk,),
        features={"numeric": ("amount",)},
        dropped={},
    )

    assert report.release is False
    assert report.blockers == (risk,)
    assert report.to_dict()["blockers"] == ["risk.target-leakage"]
    assert report.to_dict()["ready"] is False


def test_curation_report_accepts_explicit_safe_dispositions():
    """Resolved, waived, and false-positive findings are reviewable states."""
    risk = make_finding(
        identifier="risk.target-leakage",
        severity=data.Severity.ERROR,
    )
    report = data.Curation(
        manifest=data.Manifest(
            "ds_1", {"schema": "schema", "data": "data"}, 10, 2, "label"
        ),
        source_rows=10,
        rows=10,
        read=0.25,
        streamed=False,
        notes=(),
        quality=(),
        risk_findings=(
            dataclasses.replace(risk, disposition=data.Disposition.RESOLVED),
        ),
        features={"numeric": ("amount",)},
        dropped={},
    )

    assert report.release
    assert report.blockers == ()


def test_dataset_card_makes_intended_use_and_limits_explicit():
    # Enforces website/src/pages/docs/evidence.astro:82.
    report = data.Curation(
        manifest=data.Manifest(
            "ds_1",
            {"schema": "schema", "data": "data"},
            10,
            2,
            "label",
            snapshot="snapshot-1",
            task="binary_classification",
            source=data.Origin(owner="ml-platform"),
        ),
        source_rows=12,
        rows=10,
        read=0.25,
        streamed=False,
        notes=("two rows were excluded",),
        quality=(),
        risk_findings=(),
        features={"numeric": ("amount",)},
        dropped={"email": "sensitive"},
    )

    card = data.Datasheet.distill(report)
    record = card.to_dict()

    assert card.owner == "ml-platform"
    assert card.status == "machine_generated_unreviewed"
    assert record["composition"] == {
        "source_rows": 12,
        "profiled_rows": 10,
        "feature_columns": 1,
        "dropped_columns": 1,
    }
    assert "release approval" in " ".join(record["excluded"])
    assert record["schema_version"] == 2
    assert "two rows were excluded" in record["limitations"]
    assert json.loads(json.dumps(record)) == record
