"""Tests for dataset provenance and curation evidence contracts."""

import dataclasses
import json

import pytest

from scmle.contracts import data


def make_finding(
    finding_id: str = "quality.missingness",
    severity: data.FindingSeverity = data.FindingSeverity.WARNING,
) -> data.CurationFinding:
    """Builds a representative structured curation finding."""
    return data.CurationFinding(
        finding_id=finding_id,
        rule_version="missingness.v2",
        severity=severity,
        confidence=0.97,
        affected_scope={"columns": ["income"], "rows": 12},
        evidence={"missing_fraction": 0.12, "sample": "rows:0-11"},
        remediation="Review the source collection and impute explicitly.",
        owner="data-platform",
        reviewer="ml-governance",
        disposition=data.FindingDisposition.OPEN,
        blocking=severity
        in {
            data.FindingSeverity.ERROR,
            data.FindingSeverity.CRITICAL,
        },
        timestamp="2026-10-20T10:00:00+00:00",
    )


def test_curation_finding_has_stable_json_contract_and_round_trips():
    finding = make_finding()

    record = finding.to_dict()

    assert set(record) == {
        "id",
        "rule_version",
        "severity",
        "confidence",
        "affected_scope",
        "evidence",
        "remediation",
        "owner",
        "reviewer",
        "disposition",
        "blocking",
        "timestamp",
    }
    assert json.loads(json.dumps(record)) == record
    assert data.CurationFinding.from_dict(record) == finding


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan")])
def test_curation_finding_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError, match="confidence"):
        make_finding().__class__(
            finding_id="quality.invalid",
            rule_version="rule.v1",
            severity=data.FindingSeverity.INFO,
            confidence=confidence,
            affected_scope={},
            evidence={},
            remediation="Investigate the source.",
            owner="data-platform",
            reviewer=None,
            disposition=data.FindingDisposition.OPEN,
            blocking=False,
        )


def test_observation_conversion_has_deterministic_identity_and_evidence():
    first = data.CurationFinding.from_observation(
        "column 'income' has missing values", "quality"
    )
    second = data.CurationFinding.from_observation(
        "column 'income' has missing values", "quality"
    )

    assert first.finding_id == second.finding_id
    assert first.finding_id.startswith("quality.")
    assert (
        first.rule_version == second.rule_version == "pipeline-observation.v1"
    )
    assert (
        first.evidence
        == second.evidence
        == {"message": "column 'income' has missing values"}
    )


def test_dataset_manifest_carries_source_lineage_privacy_and_findings():
    finding = make_finding()
    manifest = data.DatasetManifest(
        dataset_id="ds_12345678",
        schema_hash="schema-hash",
        data_hash="data-hash",
        row_count=100,
        column_count=4,
        target="label",
        uri="s3://warehouse/events/",
        snapshot_id="snapshot-2026-10-20",
        modality="tabular",
        task="binary",
        schema_version=2,
        source=data.SourceMetadata(
            owner="analytics",
            acquisition_method="warehouse-export",
            captured_at="2026-10-20T09:00:00+00:00",
            timezone="UTC",
            sampling_frame="production customers",
            units={"income": "USD"},
            semantic_definitions={"label": "conversion within 30 days"},
            extraction_query="select * from events",
            license="internal",
            access_policy="restricted",
        ),
        lineage=data.LineageMetadata(
            source_snapshot="warehouse:events:2026-10-20",
            parent_datasets=("raw-events",),
            transformations=("deduplicate-events.v3",),
            label_source="labels/conversion.py",
            label_generator_version="labels-4.1",
            annotator_version="annotators-2.0",
            adjudication_status="complete",
            partition_ids=("date=2026-10-20",),
            object_hashes={"part-000": "abc123"},
            schema_evolution="compatible-additive",
        ),
        privacy=data.PrivacyMetadata(
            classification="confidential",
            contains_pii=True,
            sensitive_fields=("email",),
            consent_basis="service agreement",
            retention_period="90d",
            deletion_policy="delete-on-request",
            access_policy="approved-data-users",
            redaction_required=True,
        ),
        findings=(finding,),
    )

    record = manifest.to_dict()

    assert record["snapshot_id"] == "snapshot-2026-10-20"
    assert record["source"]["units"] == {"income": "USD"}
    assert record["lineage"]["partition_ids"] == ["date=2026-10-20"]
    assert record["privacy"]["sensitive_fields"] == ["email"]
    assert record["findings"] == [finding.to_dict()]
    assert json.loads(json.dumps(record)) == record
    assert data.DatasetManifest.from_dict(record) == manifest


def test_dataset_profile_serializes_typed_quality_evidence():
    """Quality evidence is JSON-native without changing existing fields."""
    profile = data.DatasetProfile(
        manifest=data.DatasetManifest("ds_1", "schema", "data", 10, 2, "y"),
        column_statistics={},
        quality_findings=("label noise: review",),
        risk_findings=(),
        target_summary={"task": "binary"},
        feature_plan=data.FeaturePlan(("x",), ()),
        quality_evidence=(
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

    assert record["quality_findings"] == ["label noise: review"]
    assert record["quality_evidence"][0]["seed"] == 3
    assert json.loads(json.dumps(record)) == record


def test_dataset_profile_rejects_non_json_quality_evidence():
    """Quality evidence must not silently stringify arbitrary objects."""
    with pytest.raises(TypeError, match="JSON-native"):
        data.DatasetProfile(
            manifest=None,
            column_statistics={},
            quality_findings=(),
            risk_findings=(),
            target_summary={},
            feature_plan=data.FeaturePlan(("x",), ()),
            quality_evidence=({"metrics": {"value": object()}},),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("dataset_id", "", "dataset_id"),
        ("schema_hash", "", "schema_hash"),
        ("data_hash", "", "data_hash"),
        ("target", "", "target"),
        ("row_count", -1, "row_count"),
        ("column_count", -1, "column_count"),
        ("schema_version", 0, "schema_version"),
    ),
)
def test_dataset_manifest_rejects_invalid_identity(field, value, message):
    """Dataset identity cannot be persisted with incomplete metadata."""
    values = {
        "dataset_id": "dataset",
        "schema_hash": "schema",
        "data_hash": "data",
        "row_count": 1,
        "column_count": 1,
        "target": "label",
        "schema_version": 1,
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        data.DatasetManifest(**values)


def test_dataset_manifest_rejects_untyped_evidence_components():
    """Lineage, privacy, and findings remain typed contract components."""
    with pytest.raises(TypeError, match="source"):
        data.DatasetManifest(
            "dataset", "schema", "data", 1, 1, "label", source={}
        )
    with pytest.raises(TypeError, match="findings"):
        data.DatasetManifest(
            "dataset", "schema", "data", 1, 1, "label", findings=(object(),)
        )


def test_curation_report_serializes_structured_quality_and_risk_findings():
    quality = make_finding()
    risk = make_finding(
        finding_id="risk.target-leakage",
        severity=data.FindingSeverity.ERROR,
    )
    report = data.Curation(
        manifest=data.DatasetManifest("ds_1", "schema", "data", 10, 2, "label"),
        source_rows=10,
        profiled_rows=10,
        read_seconds=0.25,
        streamed=False,
        cleaning_notes=("normalised headers",),
        quality_findings=(quality,),
        risk_findings=(risk,),
        feature_columns={"numeric": ("amount",)},
        dropped_columns={"email": "sensitive"},
    )

    record = report.to_dict()

    assert record["quality_findings"] == [quality.to_dict()]
    assert record["risk_findings"] == [risk.to_dict()]
    assert json.loads(json.dumps(record)) == record


def test_curation_report_is_not_release_ready_with_open_blocking_findings():
    """Open blocking observations must stop the lifecycle before training."""
    risk = make_finding(
        finding_id="risk.target-leakage",
        severity=data.FindingSeverity.ERROR,
    )
    report = data.Curation(
        manifest=data.DatasetManifest("ds_1", "schema", "data", 10, 2, "label"),
        source_rows=10,
        profiled_rows=10,
        read_seconds=0.25,
        streamed=False,
        cleaning_notes=(),
        quality_findings=(),
        risk_findings=(risk,),
        feature_columns={"numeric": ("amount",)},
        dropped_columns={},
    )

    assert report.release_ready is False
    assert report.unresolved_blocking_findings == (risk,)
    assert report.to_dict()["unresolved_blocking_findings"] == [
        "risk.target-leakage"
    ]


def test_curation_report_accepts_explicit_safe_dispositions():
    """Resolved, waived, and false-positive findings are reviewable states."""
    risk = make_finding(
        finding_id="risk.target-leakage",
        severity=data.FindingSeverity.ERROR,
    )
    report = data.Curation(
        manifest=data.DatasetManifest("ds_1", "schema", "data", 10, 2, "label"),
        source_rows=10,
        profiled_rows=10,
        read_seconds=0.25,
        streamed=False,
        cleaning_notes=(),
        quality_findings=(),
        risk_findings=(
            dataclasses.replace(
                risk, disposition=data.FindingDisposition.RESOLVED
            ),
        ),
        feature_columns={"numeric": ("amount",)},
        dropped_columns={},
    )

    assert report.release_ready
    assert report.unresolved_blocking_findings == ()


def test_dataset_card_makes_intended_use_and_limits_explicit():
    report = data.Curation(
        manifest=data.DatasetManifest(
            "ds_1",
            "schema",
            "data",
            10,
            2,
            "label",
            snapshot_id="snapshot-1",
            task="binary_classification",
            source=data.SourceMetadata(owner="ml-platform"),
        ),
        source_rows=12,
        profiled_rows=10,
        read_seconds=0.25,
        streamed=False,
        cleaning_notes=("two rows were excluded",),
        quality_findings=(),
        risk_findings=(),
        feature_columns={"numeric": ("amount",)},
        dropped_columns={"email": "sensitive"},
    )

    card = data.DatasetCard.from_curation(report)
    record = card.to_dict()

    assert card.owner == "ml-platform"
    assert card.review_status == "machine_generated_unreviewed"
    assert record["composition"] == {
        "source_rows": 12,
        "profiled_rows": 10,
        "feature_columns": 1,
        "dropped_columns": 1,
    }
    assert "release approval" in " ".join(record["out_of_scope"])
    assert "two rows were excluded" in record["limitations"]
    assert json.loads(json.dumps(record)) == record
