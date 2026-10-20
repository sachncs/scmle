"""Tests for dataset provenance and curation evidence contracts."""

import json

import pytest

from scml.contracts import data


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


def test_curation_report_serializes_structured_quality_and_risk_findings():
    quality = make_finding()
    risk = make_finding(
        finding_id="risk.target-leakage",
        severity=data.FindingSeverity.ERROR,
    )
    report = data.CurationReport(
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
