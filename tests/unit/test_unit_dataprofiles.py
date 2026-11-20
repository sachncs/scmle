"""Behaviour of data findings, feature plans, profiles, curation and cards."""

import dataclasses
import hashlib

import pytest

from scmle.contracts import errors
from scmle.contracts.data import findings
from scmle.contracts.data import profiles
from scmle.contracts.data import provenance
from scmle.contracts.tasks import graph

STAMP = "2026-11-05T10:00:00+00:00"


def finding(identifier="q.a", **changes):
    values = {
        "identifier": identifier,
        "rule": "rule.v1",
        "severity": findings.Severity.ERROR,
        "confidence": 0.9,
        "scope": {"columns": ["a"]},
        "evidence": {"message": "m"},
        "remediation": "review",
        "owner": "owner",
        "reviewer": None,
        "disposition": findings.Disposition.OPEN,
        "blocking": True,
        "timestamp": STAMP,
    }
    values.update(changes)
    return findings.Finding(**values)


def manifest(**changes):
    values = {
        "identifier": "ds_1",
        "digests": {"schema": "s", "data": "d"},
        "rows": 10,
        "columns": 2,
        "target": "label",
        "snapshot": "snap-1",
        "task": "binary_classification",
        "source": provenance.Origin(owner="ml-platform"),
    }
    values.update(changes)
    return provenance.Manifest(**values)


def curation(**changes):
    values = {
        "manifest": manifest(),
        "source_rows": 12,
        "rows": 10,
        "read": 0.25,
        "streamed": False,
        "notes": ("trimmed spaces",),
        "quality": (finding("q.a"),),
        "risk_findings": (finding("r.b", blocking=False),),
        "features": {"numeric": ("x", "y"), "categorical": ("c",)},
        "dropped": {"id": "identifier"},
    }
    values.update(changes)
    return profiles.Curation(**values)


# ---- findings --------------------------------------------------------------


def test_normalise_detaches_json_native_evidence():
    source = {"a": [1, 2], "b": {"c": None}}
    made = findings.normalise(source)
    assert made == source
    assert made is not source
    with pytest.raises(TypeError):
        findings.normalise({"a": object()})


def test_an_observation_needs_a_note_with_text():
    made = findings.Observation(None, {"x": 1})
    assert made.evidence == {"x": 1}
    assert findings.Observation("n", {}).note == "n"
    with pytest.raises(errors.InputError, match="note must not be empty"):
        findings.Observation("  ", {})


def test_severities_and_dispositions_are_the_stable_strings():
    # Enforces website/src/pages/docs/modalities.astro:41.
    assert [s.value for s in findings.Severity] == [
        "info",
        "warning",
        "error",
        "critical",
    ]
    assert findings.Disposition.FALSE_POSITIVE == "false_positive"


def test_a_finding_converts_raw_enum_values_and_round_trips():
    made = finding(severity="warning", disposition="waived")
    assert made.severity is findings.Severity.WARNING
    assert made.disposition is findings.Disposition.WAIVED
    record = made.to_dict()
    assert record == {
        "identifier": "q.a",
        "rule": "rule.v1",
        "severity": "warning",
        "confidence": 0.9,
        "scope": {"columns": ["a"]},
        "evidence": {"message": "m"},
        "remediation": "review",
        "owner": "owner",
        "reviewer": None,
        "disposition": "waived",
        "blocking": True,
        "timestamp": STAMP,
    }
    assert findings.Finding.from_dict(record) == made
    del record["reviewer"]
    assert findings.Finding.from_dict(record).reviewer is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"identifier": " "}, "identifier must not be empty"),
        ({"rule": ""}, "rule must not be empty"),
        ({"confidence": -0.01}, "confidence must be finite"),
        ({"confidence": 1.01}, "confidence must be finite"),
        ({"confidence": float("nan")}, "confidence must be finite"),
        ({"remediation": " "}, "remediation must not be empty"),
        ({"owner": ""}, "owner must not be empty"),
        ({"timestamp": " "}, "timestamp must not be empty"),
    ],
)
def test_a_finding_refuses_blank_or_impossible_fields(changes, message):
    # Enforces website/src/pages/docs/extensions.astro:211.
    with pytest.raises(errors.InputError, match=message):
        finding(**changes)


def test_a_finding_accepts_confidence_on_its_edges():
    assert finding(confidence=0.0).confidence == 0.0
    assert finding(confidence=1.0).confidence == 1.0
    with pytest.raises(ValueError):
        finding(severity="severe")


def test_glean_gives_an_observation_a_stable_identity():
    made = findings.Finding.glean("  Column x is constant ", " Quality ")
    digest = hashlib.sha256(b"Column x is constant").hexdigest()[:16]
    assert made.identifier == f"quality.{digest}"
    assert made.rule == "pipeline-observation.v1"
    assert made.severity is findings.Severity.WARNING
    assert made.confidence == 1.0
    assert made.scope == {"category": "quality"}
    assert made.evidence == {"message": "Column x is constant"}
    assert made.owner == "pipeline"
    assert made.disposition is findings.Disposition.OPEN
    assert made.blocking is False
    error = findings.Finding.glean("m", "risk", findings.Severity.ERROR)
    assert error.severity is findings.Severity.ERROR
    with pytest.raises(errors.InputError, match="message must not be empty"):
        findings.Finding.glean(" ", "q")
    with pytest.raises(errors.InputError, match="category must not be empty"):
        findings.Finding.glean("m", " ")


def test_vet_needs_finding_records():
    findings.vet((finding(),))
    findings.vet(())
    with pytest.raises(errors.KindError, match="must contain Finding records"):
        findings.vet((finding(), "text"))


# ---- profiles: blueprint and portrait --------------------------------------


def node(name, *inputs):
    return graph.Node(name=name, op="log1p", inputs=list(inputs))


def test_profile_constants_are_the_documented_values():
    assert profiles.HIGH_CARDINALITY_LEVELS == 20
    assert profiles.BLUEPRINT.renames == {
        "extra_inputs": "extras",
        "high_cardinality": "cardinal",
    }


def test_a_blueprint_names_its_features_and_sources():
    plan = profiles.Blueprint(
        numeric=("a", "d"),
        categorical=("c",),
        derived=(node("d", "raw"),),
        series=(node("s", "a"),),
        extras=("raw", "a"),
        cardinal=("c",),
    )
    assert plan.features == ("a", "d", "c")
    assert plan.engineered == frozenset({"d", "s"})
    assert plan.sourced == ("a", "c", "raw")
    assert plan.pick("numeric") == ("a", "d")
    assert plan.pick("nothing") == ()
    record = plan.to_dict()
    assert record["extras"] == ["raw", "a"]
    assert record["cardinal"] == ["c"]
    assert record["derived"][0]["name"] == "d"
    assert record["text"] == []
    assert record["dropped"] == {}


def portrait(**changes):
    values = {
        "manifest": manifest(),
        "statistics": {"a": {"mean": 1.0}},
        "quality": ("q",),
        "risk_findings": ("r",),
        "summary": {"rate": 0.5},
        "plan": profiles.Blueprint(("a",), ()),
    }
    values.update(changes)
    return profiles.Portrait(**values)


def test_a_portrait_reads_the_leakage_screen():
    # Enforces website/src/pages/docs/api-stability.astro:62.
    plain = portrait()
    assert plain.blockers == ()
    assert plain.untested == ()
    assert plain.admonition == ""
    screened = portrait(
        leakage={"blocking": ["x"], "unchecked": [1, "y"], "warning": "careful"}
    )
    assert screened.blockers == ("x",)
    assert screened.untested == ("1", "y")
    assert screened.admonition == "careful"


def test_a_portrait_normalises_evidence_and_dumps_single_words():
    made = portrait(evidence=({"k": [1]},), missingness=({"c": "a"},))
    record = made.to_dict()
    assert record["findings"] == {"quality": ["q"], "risk": ["r"]}
    assert record["plan"]["numeric"] == ["a"]
    assert record["evidence"] == [{"k": [1]}]
    assert record["missingness"] == [{"c": "a"}]
    assert record["columns"] == {"a": {"mean": 1.0}}
    assert record["statistics"] == {}
    with pytest.raises(TypeError):
        portrait(evidence=({"k": object()},))


# ---- profiles: curation and datasheet --------------------------------------


def test_blocking_findings_without_a_safe_disposition_hold_release():
    # Enforces website/src/pages/docs/api.astro:298.
    # Enforces website/src/pages/docs/data.astro:188.
    # Enforces website/src/pages/docs/extensions.astro:211.
    # Enforces website/src/pages/docs/modalities.astro:41.
    report = curation()
    assert [f.identifier for f in report.blockers] == [
        "q.a"
    ]
    assert report.release is False
    for safe in ("resolved", "waived", "false_positive"):
        cleared = curation(quality=(finding("q.a", disposition=safe),))
        assert cleared.release is True
    accepted = curation(
        quality=(finding("q.a", disposition="accepted"),)
    )
    assert accepted.release is False


def test_a_curation_dumps_single_words():
    record = curation().to_dict()
    assert record["schema_version"] == 2
    assert record["rows"] == {"source": 12, "profiled": 10}
    assert record["read"] == 0.25
    assert record["notes"] == ["trimmed spaces"]
    assert record["features"] == {"numeric": ["x", "y"], "categorical": ["c"]}
    assert record["dropped"] == {"id": "identifier"}
    assert record["ready"] is False
    assert record["blockers"] == ["q.a"]
    assert [f["identifier"] for f in record["findings"]["risk"]] == ["r.b"]


def test_a_datasheet_is_distilled_from_curation():
    card = profiles.Datasheet.distill(curation())
    assert card.dataset == "ds_1"
    assert card.snapshot == "snap-1"
    assert card.task == "binary_classification"
    assert card.target == "label"
    assert card.use == (
        "Training and evaluation for binary_classification using the "
        "declared target and feature decisions."
    )
    assert len(card.excluded) == 3
    assert card.composition == {
        "source_rows": 12,
        "profiled_rows": 10,
        "feature_columns": 3,
        "dropped_columns": 1,
    }
    assert card.limitations[2] == "trimmed spaces"
    assert len(card.limitations) == 3
    assert card.owner == "ml-platform"
    assert card.status == "machine_generated_unreviewed"
    assert card.schema_version == 2


def test_a_datasheet_without_risks_or_owner_says_so():
    bare = curation(
        manifest=manifest(task="", source=provenance.Origin()),
        risk_findings=(),
        notes=(),
    )
    card = profiles.Datasheet.distill(bare)
    assert (
        card.limitations[-1] == "No risk finding was emitted by the profiler."
    )
    assert card.owner == "unassigned"
    assert "the declared learning task" in card.use


def test_a_datasheet_dumps_single_words():
    record = profiles.Datasheet.distill(curation()).to_dict()
    assert record["dataset"] == "ds_1"
    assert record["snapshot"] == "snap-1"
    assert record["use"].startswith("Training and evaluation")
    assert record["review"] == "machine_generated_unreviewed"
    assert record["features"]["numeric"] == ["x", "y"]
    assert len(record["excluded"]) == 3
    assert record["findings"]["quality"][0]["identifier"] == "q.a"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"dataset_id": " "}, "dataset_id must not be empty"),
        ({"modality": ""}, "modality must not be empty"),
        ({"intended_use": " "}, "intended_use must not be empty"),
        ({"owner": ""}, "owner must not be empty"),
        ({"review_status": " "}, "review_status must not be empty"),
        ({"generated_at": ""}, "generated_at must not be empty"),
        ({"schema_version": 0}, "schema_version must be positive"),
        ({"composition": {"rows": -1}}, "must not be negative"),
        ({"out_of_scope": ()}, "out_of_scope must not be empty"),
        ({"limitations": ()}, "limitations must not be empty"),
    ],
)
def test_a_datasheet_refuses_an_unhelpful_card(changes, message):
    card = profiles.Datasheet.distill(curation())
    with pytest.raises(errors.InputError, match=message):
        dataclasses.replace(card, **changes)


def test_a_datasheet_accepts_a_zero_count_and_version_one():
    card = profiles.Datasheet.distill(curation())
    assert dataclasses.replace(card, composition={"rows": 0}, schema_version=1)


def test_preprocessing_defaults_are_conservative():
    made = profiles.Preprocessing()
    assert made.impute is True
    assert made.scale is False
    assert made.cardinality == "onehot"
    assert made.missing == "median"
    assert made.encoding == "tfidf"
