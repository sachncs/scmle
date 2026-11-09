"""Field provenance, as-of rules, attestations and tiers."""

import datetime
import json

import pytest

from scmle.contracts.qualification import Tier
from scmle.data.admissibility import provenance

DAY = datetime.date(2026, 6, 15)


def attestation(date="2026-01-01", expires="2026-12-31", **kw):
    fields = {
        "name_or_id": "j.doe",
        "role": "data owner",
        "statement": "written at event time, never overwritten",
        "evidence": "DDL table orders_v3, ticket DATA-42",
        "date": date,
        "scope": "a" * 64,
        "expires": expires,
    }
    fields.update(kw)
    return provenance.Attestation(**fields)


def make_field(
    name="f",
    lead=3600.0,
    source=provenance.AsOfSource.ATTESTED,
    serving=provenance.Availability.YES,
    att="default",
    log=None,
    kind=provenance.AsOfKind.LAG,
):
    if source is provenance.AsOfSource.NONE:
        as_of = provenance.AsOf.unknown()
    else:
        as_of = provenance.AsOf(kind, lead)
    if att == "default":
        att = (
            attestation() if source is provenance.AsOfSource.ATTESTED else None
        )
    if source is provenance.AsOfSource.LOG and log is None:
        log = provenance.LogCheck(100, 0, 0.0)
    return provenance.FieldProvenance(
        name=name,
        origin=provenance.Origin.INSTRUMENT,
        writer="sensor",
        written_at=provenance.WrittenAt.EVENT_TIME,
        overwritten=provenance.Overwritten.NEVER,
        definition_history=("2025-01-01 created",),
        as_of=as_of,
        as_of_source=source,
        available_at_serving=serving,
        attestation=att,
        log_check=log,
    )


def test_undeclared_field_is_unverified_with_everything_unknown():
    record = provenance.undeclared("x")
    assert record.tier() is Tier.UNVERIFIED
    assert record.as_of == provenance.AsOf.unknown()
    assert record.available_at_serving is provenance.Availability.UNKNOWN


def test_attested_field_is_attested_not_confirmed():
    assert make_field().tier(DAY) is Tier.ATTESTED


def test_log_field_with_a_passing_check_is_confirmed():
    record = make_field(
        source=provenance.AsOfSource.LOG, kind=provenance.AsOfKind.TIMESTAMP
    )
    assert record.tier() is Tier.CONFIRMED


def test_log_field_with_a_failing_check_and_no_attestation_is_unverified():
    log = provenance.LogCheck(100, 5, 0.0)
    record = make_field(source=provenance.AsOfSource.LOG, log=log, att=None)
    assert record.tier() is Tier.UNVERIFIED


def test_failing_log_with_an_attestation_falls_back_to_attested():
    log = provenance.LogCheck(100, 5, 0.0)
    record = make_field(
        source=provenance.AsOfSource.LOG, log=log, att=attestation()
    )
    assert record.tier(DAY) is Tier.ATTESTED


def test_expired_attestation_is_unverified():
    record = make_field(att=attestation(expires="2026-06-14"))
    assert record.tier(DAY) is Tier.UNVERIFIED
    assert record.tier(None) is Tier.ATTESTED  # expiry test is skipped


def test_attestation_validity_is_inclusive_at_both_ends():
    a = attestation(date="2026-06-15", expires="2026-06-20")
    assert a.valid_on(datetime.date(2026, 6, 15))
    assert a.valid_on(datetime.date(2026, 6, 20))
    assert not a.valid_on(datetime.date(2026, 6, 14))
    assert not a.valid_on(datetime.date(2026, 6, 21))


def test_attestation_needs_named_evidence_and_ordered_dates():
    with pytest.raises(ValueError, match="evidence"):
        attestation(evidence=" ")
    with pytest.raises(ValueError, match="name_or_id"):
        attestation(name_or_id="")
    with pytest.raises(ValueError, match="expires"):
        attestation(date="2026-02-01", expires="2026-01-31")
    with pytest.raises(ValueError, match="ISO date"):
        attestation(date="June 1")
    attestation(date="2026-02-01", expires="2026-02-01")  # equal is fine


def test_as_of_rule_consistency():
    with pytest.raises(ValueError, match="no lead"):
        provenance.AsOf(provenance.AsOfKind.UNKNOWN, 1.0)
    with pytest.raises(ValueError, match="needs lead_seconds"):
        provenance.AsOf(provenance.AsOfKind.LAG, None)
    with pytest.raises(ValueError):
        provenance.AsOf(provenance.AsOfKind.LAG, float("nan"))
    assert provenance.AsOf(provenance.AsOfKind.LAG, -5.0).is_known
    assert not provenance.AsOf.unknown().is_known


def test_source_and_rule_must_agree():
    with pytest.raises(ValueError, match="needs a source"):
        provenance.FieldProvenance(
            "f",
            provenance.Origin.MANUAL,
            "w",
            provenance.WrittenAt.UNKNOWN,
            provenance.Overwritten.UNKNOWN,
            (),
            provenance.AsOf(provenance.AsOfKind.LAG, 1.0),
            provenance.AsOfSource.NONE,
            provenance.Availability.YES,
        )
    with pytest.raises(ValueError, match="cannot have a source"):
        provenance.FieldProvenance(
            "f",
            provenance.Origin.MANUAL,
            "w",
            provenance.WrittenAt.UNKNOWN,
            provenance.Overwritten.UNKNOWN,
            (),
            provenance.AsOf.unknown(),
            provenance.AsOfSource.LOG,
            provenance.Availability.YES,
        )
    with pytest.raises(ValueError, match="needs an attestation"):
        make_field(att=None)
    with pytest.raises(ValueError, match="needs a log_check"):
        provenance.FieldProvenance(
            "f",
            provenance.Origin.MANUAL,
            "w",
            provenance.WrittenAt.UNKNOWN,
            provenance.Overwritten.UNKNOWN,
            (),
            provenance.AsOf(provenance.AsOfKind.TIMESTAMP, 1.0),
            provenance.AsOfSource.LOG,
            provenance.Availability.YES,
        )


def test_log_check_passes_at_the_tolerance_and_fails_above_it():
    assert provenance.LogCheck(100, 5, 0.05).passed
    assert not provenance.LogCheck(100, 6, 0.05).passed
    assert provenance.LogCheck(100, 0, 0.0).passed
    assert not provenance.LogCheck(100, 1, 0.0).passed
    assert provenance.LogCheck(200, 3, 0.0).rate == pytest.approx(0.015)
    with pytest.raises(ValueError):
        provenance.LogCheck(10, 11, 0.0)
    with pytest.raises(ValueError):
        provenance.LogCheck(0, 0, 0.0)
    with pytest.raises(ValueError):
        provenance.LogCheck(10, 1, 1.5)


def test_records_round_trip_through_json():
    records = [
        make_field("a"),
        make_field(
            "b",
            source=provenance.AsOfSource.LOG,
            kind=provenance.AsOfKind.TIMESTAMP,
        ),
        provenance.undeclared("c"),
    ]
    table = provenance.ProvenanceTable(tuple(records))
    again = provenance.ProvenanceTable.from_dict(
        json.loads(json.dumps(table.to_dict()))
    )
    assert again == table
    assert again.digest() == table.digest()


def test_digest_changes_with_any_edit():
    base = provenance.ProvenanceTable((make_field("a", lead=3600.0),))
    other = provenance.ProvenanceTable((make_field("a", lead=3601.0),))
    assert base.digest() != other.digest()


def test_from_dict_rejects_unknown_versions_and_fields():
    table = provenance.ProvenanceTable((make_field("a"),)).to_dict()
    with pytest.raises(ValueError, match="format_version"):
        provenance.ProvenanceTable.from_dict({**table, "format_version": 9})
    with pytest.raises(ValueError, match="unexpected"):
        provenance.ProvenanceTable.from_dict({**table, "extra": 1})
    record = dict(table["fields"][0])
    record["surprise"] = 1
    with pytest.raises(ValueError, match="unexpected"):
        provenance.FieldProvenance.from_dict(record)


def test_covering_adds_unverified_records_for_undeclared_columns():
    declared = [make_field("b")]
    table = provenance.ProvenanceTable.covering(["a", "b", "c"], declared)
    assert [f.name for f in table.fields] == ["a", "b", "c"]
    tiers = table.tiers(DAY)
    assert tiers == {
        "a": Tier.UNVERIFIED,
        "b": Tier.ATTESTED,
        "c": Tier.UNVERIFIED,
    }
    with pytest.raises(ValueError, match="not in the data"):
        provenance.ProvenanceTable.covering(["a"], [make_field("zzz")])
    with pytest.raises(ValueError, match="unique"):
        provenance.ProvenanceTable.covering(
            ["b"], [make_field("b"), make_field("b")]
        )
    with pytest.raises(KeyError):
        table.get("nope")
    assert table.get("b").name == "b"


def test_table_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        provenance.ProvenanceTable((make_field("a"), make_field("a")))


def test_no_field_is_confirmed_without_a_log():
    for source in (provenance.AsOfSource.ATTESTED, provenance.AsOfSource.NONE):
        assert make_field(source=source).tier(DAY) is not Tier.CONFIRMED


def test_a_table_without_fields_is_a_value_error():
    with pytest.raises(ValueError, match="fields"):
        provenance.ProvenanceTable.from_dict({"format_version": 1})


@pytest.mark.parametrize("value", [None, 5, ["x"]])
def test_an_attestation_text_field_that_is_not_text_is_a_value_error(value):
    with pytest.raises(ValueError, match="role"):
        attestation(role=value)
