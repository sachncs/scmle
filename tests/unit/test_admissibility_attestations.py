"""Per-column attestations that override a certified leakage block."""

import datetime
import json

import numpy as np
import pandas as pd
import pytest

from scmle.contracts import qualification
from scmle.data.admissibility import attestations
from scmle.data.admissibility import provenance
from scmle.data.admissibility import qualify
from scmle.data.admissibility import rules

SCOPE = "charter-hash-1"
TODAY = datetime.date(2026, 6, 1)


def table(rows=500, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=rows)
    y = pd.Series((a + rng.normal(size=rows) > 0).astype(int), name="y")
    return pd.DataFrame({"a": a, "copy": y.to_numpy() * 1.0}), y


def attest(column="copy", scope=SCOPE, date="2026-01-01", expires="2026-12-31"):
    return attestations.LeakageAttestation(
        column,
        provenance.Attestation(
            "A. Owner",
            "data owner",
            "recorded before the moment",
            "ticket DATA-1",
            date,
            scope,
            expires,
        ),
        3600.0,
    )


def run_with(items):
    features, target = table()
    return qualify.run(
        features,
        target,
        kind="predictive",
        classification=True,
        seed=0,
        attestations=items,
        scope=SCOPE,
        today=TODAY,
    )


def test_without_an_attestation_the_certified_suspect_is_blocked():
    assert set(run_with(()).quarantined) == {"copy"}


def test_a_valid_attestation_keeps_the_column_flagged_and_attested():
    found = run_with([attest()])

    assert found.quarantined == {}
    assert found.table is not None
    decision = next(d for d in found.table.decisions if d.name == "copy")
    assert decision.admission is rules.Admission.ADMIT_FLAGGED
    assert decision.tier is qualification.Tier.ATTESTED
    assert decision.suspicion_kept
    assert "copy" in found.suspects


def test_the_override_is_a_ledger_row_with_the_attester_and_the_p_value():
    found = run_with([attest()])

    rows = [c for s, c in found.checks if s == qualify.ATTESTATION_SOURCE]
    assert [c.check_id for c in rows] == ["leakage.attestation:copy"]
    row = rows[0]
    assert row.tier is qualification.Tier.ATTESTED
    assert "A. Owner" in row.null_hypothesis
    assert "ticket DATA-1" in row.null_hypothesis
    assert row.p_value is not None and row.p_value < 0.05


def test_an_expired_attestation_does_not_count():
    found = run_with([attest(expires="2026-05-31")])

    assert set(found.quarantined) == {"copy"}
    assert any("ignored" in n and "valid" in n for n in found.notes)
    assert not [c for s, c in found.checks if s == qualify.ATTESTATION_SOURCE]


def test_an_attestation_for_another_charter_does_not_count():
    found = run_with([attest(scope="other-charter")])

    assert set(found.quarantined) == {"copy"}
    assert any("not this charter" in n for n in found.notes)


def test_an_attestation_for_an_unflagged_column_is_unused_and_says_so():
    found = run_with([attest("a")])

    assert any("unused" in n for n in found.notes)
    assert not [c for s, c in found.checks if s == qualify.ATTESTATION_SOURCE]
    assert set(found.quarantined) == {"copy"}


def test_a_file_round_trips(tmp_path):
    item = attest()
    path = tmp_path / "attestations.json"
    path.write_text(
        json.dumps({"format_version": 1, "attestations": [item.to_dict()]})
    )

    assert attestations.load(path) == (item,)


@pytest.mark.parametrize(
    "document",
    [
        {"format_version": 2, "attestations": []},
        {"format_version": 1},
        [],
    ],
)
def test_a_malformed_file_is_refused(tmp_path, document):
    path = tmp_path / "a.json"
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError):
        attestations.load(path)


def test_two_attestations_for_one_column_are_refused(tmp_path):
    record = attest().to_dict()
    path = tmp_path / "a.json"
    path.write_text(
        json.dumps({"format_version": 1, "attestations": [record, record]})
    )

    with pytest.raises(ValueError, match="one attestation per column"):
        attestations.load(path)


def test_a_record_with_an_unknown_field_is_refused():
    record = {**attest().to_dict(), "extra": 1}

    with pytest.raises(ValueError):
        attestations.LeakageAttestation.from_dict(record)


def test_a_missing_file_is_a_value_error(tmp_path):
    with pytest.raises(ValueError, match="cannot read"):
        attestations.load(tmp_path / "none.json")
