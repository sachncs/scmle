"""Point-in-time reconstruction from change logs, against brute force."""

import math

import numpy
import pandas
import pytest

from scmle.contracts.qualification import Decision
from scmle.contracts.qualification import SetId
from scmle.contracts.qualification import Tier
from scmle.data.admissibility import asof
from scmle.data.admissibility import provenance
from scmle.data.detectors import certificates
from tests.unit.test_admissibility_provenance import make_field


def brute_force(rows, log):
    """Latest log value at or before the moment; ties to the later entry."""
    out = []
    for key, moment in zip(rows["key"], rows["moment"], strict=True):
        best = None
        for seq, (k, t, v) in enumerate(
            zip(log["key"], log["written_at"], log["value"], strict=True)
        ):
            later = best is None or (t, seq) >= (best[0], best[1])
            if k == key and t <= moment and later:
                best = (t, seq, v)
        out.append((math.nan, math.nan) if best is None else (best[2], best[0]))
    return out


def random_world(seed, n=60, entries=120):
    rng = numpy.random.default_rng(seed)
    log = pandas.DataFrame(
        {
            "key": rng.integers(0, n // 2, entries),
            # a coarse grid makes equal write times (ties) common
            "written_at": rng.integers(0, 20, entries).astype(float),
            "value": rng.integers(0, 1000, entries).astype(float),
        }
    )
    rows = pandas.DataFrame(
        {
            "key": rng.integers(0, n // 2, n),
            "moment": rng.integers(0, 22, n).astype(float),
        }
    )
    return rows, log


@pytest.mark.parametrize("seed", range(12))
def test_join_matches_brute_force_on_random_logs(seed):
    rows, log = random_world(seed)
    joined = asof.point_in_time(
        rows, log, "key", "moment", "written_at", "value"
    )
    expected = brute_force(rows, log)
    got = list(zip(joined["reconstructed"], joined["known_at"], strict=True))
    for (gv, gt), (ev, et) in zip(got, expected, strict=True):
        if math.isnan(ev):
            assert math.isnan(gv) and math.isnan(gt)
        else:
            assert (gv, gt) == (ev, et)
    assert list(joined.index) == list(rows.index)
    assert joined["has_entry"].tolist() == [
        not math.isnan(e[0]) for e in expected
    ]


def test_an_entry_written_exactly_at_the_moment_is_used():
    rows = pandas.DataFrame({"key": [1], "moment": [10.0]})
    log = pandas.DataFrame(
        {"key": [1, 1], "written_at": [5.0, 10.0], "value": [1.0, 2.0]}
    )
    joined = asof.point_in_time(
        rows, log, "key", "moment", "written_at", "value"
    )
    assert joined["reconstructed"].tolist() == [2.0]


def test_an_entry_just_after_the_moment_is_not_used():
    rows = pandas.DataFrame({"key": [1], "moment": [10.0]})
    log = pandas.DataFrame(
        {"key": [1, 1], "written_at": [5.0, 10.000001], "value": [1.0, 2.0]}
    )
    joined = asof.point_in_time(
        rows, log, "key", "moment", "written_at", "value"
    )
    assert joined["reconstructed"].tolist() == [1.0]


def test_equal_write_times_go_to_the_later_log_entry():
    rows = pandas.DataFrame({"key": [1], "moment": [10.0]})
    log = pandas.DataFrame(
        {"key": [1, 1, 1], "written_at": [4.0, 4.0, 4.0], "value": [7, 8, 9]}
    )
    joined = asof.point_in_time(
        rows, log, "key", "moment", "written_at", "value"
    )
    assert joined["reconstructed"].tolist() == [9]


def test_datetime_columns_agree_with_numeric_seconds():
    base = pandas.Timestamp("2026-03-01 12:00:00")
    rows = pandas.DataFrame(
        {"key": [1, 2], "moment": [base, base + pandas.Timedelta(hours=1)]}
    )
    log = pandas.DataFrame(
        {
            "key": [1, 1, 2],
            "written_at": [
                base - pandas.Timedelta(hours=2),
                base + pandas.Timedelta(minutes=1),
                base + pandas.Timedelta(hours=2),
            ],
            "value": [10, 11, 12],
        }
    )
    joined = asof.point_in_time(
        rows, log, "key", "moment", "written_at", "value"
    )
    assert joined["reconstructed"].iloc[0] == 10
    assert not joined["has_entry"].iloc[1]
    assert joined["known_at"].iloc[0] == pytest.approx(
        asof.to_seconds(pandas.Series([base - pandas.Timedelta(hours=2)]))[0]
    )


def test_timezone_aware_times_are_accepted():
    series = pandas.Series(
        pandas.to_datetime(["2026-01-01"]).tz_localize("UTC")
    )
    assert (
        asof.to_seconds(series)[0]
        == asof.to_seconds(pandas.Series(pandas.to_datetime(["2026-01-01"])))[0]
    )


def test_times_must_be_present_and_numeric_or_datetime():
    with pytest.raises(ValueError, match="missing"):
        asof.to_seconds(pandas.Series([1.0, numpy.nan]))
    with pytest.raises(ValueError, match="datetimes or numbers"):
        asof.to_seconds(pandas.Series(["a", "b"]))
    with pytest.raises(ValueError, match="datetimes or numbers"):
        asof.to_seconds(pandas.Series([True, False]))


def make_clean(n=40, seed=0):
    rng = numpy.random.default_rng(seed)
    keys = numpy.arange(n)
    times = numpy.sort(rng.uniform(0, 50, size=(n, 3)), axis=1)
    values = rng.integers(0, 10_000, size=(n, 3)).astype(float)
    moment = rng.uniform(times[:, 0] + 0.01, times[:, 2] - 0.01)
    idx = (times <= moment[:, None]).sum(axis=1) - 1
    log = pandas.DataFrame(
        {
            "key": numpy.repeat(keys, 3),
            "written_at": times.ravel(),
            "value": values.ravel(),
        }
    )
    rows = pandas.DataFrame(
        {"key": keys, "moment": moment, "stored": values[keys, idx]}
    )
    return rows, log, values


def audit(rows, log, **kw):
    return asof.audit_field(
        rows, log, "key", "moment", "stored", "written_at", "value", **kw
    )


def test_clean_table_has_no_mismatch_and_a_derived_rule():
    rows, log = make_clean()[:2]
    result = audit(rows, log)
    assert result.log_check.mismatched == 0
    assert result.counts == {
        "match": 40,
        "overwritten": 0,
        "not_yet_written": 0,
    }
    assert result.as_of.kind is provenance.AsOfKind.TIMESTAMP
    gaps = [
        m - w
        for m, w in zip(
            rows["moment"],
            asof.point_in_time(
                rows, log, "key", "moment", "written_at", "value"
            )["known_at"],
            strict=True,
        )
    ]
    assert result.as_of.lead_seconds == pytest.approx(min(gaps))
    assert result.as_of.lead_seconds > 0
    assert result.check.decision is Decision.PASS
    assert result.check.set_id is SetId.EXPLORATION


@pytest.mark.parametrize("seed", range(8))
def test_planted_overwrites_are_recovered_exactly(seed):
    rows, log, values = make_clean(60, seed)
    rng = numpy.random.default_rng(seed + 100)
    chosen = rng.choice(60, size=9, replace=False)
    planted = rows.copy()
    planted.loc[chosen, "stored"] = values[chosen, 2] + 1.0  # later, new value
    result = audit(planted, log)
    flagged = asof.reconstruction_mismatch(
        planted, log, "key", "moment", "stored", "written_at", "value"
    )
    assert set(numpy.flatnonzero(flagged)) == set(chosen)
    assert result.log_check.mismatched == 9
    assert result.log_check.rate == pytest.approx(9 / 60)
    low, high = result.interval
    assert low <= 9 / 60 <= high
    assert (low, high) == certificates.wilson_interval(9, 60)


def test_a_value_stored_before_it_was_written_is_not_yet_written():
    rows = pandas.DataFrame(
        {"key": [1, 2], "moment": [5.0, 5.0], "stored": [3.0, numpy.nan]}
    )
    log = pandas.DataFrame({"key": [1], "written_at": [9.0], "value": [3.0]})
    result = audit(rows, log)
    assert result.status.tolist() == ["not_yet_written", "match"]


def test_a_missing_stored_value_where_the_log_has_one_is_a_mismatch():
    rows = pandas.DataFrame(
        {"key": [1], "moment": [5.0], "stored": [numpy.nan]}
    )
    log = pandas.DataFrame({"key": [1], "written_at": [1.0], "value": [3.0]})
    assert audit(rows, log).status.tolist() == ["overwritten"]


def test_the_tolerance_decides_the_field_and_the_gate_the_decision():
    rows, log, values = make_clean(50)
    bad = rows.copy()
    bad.loc[:4, "stored"] = values[:5, 2] + 1.0  # 5 of 50 = 0.1
    at = audit(bad, log, tolerance=0.1)
    over = audit(bad, log, tolerance=0.0999)
    assert at.check.decision is Decision.PASS
    assert over.check.decision is Decision.WARN  # descriptive: demoted
    gated = audit(
        bad, log, tolerance=0.0999, gate=certificates.GateStatus.GATING
    )
    assert gated.check.decision is Decision.QUARANTINE
    assert gated.check.tier is Tier.UNVERIFIED


def test_log_provenance_confirms_a_clean_field_and_refuses_a_mismatched_rule():
    rows, log = make_clean()[:2]
    base = make_field("stored", source=provenance.AsOfSource.NONE)
    record = asof.log_provenance(base, audit(rows, log))
    assert record.as_of_source is provenance.AsOfSource.LOG
    assert record.tier() is Tier.CONFIRMED
    with pytest.raises(ValueError, match="another field"):
        asof.log_provenance(make_field("other"), audit(rows, log))
    empty = pandas.DataFrame(
        {"key": [1], "moment": [5.0], "stored": [numpy.nan]}
    )
    nothing = pandas.DataFrame(
        {"key": [2], "written_at": [1.0], "value": [1.0]}
    )
    with pytest.raises(ValueError, match="no row matched"):
        asof.log_provenance(base, audit(empty, nothing))


def test_a_failed_audit_gives_a_field_that_is_not_confirmed():
    rows, log, values = make_clean()
    bad = rows.copy()
    bad.loc[0, "stored"] = values[0, 2] + 1.0
    record = asof.log_provenance(
        make_field("stored", source=provenance.AsOfSource.NONE), audit(bad, log)
    )
    assert record.tier() is Tier.UNVERIFIED


def test_empty_rows_are_rejected():
    rows, log = make_clean()[:2]
    with pytest.raises(ValueError, match="empty"):
        audit(rows.iloc[:0], log)


def test_agreement_treats_two_missing_values_as_equal():
    left = pandas.Series([1.0, numpy.nan, numpy.nan, 2.0])
    right = pandas.Series([1.0, numpy.nan, 3.0, 5.0])
    assert asof.agrees(left, right).tolist() == [True, True, False, False]
