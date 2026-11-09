"""Duplicates, outlier triage, registry and pipeline order."""

import numpy
import pandas
import pytest
from scipy import stats

from scmle.data.detectors import certificates
from scmle.data.detectors import duplicates
from scmle.data.detectors import outliers
from scmle.data.detectors import pipeline
from scmle.data.detectors import registry


def table(n=200, seed=0):
    rng = numpy.random.default_rng(seed)
    return pandas.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )


def test_exact_flags_keep_the_first_copy():
    frame = table(20)
    frame.loc[15] = frame.loc[3]
    frame.loc[17] = frame.loc[3]
    flags = duplicates.exact_flags(frame, list(frame.columns))
    assert list(numpy.flatnonzero(flags)) == [15, 17]


def test_exact_flags_with_no_columns_flag_nothing():
    assert not duplicates.exact_flags(table(5), []).any()


def test_near_flags_find_jittered_copies_but_not_distinct_rows():
    frame = table(300)
    copies = frame.iloc[:30].copy() + 1e-3
    both = pandas.concat([frame, copies], ignore_index=True)
    flags = duplicates.near_flags(both, list(both.columns), 0.01)
    assert flags[300:].all() and not flags[:300].any()


def test_near_flags_cut_boundary_is_inclusive_in_rms_units():
    frame = pandas.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] * 5})
    pair = pandas.DataFrame({"a": [100.0, 101.0]})
    both = pandas.concat([frame, pair], ignore_index=True)
    gap = 1.0 / both["a"].std(ddof=0)  # the pair's distance in column sds
    assert duplicates.near_flags(both, ["a"], gap * (1 + 1e-9))[-1]
    assert not duplicates.near_flags(both, ["a"], gap * (1 - 1e-6))[-1]


def test_text_columns_block_comparisons():
    frame = pandas.DataFrame(
        {
            "k": ["x", "y"] * 10 + ["x", "y"],
            "v": [*numpy.arange(20.0), 0.0, 0.0],
        }
    )
    cols = ["k", "v"]
    flags = duplicates.near_flags(frame, cols, 0.001)
    assert list(numpy.flatnonzero(flags)) == [
        20
    ]  # row 21 differs in k from row 0
    assert not duplicates.near_flags(frame[["k"]], ["k"], 0.1).any()


def test_near_flags_validate_the_cut():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="cut"):
            duplicates.near_flags(table(5), ["a"], bad)


def test_duplicate_flags_combine_exact_and_near():
    frame = table(100)
    frame.loc[100] = frame.loc[1]
    frame.loc[101] = frame.loc[2] + 1e-4
    flags = duplicates.duplicate_flags(frame)
    assert flags[100] and flags[101] and flags.sum() == 2
    assert flags.index.equals(frame.index)


def test_assess_duplicates_report_and_exact_interval():
    frame = table(100)
    frame["y"] = numpy.arange(100) % 2
    extra = frame.iloc[:5].copy()
    conflict = frame.iloc[5:6].copy()
    conflict["y"] = 1 - conflict["y"]
    both = pandas.concat([frame, extra, conflict], ignore_index=True)
    report = duplicates.assess_duplicates(
        both, columns=["a", "b", "c"], target="y"
    )
    assert report.rows == 106 and report.exact_extra == 6 and report.groups == 6
    assert report.label_conflicts == 1
    assert report.exact_rate == pytest.approx(6 / 106)
    ref = stats.binomtest(6, 106).proportion_ci(0.95)
    assert report.exact_interval == pytest.approx((ref.low, ref.high))
    assert report.near_extra == 0 and report.cut == duplicates.DEFAULT_CUT
    assert report.to_dict()["groups"] == 6
    plain = duplicates.assess_duplicates(both)
    assert plain.label_conflicts == 0


def test_robust_z_by_hand():
    series = pandas.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    z = outliers.robust_z(series)
    assert z.iloc[4] == pytest.approx(97.0 / (1.4826 * 1.0))
    assert outliers.robust_z(pandas.Series([2.0] * 5)).sum() == 0.0


def triage_frame():
    rng = numpy.random.default_rng(0)
    frame = pandas.DataFrame(
        {"v": rng.normal(10, 1, 200), "w": rng.normal(0, 1, 200)}
    )
    frame.loc[0, "v"] = -50.0  # outside instrument range
    frame.loc[1, "v"] = 40.0  # in range, extreme, extremes possible
    frame.loc[2, "w"] = 30.0  # extreme, no provenance on w
    frame.loc[3, "v"] = 80.0  # outside population range, inside instrument
    return frame


def test_triage_assigns_class_only_from_provenance():
    frame = triage_frame()
    info = {
        "v": outliers.FieldProvenance(
            instrument_range=(-100.0, 100.0),
            population_range=(0.0, 60.0),
            extremes_possible=True,
        )
    }
    result = outliers.triage(frame, info, {"neg": lambda f: f["v"] < -10})
    assert result.classes[0] == "error"
    assert result.classes[1] == "real_extreme"
    assert result.classes[2] == "unclassified"
    assert result.classes[3] == "population"
    assert result.classes[10] == "none"
    assert result.tier == "attested"
    assert (
        result.counts()["none"] > 190 and sum(result.counts().values()) == 200
    )
    assert result.candidates[[0, 1, 2, 3]].all()


def test_without_provenance_everything_flagged_is_unclassified_and_kept():
    result = outliers.triage(triage_frame())
    assert result.tier == "unverified"
    assert set(result.classes.unique()) == {"none", "unclassified"}
    assert result.classes[0] == "unclassified"


def test_range_violation_is_an_error_even_when_not_a_ranked_candidate():
    frame = pandas.DataFrame({"v": numpy.linspace(0, 100, 101)})
    frame.loc[5, "v"] = 100.5
    result = outliers.triage(
        frame, {"v": outliers.FieldProvenance(instrument_range=(0, 100))}
    )
    assert result.classes[5] == "error" and not result.candidates[5]
    flags = outliers.error_flags(
        frame, {"v": outliers.FieldProvenance(instrument_range=(0, 100))}
    )
    assert list(numpy.flatnonzero(flags)) == [5]


def test_range_bounds_are_inclusive_and_missing_is_ignored():
    frame = pandas.DataFrame({"v": [0.0, 100.0, numpy.nan, 100.0001, -0.0001]})
    mask = outliers.outside(frame["v"], (0.0, 100.0))
    assert list(mask) == [False, False, False, True, True]


def test_extremes_possible_false_keeps_candidates_unclassified():
    frame = triage_frame()
    info = {
        "v": outliers.FieldProvenance(
            instrument_range=(-100.0, 100.0), extremes_possible=False
        )
    }
    assert outliers.triage(frame, info).classes[1] == "unclassified"


def test_triage_rejects_bad_provenance_and_contracts():
    frame = triage_frame()
    with pytest.raises(ValueError, match="unknown column"):
        outliers.triage(frame, {"zz": outliers.FieldProvenance()})
    with pytest.raises(ValueError, match="boolean"):
        outliers.triage(frame, None, {"bad": lambda f: f["v"]})


def test_isolation_rank_orders_the_outlier_first_and_is_deterministic():
    frame = triage_frame()
    rank = outliers.isolation_rank(frame, ["v", "w"], 0)
    assert rank.idxmax() in (0, 2)
    assert rank.equals(outliers.isolation_rank(frame, ["v", "w"], 0))
    assert outliers.summary(outliers.triage(frame))["tier"] == "unverified"


def test_registry_hashes_whole_modules_and_covers_every_detector():
    assert set(registry.SPECS) == set(registry.EXECUTION_ORDER)
    assert registry.EXECUTION_ORDER[0] == "f5-sentinel"
    hashes = {s.code_hash() for s in registry.SPECS.values()}
    assert len(hashes) == len(registry.SPECS)
    assert all(len(h) == 64 for h in hashes)


def sentinel_table():
    rng = numpy.random.default_rng(0)
    frame = pandas.DataFrame(
        {
            "a": rng.normal(size=200),
            "b": rng.normal(size=200),
            "y": rng.integers(0, 2, 200),
        }
    )
    frame.loc[:29, "a"] = -999.0
    return frame


def test_pipeline_runs_sentinels_first_and_recodes_before_later_detectors():
    frame = sentinel_table()
    result = pipeline.run(frame, target="y", seed=0)
    assert result.order == registry.EXECUTION_ORDER
    assert result.recode_log and result.recode_log[0]["recoded"] == 30
    # missingness saw the recoded gaps; the raw table has none
    assert result.results["f5-missingness"].rows_checked == 200
    assert any(
        i.column == "a" for i in result.results["f5-missingness"].indicators
    )
    assert (frame["a"] == -999.0).sum() == 30
    assert "f5-label-noise" in result.results
    assert "f5-duplicates" in result.results


def test_pipeline_declines_label_noise_without_a_target_or_features():
    frame = sentinel_table()
    assert "no target" in pipeline.run(frame).skipped["f5-label-noise"]
    only = frame[["y"]]
    assert (
        "numeric features"
        in pipeline.run(only, target="y").skipped["f5-label-noise"]
    )
    tiny = frame.iloc[:20]
    assert "50" in pipeline.run(tiny, target="y").skipped["f5-label-noise"]


def passing_store(detector_id, defect):
    spec = registry.SPECS[detector_id]
    from scmle.contracts import qualification

    cert = qualification.Certificate(
        detector_id, defect, 300, 300, 500, 0, (0, "c", 0)
    )
    policy = certificates.Standard(0.8, 0.05, 0.95)
    return certificates.Store(
        [
            certificates.certify(
                detector_id, spec.code_hash(), "c" * 64, (0,), [cert], policy
            )
        ]
    )


def test_gate_status_is_descriptive_without_a_current_passing_certificate():
    none = pipeline.gate_status(None, "f5-sentinel")
    assert set(none.values()) == {certificates.GateStatus.DESCRIPTIVE}
    store = passing_store("f5-sentinel", "sentinel")
    status = pipeline.gate_status(store, "f5-sentinel")
    assert status["sentinel"] is certificates.GateStatus.GATING
    empty = pipeline.gate_status(certificates.Store(), "f5-sentinel")
    assert empty["sentinel"] is certificates.GateStatus.DESCRIPTIVE
    other = passing_store("f5-duplicates", "duplicate_exact")
    assert pipeline.gate_status(other, "f5-sentinel")["sentinel"] is (
        certificates.GateStatus.DESCRIPTIVE
    )
    result = pipeline.run(sentinel_table(), store=store)
    assert (
        result.status["f5-sentinel"]["sentinel"]
        is certificates.GateStatus.GATING
    )
    assert result.status["f5-duplicates"]["duplicate_exact"] is (
        certificates.GateStatus.DESCRIPTIVE
    )


def test_a_missing_text_value_does_not_break_the_block_key():
    frame = pandas.DataFrame(
        {
            "k": ["x", None] * 10 + ["x", None],
            "v": [*numpy.arange(20.0), 0.0, 1.0],
        }
    )
    flags = duplicates.near_flags(frame, ["k", "v"], 0.01)
    assert flags[-2:].all() and not flags[:20].any()
