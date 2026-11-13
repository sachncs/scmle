"""Pins for detector-certification behaviours the main suite left unchecked.

None of these read the committed certificates, so they hold under mutation of
the hashed sources. Certification parameters (rates, magnitudes, seeds) are
pinned through the counts they produce on the bundled demo table.
"""

import pandas
import pytest

from benchmarks.qualification import f5
from benchmarks.qualification import standins
from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.detectors import certificates
from scmle.data.detectors import registry


@pytest.fixture(scope="module")
def demo():
    return standins.demo_frame()


def outcome_certificate(detector_id, defect_type, detected):
    return qualification.Certificate(
        detector_id=detector_id,
        defect_type=defect_type,
        planted=300,
        detected=detected,
        clean_rows=500,
        false_alarms=0,
        seed_path=(0, "x", 0),
    )


def current_store(detector_ids=None, detected=300):
    """A store whose certificates are current for the code under test."""
    store = certificates.Store()
    for detector_id, spec in registry.SPECS.items():
        if detector_ids is not None and detector_id not in detector_ids:
            continue
        results = [
            outcome_certificate(detector_id, defect, detected)
            for defect in spec.defect_types
        ]
        store.add(
            certificates.certify(
                detector_id,
                spec.code_hash(),
                "c" * 64,
                (0,),
                results,
                f5.POLICY,
            )
        )
    return store


def test_the_declared_constants():
    assert f5.ROW_SEEDS == (0, 1, 2, 3, 4)
    assert tuple(range(200)) == f5.MISSING_SEEDS
    assert f5.MISSING_COLUMNS == 6
    assert f5.SENTINEL_COLUMNS == ["mean radius", "mean texture"]
    assert f5.OUTLIER_COLUMNS == [
        "mean radius",
        "mean texture",
        "mean perimeter",
    ]
    assert f5.OUTLIER_PAD == 0.10
    assert f5.CUT_GRID == (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)
    assert f5.CLASS_RATES == {0: 0.20, 1: 0.05}
    assert f5.CERT_PATH.name == "detector_certificates.json"


def test_provenance_pads_the_observed_range_by_ten_percent_of_the_span():
    frame = pandas.DataFrame(
        {
            "mean radius": [10.0, 20.0],
            "mean texture": [0.0, 50.0],
            "mean perimeter": [-5.0, 5.0],
        }
    )
    provenance = f5.provenance_for(frame)
    assert list(provenance) == f5.OUTLIER_COLUMNS
    assert provenance["mean radius"].instrument_range == (9.0, 21.0)
    assert provenance["mean texture"].instrument_range == (-5.0, 55.0)
    assert provenance["mean perimeter"].instrument_range == (-6.0, 6.0)


def test_the_plan_catalogue_names_every_detector_and_defect_type():
    plans = f5.row_plans()
    assert {k: [p.defect_type for p in v] for k, v in plans.items()} == {
        "f5-sentinel": ["sentinel"],
        "f5-duplicates": ["duplicate_exact", "duplicate_near"],
        "f5-label-noise": [
            "label_flip_symmetric",
            "label_flip_class_conditional",
        ],
        "f5-outlier-triage": ["outlier_error"],
    }
    assert set(plans) == set(registry.SPECS) - {"f5-missingness"}


def test_each_plan_plants_the_declared_share_in_the_declared_columns(demo):
    plans = f5.row_plans()
    sentinel = plans["f5-sentinel"][0].plant(demo, 0)
    assert int(sentinel.row_mask.sum()) == 108
    changed = (sentinel.frame != demo).any()
    assert set(changed[changed].index) == set(f5.SENTINEL_COLUMNS)

    exact, near = plans["f5-duplicates"]
    planted = exact.plant(demo, 0)
    assert len(planted.frame) == 626
    assert int(planted.row_mask.sum()) == 57
    near_plant = near.plant(demo, 0)
    assert len(near_plant.frame) == 626
    assert int(near_plant.row_mask.sum()) == 57

    symmetric, conditional = plans["f5-label-noise"]
    flipped = symmetric.plant(demo, 0)
    assert int(flipped.row_mask.sum()) == 57
    assert int(conditional.plant(demo, 0).row_mask.sum()) == 60
    outlier = plans["f5-outlier-triage"][0].plant(demo, 0)
    assert int(outlier.row_mask.sum()) == 155
    moved = (outlier.frame != demo).any()
    assert set(moved[moved].index) == set(f5.OUTLIER_COLUMNS)


def test_near_duplicates_are_jittered_copies_not_exact_ones(demo):
    exact, near = f5.row_plans()["f5-duplicates"]
    exact_rows = exact.plant(demo, 0).frame.iloc[len(demo) :]
    near_rows = near.plant(demo, 0).frame.iloc[len(demo) :]
    key = "mean radius"
    originals = set(demo[key])
    assert all(value in originals for value in exact_rows[key])
    assert not any(value in originals for value in near_rows[key])
    nearest = [
        min(abs(value - other) / other for other in originals)
        for value in near_rows[key]
    ]
    assert max(nearest) < 1e-2


def test_label_flips_change_only_the_target_column(demo):
    for plan in f5.row_plans()["f5-label-noise"]:
        planted = plan.plant(demo, 0)
        changed = (planted.frame != demo).any()
        assert list(changed[changed].index) == [standins.TARGET]


def test_class_conditional_flips_are_heavier_on_class_zero(demo):
    plan = f5.row_plans()["f5-label-noise"][1]
    planted = plan.plant(demo, 0)
    flipped = planted.row_mask.to_numpy()
    zero = (demo[standins.TARGET] == 0).to_numpy()
    assert flipped[zero].mean() > flipped[~zero].mean()
    assert flipped[zero].sum() == round(0.20 * zero.sum())
    assert flipped[~zero].sum() == round(0.05 * (~zero).sum())


def test_renamed_plan_relabels_the_injection_and_keeps_the_rest(demo):
    original = f5.row_plans()["f5-sentinel"][0]
    plan = f5.renamed(original, "other")
    assert plan.defect_type == "other"
    injection = plan.plant(demo, 3)
    assert injection.defect_type == "other"
    base = original.plant(demo, 3)
    assert injection.frame.equals(base.frame)
    assert injection.row_mask.equals(base.row_mask)


def test_detectors_are_keyed_by_id_and_return_one_flag_per_row(demo):
    detectors = f5.row_detectors(demo)
    assert list(detectors) == [
        "f5-sentinel",
        "f5-duplicates",
        "f5-label-noise",
        "f5-outlier-triage",
    ]
    counts = {k: int(d(demo).sum()) for k, d in detectors.items()}
    assert counts == {
        "f5-sentinel": 13,
        "f5-duplicates": 0,
        "f5-label-noise": 5,
        "f5-outlier-triage": 0,
    }
    for detector in detectors.values():
        flags = detector(demo)
        assert flags.dtype == bool
        assert flags.index.equals(demo.index)


def test_the_outlier_detector_uses_the_clean_table_ranges(demo):
    detector = f5.row_detectors(demo)["f5-outlier-triage"]
    spoiled = demo.copy()
    spoiled.loc[spoiled.index[7], "mean radius"] = 1000.0
    spoiled.loc[spoiled.index[9], "mean perimeter"] = -1000.0
    flags = detector(spoiled)
    assert list(flags[flags].index) == [spoiled.index[7], spoiled.index[9]]


def test_label_detector_flags_by_position_on_the_frames_own_index(demo):
    shifted = demo.set_axis(range(1000, 1000 + len(demo)))
    flags = f5.label_detector(shifted)
    assert flags.index.equals(shifted.index)
    assert int(flags.sum()) == 5
    assert flags.equals(
        f5.label_detector(demo).set_axis(range(1000, 1000 + len(demo)))
    )


def test_select_cut_follows_the_policy_ceiling_and_confidence(demo):
    assert f5.select_cut(demo) == 0.1
    loose = certificates.Standard(0.8, 0.5, 0.95)
    assert f5.select_cut(demo, loose) == 0.3
    impossible = certificates.Standard(0.8, 0.0001, 0.95)
    assert f5.select_cut(demo, impossible) == f5.CUT_GRID[0]
    edge = certificates.Standard(0.8, 0.2, 0.90)
    assert f5.select_cut(demo, edge) == 0.3
    strict = certificates.Standard(0.8, 0.2, 0.99)
    assert f5.select_cut(demo, strict) == 0.1


def test_certify_missingness_counts_one_replicate_per_seed(demo):
    certificate = f5.certify_missingness(demo, seeds=(4, 5, 6, 7))
    assert certificate.seeds == (4, 5, 6, 7)
    assert [o.certificate.defect_type for o in certificate.outcomes] == [
        "missing_mar",
        "missing_mnar",
    ]
    base = demo.iloc[:, : f5.MISSING_COLUMNS]
    for outcome in certificate.outcomes:
        record = outcome.certificate
        assert record.detector_id == "f5-missingness"
        assert (record.planted, record.clean_rows) == (4, 4)
        assert record.detected == 4
        assert record.false_alarms == 0
        assert record.seed_path == randomness.seed_path(
            4, "certify.f5-missingness", 0
        )
    assert certificate.code_hash == registry.SPECS["f5-missingness"].code_hash()
    assert certificate.corpus_hash == certificates.corpus_hash(
        [f5.certify.frame_digest(base)]
    )


def test_certify_missingness_judges_against_the_given_policy(demo):
    policy = certificates.Standard(0.2, 0.9, 0.5)
    certificate = f5.certify_missingness(demo, seeds=(0, 1), policy=policy)
    assert certificate.policy == policy
    assert all(o.passed for o in certificate.outcomes)
    default = f5.certify_missingness(demo, seeds=(0, 1))
    assert default.policy == f5.POLICY
    assert not any(o.passed for o in default.outcomes)


def test_certify_all_restricts_to_the_requested_detectors(demo):
    only_missing = f5.certify_all(
        missing_seeds=(0, 1), frame=demo, only=["f5-missingness"]
    )
    assert [c.detector_id for c in only_missing.certificates] == [
        "f5-missingness"
    ]
    assert only_missing.certificates[0].seeds == (0, 1)
    only_rows = f5.certify_all(
        seeds=(0,), frame=demo, only=["f5-label-noise", "f5-outlier-triage"]
    )
    assert [c.detector_id for c in only_rows.certificates] == [
        "f5-label-noise",
        "f5-outlier-triage",
    ]


def test_certify_all_pools_row_counts_over_seeds_and_files_the_policy(demo):
    store = f5.certify_all(
        seeds=(0, 1),
        missing_seeds=(0,),
        frame=demo,
        only=["f5-sentinel", "f5-duplicates", "f5-missingness"],
    )
    rows = {(r["detector"], r["defect_type"]): r for r in f5.summary(store)}
    assert rows[("f5-sentinel", "sentinel")]["planted"] == 213
    assert rows[("f5-sentinel", "sentinel")]["detected"] == 213
    assert rows[("f5-sentinel", "sentinel")]["clean"] == 569
    assert rows[("f5-sentinel", "sentinel")]["false_alarms"] == 13
    assert rows[("f5-duplicates", "duplicate_near")]["planted"] == 114
    for item in store.certificates:
        assert item.policy == f5.POLICY
        assert item.seeds in {(0, 1), (0,)}
        spec = registry.SPECS[item.detector_id]
        assert item.code_hash == spec.code_hash()
    default_frame = f5.certify_all(seeds=(0,), only=["f5-sentinel"])
    assert default_frame.certificates[0].corpus_hash == (
        store.certificates[2].corpus_hash
    )


def test_problems_names_each_detector_that_is_not_current():
    assert f5.problems(current_store()) == []
    store = current_store(["f5-sentinel"])
    found = f5.problems(store)
    assert "f5-duplicates: certificate is missing" in found
    assert len(found) == len(registry.SPECS) - 1
    assert found == sorted(
        found, key=lambda m: list(registry.SPECS).index(m.split(":")[0])
    )
    stale = certificates.Store(
        [
            certificates.certify(
                "f5-sentinel",
                "d" * 64,
                "c" * 64,
                (0,),
                [outcome_certificate("f5-sentinel", "sentinel", 300)],
                f5.POLICY,
            )
        ]
    )
    assert "f5-sentinel: certificate is stale" in f5.problems(stale)


def test_gating_lists_detector_hash_and_type_for_passing_outcomes_only():
    store = current_store()
    expected = sorted(
        (detector_id, spec.code_hash(), defect)
        for detector_id, spec in registry.SPECS.items()
        for defect in spec.defect_types
    )
    assert sorted(f5.gating(store)) == expected
    failing = current_store(detected=100)
    assert f5.gating(failing) == []
    mixed = certificates.Store(
        [
            certificates.certify(
                "f5-duplicates",
                registry.SPECS["f5-duplicates"].code_hash(),
                "c" * 64,
                (0,),
                [
                    outcome_certificate(
                        "f5-duplicates", "duplicate_exact", 300
                    ),
                    outcome_certificate("f5-duplicates", "duplicate_near", 100),
                ],
                f5.POLICY,
            )
        ]
    )
    assert f5.gating(mixed) == [
        (
            "f5-duplicates",
            registry.SPECS["f5-duplicates"].code_hash(),
            "duplicate_exact",
        )
    ]


def test_audit_reports_each_pair_the_store_would_refuse(monkeypatch):
    store = current_store(["f5-sentinel"])
    assert f5.audit_problems(store) == []

    def refuse(self, detector_id, digest, defect):
        raise certificates.CertificateError(f"no {defect}")

    monkeypatch.setattr(certificates.Store, "mark_gating", refuse)
    assert f5.audit_problems(store) == ["f5-sentinel/sentinel: no sentinel"]


def test_summary_rounds_bounds_and_reports_every_outcome():
    certificate = qualification.Certificate(
        detector_id="f5-sentinel",
        defect_type="sentinel",
        planted=250,
        detected=240,
        clean_rows=700,
        false_alarms=3,
        seed_path=(0, "x", 0),
    )
    store = certificates.Store(
        [
            certificates.certify(
                "f5-sentinel",
                "a" * 64,
                "b" * 64,
                (0,),
                [certificate],
                f5.POLICY,
            )
        ]
    )
    outcome = store.certificates[0].outcomes[0]
    (row,) = f5.summary(store)
    assert row == {
        "detector": "f5-sentinel",
        "defect_type": "sentinel",
        "planted": 250,
        "detected": 240,
        "recall_lower": round(outcome.recall_interval[0], 4),
        "clean": 700,
        "false_alarms": 3,
        "false_alarm_upper": round(outcome.false_alarm_interval[1], 4),
        "gates": outcome.passed,
    }
    assert row["recall_lower"] != outcome.recall_interval[0]
    assert row["false_alarm_upper"] != outcome.false_alarm_interval[1]
    assert list(row) == [
        "detector",
        "defect_type",
        "planted",
        "detected",
        "recall_lower",
        "clean",
        "false_alarms",
        "false_alarm_upper",
        "gates",
    ]


def test_certify_all_plants_into_the_frame_it_is_given(demo):
    small = demo.iloc[:200].reset_index(drop=True)
    store = f5.certify_all(seeds=(0,), frame=small, only=["f5-sentinel"])
    (row,) = f5.summary(store)
    assert row["clean"] == 200
    assert row["planted"] == 39
