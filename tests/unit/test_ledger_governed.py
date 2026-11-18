"""The governed run: exploration, negative control and confirmation."""

import json

import numpy
import pandas
import pytest

from scmle.assurance.custody import sealed
from scmle.assurance.ledger import confirm
from scmle.assurance.ledger import explore
from scmle.assurance.ledger import governed
from scmle.assurance.ledger import online_fdr
import scmle.assurance.ledger.ledger
from scmle.data.splitting.splitters import lockbox
from tests.unit.test_charter_contracts import make_charter


def build(rows=500, seed=0, effect=0.7, kind="regression"):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame({f"f{i}": rng.normal(size=rows) for i in range(8)})
    frame["label"] = ["a", "b"] * (rows // 2)
    signal = effect * (frame["f0"] + frame["f1"]) + rng.normal(size=rows)
    if kind == "binary":
        return frame.drop(columns="label"), pandas.Series(
            numpy.where(signal > 0, "yes", "no")
        )
    return frame.drop(columns="label"), pandas.Series(signal)


def locate(features, target):
    log = sealed.Journal()
    locked = lockbox.Resident(features, target, tuple(range(len(target))))
    return sealed.Locker("C", locked, log)


def test_count_discoveries_matches_the_ledger_on_the_same_data():
    features, target = build()
    charter = make_charter()
    plan = explore.hypothesize(features, explore.NUMERIC)
    config = online_fdr.Settings(
        online_fdr.ELOND, charter.budget.exploration.initial
    )
    ledger = scmle.assurance.ledger.ledger.Ledger.create(
        None,
        charter_hash=charter.charter_hash,
        config=config,
        alpha_confirm=charter.budget.confirmation.alpha_confirm,
        seed=1,
    )
    results = explore.run_exploration(ledger, features, target, plan)
    assert sum(r.rejected for r in results) == explore.tally(
        config, plan, features, target
    )
    assert sum(r.rejected for r in results) >= 2


def test_a_planted_signal_is_found_controlled_and_confirmed(tmp_path):
    features, target = build()
    cfeatures, ctarget = build(rows=400, seed=9)
    source = locate(cfeatures, ctarget)
    run = governed.run_governed(
        charter=make_charter(),
        features=features,
        target=target,
        classification=False,
        source=source,
        run_dir=tmp_path,
    )
    assert run.control.exceeds and run.control.b == 20
    assert run.control.scope == "global"
    assert run.confirmation.outcome == confirm.CONFIRMED
    columns = {
        run.ledger.hypotheses[c.identifier].spec.column
        for c in run.confirmation.confirmed
    }
    assert columns == {"f0", "f1"}
    for name in (
        scmle.assurance.ledger.ledger.FILE_NAME,
        governed.CONTROL_FILE,
        confirm.CLAIMS_FILE,
    ):
        assert (tmp_path / name).exists()
    control = json.loads((tmp_path / governed.CONTROL_FILE).read_text())
    assert control["permuted"] == list(run.control.permuted)
    reloaded = scmle.assurance.ledger.ledger.Ledger.load(
        tmp_path / scmle.assurance.ledger.ledger.FILE_NAME,
        expected_head=run.ledger.head,
    )
    assert reloaded.confirmation["outcome"] == confirm.CONFIRMED
    summary = run.summary()
    assert summary["confirmed"] == 2 and summary["control_exceeds"] is True
    assert summary["skipped_features"] == 0


def test_a_pure_noise_target_is_not_promoted():
    features, target = build(effect=0.0)
    cfeatures, ctarget = build(rows=400, seed=9, effect=0.0)
    source = locate(cfeatures, ctarget)
    run = governed.run_governed(
        charter=make_charter(),
        features=features,
        target=target,
        classification=False,
        source=source,
        run_dir=None,
    )
    assert run.confirmation.confirmed == ()
    assert run.confirmation.outcome in (
        confirm.NO_SURVIVORS,
        confirm.FLOOR_NOT_EXCEEDED,
        confirm.CONFIRMED,
    )


def test_binary_labels_are_coded_consistently_for_c():
    features, target = build(kind="binary")
    cfeatures, ctarget = build(rows=400, seed=9, kind="binary")
    run = governed.run_governed(
        charter=make_charter(),
        features=features,
        target=target,
        classification=True,
        source=locate(cfeatures, ctarget),
        run_dir=None,
    )
    assert {
        run.ledger.hypotheses[c.identifier].spec.column
        for c in run.confirmation.confirmed
    } == {"f0", "f1"}


def test_multiclass_targets_use_the_kruskal_statistic():
    rng = numpy.random.default_rng(4)
    features = pandas.DataFrame(
        {"a": rng.normal(size=600), "b": rng.normal(size=600)}
    )
    target = pandas.Series(
        pandas.cut(
            features["a"], [-9, -0.5, 0.5, 9], labels=["lo", "mid", "hi"]
        )
    ).astype(str)
    plan = explore.hypothesize(features, explore.target_kind(target, True))
    assert {s.statistic for s in plan.specs} == {"kruskal"}
    run = governed.run_governed(
        charter=make_charter(),
        features=features,
        target=target,
        classification=True,
        source=None,
        run_dir=None,
    )
    assert run.confirmation.outcome == confirm.NO_CONFIRMATION_SET
    assert "a" in {
        run.ledger.hypotheses[i].spec.column
        for i in run.ledger.survivor_set.hypothesis_ids
    }


def test_structure_columns_and_non_numeric_features_are_not_candidates():
    features = build()[0]
    features["site"] = ["x", "y"] * (len(features) // 2)
    features["const"] = 1.0
    features["entity"] = numpy.arange(len(features)) // 10
    plan = explore.hypothesize(features, explore.NUMERIC, ("entity",))
    assert dict(plan.skipped) == {
        "site": "not numeric: descriptive only",
        "const": "constant or entirely missing",
        "entity": "structure column, not a candidate",
    }
    assert len(plan.specs) == 8


def test_strata_come_from_groups_then_time_blocks():
    frame = pandas.DataFrame(
        {"g": [1, 1, 2, 2, 3, 3], "t": [5.0, 1.0, 3.0, 2.0, 6.0, 4.0]}
    )
    groups = governed.strata_labels(frame, "g", "t")
    assert list(groups) == ["1", "1", "2", "2", "3", "3"]
    blocks = governed.strata_labels(frame, None, "t", blocks=3)
    assert len(set(blocks)) == 3
    assert blocks[1] == 0 and blocks[0] == 4 // 2  # lowest time -> block 0
    assert governed.strata_labels(frame, None, None) is None
    assert governed.strata_labels(frame, "missing", None) is None


def test_a_grouped_target_control_reports_strata_scope():
    features, target = build(rows=300)
    strata = numpy.repeat(numpy.arange(30), 10)
    run = governed.run_governed(
        charter=make_charter(),
        features=features,
        target=target,
        classification=False,
        source=None,
        run_dir=None,
        strata=strata,
    )
    assert run.scope == "strata" and run.control.scope == "strata"


def test_unavailable_failure_policies_are_refused_up_front():
    features, target = build()
    with pytest.raises(Exception, match="not implemented"):
        governed.run_governed(
            charter=make_charter(),
            features=features,
            target=target,
            classification=False,
            source=None,
            run_dir=None,
            policy=confirm.THRESHOLDOUT,
        )
