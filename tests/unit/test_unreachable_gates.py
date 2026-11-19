"""Two removed gates guarded conditions no input can reach.

`ingest.multiplicity` (the run blocked when the ledger did not register
exactly the synthesized columns that were not quarantined or skipped) and
`robustness.assessment_present` (the run blocked when the stability stage
held no assessment) were removed on 2026-11-19 (CHANGELOG). Each link of
the argument that their conditions are impossible is checked here:

* The declared family size is the number of synthesized columns: both are
  counted from the one dictionary of kept columns
  (`tests/unit/test_ingest_synthesize.py` checks the count).
* The synthesized columns are appended to the frame, and the split drops
  only the target, so every one is a column of the exploration features.
* The exploration plan puts every feature column either among its
  hypotheses or in its skipped list, and the structure and quarantined
  columns it is told to exclude are always skipped.
* The governed exploration registers every planned hypothesis.
* The governed exploration always calls the second stage, which sets its
  assessment before it returns or raises, so the run stops first.
"""

import numpy
import pandas
import pytest

from scmle.assurance.ledger import explore
from scmle.assurance.ledger import governed
from scmle.assurance.robustness import prefilter
from scmle.contracts import tasks
from scmle.data.splitting.splitters import splitter
from tests.unit.test_charter_contracts import make_charter
from tests.unit.test_ledger_governed import build
from tests.unit.test_robustness_prefilter import SEED
from tests.unit.test_robustness_prefilter import stage_for
from tests.unit.test_robustness_prefilter import world


def mixed(rows=60, seed=0):
    """Numeric, constant, missing, integer and text columns."""
    rng = numpy.random.default_rng(seed)
    return pandas.DataFrame(
        {
            "a": rng.normal(size=rows),
            "b": rng.integers(0, 3, rows),
            "flat": numpy.ones(rows),
            "empty": numpy.full(rows, numpy.nan),
            "text": rng.choice(["x", "y"], rows),
            "group": rng.integers(0, 5, rows),
        }
    )


@pytest.mark.parametrize("kind", [explore.NUMERIC, explore.MULTICLASS])
@pytest.mark.parametrize(
    "exclude", [(), ("group",), ("group", "a"), ("a", "text", "flat")]
)
def test_every_feature_column_is_a_hypothesis_or_skipped(kind, exclude):
    features = mixed()
    plan = explore.hypothesize(features, kind, exclude)
    tested = [spec.column for spec in plan.specs]
    skipped = [pair[0] for pair in plan.skipped]
    assert sorted(tested + skipped) == sorted(features.columns)
    assert not set(tested) & set(skipped)
    assert not set(tested) & set(exclude)


def test_the_exploration_registers_every_planned_hypothesis():
    features, target = build()
    explored = governed.scout(
        charter=make_charter(),
        features=features,
        target=target,
        classification=False,
        run_dir=None,
        exclude=("f7",),
    )
    registered = sorted(
        h.spec.column
        for h in explored.ledger.hypotheses.values()
        if h.spec.column
    )
    assert registered == sorted(spec.column for spec in explored.plan.specs)
    skipped = {pair[0] for pair in explored.plan.skipped}
    assert set(registered) | skipped == set(features.columns)


def test_the_split_keeps_every_column_but_the_target():
    frame = mixed(rows=200).assign(
        y=numpy.arange(200) % 2, key=numpy.arange(200)
    )
    spec = tasks.Config(
        data=tasks.Dataset(
            uri="m",
            target="y",
            id_columns=["key"],
            excluded_columns=["text"],
        ),
    )
    result = splitter.Splitter(spec, tasks.Task.BINARY).split(frame)
    assert list(result.dev_features.columns) == [
        c for c in frame.columns if c != "y"
    ]


def test_the_exploration_always_runs_the_second_stage():
    features, target = world(SEED)
    stage = stage_for(features, target, requested=False)
    assert stage.found is None
    governed.scout(
        charter=make_charter(),
        features=features,
        target=target,
        classification=False,
        run_dir=None,
        second_stage=stage,
    )
    assert stage.found is not None


def test_a_second_stage_that_fails_stops_the_exploration():
    def fail(columns):
        raise ArithmeticError(f"no assessment for {columns}")

    features, target = world(SEED)
    stage = prefilter.Prefilter(fail, requested=True, rows=len(target), seed=1)
    with pytest.raises(ArithmeticError):
        governed.scout(
            charter=make_charter(),
            features=features,
            target=target,
            classification=False,
            run_dir=None,
            second_stage=stage,
        )
    assert stage.found is None
