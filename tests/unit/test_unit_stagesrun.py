"""The standard stage sequence, run once end to end on a small table.

The stages are mixins: the profile, final-evaluation and packaging stages
inherit the work of the modules below them. One short run exercises every
stage module; the tests then read what each stage left in the run
directory, and pin the composition the sequence relies on.
"""

import json

import numpy
import pandas
import pytest

import scmle.api.project
from scmle.pipeline.stages import admission
from scmle.pipeline.stages import assessment
from scmle.pipeline.stages import bracketing
from scmle.pipeline.stages import calibration
from scmle.pipeline.stages import cost
from scmle.pipeline.stages import defaults
from scmle.pipeline.stages import diagnosis
from scmle.pipeline.stages import documentation
from scmle.pipeline.stages import examination
from scmle.pipeline.stages import exploration
from scmle.pipeline.stages import identification
from scmle.pipeline.stages import impounding
from scmle.pipeline.stages import labelling
from scmle.pipeline.stages import packaging
from scmle.pipeline.stages import planning
from scmle.pipeline.stages import profiling
from scmle.pipeline.stages import qualification
from scmle.pipeline.stages import relabelling
from scmle.pipeline.stages import releasing
from scmle.pipeline.stages import remedy
from scmle.pipeline.stages import reporting
from scmle.pipeline.stages import reservation
from scmle.pipeline.stages import scoring
from scmle.pipeline.stages import screening
from scmle.pipeline.stages import selecting
from scmle.pipeline.stages import summary
from scmle.pipeline.stages import surveying
from scmle.pipeline.stages import tracing


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    rng = numpy.random.default_rng(0)
    frame = pandas.DataFrame(rng.normal(size=(240, 3)), columns=["a", "b", "c"])
    frame["y"] = (frame.a + rng.normal(scale=0.5, size=240) > 0).astype(int)
    root = tmp_path_factory.mktemp("stages")
    result = scmle.api.project.Project(
        frame, "y", time_budget=30, output_dir=root
    ).run()
    return result, frame


def read(result, relative):
    return json.loads((result.run_dir / relative).read_text("utf-8"))


def kinds(result):
    lines = (result.run_dir / "events.jsonl").read_text("utf-8").splitlines()
    return [json.loads(line)["kind"] for line in lines]


def test_the_default_sequence_is_in_lifecycle_order():
    assert [type(s).__name__ for s in defaults.default_stages()] == [
        "IngestStage",
        "Reservation",
        "ProfileStage",
        "Planning",
        "TrainStage",
        "SelectStage",
        "FinalEvaluationStage",
        "PackageStage",
    ]


def test_the_profile_stage_inherits_the_charter_work_in_order():
    assert profiling.ProfileStage.__mro__[1:10] == (
        impounding.Impounding,
        exploration.Exploration,
        admission.Admission,
        identification.Identification,
        qualification.Qualification,
        screening.Screening,
        surveying.Surveying,
        labelling.Labelling,
        profiling.ProfileStage.__mro__[9],
    )


def test_the_final_evaluation_inherits_scoring_and_remedies_in_order():
    assert assessment.FinalEvaluationStage.__mro__[1:6] == (
        summary.Summary,
        reporting.Reporting,
        relabelling.Relabelling,
        remedy.Remedy,
        examination.Examination,
    )


def test_packaging_inherits_the_diagnostics_and_release_in_order():
    assert packaging.PackageStage.__mro__[1:7] == (
        tracing.Tracing,
        bracketing.Bracketing,
        diagnosis.Diagnosis,
        calibration.Calibration,
        documentation.Documentation,
        releasing.Releasing,
    )


def test_each_stage_names_what_it_needs_and_makes():
    assert reservation.Reservation.makes is reservation.Allowance
    assert selecting.SelectStage.makes is selecting.Choice
    assert planning.Planning.makes is planning.Blueprint
    assert profiling.ProfileStage.makes is profiling.Partition
    assert assessment.FinalEvaluationStage.makes is assessment.Exam
    assert packaging.PackageStage.makes is packaging.Package


def test_every_stage_left_a_record_in_order(run):
    result = run[0]
    records = sorted((result.run_dir / "stages").glob("*.json"))
    assert [p.name for p in records] == [
        "00-ingeststage.json",
        "01-reservation.json",
        "02-profilestage.json",
        "03-planning.json",
        "04-trainstage.json",
        "05-selectstage.json",
        "06-finalevaluationstage.json",
        "07-packagestage.json",
    ]
    loaded = [json.loads(p.read_text("utf-8")) for p in records]
    assert [r["index"] for r in loaded] == list(range(8))
    assert [r["state"] for r in loaded] == [
        "VALIDATING",
        "VALIDATING",
        "PROFILING",
        "PLANNING",
        "TRAINING",
        "SELECTING",
        "TESTING",
        "PACKAGING",
    ]
    assert loaded[1]["needs"] == {"IngestStage": loaded[0]["product"]}
    assert set(loaded[7]["needs"]) == {
        "IngestStage",
        "Reservation",
        "ProfileStage",
        "Planning",
        "TrainStage",
        "SelectStage",
        "FinalEvaluationStage",
    }


def test_the_states_are_announced_in_order(run):
    result = run[0]
    states = [k for k in kinds(result) if k.startswith("state.")]
    assert states == [
        "state.CREATED",
        "state.VALIDATING",
        "state.PROFILING",
        "state.PLANNING",
        "state.TRAINING",
        "state.SELECTING",
        "state.TESTING",
        "state.PACKAGING",
        "state.COMPLETED",
    ]


def test_the_reserve_comes_before_profiling_and_selection_before_test(run):
    result = run[0]
    found = kinds(result)
    assert found.index("budget.reserved") < found.index("state.PROFILING")
    assert found.index("split.created") < found.index("state.PLANNING")
    assert found.index("selection.frozen") < found.index("test.opened")
    assert found.count("test.opened") == 1
    assert found.index("test.opened") < found.index("state.PACKAGING")


def test_the_reserve_event_reports_the_seconds_held(run):
    result = run[0]
    lines = (result.run_dir / "events.jsonl").read_text("utf-8").splitlines()
    reserved = next(
        json.loads(line)
        for line in lines
        if json.loads(line)["kind"] == "budget.reserved"
    )
    assert reserved["reason"].endswith(
        " s budget held for the final refit, diagnostics and packaging"
    )
    assert "of the 30 s budget" in reserved["reason"]
    assert set(reserved["data"]["reserve_seconds"]) >= {"package"}


def test_each_trial_is_written_once(run):
    result = run[0]
    written = sorted(p.stem for p in (result.run_dir / "trials").glob("*.json"))
    assert written == sorted(t.trial_id for t in result.trials)
    assert len(set(written)) == len(written)


def test_the_selected_trial_is_evaluated_on_the_test_rows(run):
    result = run[0]
    evaluation = read(result, "evaluation.json")
    chosen = evaluation["selected"]
    assert chosen["model"] == result.selected_model
    assert chosen["trial"] in {t.trial_id for t in result.trials}
    assert set(evaluation["metrics"]) == {
        "roc_auc",
        "pr_auc",
        "log_loss",
        "f1",
        "balanced_accuracy",
        "accuracy",
    }
    assert 0.5 < evaluation["metrics"]["roc_auc"] <= 1.0
    assert evaluation["task"] == "binary_classification"


def test_the_package_serves_the_selected_model(run):
    result, frame = run
    rows = frame.drop(columns="y").head(4)
    predictions = result.predict(rows)
    assert len(predictions) == 4
    assert set(numpy.asarray(predictions).tolist()) <= {0, 1}
    probabilities = numpy.asarray(result.predict_proba(rows))
    numpy.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert (result.run_dir / "model" / "model_manifest.json").exists()
    assert (
        (result.run_dir / "report.html")
        .read_text("utf-8")
        .startswith("<!doctype html>")
    )


def test_the_run_completes_with_a_record_of_the_whole(run):
    result = run[0]
    complete = read(result, "complete.json")
    assert complete["schema_version"] == 1
    assert complete["elapsed"] > 0
    assert str(result.state).rsplit(".", 1)[-1].upper() == "COMPLETED"


def test_the_cost_helpers_size_this_run(run):
    result = run[0]
    chosen = next(
        t
        for t in result.trials
        if t.trial_id == read(result, "evaluation.json")["selected"]["trial"]
    )
    assert cost.foldcost(chosen) == pytest.approx(
        cost.FIT_MARGIN * chosen.fit_seconds / max(len(chosen.fold_scores), 1)
    )
    assert len(scoring.pin(chosen, None)) == 64
