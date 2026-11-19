"""The modality lifecycle: plan checks, stages, explanation and the predictor.

One text lifecycle runs once for the module; the tests read the stage
evidence of its signed report, load its package through the predictor, and
drive the explanation and planning helpers directly.
"""

import json

import numpy
import pytest
from sklearn.linear_model import LogisticRegression

from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.contracts import performance
import scmle.data.modalities.document
from scmle.pipeline.modality_lifecycle import containers
from scmle.pipeline.modality_lifecycle import explanation
from scmle.pipeline.modality_lifecycle import lifecycle
from scmle.pipeline.modality_lifecycle import plans
from scmle.pipeline.modality_lifecycle import predictors
from scmle.pipeline.modality_lifecycle import state
from scmle.pipeline.modality_lifecycle import steps
from scmle.pipeline.modality_lifecycle.preparers import document

KEY = "lifecycle-key"


def adapter():
    samples = tuple(
        scmle.data.modalities.document.Sample(
            identifier=f"text-{index}",
            content=("alpha signal " if index % 2 else "beta signal ")
            * (index + 2),
            label="alpha" if index % 2 else "beta",
        )
        for index in range(12)
    )
    return document.Plaintext(scmle.data.modalities.document.Dataset(samples))


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    root = tmp_path_factory.mktemp("lifecycle")
    source = adapter()
    report = lifecycle.Lifecycle().run(
        source,
        root / "package",
        signing_key=KEY,
        seed=17,
        report_path=root / "report.json",
    )
    return report, source, root


def evidence(report):
    return {item.stage: item.evidence for item in report.evidence}


def test_every_stage_is_evidenced_in_the_required_order(ran):
    report, source = ran[0], ran[1]
    assert tuple(item.stage for item in report.evidence) == (
        modalities.REQUIRED_LIFECYCLE_STAGES
    )
    assert all(item.evidenced for item in report.evidence)
    found = evidence(report)
    assert found[modalities.Phase.INGESTION] == {
        "dataset_hash": source.digest,
        "sample_count": 12,
    }


def test_the_report_is_signed_and_written(ran):
    report, source, root = ran
    assert report.verify_signature(KEY) is True
    assert report.verify_signature("other") is False
    assert json.loads((root / "report.json").read_text("utf-8")) == (
        report.to_dict()
    )
    assert report.digests == {"dataset": source.digest}


def test_inference_replays_the_package_predictions(ran):
    report = ran[0]
    assert report.inference == {
        "package_verified": True,
        "prediction_count": report.evaluation["samples"],
        "predictions_match": True,
    }
    assert report.recovery["tampered_manifest_rejected"] is True


def test_the_performance_evidence_covers_every_operation(ran):
    report = ran[0]
    observations = report.performance["observations"]
    assert [o["operation"] for o in observations] == [
        operation.value for operation in performance.Operation
    ]
    assert all(o["samples"] > 0 for o in observations)
    assert report.performance["budget"] is None
    assert report.performance["comparison"] is None


def test_the_declaration_is_complete_but_experimental(ran):
    report = ran[0]
    declaration = lifecycle.Lifecycle().declaration(report)
    assert declaration.status is modalities.Status.EXPERIMENTAL
    assert declaration.complete


def test_the_model_explanation_passes_its_quality_checks(ran):
    report = ran[0]
    found = evidence(report)[modalities.Phase.MODEL_EXPLAINABILITY]
    assert found["quality"]["passed"] is True
    assert found["quality"]["fidelity"] == pytest.approx(1.0)
    repeated = found["repeated_perturbation"]
    assert repeated["evaluated_sample_count"] == min(
        repeated["sample_count"], 128
    )
    assert repeated["feature_space_only"] is True


def test_data_explanation_traces_every_sample(ran):
    report, source = ran[0], ran[1]
    found = evidence(report)[modalities.Phase.DATA_EXPLAINABILITY]
    lineage = found["sample_lineage"]
    assert {item["sample_id"] for item in lineage} == {
        source.sample_id(sample) for sample in source.samples
    }
    assert sum(found["partition_counts"].values()) == 12
    assert all(
        stats["nonfinite_count"] == 0
        for stats in found["feature_statistics"].values()
    )


def test_the_predictor_loads_the_package_and_scores_samples(ran):
    source, root = ran[1], ran[2]
    predictor = predictors.Predictor.from_package(
        source,
        root / "package",
        signing_key=KEY,
        lifecycle_report_path=root / "report.json",
    )
    predicted = predictor.predict(source.samples[:4])
    assert len(predicted) == 4
    assert set(map(str, predicted)) <= {"alpha", "beta"}
    probabilities = predictor.predict_proba(source.samples[:4])
    numpy.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    batch = predictor.features_for(source.samples[:2])
    assert batch.sample_ids == tuple(
        source.sample_id(sample) for sample in source.samples[:2]
    )
    health = predictor.health()
    assert health["package_verified"] is True
    assert health["modality"] == "text"
    assert health["training_dataset_hash"] == source.digest
    assert health["feature_names"] == list(predictor.feature_names)
    assert health["lifecycle_report_verified"] is True
    assert len(health["lifecycle_report_digest"]) == 64


def test_the_predictor_refuses_a_wrong_key(ran):
    source, root = ran[1], ran[2]
    with pytest.raises(errors.InputError, match="does not verify"):
        predictors.Predictor.from_package(
            source, root / "package", signing_key="wrong"
        )


def test_the_package_must_be_for_the_adapter_modality(ran):
    source, root = ran[1], ran[2]
    other = document.DocumentPreparer(source.dataset)
    with pytest.raises(errors.InputError, match="does not match adapter"):
        predictors.Predictor.from_package(other, root / "package", KEY)


def test_typecheck_rejects_options_of_the_wrong_type():
    plans.typecheck(True, None, False)
    plans.typecheck(False, lambda: False, True)
    with pytest.raises(errors.KindError, match="require_independent_ground"):
        plans.typecheck(1, None, False)
    with pytest.raises(errors.KindError, match="cancellation_check must be"):
        plans.typecheck(True, "not callable", False)
    with pytest.raises(errors.KindError, match="resume must be bool"):
        plans.typecheck(True, None, "yes")


def test_coherent_rejects_inconsistent_paths(tmp_path):
    plans.coherent(False, None, None)
    plans.coherent(True, tmp_path / "c", tmp_path / "r")
    with pytest.raises(errors.InputError, match="resume requires"):
        plans.coherent(True, None, tmp_path / "r")
    with pytest.raises(errors.InputError, match="requires report_path"):
        plans.coherent(False, tmp_path / "c", None)
    with pytest.raises(errors.InputError, match="must differ"):
        plans.coherent(False, tmp_path / "c", tmp_path / "c")


def test_exported_turns_a_record_into_its_mapping():
    class Record:
        def to_dict(self):
            return {"k": 1}

    assert plans.exported(None) is None
    assert plans.exported(Record()) == {"k": 1}


def test_prepare_fixes_the_inputs_and_what_existed(tmp_path):
    source = adapter()
    (tmp_path / "package").mkdir()
    plan = plans.prepare(
        source,
        tmp_path / "package",
        KEY,
        3,
        None,
        tmp_path / "report.json",
        None,
        False,
        None,
        None,
        False,
    )
    assert plan.observed is source
    assert plan.package_preexisting is True
    assert plan.report_preexisting is False
    assert plan.checkpoint_target is None
    assert plan.digest is None
    assert plan.seed == 3


def test_statistics_summarise_a_column():
    found = steps.statistics(numpy.array([1.0, 2.0, 3.0, 4.0, numpy.inf]))
    assert found["min"] == 1.0
    assert found["max"] == numpy.inf
    assert found["nonfinite_count"] == 1
    plain = steps.statistics(numpy.array([0.0, 10.0]))
    assert plain == {
        "min": 0.0,
        "max": 10.0,
        "mean": 5.0,
        "std": 5.0,
        "q05": 0.5,
        "median": 5.0,
        "q95": 9.5,
        "nonfinite_count": 0,
    }


def batch(values):
    rows = numpy.asarray(values, dtype=float)
    return containers.Batch(
        tuple(f"s{i}" for i in range(len(rows))),
        tuple(f"f{j}" for j in range(rows.shape[1])),
        rows,
    )


def linear():
    rows = numpy.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.9], [0.9, 0.1]])
    return LogisticRegression().fit(rows, [0, 1, 0, 1]), batch(rows)


def test_replicate_reproduces_the_decision_exactly():
    model, rows = linear()
    assert explanation.replicate(model, rows) == pytest.approx(1.0)


def test_ablate_counts_sign_consistent_zeroings():
    model, rows = linear()
    row = rows.values[1]
    decision = numpy.asarray(model.decision_function(row[None]))
    assert explanation.ablate(model, row, model.coef_[0], decision, 0) == 1.0
    zero = numpy.zeros(2)
    assert explanation.ablate(model, zero, model.coef_[0], decision, 0) == 0.0


def test_sensitivities_nudge_each_feature_both_ways():
    model, rows = linear()
    reports = explanation.sensitivities(
        rows, 0, model.coef_[0], numpy.array([0.1, 0.1])
    )
    assert len(reports) == 1 + 2 * 2


def test_perturb_evaluates_the_leading_rows():
    model, rows = linear()
    found = explanation.perturb(model, rows, numpy.array([1e-6, 1e-6]), 3)
    assert len(found.stabilities) == 3
    assert len(found.reports) == 3
    assert len(found.perturbation_reports) == 5
    numpy.testing.assert_array_equal(found.coefficients, model.coef_[0])
    assert found.perturbation_stability == found.stabilities[0]
    with pytest.raises(RuntimeError, match="produced no report"):
        explanation.perturb(model, rows, numpy.array([0.1, 0.1]), 0)


def test_each_stage_belongs_to_the_state_of_its_phase():
    assert state.STATES[modalities.Phase.INGESTION].name == "VALIDATING"
    assert state.STATES[modalities.Phase.PACKAGING].name == "PACKAGING"
    assert set(state.STATES) == set(modalities.REQUIRED_LIFECYCLE_STAGES)
    assert steps.Packaging.state is state.STATES[modalities.Phase.PACKAGING]
    assert (
        explanation.ModelExplainabilityStage.state
        is state.STATES[modalities.Phase.MODEL_EXPLAINABILITY]
    )


def test_a_board_hands_out_only_finished_products():
    board = state.Board(plan=None)
    with pytest.raises(errors.OrderError, match="Intake is not made yet"):
        board.get(state.Intake)
    made = state.Intake({"sample_count": 1})
    made.apply(board)
    assert board.get(state.Intake) is made
    assert made.summary() == {"sample_count": 1}
    with pytest.raises(errors.KindError, match="applies to a Board"):
        made.apply(object())
    assert board.replaying(modalities.Phase.INGESTION) is False
    resumed = state.Board(plan=None, resumed=("ingestion",))
    assert resumed.replaying(modalities.Phase.INGESTION) is (
        modalities.Phase.INGESTION.value == "ingestion"
    )
