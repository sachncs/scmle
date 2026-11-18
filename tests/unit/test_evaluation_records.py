"""Renamed evaluation and model records load their old versions unchanged.

Each fixture in `tests/fixtures/records/v1/evaluation` was written by the
code before the R6 renames: the conformal scores, the three estimation
envelopes and the modality checkpoint and curation at version 1, and the
modality report at version 2 (its previous layout). Loading one migrates it
to the current field names; the identity digests of the estimations are
anchored at version 1 and the HMAC signatures cover the layout a record was
stored in, so every pinned digest and signature still verifies.
"""

import dataclasses
import json
import pathlib

import hypothesis
from hypothesis import strategies
import pytest

from scmle.contracts import errors
from scmle.contracts import records
from scmle.contracts import signing
from scmle.evaluation import conformal
from scmle.evaluation import envelope
from scmle.evaluation import longitudinal
from scmle.evaluation import parameter_estimation
from scmle.evaluation import sequential
from scmle.pipeline import modality_lifecycle
from scmle.pipeline.modality_lifecycle import checkpoints
from scmle.pipeline.modality_lifecycle import curators
from scmle.pipeline.modality_lifecycle import reporting

OLD = pathlib.Path(__file__).parents[1] / "fixtures" / "records" / "v1"
OLD = OLD / "evaluation"
KEY = "fixture-key"
ESTIMATIONS = {
    "parameter": parameter_estimation.Estimation,
    "sequential": sequential.Estimation,
    "longitudinal": longitudinal.Estimation,
}


def stored(name):
    """The parsed old-version fixture."""
    return json.loads((OLD / f"{name}.json").read_text("utf-8"))


def test_old_conformal_scores_load_under_the_new_name():
    old = stored("conformal")
    calibrator = conformal.ConformalCalibrator.from_dict(old)
    assert calibrator.conditional == tuple(
        tuple(scores) for scores in old["class_scores"]
    )
    current = calibrator.to_dict()
    assert current[records.KEY] == 2
    assert "class_scores" not in current
    del current[records.KEY]
    restored = conformal.ConformalCalibrator.migrations[0].downgrade(current)
    assert restored == old


def test_a_conformal_record_newer_than_the_build_is_refused():
    newer = {**stored("conformal"), records.KEY: 3}
    with pytest.raises(errors.InputError, match="schema version"):
        conformal.ConformalCalibrator.from_dict(newer)


@hypothesis.given(
    scores=strategies.lists(
        strategies.floats(0.0, 1.0), min_size=20, max_size=40
    )
)
def test_a_conformal_record_survives_a_round_trip(scores):
    calibrator = conformal.ConformalCalibrator(
        "classification", tuple(scores), (0, 1), (tuple(scores), ())
    )
    layout = calibrator.to_dict()
    assert conformal.ConformalCalibrator.from_dict(layout) == calibrator
    del layout[records.KEY]
    step = conformal.ConformalCalibrator.migrations[0]
    assert step.upgrade(step.downgrade(layout)) == layout


@pytest.mark.parametrize("name", sorted(ESTIMATIONS))
def test_an_old_estimation_verifies_and_keeps_its_digest(name, tmp_path):
    kind = ESTIMATIONS[name]
    old = stored(name)
    assert old[records.KEY] == 1
    result = kind.read(OLD / f"{name}.json")
    assert result.digest() == old["digest"]
    layout = json.loads(json.dumps(result.to_dict()))
    assert records.downgrade(kind.migrations, layout) == old["report"]
    path = tmp_path / f"{name}.json"
    assert result.write(path) == old["digest"]
    current = json.loads(path.read_text("utf-8"))
    assert current[records.KEY] == envelope.VERSION
    assert current["digest"] == old["digest"]
    assert current["report"]["rows"] == dataclasses.asdict(result.rows)
    assert current["report"]["digests"] == {
        "frame": old["report"]["raw_input_fingerprint"]
    }
    assert kind.read(path) == result


@pytest.mark.parametrize("name", sorted(ESTIMATIONS))
def test_a_tampered_old_estimation_is_refused(name, tmp_path):
    old = stored(name)
    old["report"]["rows_used"] += 1
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        ESTIMATIONS[name].read(path)


def test_the_sequential_contrast_and_subjects_are_nested():
    old = stored("sequential")["report"]
    result = sequential.Estimation.from_dict(
        records.upgrade(sequential.Estimation.migrations, old, records.ORIGIN)
    )
    assert result.contrast.estimate == old["contrast_estimate"]
    assert result.contrast.se == old["contrast_standard_error"]
    assert list(result.contrast.interval) == old["contrast_interval"]
    assert dataclasses.asdict(result.subjects) == {
        "received": old["subjects_input"],
        "used": old["subjects_used"],
        "dropped": old["subjects_dropped"],
    }


@pytest.mark.parametrize("name", sorted(ESTIMATIONS))
@hypothesis.settings(max_examples=25)
@hypothesis.given(dropped=strategies.integers(0, 10_000))
def test_an_estimation_survives_a_round_trip(name, dropped):
    kind = ESTIMATIONS[name]
    base = kind.from_dict(
        records.upgrade(kind.migrations, stored(name)["report"], records.ORIGIN)
    )
    rows = envelope.Counts(base.rows.used + dropped, base.rows.used, dropped)
    result = dataclasses.replace(base, rows=rows)
    again = kind.from_dict(json.loads(json.dumps(result.to_dict())))
    assert again == result
    assert again.digest() == result.digest()
    layout = result.to_dict()
    older = records.downgrade(kind.migrations, layout)
    assert records.upgrade(kind.migrations, older, records.ORIGIN) == layout


def test_unfold_refuses_unknown_nested_keys():
    with pytest.raises(errors.InputError, match="unknown keys"):
        envelope.unfold(
            {"rows": {"spare": 1}}, {"rows": {"rows_input": "received"}}
        )


def test_an_old_checkpoint_still_verifies_and_is_not_resigned(tmp_path):
    old = stored("checkpoint")
    path = OLD / "checkpoint.json"
    checkpoint = modality_lifecycle.Checkpoint.read_verified(path, KEY)
    assert checkpoint.schema_version == 1
    assert checkpoint.digests.stages == tuple(old["stage_digests"])
    assert checkpoint.independent is old["require_independent_ground_truth"]
    assert checkpoint.to_dict() == old
    tampered = {**old, "seed": old["seed"] + 1}
    copy = tmp_path / "checkpoint.json"
    copy.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="signature does not verify"):
        modality_lifecycle.Checkpoint.read_verified(copy, KEY)


def test_a_new_checkpoint_is_signed_in_the_current_layout():
    old = modality_lifecycle.Checkpoint.from_dict(stored("checkpoint"))
    current = dataclasses.replace(old, schema_version=checkpoints.VERSION).sign(
        KEY
    )
    record = current.to_dict()
    assert record[records.KEY] == 2
    assert record["digests"]["dataset"] == old.digests.dataset
    assert "dataset_hash" not in record
    assert modality_lifecycle.Checkpoint.from_dict(record) == current
    assert current.verify_signature(KEY)
    assert current.signature != old.signature


@hypothesis.settings(max_examples=25)
@hypothesis.given(
    seed=strategies.integers(0, 2**31),
    independent=strategies.booleans(),
    version=strategies.sampled_from((1, 2)),
)
def test_a_checkpoint_survives_a_round_trip(seed, independent, version):
    base = modality_lifecycle.Checkpoint.from_dict(stored("checkpoint"))
    checkpoint = dataclasses.replace(
        base, seed=seed, independent=independent, schema_version=version
    ).sign(KEY)
    record = json.loads(json.dumps(checkpoint.to_dict()))
    again = modality_lifecycle.Checkpoint.from_dict(record)
    assert again == checkpoint
    assert again.verify_signature(KEY)


def test_an_old_curation_still_verifies_and_keeps_its_content_digest():
    old = stored("curation")
    curation = modality_lifecycle.Curation.read_verified(
        OLD / "curation.json", KEY
    )
    assert curation.schema_version == 1
    assert curation.digests == {"dataset": old["dataset_hash"]}
    assert curation.samples == old["sample_count"]
    assert curation.card == old["dataset_card"]
    assert curation.to_dict() == old
    assert curation.content_digest == old["content_digest"]
    current = dataclasses.replace(
        curation, schema_version=curators.VERSIONS[-1]
    )
    assert current.content_digest == old["content_digest"]
    assert "samples" in current.to_dict()


@hypothesis.settings(max_examples=25)
@hypothesis.given(
    samples=strategies.integers(0, 10_000),
    version=strategies.sampled_from(curators.VERSIONS),
)
def test_a_curation_survives_a_round_trip(samples, version):
    base = modality_lifecycle.Curation.from_dict(stored("curation"))
    draft = dataclasses.replace(
        base,
        samples=samples,
        curation={**base.curation, "sample_count": samples},
        card={**base.card, "sample_count": samples},
        schema_version=version,
    )
    curation = dataclasses.replace(
        draft, signature=signing.mac(draft.preimage(), KEY)
    )
    record = json.loads(json.dumps(curation.to_dict()))
    again = modality_lifecycle.Curation.from_dict(record)
    assert again == curation
    assert again.verify_signature(KEY)


def test_a_previous_version_report_still_verifies():
    old = stored("report")
    assert old[records.KEY] == reporting.PREVIOUS
    report = modality_lifecycle.Record.read_verified(OLD / "report.json", KEY)
    assert report.digests == {"dataset": old["dataset_hash"]}
    assert len(report.evidence) == len(old["stage_evidence"])
    assert report.to_dict() == old


@hypothesis.settings(max_examples=25)
@hypothesis.given(
    accuracy=strategies.floats(0.0, 1.0),
    version=strategies.sampled_from((2, 3)),
)
def test_a_report_survives_a_round_trip(accuracy, version):
    base = modality_lifecycle.Record.from_dict(stored("report"))
    report = dataclasses.replace(
        base, evaluation={"accuracy": accuracy}, schema_version=version
    ).sign(KEY)
    record = json.loads(json.dumps(report.to_dict()))
    again = modality_lifecycle.Record.from_dict(record)
    assert again == report
    assert again.verify_signature(KEY)
    assert ("evidence" in record) is (version == 3)
