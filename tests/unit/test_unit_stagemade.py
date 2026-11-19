"""The first stages return frozen products; the context is filled from them.

Each stage runs on fake services (an in-memory frame, a fixed clock, no
run directory), returns its product, and the product's `apply` sets the
context fields the later, unconverted stages still read.
"""

import dataclasses
import datetime
import hashlib
import json
import types

import numpy
import pandas
import pytest

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.evaluation import metrics
from scmle.evaluation import selection
import scmle.models.registry
from scmle.pipeline import products
from scmle.pipeline import services
import scmle.pipeline.context
from scmle.pipeline.stages import admission
from scmle.pipeline.stages import ingestion
from scmle.pipeline.stages import labelling
from scmle.pipeline.stages import profiling
from scmle.pipeline.stages import reservation
from scmle.pipeline.stages import screening
from scmle.pipeline.stages import surveying
from scmle.runtime import limits

ROWS = 240
LEVELS = ("low", "middle", "high")


class Fixed(services.Clock):
    """A clock that advances `step` seconds on every read."""

    def __init__(self, step):
        self.step = step
        self.reads = 0

    def monotonic(self):
        self.reads += 1
        return self.reads * self.step

    def now(self):
        return datetime.datetime(2026, 1, 2, tzinfo=datetime.UTC)


def rows(missing=0):
    rng = numpy.random.default_rng(3)
    frame = pandas.DataFrame(
        {
            "a": rng.normal(size=ROWS),
            "b": rng.normal(size=ROWS),
            "y": rng.integers(0, 2, size=ROWS).astype(float),
        }
    )
    frame.loc[0, "b"] = numpy.inf
    if missing:
        frame.loc[: missing - 1, "y"] = numpy.nan
    return frame


def config(**data):
    return tasks.Config(
        data=tasks.Dataset(uri="memory.csv", target="y", **data),
        resources=tasks.Resources(time_budget_seconds=600),
    )


def board(frame=None, spec=None):
    spec = spec or config()
    given = services.Services(
        spec,
        scmle.models.registry.Catalog.stock(),
        metrics.CATALOG,
        selection.Averager(spec.selection),
        limits.Governor(spec.resources, 0.0),
        input_frame=rows() if frame is None else frame,
        clock=Fixed(0.5),
    )
    return scmle.pipeline.context.Context.start(given)


def digest(summary):
    text = json.dumps(
        products.untimed(summary),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_ingestion_returns_the_intake_and_applies_it():
    context = board()
    made = ingestion.IngestStage().execute(context)
    assert isinstance(made, ingestion.Intake)
    with pytest.raises(dataclasses.FrozenInstanceError):
        made.read_seconds = 0.0
    assert made.read_seconds == 0.5
    assert made.stream is None
    assert made.cleaning_notes == [
        "column 'b': 1 infinite value(s) treated as missing"
    ]
    assert context.frame is made.frame
    assert context.manifest is made.manifest
    assert context.cleaning_notes == made.cleaning_notes
    assert context.read_seconds == made.read_seconds


def test_the_intake_digest_is_that_of_its_old_summary():
    made = ingestion.IngestStage().execute(board())
    assert made.save(None) == digest(
        {
            "digests": dict(made.manifest.digests),
            "notes": list(made.cleaning_notes),
            "streamed": False,
        }
    )


def test_a_file_backed_modality_without_an_adapter_is_refused():
    context = board(spec=config(modality="image"))
    context = dataclasses.replace(
        context,
        services=dataclasses.replace(context.services, input_frame=None),
    )
    with pytest.raises(errors.ScmleError, match="ingestion adapter"):
        ingestion.IngestStage().execute(context)


def test_the_reservation_returns_the_reserve_with_its_old_digest():
    context = board()
    ingestion.IngestStage().execute(context)
    made = reservation.Reservation().execute(context)
    assert reservation.Reservation.makes is reservation.Reserve
    assert isinstance(made, reservation.Reserve)
    assert made.governor is context.services.governor
    reserve = context.services.governor.reserve
    assert reserve.shape is not None
    assert made.save(None) == digest(
        {"shape": reserve.shape, "parts": reserve.parts}
    )


def test_labelling_resolves_without_touching_the_intake_notes():
    frame = rows(missing=4)
    context = board(frame, config(unlabeled="ignore"))
    intake = ingestion.IngestStage().execute(context)
    before = list(intake.cleaning_notes)
    stage = profiling.ProfileStage()
    made = stage.infer(context, context.frame)
    assert isinstance(made, labelling.Resolution)
    assert made.task is tasks.Task.BINARY
    assert made.metric is not None
    assert len(made.frame) == ROWS - 4
    assert made.unlabeled is not None
    assert len(made.unlabeled) == 4
    assert "y" not in made.unlabeled
    assert made.notes == [
        *before,
        "4 row(s) without a target were set aside (policy 'ignore')",
    ]
    assert intake.cleaning_notes == before
    assert made.labels is None
    made.apply(context)
    assert context.cleaning_notes == made.notes
    assert context.unlabeled is made.unlabeled


def test_the_unlabeled_gate_blocks_by_default():
    context = board(rows(missing=2))
    ingestion.IngestStage().execute(context)
    with pytest.raises(errors.BlockedError, match="2 missing"):
        profiling.ProfileStage().infer(context, context.frame)


def test_an_ordinal_target_is_resolved_to_ranks():
    frame = rows()
    frame["y"] = pandas.cut(
        frame["a"], [-numpy.inf, -0.5, 0.5, numpy.inf], labels=LEVELS
    ).astype(str)
    spec = tasks.Config(
        data=tasks.Dataset(
            uri="memory.csv", target="y", ordinal_order=list(LEVELS)
        ),
        task="ordinal",
        resources=tasks.Resources(time_budget_seconds=600),
    )
    context = board(frame, spec)
    ingestion.IngestStage().execute(context)
    made = profiling.ProfileStage().infer(context, context.frame)
    assert made.task is tasks.Task.ORDINAL
    assert made.labels == LEVELS
    made.apply(context)
    assert context.ordinal_labels == LEVELS
    assert set(made.frame["y"]) == {0, 1, 2}


def test_screening_without_a_plan_keeps_the_frame():
    context = board()
    ingestion.IngestStage().execute(context)
    made = profiling.ProfileStage().synthesize(context, context.frame)
    assert isinstance(made, screening.Screen)
    assert made.frame is context.frame
    assert made.ingest is None


def test_the_profile_stage_returns_the_partition_and_its_parts():
    context = board()
    ingestion.IngestStage().execute(context)
    made = profiling.ProfileStage().execute(context)
    try:
        assert isinstance(made, profiling.Partition)
        assert made.task is tasks.Task.BINARY
        assert context.split is made.split
        assert context.profile is made.profile
        assert context.curation is made.curation
        assert context.frame is made.frame
        assert made.propensity is None
        assert made.seasonal is None
        assert context.seasonal_baseline is None
        assert made.save(None) == digest(
            {
                "task": made.task,
                "metric": made.metric.name,
                "split": made.split.manifest.fingerprint,
                "profile": made.profile.to_dict(),
            }
        )
    finally:
        context.close()


def test_surveying_curates_into_a_portrait():
    context = board()
    ingestion.IngestStage().execute(context)
    stage = profiling.ProfileStage()
    made = stage.execute(context)
    try:
        development = made.split.dev_features.assign(y=made.split.dev_target)
        portrait = stage.curate(
            context, context.frame, development, made.profile
        )
        assert isinstance(portrait, surveying.Portrait)
        assert portrait.profile is made.profile
        assert portrait.curation.profiled_rows == len(development)
    finally:
        context.close()


def test_a_failure_after_the_split_closes_the_locked_rows(monkeypatch):
    closed = []
    context = board()
    ingestion.IngestStage().execute(context)
    stage = profiling.ProfileStage()
    divide = stage.divide

    def shut(box):
        closed.append(box)

    def spy(*arguments):
        made = divide(*arguments)
        monkeypatch.setattr(type(made[0].locked_test), "close", shut)
        return made

    def refuse(*arguments):
        raise errors.BlockedError("refused")

    monkeypatch.setattr(stage, "divide", spy)
    monkeypatch.setattr(stage, "enforce_leakage_policy", refuse)
    with pytest.raises(errors.BlockedError, match="refused"):
        stage.execute(context)
    assert len(closed) == 1
    assert context.split is None


def test_a_bridge_reaches_the_renamed_context_attributes():
    made = labelling.Resolution(
        tasks.Task.BINARY,
        metrics.CATALOG.resolve(None, tasks.Task.BINARY),
        rows(),
        None,
        ("a", "b", "c"),
        ["note"],
    )
    target = types.SimpleNamespace()
    made.apply(target)
    assert target.ordinal_labels == ("a", "b", "c")
    assert target.cleaning_notes == ["note"]
    assert not hasattr(target, "labels")
    again = labelling.Resolution.capture(target)
    assert again.frame is made.frame
    assert (again.labels, again.notes) == (made.labels, made.notes)


def test_amend_applies_the_amended_portrait():
    context = board()
    ingestion.IngestStage().execute(context)
    made = profiling.ProfileStage().execute(context)
    try:
        plan = dataclasses.replace(
            made.profile.feature_plan, numeric=("a",), dropped={"b": "gone"}
        )
        written = admission.amend(context, plan)
        assert written == ["curation_report.json", "dataset_card.json"]
        assert context.profile.feature_plan is plan
        assert context.curation.dropped_columns == {"b": "gone"}
        assert made.profile.feature_plan is not plan
    finally:
        context.close()
