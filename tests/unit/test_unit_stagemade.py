"""The first stages return frozen products; the board holds them by type.

Each stage runs on fake services (an in-memory frame, a fixed clock, no
run directory), returns its product, and the product's `apply` holds it on
the run's board, where the later stages read it (`engine.Board.get`).
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
from scmle.pipeline import engine
from scmle.pipeline import products
from scmle.pipeline import services
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


def start(frame=None, spec=None):
    spec = spec or config()
    given = services.Services(
        spec,
        scmle.models.registry.Catalog.stock(),
        metrics.CATALOG,
        selection.Averager(spec.selection),
        limits.Governor(spec.resources, 0.0),
        frame=rows() if frame is None else frame,
        clock=Fixed(0.5),
    )
    return engine.Board.start(given)


def digest(summary):
    text = json.dumps(
        products.untimed(summary),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_ingestion_returns_the_intake_and_applies_it():
    board = start()
    made = ingestion.Ingest().execute(board)
    assert isinstance(made, ingestion.Intake)
    with pytest.raises(dataclasses.FrozenInstanceError):
        made.read = 0.0
    assert made.read == 0.5
    assert made.stream is None
    assert made.notes == ["column 'b': 1 infinite value(s) treated as missing"]
    assert board.get(ingestion.Intake) is made


def test_the_intake_digest_is_that_of_its_old_summary():
    made = ingestion.Ingest().execute(start())
    assert made.save(None) == digest(
        {
            "digests": dict(made.manifest.digests),
            "notes": list(made.notes),
            "streamed": False,
        }
    )


def test_a_file_backed_modality_without_an_adapter_is_refused():
    board = start(spec=config(modality="text"))
    board = dataclasses.replace(
        board,
        services=dataclasses.replace(board.services, frame=None),
    )
    with pytest.raises(errors.ScmleError, match="ingestion adapter"):
        ingestion.Ingest().execute(board)


def test_the_reservation_returns_the_reserve_with_its_old_digest():
    board = start()
    ingestion.Ingest().execute(board)
    made = reservation.Reservation().execute(board)
    assert reservation.Reservation.makes is reservation.Reserve
    assert isinstance(made, reservation.Reserve)
    assert made.governor is board.services.governor
    reserve = board.services.governor.reserve
    assert reserve.shape is not None
    assert made.save(None) == digest(
        {"shape": reserve.shape, "parts": reserve.parts}
    )


def test_labelling_resolves_without_touching_the_intake_notes():
    frame = rows(missing=4)
    board = start(frame, config(unlabeled="ignore"))
    intake = ingestion.Ingest().execute(board)
    before = list(intake.notes)
    stage = profiling.Profiling()
    made = stage.infer(board, intake.frame)
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
    assert intake.notes == before
    assert made.labels is None
    made.apply(board)
    assert board.get(labelling.Resolution) is made


def test_the_unlabeled_gate_blocks_by_default():
    board = start(rows(missing=2))
    intake = ingestion.Ingest().execute(board)
    with pytest.raises(errors.BlockedError, match="2 missing"):
        profiling.Profiling().infer(board, intake.frame)


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
    board = start(frame, spec)
    intake = ingestion.Ingest().execute(board)
    made = profiling.Profiling().infer(board, intake.frame)
    assert made.task is tasks.Task.ORDINAL
    assert made.labels == LEVELS
    assert set(made.frame["y"]) == {0, 1, 2}


def test_screening_without_a_plan_keeps_the_frame():
    board = start()
    intake = ingestion.Ingest().execute(board)
    made = profiling.Profiling().synthesize(board, intake.frame)
    assert isinstance(made, screening.Screen)
    assert made.frame is intake.frame
    assert made.ingest is None


def test_the_profile_stage_returns_the_partition_and_its_parts():
    board = start()
    ingestion.Ingest().execute(board)
    made = profiling.Profiling().execute(board)
    try:
        assert isinstance(made, profiling.Partition)
        assert made.task is tasks.Task.BINARY
        assert board.get(profiling.Partition) is made
        assert board.get(surveying.Portrait).profile is made.profile
        assert board.get(surveying.Portrait).curation is made.curation
        assert board.get(labelling.Resolution).task is made.task
        assert made.propensity is None
        assert made.seasonal is None
        assert made.save(None) == digest(
            {
                "task": made.task,
                "metric": made.metric.name,
                "split": made.split.manifest.fingerprint,
                "profile": made.profile.to_dict(),
            }
        )
    finally:
        board.close()


def test_surveying_curates_into_a_portrait():
    board = start()
    ingestion.Ingest().execute(board)
    stage = profiling.Profiling()
    made = stage.execute(board)
    try:
        development = made.split.features.assign(y=made.split.target)
        portrait = stage.curate(board, made.frame, development, made.profile)
        assert isinstance(portrait, surveying.Portrait)
        assert portrait.profile is made.profile
        assert portrait.curation.profiled_rows == len(development)
    finally:
        board.close()


def test_a_failure_after_the_split_closes_the_locked_rows(monkeypatch):
    closed = []
    board = start()
    ingestion.Ingest().execute(board)
    stage = profiling.Profiling()
    divide = stage.divide
    originals = []

    def shut(box):
        closed.append(box)
        originals[0](box)

    def spy(*arguments):
        made = divide(*arguments)
        originals.append(type(made[0].test).close)
        monkeypatch.setattr(type(made[0].test), "close", shut)
        return made

    def refuse(*arguments):
        raise errors.BlockedError("refused")

    monkeypatch.setattr(stage, "divide", spy)
    monkeypatch.setattr(stage, "enforce_leakage_policy", refuse)
    with pytest.raises(errors.BlockedError, match="refused"):
        stage.execute(board)
    assert len(closed) == 1
    assert board.find(profiling.Partition) is None


def test_closing_the_board_closes_the_partitions_locked_rows(monkeypatch):
    closed = []
    board = start()
    ingestion.Ingest().execute(board)
    made = profiling.Profiling().execute(board)
    original = type(made.split.test).close

    def shut(box):
        closed.append(box)
        original(box)

    monkeypatch.setattr(type(made.split.test), "close", shut)
    board.close()
    assert closed == [made.split.test]


def test_a_resolution_is_held_by_type_with_its_own_field_names():
    made = labelling.Resolution(
        tasks.Task.BINARY,
        metrics.CATALOG.resolve(None, tasks.Task.BINARY),
        rows(),
        None,
        ("a", "b", "c"),
        ["note"],
    )
    target = types.SimpleNamespace(made={})
    made.apply(target)
    assert target.made == {labelling.Resolution: made}
    assert not hasattr(target, "ordinal_labels")


def test_amend_applies_the_amended_portrait():
    board = start()
    ingestion.Ingest().execute(board)
    made = profiling.Profiling().execute(board)
    try:
        plan = dataclasses.replace(
            made.profile.feature_plan, numeric=("a",), dropped={"b": "gone"}
        )
        written = admission.amend(board, plan)
        assert written == ["curation_report.json", "dataset_card.json"]
        portrait = board.get(surveying.Portrait)
        assert portrait.profile.feature_plan is plan
        assert portrait.curation.dropped_columns == {"b": "gone"}
        assert made.profile.feature_plan is not plan
    finally:
        board.close()
