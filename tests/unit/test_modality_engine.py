"""The modality lifecycle on the one pipeline engine.

The fifteen modality stages are engine nodes: the engine orders them, checks
cancellation before each, times them and hands each finished stage to the
signed checkpoint archive. Each stage returns a frozen product; the run's
board holds the fixed inputs and the products and has no attribute a stage
assigns. The checkpoints in `tests/fixtures/modality/` were signed by the
lifecycle before it ran on the engine (layouts 1 and 2, six stages done);
they still verify and resume, and the replayed stages reproduce their
digests byte for byte.
"""

import dataclasses
import json
from pathlib import Path
import shutil

import pytest

from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.contracts import performance
from scmle.data.modalities import document
from scmle.pipeline import modality_lifecycle
from scmle.pipeline import products
from scmle.pipeline.modality_lifecycle import checkpoints
from scmle.pipeline.modality_lifecycle import state
from scmle.runtime import engine
import scmle.runtime.state

FIXTURES = Path(__file__).parents[1] / "fixtures" / "modality"
KEY = "checkpoint-key"


def adapter() -> modality_lifecycle.Plaintext:
    """The labelled text data the fixture checkpoints were made from."""
    samples = tuple(
        document.Sample(
            identifier=f"text-{index}",
            content=("alpha signal " if index % 2 else "beta signal ")
            * (index + 2),
            label="alpha" if index % 2 else "beta",
        )
        for index in range(12)
    )
    return modality_lifecycle.Plaintext(document.Dataset(samples))


def test_the_engine_orders_the_stages_in_the_required_lifecycle_order():
    ordered = engine.order(modality_lifecycle.Lifecycle.stages)
    assert tuple(stage.stage for stage in ordered) == (
        modalities.REQUIRED_LIFECYCLE_STAGES
    )


def test_every_stage_is_an_engine_node_that_makes_a_frozen_product():
    for stage in modality_lifecycle.Lifecycle.stages:
        assert stage.makes is not None
        assert issubclass(stage.makes, products.Derived)
        assert dataclasses.is_dataclass(stage.makes)
        assert stage.makes.__dataclass_params__.frozen
        assert stage.describe(None)
        assert all(isinstance(need, type) for need in stage.requires)


def test_the_board_has_no_attribute_a_stage_can_assign(tmp_path: Path):
    plan = modality_lifecycle.plans.prepare(
        adapter(),
        tmp_path / "package",
        KEY,
        0,
        None,
        None,
        None,
        False,
        None,
        None,
        False,
    )
    board = state.Board(plan)
    with pytest.raises(dataclasses.FrozenInstanceError):
        board.plan = plan  # type: ignore[misc]
    with pytest.raises(errors.OrderError, match="Partition"):
        board.get(state.Partition)


def test_a_run_times_its_stages_through_the_engine(tmp_path: Path):
    report = modality_lifecycle.Lifecycle().run(
        adapter(), tmp_path / "package", signing_key=KEY
    )
    measured = {
        item["operation"]
        for item in report.performance["observations"]  # type: ignore[union-attr]
    }
    assert measured == {
        operation.value
        for operation in (
            performance.Operation.INGEST,
            performance.Operation.FIT,
            performance.Operation.PACKAGE,
            performance.Operation.LOAD,
            performance.Operation.INFERENCE,
        )
    }


def test_the_lifecycle_table_runs_its_states_forward():
    progress = scmle.runtime.state.Progress
    assert (
        scmle.runtime.state.chain(
            (
                progress.CREATED,
                progress.VALIDATING,
                progress.PROFILING,
                progress.TRAINING,
                progress.TESTING,
                progress.PACKAGING,
                progress.COMPLETED,
            ),
            scmle.runtime.state.ABORT_STATES,
        )
        == state.TABLE
    )


def test_a_run_moves_the_shared_machine_along_the_lifecycle_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    entered = []

    def enter(machine, progress, reason, artifacts):
        assert machine.table is state.TABLE
        entered.append(progress)

    monkeypatch.setattr(scmle.runtime.state.Machine, "enter", enter)
    modality_lifecycle.Lifecycle().run(
        adapter(), tmp_path / "package", signing_key=KEY
    )
    assert entered == [
        target for target in state.TABLE if target in set(state.STATES.values())
    ] + [scmle.runtime.state.Progress.COMPLETED]


@pytest.mark.parametrize("version", [1, 2])
def test_a_checkpoint_signed_before_the_engine_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: int
):
    monkeypatch.chdir(tmp_path)
    shutil.copy(FIXTURES / f"checkpoint_v{version}.json", "checkpoint.json")
    before = modality_lifecycle.Checkpoint.read_verified("checkpoint.json", KEY)
    assert before.schema_version == version

    report = modality_lifecycle.Lifecycle().run(
        adapter(),
        "package",
        signing_key=KEY,
        report_path="report.json",
        checkpoint_path="checkpoint.json",
        resume=True,
    )

    assert report.verify_signature(KEY)
    after = modality_lifecycle.Checkpoint.read_verified("checkpoint.json", KEY)
    assert after.status == "completed"
    assert after.schema_version == checkpoints.VERSION
    assert after.digests.stages[: len(before.completed)] == (
        before.digests.stages
    )
    assert len(after.completed) == len(modalities.REQUIRED_LIFECYCLE_STAGES)


def test_a_replay_that_differs_from_its_checkpoint_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    stored = json.loads(
        (FIXTURES / "checkpoint_v2.json").read_text(encoding="utf-8")
    )
    old = modality_lifecycle.Checkpoint.from_dict(stored)
    stages = ("0" * 64, *old.digests.stages[1:])
    edited = dataclasses.replace(
        old, digests=dataclasses.replace(old.digests, stages=stages)
    ).sign(KEY)
    Path("checkpoint.json").write_text(
        json.dumps(edited.to_dict()), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="replay diverged at stage ingestion"):
        modality_lifecycle.Lifecycle().run(
            adapter(),
            "package",
            signing_key=KEY,
            report_path="report.json",
            checkpoint_path="checkpoint.json",
            resume=True,
        )
