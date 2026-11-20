"""The engine keeps each stage's product and a resumed run reuses it.

A stage whose product is kept (`products.Kept`) and whose record names the
same digests of the products it needs is not executed again: its product
is loaded and applied. A derived product (`products.Derived`) is made again
every time; when what it is derived from changed, every stage after it runs
again.
"""

import contextlib
import dataclasses

import pytest

from scmle.contracts import errors
from scmle.contracts import signing
from scmle.pipeline import archive
from scmle.pipeline import engine
from scmle.pipeline import products
from scmle.runtime import state


@dataclasses.dataclass(frozen=True)
class Source(products.Derived):
    seed: int

    def summary(self):
        return {"seed": self.seed}


@dataclasses.dataclass(frozen=True)
class Tally(products.Kept):
    count: int


class Regulator:
    def cap(self):
        return contextlib.nullcontext()


class Board:
    def __init__(self, origin):
        self.origin = origin
        self.made = {}
        self.ran = []
        self.regulator = Regulator()

    @property
    def count(self):
        return self.made[Tally].count

    def check(self):
        pass


class First:
    state = state.Progress.VALIDATING
    requires = ()
    makes = Source

    def describe(self, context):
        return "first"

    def execute(self, context):
        context.ran.append("First")
        return Source(context.origin)


class Second:
    state = state.Progress.PROFILING
    requires = (First,)
    makes = Tally

    def describe(self, context):
        return "second"

    def execute(self, context):
        context.ran.append("Second")
        return Tally(context.made[Source].seed * 10)


@pytest.fixture(autouse=True)
def keyed(monkeypatch):
    monkeypatch.setenv(signing.SIGNING_KEY_ENV, "a key long enough for tests")


def attempt(directory, origin):
    board = Board(origin)
    docket = archive.Directory(directory, "config")
    docket.keep(docket.read())
    pipeline = engine.Pipeline((First(), Second()), docket)
    pipeline.advance(board)
    return board, pipeline


def test_a_kept_product_is_loaded_instead_of_running_its_stage(tmp_path):
    board, pipeline = attempt(tmp_path, 2)
    assert board.ran == ["First", "Second"]
    assert pipeline.restored == []
    record = archive.Directory(tmp_path, "config").read()[1]
    assert set(record["needs"]) == {"First"}
    assert isinstance(record["product"], str)

    board, pipeline = attempt(tmp_path, 2)
    assert board.ran == ["First"]
    assert board.count == 20
    assert pipeline.restored == ["Second"]


def test_a_stage_whose_inputs_changed_runs_again(tmp_path):
    attempt(tmp_path, 2)
    board, pipeline = attempt(tmp_path, 3)
    assert board.ran == ["First", "Second"]
    assert board.count == 30
    assert pipeline.restored == []


def test_a_record_not_verified_is_not_restored(tmp_path):
    attempt(tmp_path, 2)
    board = Board(2)
    unread = archive.Directory(tmp_path, "config")
    engine.Pipeline((First(), Second()), unread).advance(board)
    assert board.ran == ["First", "Second"]


def test_a_product_file_that_changed_is_not_loaded(tmp_path):
    attempt(tmp_path, 2)
    (kept,) = (tmp_path / archive.FOLDER).glob("*.product")
    kept.write_bytes(kept.read_bytes() + b"x")
    board, pipeline = attempt(tmp_path, 2)
    assert board.ran == ["First", "Second"]
    assert pipeline.restored == []


class Silent(First):
    def execute(self, context):
        context.ran.append("Silent")


def test_a_stage_that_returns_no_declared_product_is_refused():
    with pytest.raises(errors.KindError, match="Silent returned NoneType"):
        engine.Pipeline((Silent(),)).advance(Board(1))
