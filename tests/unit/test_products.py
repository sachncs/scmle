"""Stage products: what is kept, what is derived again, and what is refused."""

import dataclasses
import types

import pytest

from scmle.contracts import signing
from scmle.pipeline import products
from scmle.pipeline.stages import ingestion
from scmle.pipeline.stages import training

KEY = "a test signing key that is long enough"


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setenv(signing.SIGNING_KEY_ENV, KEY)


def search():
    return training.Search(
        trials=[{"trial_id": "trial_001"}],
        outcomes=[],
        trial_counter=1,
        skipped=[{"work": "tuning", "reason": "late"}],
    )


def test_a_kept_product_loads_back_equal_and_applies_to_a_board(
    keyed, tmp_path
):
    path = tmp_path / "04-trainstage.product"
    made = search()
    digest = made.save(path)
    loaded = training.Search.load(path, digest)
    assert loaded == made
    board = types.SimpleNamespace()
    loaded.apply(board)
    assert board.trial_counter == 1
    assert training.Search.capture(board) == made


def test_an_unsigned_product_is_never_unpickled(monkeypatch, tmp_path):
    monkeypatch.delenv(signing.SIGNING_KEY_ENV, raising=False)
    monkeypatch.delenv(signing.SIGNING_KEY_FILE_ENV, raising=False)
    path = tmp_path / "04-trainstage.product"
    digest = search().save(path)
    assert isinstance(digest, str)
    assert training.Search.load(path, digest) is None


def test_a_product_whose_digest_or_kind_differs_is_not_loaded(keyed, tmp_path):
    path = tmp_path / "04-trainstage.product"
    digest = search().save(path)
    assert training.Search.load(path, "0" * 64) is None
    assert training.Search.load(tmp_path / "missing", digest) is None

    @dataclasses.dataclass(frozen=True)
    class Other(products.Kept):
        trials: list
        outcomes: list
        trial_counter: int
        skipped: list

    assert Other.load(path, digest) is None


def test_an_unpicklable_product_has_no_digest_and_leaves_no_file(
    keyed, tmp_path
):
    path = tmp_path / "04-trainstage.product"
    search().save(path)
    made = training.Search(
        trials=[lambda: None], outcomes=[], trial_counter=0, skipped=[]
    )
    assert made.save(path) is None
    assert not path.exists()


def test_a_derived_product_is_digested_by_its_identity_and_never_loaded(
    tmp_path,
):
    def intake(seconds, data="d1"):
        return ingestion.Intake(
            frame=None,
            manifest=types.SimpleNamespace(
                digests={"data": data, "schema": "s"}
            ),
            cleaning_notes=[],
            read_seconds=seconds,
            stream=None,
        )

    path = tmp_path / "00-ingeststage.product"
    first = intake(0.1).save(path)
    assert first == intake(9.0).save(path)
    assert first != intake(0.1, "d2").save(path)
    assert not path.exists()
    assert ingestion.Intake.load(path, first) is None


@dataclasses.dataclass(frozen=True)
class Profiled(products.Derived):
    seconds: float
    rows: int

    def summary(self):
        return {"profile": [{"rows": self.rows, "fit_seconds": self.seconds}]}


def test_a_derived_digest_leaves_out_measured_durations(tmp_path):
    path = tmp_path / "02-profilestage.product"
    assert Profiled(0.1, 3).save(path) == Profiled(0.2, 3).save(path)
    assert Profiled(0.1, 3).save(path) != Profiled(0.1, 4).save(path)
