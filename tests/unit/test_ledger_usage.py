"""A confirmation set is spent once per charter and data (C8 charter_rerun)."""

import json

import pytest

from scmle.assurance.ledger import confirm
from scmle.assurance.ledger import governed
from scmle.assurance.ledger import usage
from scmle.contracts import errors
from tests.unit.test_charter_contracts import make_charter
from tests.unit.test_ledger_governed import build
from tests.unit.test_ledger_governed import locate


def govern(folder, seed=0):
    features, target = build(seed=seed)
    cfeatures, ctarget = build(rows=400, seed=9)
    source = locate(cfeatures, ctarget)
    explored = governed.scout(
        charter=make_charter(),
        features=features,
        target=target,
        classification=False,
        run_dir=folder,
    )
    registry = None if folder is None else folder.parent
    return governed.finalise(explored, source=source, registry=registry), source


def test_the_same_charter_on_the_same_data_is_refused_a_second_time(
    tmp_path,
):
    first = govern(tmp_path / "one")[0]
    assert first.confirmation.outcome == confirm.CONFIRMED
    features, target = build()
    cfeatures, ctarget = build(rows=400, seed=9)
    source = locate(cfeatures, ctarget)
    explored = governed.scout(
        charter=make_charter(),
        features=features,
        target=target,
        classification=False,
        run_dir=tmp_path / "two",
    )
    with pytest.raises(errors.LeakageError, match="already confirmed"):
        governed.finalise(explored, source=source, registry=tmp_path)
    assert not source.opened
    assert [e["event"] for e in source.log.entries][-1] == "denied"
    assert not (tmp_path / "two" / confirm.CLAIMS_FILE).exists()


def test_other_data_under_the_same_charter_is_confirmed(tmp_path):
    govern(tmp_path / "one")
    second = govern(tmp_path / "two", seed=1)[0]
    assert second.confirmation.outcome == confirm.CONFIRMED


def test_the_use_is_recorded_durably_with_its_run(tmp_path):
    govern(tmp_path / "one")
    records = list((tmp_path / usage.FOLDER).iterdir())
    assert len(records) == 1
    body = json.loads(records[0].read_text(encoding="utf-8"))
    assert body["run"] == "one"
    assert body["charter_hash"] == make_charter().charter_hash


def test_the_same_run_may_record_its_own_use_again(tmp_path):
    register = usage.Register(tmp_path, "k" * 64, "one")
    assert register.record() is None
    assert register.record() is None
    assert "run one" in usage.Register(tmp_path, "k" * 64, "two").record()
    register.path.write_text("{", encoding="utf-8")
    assert "unreadable" in register.record()


def test_the_key_depends_on_charter_data_and_rows():
    base = usage.key("c", "d", (1, 2))
    assert base == usage.key("c", "d", (1, 2))
    assert (
        len(
            {
                base,
                usage.key("x", "d", (1, 2)),
                usage.key("c", "x", (1, 2)),
                usage.key("c", "d", (1, 3)),
            }
        )
        == 4
    )


def test_an_in_memory_run_records_nothing():
    govern(None)
    govern(None)


def test_without_a_registry_nothing_is_recorded(tmp_path):
    features, target = build()
    cfeatures, ctarget = build(rows=400, seed=9)
    governed.run_governed(
        charter=make_charter(),
        features=features,
        target=target,
        classification=False,
        source=locate(cfeatures, ctarget),
        run_dir=tmp_path / "one",
    )
    assert not (tmp_path / usage.FOLDER).exists()
