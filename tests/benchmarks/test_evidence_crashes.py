"""Kill -9 at every durable operation and resume: evidence for claim C7."""

import collections
import dataclasses
import enum
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

import joblib
import numpy
import pandas
import pytest

from benchmarks.evidence import crashes
from scmle import envelope
from scmle.contracts import errors
from scmle.pipeline import archive
from scmle.pipeline import products

Engines = tuple[Path, crashes.Baseline]
Batches = tuple[Path, crashes.Batch, crashes.Baseline]


@pytest.fixture(scope="module")
def engine(tmp_path_factory: pytest.TempPathFactory) -> Engines:
    """The engine's baseline, shared by the tests that need its runs."""
    scratch = tmp_path_factory.mktemp("engine")
    base = crashes.baseline(crashes.Engine(), scratch)
    assert base.error == ""
    return scratch, base


@pytest.fixture(scope="module")
def batch(engine: Engines, tmp_path_factory: pytest.TempPathFactory) -> Batches:
    """The batch scorer's baseline over the engine's reference model."""
    scratch = tmp_path_factory.mktemp("batch")
    subject = crashes.Batch()
    subject.sources = {"engine": engine[1].folder}
    base = crashes.baseline(subject, scratch)
    assert base.error == ""
    return scratch, subject, base


def row(
    status: str = "identical",
    kind: str = crashes.APPEND,
    point: str = "event:1:before",
) -> dict[str, Any]:
    return {
        "point": point,
        "kind": kind,
        "status": status,
        "differing": [] if status != "differs" else ["changed run/x"],
        "duplicates": [] if status != "duplicated" else ["trials/t.json"],
        "error": "" if status != "refused" else "BlockedError: opened",
        "seconds": 1.0,
    }


def document(
    rows: list[dict[str, Any]] | None = None,
    volatile: list[str] | None = None,
    error: str = "",
) -> dict[str, Any]:
    rows = [row()] if rows is None else rows
    return {
        "settings": crashes.settings(),
        "code": crashes.current(),
        "subjects": {
            name: {
                "essential": list(kind.essential),
                "volatile": list(volatile or []) if name == "engine" else [],
                "fields": {"run/a.json": ["/seconds"]}
                if name == "engine"
                else {},
                "outputs": 3,
                "reference_seconds": 1.0,
                "error": error if name == "engine" else "",
                "points": rows,
            }
            for name, kind in crashes.SUBJECTS.items()
        },
    }


def test_a_point_reads_back_from_its_text() -> None:
    point = crashes.Point("lease", 3, crashes.AFTER)
    assert point.text == "lease:3:after"
    assert crashes.Point.parse(point.text) == point


@pytest.mark.parametrize(
    "text", ["lease:0:after", "lease:3", "lease:x:after", "lease:3:during"]
)
def test_a_malformed_point_is_refused(text: str) -> None:
    with pytest.raises(errors.InputError, match="not a kill point"):
        crashes.Point.parse(text)


def test_points_come_from_the_trace_once_each_in_order(tmp_path: Path) -> None:
    trace = tmp_path / "t.trace"
    trace.write_text(
        "stage:1:before\nevent:1:before\nevent:1:after\nstage:1:after\n"
        "stage:1:before\n",
        encoding="utf-8",
    )
    assert [p.text for p in crashes.points(trace)] == [
        "stage:1:before",
        "event:1:before",
        "event:1:after",
        "stage:1:after",
    ]
    assert crashes.points(tmp_path / "absent") == []


def test_differences_name_missing_extra_and_changed_outputs() -> None:
    reference = {"a": b"1", "b": b"2", "c": b"3", "v": b"\x00"}
    found = {"a": b"1", "c": b"9", "d": b"5", "v": b"\x01"}
    assert crashes.differences(reference, found, ["v"], {}) == [
        "missing b",
        "changed c",
        "extra d",
    ]
    # a volatile output still has to exist
    assert crashes.differences(reference, {}, ["v"], {}) == [
        "missing a",
        "missing b",
        "missing c",
        "missing v",
    ]


def test_json_and_json_lines_parse_and_other_bytes_do_not() -> None:
    assert crashes.parse(b'{"a": [1, 2]}') == {"a": [1, 2]}
    assert crashes.parse(b'{"a": 1}\n{"a": 2}\n') == [{"a": 1}, {"a": 2}]
    assert crashes.parse(b"null") is None
    with pytest.raises(ValueError):
        crashes.parse(b"PAR1\x00\x01")
    with pytest.raises(ValueError):
        crashes.parse(b"<html></html>")


def test_drift_names_every_differing_leaf_key_and_list() -> None:
    first = {"a": 1, "b": {"t": 1.5, "k": "x"}, "c": [1, 2], "d": [{"s": 1}]}
    second = {
        "a": 1,
        "b": {"t": 2.5, "k": "x"},
        "c": [1],
        "d": [{"s": 2}],
        "e": 0,
    }
    assert crashes.drift(first, second) == [
        ("b", "t"),
        ("c",),
        ("d", 0, "s"),
        ("e",),
    ]
    assert crashes.drift([1], {"a": 1}) == [()]
    assert crashes.drift({"a": 1}, {"a": 1}) == []


def test_a_pointer_escapes_its_tokens() -> None:
    assert crashes.pointer(("b", 0, "x/y~z")) == "/b/0/x~1y~0z"
    assert crashes.pointer(()) == ""


def test_masked_fields_are_ignored_and_every_other_field_compared() -> None:
    reference = {
        "s.json": b'{"seconds": 1.0, "loss": 0.5}',
        "e.jsonl": b'{"at": 1, "kind": "x"}\n{"at": 2, "kind": "y"}\n',
    }
    fields: dict[str, list[Any]] = {
        "s.json": [("seconds",)],
        "e.jsonl": [(0, "at"), (1, "at")],
    }
    timing = {
        "s.json": b'{"loss": 0.5, "seconds": 9.0}',
        "e.jsonl": b'{"at": 7, "kind": "x"}\n{"at": 8, "kind": "y"}\n',
    }
    assert crashes.differences(reference, timing, [], fields) == []
    corrupt = {
        "s.json": b'{"seconds": 1.0, "loss": 0.6}',
        "e.jsonl": b'{"at": 1, "kind": "x"}\n{"at": 2, "kind": "y"}\n'
        b'{"at": 3, "kind": "y"}\n',
    }
    assert crashes.differences(reference, corrupt, [], fields) == [
        "changed e.jsonl",
        "changed s.json",
    ]
    # a masked field must still be there
    dropped = {"s.json": b'{"loss": 0.5}', "e.jsonl": reference["e.jsonl"]}
    assert crashes.differences(reference, dropped, [], fields) == [
        "changed s.json"
    ]
    # bytes that no longer parse are changed
    broken = {"s.json": b"{", "e.jsonl": reference["e.jsonl"]}
    assert crashes.differences(reference, broken, [], fields) == [
        "changed s.json"
    ]


def test_volatility_is_masked_per_field_and_whole_only_when_unparsed() -> None:
    reference = {
        "same": b"1",
        "t.json": b'{"seconds": 1, "loss": 2}',
        "m.bin": b"\x00",
        "l.json": b"[1]",
    }
    control = {
        "same": b"1",
        "t.json": b'{"seconds": 3, "loss": 2}',
        "m.bin": b"\x01",
        "l.json": b'{"a": 1}',
    }
    volatile, fields = crashes.volatility(reference, control)
    assert volatile == ["l.json", "m.bin"]
    assert fields == {"t.json": [("seconds",)]}


def test_volatility_is_the_union_over_every_control() -> None:
    reference = {
        "t.json": b'{"seconds": 1, "at": 1, "loss": 2}',
        "u.json": b'{"a": 1}',
        "same": b"1",
    }
    agreeing = {"t.json": reference["t.json"], "u.json": b'{"a": 1}'}
    timed = {"t.json": b'{"seconds": 2, "at": 1, "loss": 2}', "u.json": b"x"}
    stamped = {"t.json": b'{"seconds": 1, "at": 5, "loss": 2}', "same": b"1"}
    assert crashes.volatility(reference, agreeing) == ([], {})
    assert crashes.volatility(reference, agreeing, timed, stamped) == (
        ["u.json"],
        {"t.json": [("seconds",), ("at",)]},
    )
    # an output volatile in one control is not also masked
    whole = {"u.json": b"[1]"}
    fielded = {"u.json": b'{"a": 2}'}
    assert crashes.volatility(reference, fielded, whole) == (["u.json"], {})
    assert crashes.CONTROLS == 3
    assert crashes.settings()["controls"] == crashes.CONTROLS


def test_the_hook_counts_each_label_and_kills_only_where_armed(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "h.trace"
    hook = crashes.Hook(trace, None)
    assert hook.visit("put", crashes.BEFORE) == 1
    assert hook.visit("put", crashes.AFTER, 1) == 1
    assert hook.visit("put", crashes.BEFORE) == 2
    assert hook.visit("delete", crashes.BEFORE) == 1
    assert trace.read_text(encoding="utf-8").split() == [
        "put:1:before",
        "put:1:after",
        "put:2:before",
        "delete:1:before",
    ]


def test_an_operation_matches_on_the_end_of_its_key() -> None:
    lease = crashes.Operation("lease", crashes.LEASE, "os", "stat", "/lock")
    assert lease.matches("workflows/w/runs/r/lock")
    assert not lease.matches("workflows/w/runs/r/run.json")
    assert crashes.Operation("x", crashes.APPEND, "os", "stat").matches("a")
    assert lease.target() is os


def test_every_subject_names_its_kill_points_and_kinds() -> None:
    labels = {
        name: [o.label for o in kind.operations]
        for name, kind in crashes.SUBJECTS.items()
    }
    assert labels == {
        "engine": ["stage", "marker", "event", "trial", "custody"],
        "executor": ["lease", "release", "record"],
        "batch": ["chunk", "journal", "manifest", "swap"],
        "erasure": ["plan", "step", "receipt"],
    }
    kinds = {
        o.kind for kind in crashes.SUBJECTS.values() for o in kind.operations
    }
    assert kinds == set(crashes.KINDS)
    # every operation resolves to a real product attribute
    for kind in crashes.SUBJECTS.values():
        for operation in kind.operations:
            assert callable(getattr(operation.target(), operation.attribute))


def test_needs_close_over_the_reference_runs_in_order() -> None:
    assert crashes.closure(["erasure"]) == ["engine", "batch", "erasure"]
    assert crashes.closure(["executor"]) == ["executor"]
    with pytest.raises(errors.InputError, match="unknown subjects"):
        crashes.closure(["nothing"])


def test_all_identical_points_pass() -> None:
    value, verdict = crashes.summary(document())
    assert verdict == "pass"
    assert value.startswith("kill -9 at 4 points over 4 subjects")
    assert "4 identical" in value


@pytest.mark.parametrize(
    "status", ["differs", "duplicated", "refused", "unreached", "failed"]
)
def test_any_other_status_fails(status: str) -> None:
    value, verdict = crashes.summary(document([row(), row(status)]))
    assert verdict == "fail"
    assert f"engine event:1:before: {status}" in value


def test_a_subject_without_points_fails() -> None:
    assert crashes.summary(document([]))[1] == "fail"


def test_a_failed_baseline_fails() -> None:
    value, verdict = crashes.summary(document(error="exit 1"))
    assert verdict == "fail"
    assert "engine: baseline failed: exit 1" in value


def test_a_volatile_essential_output_fails_and_others_are_counted() -> None:
    value, verdict = crashes.summary(document(volatile=["run/events.jsonl"]))
    assert verdict == "pass"
    assert (
        "1 volatile outputs compared for presence only, "
        "1 volatile fields masked in 1 outputs"
    ) in value
    value, verdict = crashes.summary(
        document(volatile=["run/model/pipeline.joblib"])
    )
    assert verdict == "fail"
    assert "essential output run/model/pipeline.joblib is volatile" in value


def test_stale_or_missing_measurements_are_not_measured() -> None:
    stale = document()
    stale["code"] = "other"
    assert crashes.summary(stale) == (
        "crashes: measurements are for other code",
        "not measured",
    )
    partial = document()
    del partial["subjects"]["batch"]
    assert crashes.problems(partial) == [
        "crashes: subjects not measured: ['batch']"
    ]
    assert crashes.summary({"code": "x"})[1] == "not measured"


def test_write_refuses_a_subset_and_workers_must_be_positive() -> None:
    with pytest.raises(errors.InputError, match="every subject"):
        crashes.main(["--write", "--subjects", "executor"])
    with pytest.raises(errors.InputError, match="positive"):
        crashes.main(["--measure", "x.json", "--workers", "0"])


def test_an_armed_child_kills_its_whole_group_exactly_there(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "run"
    subject = crashes.Executor()
    subject.prepare(folder)
    trace = tmp_path / "t.trace"
    traced = crashes.spawn("executor", "start", folder, trace=trace)
    assert traced.code == 0, traced.error
    found = crashes.points(trace)
    assert found[:4] == [
        crashes.Point("lease", 1, crashes.BEFORE),
        crashes.Point("lease", 1, crashes.AFTER),
        crashes.Point("record", 1, crashes.BEFORE),
        crashes.Point("record", 1, crashes.AFTER),
    ]
    assert found[-1] == crashes.Point("release", 1, crashes.AFTER)
    again = tmp_path / "again"
    subject.prepare(again)
    killed = crashes.spawn("executor", "start", again, armed=found[2])
    assert killed.code == -signal.SIGKILL
    assert killed.killed
    # killed before the first record: only the lease is in the store
    assert list(subject.records(again)) == [
        "workflows/crashes/runs/crashes-39ee12a6a2fec502/lock"
    ]


def test_a_killed_executor_run_resumes_to_the_reference(tmp_path: Path) -> None:
    subject = crashes.Executor()
    base = crashes.baseline(subject, tmp_path)
    assert base.error == ""
    assert base.volatile == []
    assert base.fields == {}
    assert len(base.points) == 34
    # killed between a step's running record and its success record
    point = crashes.Point("record", 4, crashes.BEFORE)
    found = crashes.trial(subject, point, tmp_path, base)
    assert found["status"] == "identical", found
    assert found["kind"] == crashes.APPEND


def test_a_finished_run_killed_before_its_release_keeps_a_stale_lease(
    tmp_path: Path,
) -> None:
    # Product finding: `Executor.start` returns a finished run's record
    # without releasing the lease the killed process left.
    subject = crashes.Executor()
    base = crashes.baseline(subject, tmp_path)
    point = crashes.Point("release", 1, crashes.BEFORE)
    found = crashes.trial(subject, point, tmp_path, base)
    assert found["status"] == "differs"
    assert found["differing"] == [
        "extra workflows/crashes/runs/crashes-39ee12a6a2fec502/lock"
    ]


def test_the_executor_ledger_counts_reruns_of_recorded_steps(
    tmp_path: Path,
) -> None:
    subject = crashes.Executor()
    folder = tmp_path / "run"
    subject.prepare(folder)
    assert crashes.spawn("executor", "start", folder).code == 0
    assert subject.ledger(folder) == collections.Counter(
        dict.fromkeys(crashes.Executor.STEPS, 1)
    )


def test_the_check_command_reports_problems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(crashes, "PATH", tmp_path / "absent.json")
    assert crashes.main(["--check"]) == 1


def test_the_module_runs_as_a_command() -> None:
    finished = subprocess.run(
        [sys.executable, "-m", crashes.MODULE, "--help"],
        cwd=crashes.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert finished.returncode == 0
    assert "--measure" in finished.stdout


def test_settings_carry_every_operation_as_json() -> None:
    assert json.loads(json.dumps(crashes.settings()))["operations"]["executor"][
        0
    ] == {
        "label": "lease",
        "kind": crashes.LEASE,
        "owner": "scmle.storage.objectstore:LocalObjectStore",
        "attribute": "put",
        "suffix": "/lock",
    }


def test_a_long_error_keeps_its_first_line_and_its_end() -> None:
    text = "Traceback (most recent call last):\n" + "x" * 5000 + "\nBoom"
    clipped = crashes.clip(text)
    assert clipped.startswith("Traceback (most recent call last):\n...\n")
    assert clipped.endswith("\nBoom")
    assert len(clipped) == len("Traceback (most recent call last):\n...\n") + (
        crashes.TAIL
    )
    assert crashes.clip("short") == "short"


class Marker:
    def finish(self) -> str:
        return "done"


def test_the_hook_wraps_a_method_that_takes_no_argument(tmp_path: Path) -> None:
    trace = tmp_path / "m.trace"
    operation = crashes.Operation(
        "marker", crashes.BOUNDARY, f"{__name__}:Marker", "finish"
    )
    original = Marker.finish
    try:
        crashes.Hook(trace, None).install((operation,))
        assert Marker().finish() == "done"
    finally:
        setattr(Marker, "finish", original)  # noqa: B010 -- restores it
    assert trace.read_text(encoding="utf-8").split() == [
        "marker:1:before",
        "marker:1:after",
    ]


def test_a_batch_killed_before_its_swap_leaves_an_orphaned_temporary(
    batch: Batches,
) -> None:
    # Product finding: `durability.write_atomically` cleans its temporary
    # file only on an exception, so SIGKILL between the write and the swap
    # leaves `.<name>.<random>.tmp` beside the output, and nothing removes it.
    scratch, subject, base = batch
    assert base.volatile == []
    assert base.fields == {}
    point = crashes.Point("swap", 1, crashes.BEFORE)
    found = crashes.trial(subject, point, scratch, base)
    assert found["status"] == "differs"
    assert len(found["differing"]) == 1
    assert found["differing"][0].startswith(
        "extra scores/.predictions.csv.progress.json."
    )
    assert found["differing"][0].endswith(".tmp")


def test_an_essential_output_may_mask_only_timings() -> None:
    found = document()
    found["subjects"]["engine"]["fields"] = {
        "run/stages/04-trainstage.json": ["/seconds", "/elapsed"],
        "run/trials/trial_001.json": ["/access/0/timestamp"],
        "run/evaluation.json": ["/card/resources/latency"],
    }
    assert crashes.summary(found)[1] == "pass"
    found["subjects"]["engine"]["fields"]["run/stages/04-trainstage.json"] = [
        "/seconds",
        "/digest",
        "/needs/Planning",
    ]
    value, verdict = crashes.summary(found)
    assert verdict == "fail"
    assert (
        "engine: essential output run/stages/04-trainstage.json masks /digest"
    ) in value
    assert "masks /needs/Planning" in value
    assert "masks /seconds" not in value


@pytest.mark.parametrize(
    ("field", "timed"),
    [
        ("/seconds", True),
        ("/statistics/operational/read_seconds", True),
        ("/0/timestamp", True),
        ("/card/generated", True),
        ("/sources/0/snapshots/0/captured", True),
        ("/elapsed", True),
        ("/digest", False),
        ("/product", False),
        ("/0/fit", False),
        ("/seconds/total", False),
    ],
)
def test_a_timing_is_named_by_its_last_token(field: str, timed: bool) -> None:
    assert crashes.timing(field) is timed


def test_the_engine_names_its_records_essential() -> None:
    assert crashes.Engine.essential == (
        "run/model/",
        "run/stages/",
        "run/leaderboard.parquet",
        "run/trials/",
    )


class Colour(enum.Enum):
    RED = "red"


@dataclasses.dataclass
class Leaf:
    name: str
    fit_seconds: float
    colour: Colour


class Holder:
    def __init__(self) -> None:
        self.leaf = Leaf("a", 1.5, Colour.RED)
        self.loop = self
        self.when_timestamp = "now"


def test_plain_turns_unpickled_values_into_json_without_timings() -> None:
    value = {
        "holder": Holder(),
        "array": numpy.array([1, 2]),
        "scalar": numpy.float64(0.5),
        "count": numpy.int64(3),
        "flag": True,
        "nan": float("nan"),
        "set": {"b", "a"},
        "pair": (1, None),
        "frame": pandas.DataFrame({"x": [1]}),
        "series": pandas.Series([2.0]),
        "fit_seconds": 9.0,
    }
    assert crashes.plain(value) == {
        "holder": {
            "type": "Holder",
            "leaf": {"type": "Leaf", "name": "a", "colour": "red"},
            "loop": "<cycle>",
        },
        "array": [1, 2],
        "scalar": 0.5,
        "count": 3,
        "flag": True,
        "nan": "nan",
        "set": ["a", "b"],
        "pair": [1, None],
        "frame": {"index": [0], "columns": ["x"], "data": [[1]]},
        "series": [2.0],
    }
    assert json.loads(json.dumps(crashes.plain(value))) == crashes.plain(value)
    assert crashes.plain(object()).startswith("<object object at")


def test_a_kept_product_compares_by_its_fields_without_timings(
    tmp_path: Path,
) -> None:
    path = tmp_path / "03-planning.product"
    buffer = io.BytesIO()
    joblib.dump({"plan": Leaf("p", 2.0, Colour.RED), "count": 3}, buffer)
    envelope.seal(path, {"product": "Blueprint"}, buffer.getvalue())
    assert json.loads(crashes.product(path, path.read_bytes())) == {
        "meta": {"product": "Blueprint"},
        "fields": {
            "plan": {"type": "Leaf", "name": "p", "colour": "red"},
            "count": 3,
        },
    }
    # bytes that do not unseal compare as they are
    path.write_bytes(b"torn")
    assert crashes.product(path, b"torn") == b"torn"


def test_a_leaderboard_compares_row_by_row(tmp_path: Path) -> None:
    path = tmp_path / "leaderboard.parquet"
    pandas.DataFrame({"model": ["a", "b"], "score": [0.5, 0.25]}).to_parquet(
        path
    )
    assert json.loads(crashes.table(path, path.read_bytes())) == [
        {"model": "a", "score": 0.5},
        {"model": "b", "score": 0.25},
    ]
    path.write_bytes(b"torn")
    assert crashes.table(path, b"torn") == b"torn"


def record(
    folder: Path,
    index: int,
    name: str,
    product: str | None,
    needs: dict[str, str],
) -> Path:
    body: dict[str, Any] = {
        "schema_version": 1,
        "stage": name,
        "index": index,
        "seconds": 1.0,
        "product": product,
        "needs": needs,
    }
    body["digest"] = archive.digest(body)
    path = folder / f"{index:02d}-{name.lower()}.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_a_stage_record_compares_whether_its_digests_hold(
    tmp_path: Path,
) -> None:
    payload = b"pickled"
    envelope.seal(tmp_path / "00-planning.product", {"product": "P"}, payload)
    made = products.digest(payload)
    first = record(tmp_path, 0, "Planning", made, {})
    second = record(tmp_path, 1, "Train", None, {"Planning": made})
    assert json.loads(crashes.attest(first, first.read_bytes())) == {
        "schema_version": 1,
        "stage": "Planning",
        "index": 0,
        "seconds": 1.0,
        "product": {"holds": True},
        "needs": {},
        "digest": {"holds": True},
    }
    attested = json.loads(crashes.attest(second, second.read_bytes()))
    assert attested["product"] is None
    assert attested["needs"] == {"Planning": {"holds": True}}
    assert attested["digest"] == {"holds": True}
    # a stale need, an edited record and another product do not hold
    stale = record(tmp_path, 1, "Train", None, {"Planning": "0" * 64})
    assert json.loads(crashes.attest(stale, stale.read_bytes()))["needs"] == {
        "Planning": {"holds": False}
    }
    edited = json.loads(first.read_text(encoding="utf-8"))
    edited["seconds"] = 2.0
    first.write_text(json.dumps(edited), encoding="utf-8")
    assert json.loads(crashes.attest(first, first.read_bytes()))["digest"] == {
        "holds": False
    }
    envelope.seal(tmp_path / "00-planning.product", {"product": "P"}, b"other")
    assert json.loads(crashes.attest(first, first.read_bytes()))["product"] == {
        "holds": False
    }
    assert crashes.attest(first, b"{") == b"{"


def test_the_engine_reference_records_hold_and_compare_by_field(
    engine: Engines,
) -> None:
    scratch, base = engine
    del scratch
    stages = {
        name: json.loads(content)
        for name, content in base.outputs.items()
        if name.startswith("run/stages/") and name.endswith(".json")
    }
    assert len(stages) == 8
    for body in stages.values():
        assert body["digest"] == {"holds": True}
        assert all(need == {"holds": True} for need in body["needs"].values())
    kept = [n for n in base.outputs if n.endswith(archive.SUFFIX)]
    assert kept == [
        "run/stages/03-planning.product",
        "run/stages/04-trainstage.product",
        "run/stages/05-selectstage.product",
    ]
    for name in kept:
        assert json.loads(base.outputs[name])["meta"]["product"]
        assert name not in base.volatile
    # Product finding: the planning product carries a free-disk reading, so
    # it may differ between two uninterrupted runs; nothing else may.
    reading = (
        "fields",
        "registry",
        "runtime_environment",
        "disk_gb",
    )
    assert {path for name in kept for path in base.fields.get(name, [])} <= {
        reading
    }
    assert "run/leaderboard.parquet" not in base.volatile
    assert json.loads(base.outputs["run/leaderboard.parquet"])


def test_an_engine_killed_after_custody_opens_the_test_set_is_refused(
    engine: Engines,
) -> None:
    # Product finding: once the final evaluation opens the locked test set,
    # `Journal.admit` refuses to resume the run, even with every stage
    # record durable.
    scratch, base = engine
    point = crashes.Point("custody", 2, crashes.AFTER)
    assert point in base.points
    found = crashes.trial(crashes.Engine(), point, scratch, base)
    assert found["status"] == "refused"
    assert "already opened" in found["error"]


def test_an_erasure_killed_before_a_step_records_it_unremoved(
    engine: Engines, batch: Batches, tmp_path: Path
) -> None:
    # Product finding: a deletion step the killed run did is redone on
    # resume and journaled `removed: false`, so the journal and the signed
    # receipt differ from an uninterrupted erasure.
    subject = crashes.Erasure()
    subject.sources = {"engine": engine[1].folder, "batch": batch[2].folder}
    base = crashes.baseline(subject, tmp_path)
    assert base.error == ""
    # step 1 removes nothing, so redoing it records the same; step 2 is
    # the first step that deletes a file
    point = crashes.Point("step", 2, crashes.BEFORE)
    assert point in base.points
    found = crashes.trial(subject, point, tmp_path, base)
    assert found["status"] == "differs"
    changed = [
        entry.removeprefix("changed receipts/forget-").split("/", 1)[-1]
        for entry in found["differing"]
    ]
    assert len(changed) == 2
    assert changed[1] == "journal.jsonl"
    assert changed[0].endswith(".json")


def test_an_engine_resume_starts_again_skips_or_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scmle.api import project

    started = []
    resumed = []
    monkeypatch.setattr(
        crashes.Engine, "start", lambda self, folder: started.append(folder)
    )
    monkeypatch.setattr(
        project.Project,
        "resume",
        classmethod(lambda cls, run, data: resumed.append((run, len(data)))),
    )
    subject = crashes.Engine()
    subject.resume(tmp_path)
    assert started == [tmp_path]
    assert resumed == []
    run = tmp_path / "runs" / "run_20260101_001"
    run.mkdir(parents=True)
    subject.resume(tmp_path)
    assert resumed == [(run, crashes.ROWS)]
    (run / archive.MARKER).write_text("{}", encoding="utf-8")
    subject.resume(tmp_path)
    assert resumed == [(run, crashes.ROWS)]
    assert started == [tmp_path]


LOCK = "workflows/w/runs/r/lock"
RUN = "workflows/w/runs/r/run.json"
STEP = "workflows/w/runs/r/steps/a.json"


def test_a_takeover_must_raise_the_fence_by_one() -> None:
    subject = crashes.Executor()
    reference = {RUN: {"state": "done", "fence": 1}, STEP: {"fence": 1}}
    kept = {
        LOCK: {"fence": 1, "expires": 1.0},
        RUN: {"state": "running", "fence": 1},
        STEP: {"fence": 1},
    }
    raised = {RUN: {"state": "done", "fence": 2}, STEP: {"fence": 1}}
    assert subject.audit(raised, kept, reference) == []
    # rewritten under the old fence, or an untouched record moved
    reused = {RUN: {"state": "done", "fence": 1}, STEP: {"fence": 1}}
    assert subject.audit(reused, kept, reference) == [f"fence {RUN}"]
    skipped = {RUN: {"state": "done", "fence": 3}, STEP: {"fence": 1}}
    assert subject.audit(skipped, kept, reference) == [f"fence {RUN}"]
    # killed before any lease: the resume takes the first one
    fresh = {RUN: {"state": "done", "fence": 1}, STEP: {"fence": 1}}
    assert subject.audit(fresh, {}, reference) == []
    assert subject.audit(raised, {}, reference) == [f"fence {RUN}"]


def test_executor_outputs_leave_out_the_clock_and_the_fence(
    tmp_path: Path,
) -> None:
    subject = crashes.Executor()
    folder = tmp_path / "run"
    subject.prepare(folder)
    assert crashes.spawn("executor", "start", folder).code == 0
    state = subject.state(folder)
    assert {record["fence"] for record in state.values()} == {1}
    for content in subject.contents(folder).values():
        assert not {*crashes.CLOCKED, crashes.FENCE} & set(json.loads(content))
    assert "fence" not in crashes.CLOCKED


def test_measure_runs_every_point_of_a_subject_once(tmp_path: Path) -> None:
    found = crashes.measure(tmp_path, ["executor"], workers=2)
    assert list(found["subjects"]) == ["executor"]
    measured = found["subjects"]["executor"]
    assert measured["error"] == ""
    assert measured["volatile"] == []
    assert measured["fields"] == {}
    texts = [row["point"] for row in measured["points"]]
    assert len(texts) == 34
    assert len(set(texts)) == 34
    assert texts[0] == "lease:1:before"
    assert texts[-1] == "release:1:after"
    assert found["settings"] == crashes.settings()
    assert found["code"] == crashes.current()


def test_main_measures_into_a_file_and_writes_the_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = []

    def fake(
        scratch: Path, names: list[str] | None, workers: int
    ) -> dict[str, Any]:
        calls.append((names, workers))
        return document()

    monkeypatch.setattr(crashes, "measure", fake)
    out = tmp_path / "out.json"
    assert crashes.main(["--measure", str(out), "--subjects", "executor"]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == document()
    printed = json.loads(capsys.readouterr().out)
    assert printed["verdict"] == ""
    assert json.loads(printed["value"])["statuses"] == {"identical": 4}
    monkeypatch.setattr(crashes, "PATH", tmp_path / "crashes.json")
    assert crashes.main(["--write", "--workers", "3"]) == 0
    assert calls == [(["executor"], crashes.WORKERS), (None, 3)]
    assert json.loads(crashes.PATH.read_text(encoding="utf-8")) == document()
    printed = json.loads(capsys.readouterr().out)
    assert printed["verdict"] == "pass"
    assert crashes.main(["--check"]) == 0
