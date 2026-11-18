"""The resource governor's messages and report, and the process tables."""

import multiprocessing
import os
import pathlib
import sys
import time
import types

import pytest

from scmle.contracts import tasks
from scmle.runtime import limits
from tests.unit.test_trial_supervision import explode

HUGE = float("inf")


def governor(**changes) -> limits.Governor:
    return limits.Governor(tasks.Resources(**changes))


def late(**changes) -> limits.Governor:
    """A governor whose budget ran out long ago."""
    found = governor(**changes)
    found.started_at = time.monotonic() - 10 * found.spec.time_budget_seconds
    return found


def test_cooperative_enforcement_is_stated_word_for_word() -> None:
    found = governor(enforcement="cooperative")
    assert found.explain("killed") == (
        "cooperative: no trial starts past the limit"
    )


def test_late_work_reports_no_negative_time_left() -> None:
    assert late().overdue(5.0) == (
        "needs about 5.0 s but 0.0 s remain before the run's end"
    )
    assert late().deadline() == 0.0


def test_work_that_fits_is_stopped_by_the_memory_limit(monkeypatch) -> None:
    monkeypatch.setattr(limits, "peak_memory_gb", lambda: HUGE)
    assert governor().overdue(0.0) == "memory limit reached"


def test_the_first_trial_runs_above_the_memory_limit(monkeypatch) -> None:
    monkeypatch.setattr(limits, "peak_memory_gb", lambda: HUGE)
    found = governor()
    assert found.skip_reason(first=True) is None
    assert found.skip_reason(first=False) == (
        "memory limit reached before this trial started"
    )


def test_a_closed_trial_window_is_named() -> None:
    reason = late().skip_reason(first=True)
    assert reason is not None
    assert reason.startswith("trial window closed: 0.0 s left")


def test_the_report_names_every_limit_and_its_enforcement() -> None:
    found = governor(time_budget_seconds=100).report()
    assert found["time_budget"]["within_limit"] is True
    assert 0.0 <= found["time_budget"]["elapsed_seconds"] < 100
    assert found["cpu"]["enforcement"] == "no cap configured"
    assert found["memory"]["enforcement"] == (
        "no trial starts past the limit; running trials above the limit are "
        "killed (resident memory sampled while they run)"
    )
    assert found["gpu"] == {
        "mode": "auto",
        "enforcement": "no built-in model uses a GPU",
    }


def test_peak_memory_is_a_positive_number_of_gigabytes() -> None:
    assert 0.0 < limits.peak_memory_gb() < HUGE


def test_the_footprint_follows_children_only(monkeypatch) -> None:
    table = {1: (0, 10), 2: (1, 20), 3: (2, 30), 4: (9, 40)}
    monkeypatch.setattr(limits, "pids", lambda: table)
    assert limits.footprint(1) == {1: 10, 2: 20, 3: 30}


def test_the_ps_table_lists_this_process() -> None:
    table = limits.ps_table()
    assert table[os.getpid()][0] == os.getppid()


def test_the_psutil_table_skips_processes_without_memory(monkeypatch) -> None:
    rows = [
        {"pid": 5, "ppid": 1, "memory_info": types.SimpleNamespace(rss=7)},
        {"pid": 6, "ppid": 1, "memory_info": None},
    ]
    fake = types.SimpleNamespace(
        process_iter=lambda fields: [
            types.SimpleNamespace(info=r) for r in rows
        ]
    )
    monkeypatch.setitem(sys.modules, "psutil", fake)
    assert limits.psutil_table() == {5: (1, 7)}


def test_the_proc_table_reads_parent_and_resident_pages(
    monkeypatch, tmp_path
) -> None:
    entry = tmp_path / "proc" / "42"
    entry.mkdir(parents=True)
    (entry / "stat").write_text("42 (a b) S 7 0 0", encoding="ascii")
    (entry / "statm").write_text("100 25 0", encoding="ascii")
    real = pathlib.Path

    def path(text):
        return tmp_path / "proc" if text == "/proc" else real(text)

    monkeypatch.setattr(limits, "pathlib", types.SimpleNamespace(Path=path))
    page = limits.resource.getpagesize()
    assert limits.tabulate() == {42: (7, 25 * page)}


def test_the_child_sends_its_value_or_its_failure() -> None:
    receiver, sender = multiprocessing.Pipe(duplex=False)
    limits.execute(sender, abs, -3)
    assert receiver.recv() == (True, 3)
    receiver, sender = multiprocessing.Pipe(duplex=False)
    limits.execute(sender, explode, 1)
    succeeded, failure = receiver.recv()
    assert succeeded is False
    assert failure["kind"] == "ValueError"


def test_a_failed_trial_carries_the_child_failure() -> None:
    with pytest.raises(
        limits.TrialError, match=r"^ValueError: boom$"
    ) as raised:
        limits.Supervisor(30.0, None).run(explode, 1)
    assert raised.value.details["failure"]["message"] == "boom"


class Entry:
    """A `/proc` entry whose file names match case-sensitively, as on Linux."""

    def __init__(self, name: str, files: dict[str, pathlib.Path]) -> None:
        self.name = name
        self.files = files

    def __truediv__(self, name: str) -> pathlib.Path:
        return self.files.get(name, self.files["stat"].parent / "absent")


def test_the_proc_table_reads_the_lowercase_statm(
    monkeypatch, tmp_path
) -> None:
    stat = tmp_path / "stat"
    stat.write_text("42 (a b) S 7 0 0", encoding="ascii")
    statm = tmp_path / "statm"
    statm.write_text("100 25 0", encoding="ascii")
    entry = Entry("42", {"stat": stat, "statm": statm})
    root = types.SimpleNamespace(iterdir=lambda: [entry])
    monkeypatch.setattr(
        limits, "pathlib", types.SimpleNamespace(Path=lambda text: root)
    )
    page = limits.resource.getpagesize()
    assert limits.tabulate() == {42: (7, 25 * page)}


def test_a_process_with_a_non_ascii_name_is_still_counted(
    tmp_path, monkeypatch
):
    # /proc/<pid>/stat holds the command name in parentheses; a name with
    # non-ASCII bytes used to fail decoding, the process was skipped and the
    # memory check undercounted.
    proc = tmp_path / "proc"
    entry = proc / "4242"
    entry.mkdir(parents=True)
    (entry / "stat").write_bytes("4242 (café) S 1 4242".encode())
    (entry / "statm").write_text("10 3 0 0 0 0 0", encoding="ascii")
    real = pathlib.Path
    monkeypatch.setattr(
        limits.pathlib,
        "Path",
        lambda path: real(proc) if path == "/proc" else real(path),
    )
    table = limits.tabulate()
    assert table[4242] == (1, 3 * limits.resource.getpagesize())
