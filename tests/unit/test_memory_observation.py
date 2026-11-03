"""Physical memory observation on POSIX and on Windows.

`os.sysconf` does not exist on Windows, so memory was reported unknown and the
capability gate refused every run there ("memory_gb is required").
"""

import os
import sys
import types

import pytest

from scmle.core import resource_observation

GIB = 1024**3


def fake_psutil(total: int) -> types.ModuleType:
    """A stand-in `psutil` whose `virtual_memory().total` is `total`."""
    module = types.ModuleType("psutil")
    module.virtual_memory = lambda: types.SimpleNamespace(total=total)
    return module


def no_sysconf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `os.sysconf` absent, as it is on Windows."""
    monkeypatch.delattr(os, "sysconf", raising=False)


def test_posix_uses_sysconf(monkeypatch):
    monkeypatch.setattr(
        os,
        "sysconf",
        lambda name: {"SC_PHYS_PAGES": 4 * 1024, "SC_PAGE_SIZE": 1024**2}[name],
        raising=False,
    )

    memory, source = resource_observation.observe_memory_gb()

    assert memory == pytest.approx(4.0)
    assert source == "os.sysconf(SC_PHYS_PAGES)"


def test_windows_reads_total_memory_from_psutil(monkeypatch):
    no_sysconf(monkeypatch)
    monkeypatch.setattr(resource_observation, "WINDOWS", True)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil(16 * GIB))

    memory, source = resource_observation.observe_memory_gb()

    assert memory == pytest.approx(16.0)
    assert source == "psutil.virtual_memory()"


def test_windows_without_psutil_stays_unknown(monkeypatch):
    no_sysconf(monkeypatch)
    monkeypatch.setattr(resource_observation, "WINDOWS", True)
    monkeypatch.setitem(sys.modules, "psutil", None)  # import raises

    assert resource_observation.observe_memory_gb() == (None, "unknown")


def test_non_windows_without_sysconf_never_consults_psutil(monkeypatch):
    no_sysconf(monkeypatch)
    monkeypatch.setattr(resource_observation, "WINDOWS", False)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil(16 * GIB))

    assert resource_observation.observe_memory_gb() == (None, "unknown")


@pytest.mark.parametrize("total", [0, -1])
def test_windows_rejects_non_positive_totals(monkeypatch, total):
    no_sysconf(monkeypatch)
    monkeypatch.setattr(resource_observation, "WINDOWS", True)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil(total))

    assert resource_observation.observe_memory_gb() == (None, "unknown")
