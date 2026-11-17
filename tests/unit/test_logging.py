"""The logging convention rolled out (task 2.1, ERR-22, ERR-24).

The CLI installs the convention's handler, with JSON lines on `--log-json`;
the service runtime installs it too; every logger in the rolled-out packages
is named after its module; the ERR-22 reduced paths in sampling/ and
operations/ log a WARNING and raise `Degraded`.
"""

import json
import logging
import pathlib
import re

import numpy
import pandas
import pytest

from scmle import cli
from scmle.core import logs
from scmle.operations import metrics
from scmle.sampling import synthesis

ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGES = (
    "scmle/cli.py",
    "scmle/sampling",
    "scmle/operations",
    "scmle/monitoring",
    "scmle/governance",
    "scmle/deploy",
)
NAMED = re.compile(r"getLogger\(([^)]*)\)")


def sources():
    """The Python files of the rolled-out packages."""
    for name in PACKAGES:
        path = ROOT / name
        yield from [path] if path.is_file() else sorted(path.rglob("*.py"))


@pytest.fixture
def package():
    """Restores the package logger's handlers and level after the test."""
    logger = logging.getLogger(logs.PACKAGE)
    handlers, level = list(logger.handlers), logger.level
    yield logger
    logger.handlers[:] = handlers
    logger.setLevel(level)


def test_log_json_writes_one_json_object_per_record(package, capsys):
    status = cli.main(["--log-json", "trail", "show", "/absent/run"])
    assert status != cli.EXIT_OK
    lines = [x for x in capsys.readouterr().err.splitlines() if x.strip()]
    records = [json.loads(x) for x in lines]
    assert records
    assert records[-1]["level"] == "ERROR"
    assert records[-1]["logger"] == "scmle.cli"
    assert {"run", "request", "time"} <= set(records[-1])


def test_the_cli_installs_the_convention_handler(package):
    cli.main(["trail", "show", "/absent/run"])
    installed = [h for h in package.handlers if getattr(h, "convention", 0)]
    assert len(installed) == 1
    assert not isinstance(installed[0].formatter, logs.Lines)


def test_the_service_runtime_uses_the_convention():
    text = (ROOT / "scmle/deploy/runtime.py").read_text(encoding="utf-8")
    assert "logs.configure(" in text
    assert "basicConfig" not in text


@pytest.mark.parametrize("path", list(sources()), ids=str)
def test_every_logger_is_named_after_its_module(path):
    names = NAMED.findall(path.read_text(encoding="utf-8"))
    assert all(x == "__name__" for x in names), names


def test_an_undersized_class_is_reported_as_degraded(caplog):
    rng = numpy.random.default_rng(0)
    features = pandas.DataFrame({"x": rng.normal(size=23)})
    target = pandas.Series(["a"] * 20 + ["b"] * 3)
    sampler = synthesis.CopulaOverSampler()
    with (
        caplog.at_level(logging.WARNING, logger="scmle.sampling.synthesis"),
        pytest.warns(logs.Degraded, match="'b'"),
    ):
        sampler.resample(features, target, seed=0)
    assert any("'b'" in r.getMessage() for r in caplog.records)


def test_a_missing_resource_module_is_reported_once(monkeypatch, caplog):
    monkeypatch.setattr(metrics, "resource", None)
    metrics.forgo.cache_clear()
    with caplog.at_level(logging.WARNING, logger="scmle.operations.metrics"):
        with pytest.warns(logs.Degraded, match="resource"):
            peak = metrics.residency()
        metrics.cpu()
    assert peak == 0
    assert len(caplog.records) == 1
    metrics.forgo.cache_clear()
