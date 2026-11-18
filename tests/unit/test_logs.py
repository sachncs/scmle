"""The package logging convention (task 2.1, ERR-24, ERR-22).

Every record carries the run id and request id in scope; the structured
format writes one JSON object per line; a degraded optional path both logs a
WARNING and raises a `Degraded` warning; core fallbacks and gate downgrades
log a WARNING.
"""

import io
import json
import logging
import pathlib
import warnings

import pytest

from scmle.runtime import logs
from scmle.runtime import resource_observation
from scmle.runtime import retry

ROOT = pathlib.Path(__file__).resolve().parents[2]


def capture(structured):
    """A configured stream handler on a fresh logger, and its stream."""
    stream = io.StringIO()
    handler = logs.handler(structured=structured, stream=stream)
    logger = logging.getLogger("scmle.tests.logs")
    logger.handlers[:] = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, stream


def test_structured_records_carry_the_run_and_request_ids():
    logger, stream = capture(structured=True)
    with logs.bind(run="run-1"):
        with logs.bind(request="req-9"):
            logger.warning("fell back to %s", "tf-idf")
        logger.info("after the request")
    logger.info("outside the run")
    first, second, third = (
        json.loads(x) for x in stream.getvalue().splitlines()
    )
    assert first["run"] == "run-1"
    assert first["request"] == "req-9"
    assert first["level"] == "WARNING"
    assert first["logger"] == "scmle.tests.logs"
    assert first["message"] == "fell back to tf-idf"
    assert "time" in first
    assert (second["run"], second["request"]) == ("run-1", "")
    assert (third["run"], third["request"]) == ("", "")


def test_structured_records_keep_the_traceback():
    logger, stream = capture(structured=True)
    try:
        raise KeyError("lost")
    except KeyError:
        logger.exception("swallowed")
    record = json.loads(stream.getvalue())
    assert "KeyError" in record["traceback"]


def test_text_records_name_the_ids():
    logger, stream = capture(structured=False)
    with logs.bind(run="run-1", request="req-9"):
        logger.warning("downgraded")
    text = stream.getvalue()
    assert "run=run-1" in text
    assert "request=req-9" in text
    assert "WARNING" in text


def test_degrade_logs_a_warning_and_warns(caplog):
    logger = logging.getLogger("scmle.tests.degrade")
    with (
        caplog.at_level(logging.WARNING, logger=logger.name),
        pytest.warns(logs.Degraded, match="no psutil"),
    ):
        logs.degrade(logger, "no psutil; %s", "memory unread")
    assert caplog.records[0].levelno == logging.WARNING
    assert caplog.records[0].getMessage() == "no psutil; memory unread"


def test_configure_installs_one_handler_on_the_package_logger():
    package = logging.getLogger("scmle")
    before = list(package.handlers)
    try:
        logs.configure(logging.INFO, structured=True)
        logs.configure(logging.INFO, structured=True)
        installed = [h for h in package.handlers if h not in before]
        assert len(installed) == 1
    finally:
        package.handlers[:] = before


def test_a_retried_transient_failure_is_logged(caplog):
    outcomes = iter([TimeoutError("slow"), "done"])

    def operation():
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    with caplog.at_level(logging.WARNING, logger="scmle.runtime.retry"):
        result = retry.call(
            operation, retry.Retries(max_attempts=2), sleep=lambda seconds: None
        )
    assert result == "done"
    assert any("slow" in r.getMessage() for r in caplog.records)


def test_degrade_is_not_silenced_by_the_package_filter():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        logs.degrade(logging.getLogger("scmle.tests.x"), "optional path off")
    assert [w.category for w in caught] == [logs.Degraded]


def test_an_unreadable_disk_is_logged(caplog, tmp_path):
    missing = tmp_path / "absent"
    with caplog.at_level(logging.WARNING, logger="scmle.pipeline"):
        assert resource_observation.freespace(missing) == (None, "unknown")
    assert any("absent" in r.getMessage() for r in caplog.records)


def test_a_capability_downgrade_and_a_budget_skip_log_a_warning():
    stages = (ROOT / "scmle/pipeline/stages/planning.py").read_text(
        encoding="utf-8"
    )
    context = (ROOT / "scmle/pipeline/journal.py").read_text(encoding="utf-8")
    assert "accepted as extension-only" in stages
    assert 'LOGGER.warning("%s skipped or reduced' in context
