"""Every record logged while handling a request carries its id (ERR-24)."""

import logging

from scmle.core import logs
from scmle.inference import serving
from tests.unit.test_serving_contract import bundle  # noqa: F401
from tests.unit.test_serving_contract import call
from tests.unit.test_serving_contract import golden  # noqa: F401
from tests.unit.test_serving_contract import running  # noqa: F401


class Collect(logging.Handler):
    """Keeps every record it sees."""

    def __init__(self):
        super().__init__()
        self.records = []
        self.addFilter(logs.Stamp())

    def emit(self, record):
        self.records.append(record)


def test_a_failing_request_logs_its_request_id(running, golden, monkeypatch):  # noqa: F811
    def broken(request):
        raise RuntimeError("boom")

    monkeypatch.setattr(running, "score", broken)
    collect = Collect()
    serving.LOGGER.addHandler(collect)
    try:
        status = call(
            f"{running.address}/predict",
            {"records": golden.records},
            headers={"X-Request-ID": "trace-42"},
        )[0]
    finally:
        serving.LOGGER.removeHandler(collect)
    assert status == 500
    assert collect.records
    assert all(record.request == "trace-42" for record in collect.records)
