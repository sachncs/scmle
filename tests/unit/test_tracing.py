"""OpenTelemetry tracing: off by default, spans when enabled, no-op without."""

import contextlib
import json
import pathlib
import sys
import urllib.request

import pandas
import pytest

from scmle import cli
from scmle.deploy import parity
from scmle.pipeline import engine
from scmle.runtime import logs
from scmle.runtime import state
from scmle.runtime import tracing
from scmle.serving import server

sdk = pytest.importorskip("opentelemetry.sdk.trace")
memory = pytest.importorskip(
    "opentelemetry.sdk.trace.export.in_memory_span_exporter"
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "deploy" / "local" / "golden"
PARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


@pytest.fixture(scope="module")
def golden():
    return parity.goldens(GOLDEN)


@pytest.fixture(scope="module")
def bundle(golden):
    return parity.handwrite(golden.spec)


@pytest.fixture
def exported():
    sink = memory.InMemorySpanExporter()
    assert tracing.TRACING.enable(sink, environ={"OTEL_SERVICE_NAME": "svc"})
    yield sink
    tracing.TRACING.disable()


def spans(sink):
    tracing.TRACING.flush()
    return {span.name: span for span in sink.get_finished_spans()}


def test_tracing_is_off_by_default_and_spans_are_no_ops():
    assert tracing.TRACING.active is False
    with tracing.TRACING.span("predict", {"scmle.rows": 1}):
        tracing.TRACING.annotate("key", 1)
    tracing.TRACING.flush()
    tracing.TRACING.disable()


def test_a_request_span_continues_the_callers_trace_and_parents_predict(
    exported, golden, bundle
):
    adapter = server.HttpServing(bundle)
    adapter.start()
    try:
        request = urllib.request.Request(
            f"{adapter.address}/predict",
            json.dumps({"records": golden.records[:2]}).encode(),
            {"Content-Type": "application/json", "traceparent": PARENT},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            assert response.status == 200
        urllib.request.urlopen(f"{adapter.address}/livez", timeout=20).close()
    finally:
        adapter.stop()
    found = spans(exported)
    assert set(found) == {"request", "predict"}
    outer, inner = found["request"], found["predict"]
    assert format(outer.context.trace_id, "032x") == PARENT.split("-")[1]
    assert format(outer.parent.span_id, "016x") == PARENT.split("-")[2]
    assert inner.parent.span_id == outer.context.span_id
    assert inner.context.trace_id == outer.context.trace_id
    assert dict(outer.attributes) == {
        "http.request.method": "POST",
        "url.path": "/predict",
        "http.response.status_code": 200,
    }
    assert dict(inner.attributes) == {
        "scmle.rows": 2,
        "scmle.model_version": "golden-1",
    }
    assert outer.resource.attributes["service.name"] == "svc"


def test_a_failing_span_records_the_error(exported):
    with (
        pytest.raises(KeyError),
        tracing.TRACING.span("stage Broken", {"scmle.stage": "Broken"}),
    ):
        raise KeyError("missing")
    found = spans(exported)["stage Broken"]
    assert found.status.status_code.name == "ERROR"
    assert [event.name for event in found.events] == ["exception"]


def test_pipeline_stages_and_batch_files_are_spans(
    exported, golden, bundle, tmp_path
):
    class Board:
        regulator = type("R", (), {"cap": contextlib.nullcontext})()

        def check_cancelled(self):
            """Never cancelled."""

    class Load:
        state = state.Progress.VALIDATING
        requires = ()
        makes = None

        def describe(self, context):
            return "load"

        def execute(self, context):
            return None

    engine.Pipeline((Load(),)).advance(Board())
    source = tmp_path / "records.csv"
    pandas.DataFrame(golden.records).to_csv(source, index=False)
    server.Batcher(bundle).score_file(source, tmp_path / "out.csv", 3)
    tracing.TRACING.flush()
    finished = exported.get_finished_spans()
    names = [span.name for span in finished]
    assert names.count("stage Load") == 1
    assert names.count("batch") == 1
    rows = len(golden.records)
    assert names.count("predict") == -(-rows // 3)
    batch = next(span for span in finished if span.name == "batch")
    assert all(
        span.parent.span_id == batch.context.span_id
        for span in finished
        if span.name == "predict"
    )


def test_without_the_extra_enabling_degrades_to_no_ops(monkeypatch):
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk", None)
    with pytest.warns(logs.Degraded, match="'tracing' extra"):
        assert tracing.TRACING.enable() is False
    assert tracing.TRACING.active is False


def test_the_cli_switch_enables_tracing_for_the_command(monkeypatch):
    seen = []
    monkeypatch.setattr(
        tracing.TRACING, "enable", lambda: seen.append("on") or True
    )
    monkeypatch.setattr(tracing.TRACING, "disable", lambda: seen.append("off"))
    code = cli.main(["--trace", "batch", "missing-bundle", "missing.csv"])
    assert code == cli.EXIT_FAILED
    assert seen == ["on", "off"]
    seen.clear()
    monkeypatch.setenv(tracing.SWITCH, "1")
    cli.main(["batch", "missing-bundle", "missing.csv"])
    assert seen == ["on", "off"]
