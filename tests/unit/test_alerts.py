"""Operational alert routing and delivery evidence tests."""

from http import server
import json
from pathlib import Path
import threading

import pytest

from scmle.monitoring import alerts


def make_alert(route: str = "on-call") -> alerts.Alert:
    """Build deterministic alert evidence."""
    return alerts.Alert.from_evidence(
        route=route,
        severity="high",
        title="model SLO breached",
        details={"run_id": "run-1", "rollback_target": "model-0"},
        evidence={"slo_status": "breached", "observed_maximum": 0.9},
        created_at="2026-10-20T12:00:00+00:00",
    )


def test_dispatch_is_idempotent_and_journaled(tmp_path: Path):
    """A successful alert is not sent twice after a replay."""
    sink_path = tmp_path / "sink.jsonl"
    dispatcher = alerts.AlertDispatcher(
        {"on-call": alerts.JsonlAlertSink(sink_path)},
        tmp_path / "deliveries.jsonl",
    )
    alert = make_alert()

    first = dispatcher.dispatch(alert)
    second = dispatcher.dispatch(alert)

    assert first.status == "delivered"
    assert second.status == "deduplicated"
    assert len(sink_path.read_text().splitlines()) == 1
    records = [
        json.loads(line)
        for line in (tmp_path / "deliveries.jsonl").read_text().splitlines()
    ]
    assert records[0]["delivery"]["status"] == "delivered"


def test_missing_route_is_a_durable_failure(tmp_path: Path):
    """An unconfigured paging route cannot be reported as delivered."""
    dispatcher = alerts.AlertDispatcher(
        {"on-call": alerts.JsonlAlertSink(tmp_path / "sink.jsonl")},
        tmp_path / "deliveries.jsonl",
    )

    delivery = dispatcher.dispatch(make_alert("missing-route"))

    assert delivery.status == "failed"
    assert "no sink" in delivery.reason


def test_webhook_requires_https_except_for_explicit_local_test_mode():
    """Production webhook configuration rejects accidental plaintext URLs."""
    with pytest.raises(ValueError, match="HTTPS"):
        alerts.WebhookAlertSink("http://example.test/alerts")
    alerts.WebhookAlertSink(
        "http://127.0.0.1:8123/alerts", allow_insecure_localhost=True
    )


def test_webhook_delivers_json_with_bounded_attempts():
    """The webhook sink sends the signed alert payload and records success."""
    received: list[dict[str, object]] = []

    class Handler(server.BaseHTTPRequestHandler):
        """Small local receiver for a deterministic delivery test."""

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            received.append(json.loads(self.rfile.read(length)))
            self.send_response(204)
            self.end_headers()

        def log_message(self, format, *args):
            """Keep the test receiver silent."""

    receiver = server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=receiver.serve_forever, daemon=True)
    thread.start()
    try:
        sink = alerts.WebhookAlertSink(
            f"http://127.0.0.1:{receiver.server_port}/alerts",
            token="test-token",
            timeout_seconds=1.0,
            max_attempts=2,
            allow_insecure_localhost=True,
        )
        delivery = sink.deliver(make_alert())
    finally:
        receiver.shutdown()
        receiver.server_close()
        thread.join(timeout=2)

    assert delivery.status == "delivered"
    assert delivery.attempts == 1
    assert received[0]["alert_id"] == make_alert().alert_id
