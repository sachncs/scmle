"""Metrics, logs, windowed signals, webhooks, classifier, cost."""

import json
import pathlib

import pytest

from scmle.contracts import observability
from scmle.operations import classify
from scmle.operations import cost
from scmle.operations import logs
from scmle.operations import metrics
from scmle.operations import notify
from scmle.operations import respond
from scmle.operations import signals
from scmle.security import audit
from tests import support_ops

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "ops"
STAMP = "2026-11-05T10:00:00+00:00"


def fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ---- metrics -----------------------------------------------------------


def make_service():
    registry = metrics.MetricsRegistry()
    return registry, metrics.ServiceMetrics(
        registry, "v1", "sha256:" + "a" * 64
    )


def test_counts_error_rates_availability_and_latency_are_kept_apart():
    registry, service = make_service()
    for status in [200] * 90 + [422] * 4 + [429] * 2 + [500] * 3 + [503]:
        service.record_request("/predict", status, 0.05)
    summary = service.summary()
    assert summary["requests"] == 100
    assert summary["server_error_rate"] == pytest.approx(0.04)
    assert summary["client_error_rate"] == pytest.approx(0.04)
    assert summary["availability"] == pytest.approx(0.94)
    assert summary["latency_p95_seconds"] <= 0.1
    assert registry.total(service.REQUESTS, status="500") == 3


def test_a_service_with_no_requests_reports_none_not_perfect():
    ignored, service = make_service()
    summary = service.summary()
    assert summary["availability"] is None and summary["requests"] == 0


def test_the_quantile_estimate_follows_the_buckets():
    registry = metrics.MetricsRegistry(buckets=(0.1, 1.0, 10.0))
    for value in [0.05] * 50 + [0.5] * 40 + [5.0] * 10:
        registry.observe("t", value)
    hist = registry.histogram("t")
    assert 0.1 <= hist.quantile(0.7) <= 1.0
    assert 1.0 < hist.quantile(0.99) <= 10.0
    assert metrics.MetricsRegistry().histogram("t") is None


def test_the_text_exposition_is_valid_prometheus():
    registry, service = make_service()
    service.record_request("/predict", 200, 0.02)
    service.record_request("/predict", 500, 0.3)
    service.record_resources()
    text = registry.render()
    assert text.endswith("\n")
    assert "# TYPE scmle_http_requests_total counter" in text
    assert (
        'scmle_http_requests_total{model_version="v1",route="/predict",'
        'status="500"} 1'
    ) in text
    assert 'le="+Inf"} 2' in text
    assert "scmle_http_request_duration_seconds_count" in text
    assert "scmle_process_resident_memory_peak_bytes" in text
    assert "artifact_digest=" in text
    for row in text.splitlines():
        if row and not row.startswith("#"):
            name_and_labels, value = row.rsplit(" ", 1)
            float(value.replace("+Inf", "inf"))
            assert name_and_labels[0].isalpha() or name_and_labels[0] == "_"


def test_the_registry_refuses_bad_names_labels_and_values():
    registry = metrics.MetricsRegistry()
    with pytest.raises(ValueError):
        registry.increment("1bad")
    with pytest.raises(ValueError):
        registry.increment("ok", {"__x": "y"})
    with pytest.raises(ValueError):
        registry.increment("ok", amount=-1)
    registry.increment("ok")
    with pytest.raises(ValueError):
        registry.set("ok", 1.0)
    with pytest.raises(ValueError):
        registry.observe("h", float("nan"))
    with pytest.raises(ValueError):
        metrics.MetricsRegistry(buckets=(1.0, 0.5))
    registry.set("g", 1.0, {"a": 'x"\n\\y'})
    assert 'a="x\\"\\n\\\\y"' in registry.render()


# ---- logs --------------------------------------------------------------


def test_logs_carry_trace_version_and_digest_and_never_a_secret():
    lines = []
    log = logs.StructuredLog(
        "serve", "v1", "sha256:" + "a" * 64, sink=lines.append
    )
    trace = logs.new_trace_id()
    log.request(
        trace,
        "/predict",
        500,
        0.25,
        authorization="Bearer not-a-digest-zq",
        n=3,
    )
    record = json.loads(lines[0])
    assert record["trace_id"] == trace and record["level"] == "error"
    assert record["model_version"] == "v1"
    assert record["artifact_digest"].startswith("sha256:")
    assert record["duration_ms"] == 250.0
    assert record["authorization"] == logs.REDACTED
    assert "not-a-digest-zq" not in lines[0]
    nested = log.emit("info", "x", None, context={"api_key": "k", "ok": 1})
    assert nested["context"] == {"api_key": logs.REDACTED, "ok": 1}


def test_logs_refuse_reserved_fields_and_unknown_levels(tmp_path):
    log = logs.StructuredLog("s", "v", "d", path=tmp_path / "x.log")
    with pytest.raises(ValueError):
        log.emit("info", "m", model_version="other")
    with pytest.raises(ValueError):
        log.emit("loud", "m")
    log.emit("info", "written")
    assert json.loads((tmp_path / "x.log").read_text())["message"] == "written"
    with pytest.raises(ValueError):
        logs.StructuredLog("s", "v", "d")


def test_traceparent_round_trips_and_bad_headers_get_a_new_id():
    trace = logs.new_trace_id()
    header = logs.traceparent_for(trace)
    assert logs.parse_traceparent(header) == trace
    assert logs.trace_for(header) == trace
    for bad in (None, "", "00-xyz", f"00-{'0' * 32}-{'1' * 16}-01"):
        assert logs.parse_traceparent(bad) is None
        assert len(logs.trace_for(bad)) == 32
    with pytest.raises(ValueError):
        logs.traceparent_for("short")


# ---- windowed signals --------------------------------------------------


def test_healthy_traffic_and_data_breach_nothing():
    watch = support_ops.monitor()
    support_ops.healthy_traffic(watch)
    watch.observe_features(support_ops.START, support_ops.frame(300, seed=1))
    watch.observe_outcomes(support_ops.START, [0.0] * 90 + [1.0] * 10)
    rows = watch.evaluate(support_ops.START + 20)
    assert not [r for r in rows if r.breached], [r.to_dict() for r in rows]
    assert classify.classify(rows).verdict is classify.Verdict.NONE


def test_too_little_data_is_reported_not_called_healthy():
    watch = support_ops.monitor(minimum=50)
    support_ops.healthy_traffic(watch, count=10)
    rows = watch.evaluate(support_ops.START + 5)
    assert all(r.status is signals.Status.INSUFFICIENT_DATA for r in rows)
    assert classify.classify(rows).verdict is classify.Verdict.INDETERMINATE


def test_old_events_leave_the_window():
    watch = support_ops.monitor(window=60.0)
    support_ops.healthy_traffic(watch, status=500, count=100)
    assert named(
        watch.evaluate(support_ops.START + 20), "server_error_rate"
    ).breached
    later = named(watch.evaluate(support_ops.START + 500), "server_error_rate")
    assert later.status is signals.Status.INSUFFICIENT_DATA


def named(rows, name):
    return support_ops.named(rows, name)


def test_feature_quality_and_drift_are_each_detected():
    watch = support_ops.monitor()
    watch.observe_features(
        support_ops.START, support_ops.frame(400, seed=2, shift=25.0)
    )
    rows = watch.evaluate(support_ops.START + 1)
    assert named(rows, "feature_psi_max").breached
    assert named(rows, "feature_psi_max").detail["worst"][0][0] in {
        "age",
        "balance",
    }
    watch = support_ops.monitor()
    watch.observe_features(
        support_ops.START, support_ops.frame(400, seed=3, missing=0.4)
    )
    rows = watch.evaluate(support_ops.START + 1)
    assert named(rows, "feature_missing_increase").breached
    assert named(rows, "feature_missing_increase").detail["features"] == ["age"]
    watch = support_ops.monitor()
    watch.observe_features(
        support_ops.START, support_ops.frame(400, seed=4, unseen=0.5)
    )
    assert named(
        watch.evaluate(support_ops.START + 1), "unseen_category_rate"
    ).breached


def test_a_feature_missing_from_the_batch_counts_as_missing():
    watch = support_ops.monitor()
    watch.observe_features(
        support_ops.START,
        support_ops.frame(300, seed=5).drop(columns=["balance"]),
    )
    row = named(
        watch.evaluate(support_ops.START + 1), "feature_missing_increase"
    )
    assert row.breached and "balance" in row.detail["features"]


def test_delayed_label_loss_breaches_above_reference_plus_tolerance():
    watch = support_ops.monitor()
    watch.observe_outcomes(support_ops.START, [1.0] * 40 + [0.0] * 60)
    row = named(watch.evaluate(support_ops.START + 1), "windowed_loss")
    assert row.breached and row.value == pytest.approx(0.4)
    assert row.threshold == pytest.approx(0.15)
    with pytest.raises(ValueError):
        watch.observe_outcomes(support_ops.START, [1.5])
    assert list(signals.zero_one_losses([1, 0, 1], [1, 1, 1])) == [0, 1, 0]
    with pytest.raises(ValueError):
        signals.zero_one_losses([1], [1, 2])


def test_sampling_keeps_a_share_and_is_deterministic():
    kept = []
    for ignored in range(2):
        watch = signals.SignalMonitor(
            signals.WindowPolicy(sample_rate=0.25, seed=7),
            profile=support_ops.profile(),
        )
        kept.append(
            watch.observe_features(
                support_ops.START, support_ops.frame(4000, seed=1)
            )
        )
    assert kept[0] == kept[1] and 800 < kept[0] < 1200
    with pytest.raises(ValueError):
        signals.WindowPolicy(sample_rate=0.0)
    with pytest.raises(ValueError):
        signals.WindowPolicy(window_seconds=0)


def test_signals_become_metric_samples_for_export():
    watch = support_ops.monitor()
    support_ops.healthy_traffic(watch)
    samples = signals.as_metric_samples(
        watch.evaluate(support_ops.START + 20), STAMP, {"model_version": "v1"}
    )
    names = {s.name for s in samples}
    assert {"server_error_rate", "availability", "latency_p95_seconds"} <= names
    unit = {s.name: s.unit for s in samples}
    assert (
        unit["latency_p95_seconds"] == "s" and unit["availability"] == "ratio"
    )


# ---- samples -----------------------------------------------------------


def samples():
    return [
        observability.MetricSample(
            "latency_p95_seconds",
            observability.SignalKind.LATENCY,
            0.35,
            "s",
            STAMP,
            {"model_version": "v1"},
        ),
        observability.MetricSample(
            "server_error_rate",
            observability.SignalKind.ERROR,
            0.02,
            "ratio",
            STAMP,
            {"model_version": "v1"},
        ),
    ]


# ---- webhooks ----------------------------------------------------------


NOTE = {"severity": "critical", "summary": "down", "incident_id": "inc-1"}


def sender(script, kind="pager", attempts=3, **kwargs):
    seen = []
    waits = []

    def transport(request):
        seen.append(request)
        step = script[min(len(seen) - 1, len(script) - 1)]
        if isinstance(step, Exception):
            raise step
        return step

    made = notify.WebhookSender(
        kind,
        "https://hooks.example.test/x",
        "shared-secret",
        transport,
        max_attempts=attempts,
        sleep=waits.append,
        clock=lambda: 1000.0,
        **kwargs,
    )
    return made, seen, waits


def test_a_delivery_is_signed_and_the_receiver_can_verify_it():
    made, seen, waits = sender([202])
    result = made.send(NOTE)
    assert result.delivered and result.attempts == 1 and not waits
    request = seen[0]
    assert notify.verify_signature(
        "shared-secret", request.headers, request.body, 1000.0
    )
    assert not notify.verify_signature(
        "other", request.headers, request.body, 1000.0
    )
    assert not notify.verify_signature(
        "shared-secret", request.headers, request.body + b" ", 1000.0
    )
    assert not notify.verify_signature(
        "shared-secret", request.headers, request.body, 5000.0
    )
    assert not notify.verify_signature(
        "shared-secret", {}, request.body, 1000.0
    )
    assert json.loads(request.body)["kind"] == "pager"
    assert "shared-secret" not in json.dumps(result.to_dict())


def test_server_errors_and_timeouts_are_retried_with_one_idempotency_key():
    made, seen, waits = sender([503, TimeoutError("slow"), 200])
    result = made.send(NOTE)
    assert result.delivered and result.attempts == 3
    assert waits == [0.5, 1.0]
    keys = {r.headers[notify.IDEMPOTENCY_HEADER] for r in seen}
    assert keys == {result.idempotency_key}


def test_client_errors_are_final_and_exhausted_retries_are_reported():
    made, ignored, ignored = sender([400])
    result = made.send(NOTE)
    assert (
        not result.delivered
        and result.attempts == 1
        and result.http_status == 400
    )
    made, ignored, ignored = sender([500])
    result = made.send(NOTE)
    assert not result.delivered and result.attempts == 3
    assert "after 3 attempts" in result.reason
    made, ignored, ignored = sender([ConnectionError("down")])
    result = made.send(NOTE)
    assert not result.delivered and result.http_status is None
    made, ignored, ignored = sender([429, 200])
    assert made.send(NOTE).delivered


def test_the_sender_rejects_bad_configuration_and_notifications():
    with pytest.raises(ValueError):
        notify.WebhookSender("sms", "https://x", "k")
    with pytest.raises(ValueError):
        notify.WebhookSender("pager", "https://x", "")
    with pytest.raises(ValueError):
        notify.WebhookSender("pager", "https://x", "k", max_attempts=0)
    made, ignored, ignored = sender([200])
    with pytest.raises(ValueError):
        made.send({"severity": "apocalyptic"})
    with pytest.raises(ValueError):
        made.send({"severity": "info", "blob": "x" * 70000})
    plain = notify.urllib_transport()
    with pytest.raises(ValueError):
        plain(notify.Request("http://insecure.test", {}, b"{}"))


def test_routing_follows_severity_and_reports_a_missing_sender():
    pager, ignored, ignored = sender([200], "pager")
    ticket, ignored, ignored = sender([200], "ticket")
    both = notify.Notifier(pager, ticket)
    assert [d.kind for d in both.notify(NOTE)] == ["pager", "ticket"]
    warning = {**NOTE, "severity": "warning"}
    assert [d.kind for d in both.notify(warning)] == ["ticket"]
    assert both.notify({**NOTE, "severity": "info"}) == []
    missing = notify.Notifier(None, ticket).notify(NOTE)
    assert [d.delivered for d in missing] == [False, True]
    assert missing[0].reason == "no sender configured"


# ---- classifier --------------------------------------------------------


def window(**breach):
    """Signals where the named ones are breached and the rest are fine."""
    rows = []
    for name, category, threshold, direction in [
        ("server_error_rate", signals.Category.INFRASTRUCTURE, 0.05, "max"),
        ("availability", signals.Category.INFRASTRUCTURE, 0.99, "min"),
        ("latency_p95_seconds", signals.Category.INFRASTRUCTURE, 1.0, "max"),
        ("invalid_request_rate", signals.Category.DATA_QUALITY, 0.05, "max"),
        (
            "feature_missing_increase",
            signals.Category.DATA_QUALITY,
            0.05,
            "max",
        ),
        ("unseen_category_rate", signals.Category.DATA_QUALITY, 0.2, "max"),
        ("feature_psi_max", signals.Category.DATA_QUALITY, 0.25, "max"),
        ("windowed_loss", signals.Category.MODEL_PERFORMANCE, 0.15, "max"),
    ]:
        state = breach.get(name, "ok")
        status = {
            "ok": signals.Status.OK,
            "breach": signals.Status.BREACHED,
            "unseen": signals.Status.INSUFFICIENT_DATA,
        }[state]
        value = (
            None
            if state == "unseen"
            else (threshold * 2 if state == "breach" else threshold / 2)
        )
        rows.append(
            signals.Signal(
                name, category, status, value, threshold, direction, 100
            )
        )
    return rows


def test_each_kind_of_fault_gets_its_own_verdict():
    infra = classify.classify(window(server_error_rate="breach"))
    assert infra.verdict is classify.Verdict.INFRASTRUCTURE
    assert infra.severity == "warning"
    outage = classify.classify(
        window(availability="breach", server_error_rate="breach")
    )
    assert (
        outage.verdict is classify.Verdict.INFRASTRUCTURE
        and outage.severity == "critical"
    )
    data = classify.classify(window(feature_missing_increase="breach"))
    assert data.verdict is classify.Verdict.DATA_QUALITY
    model = classify.classify(window(windowed_loss="breach"))
    assert model.verdict is classify.Verdict.MODEL_PERFORMANCE
    assert "escalation" in model.response
    assert classify.classify(window()).verdict is classify.Verdict.NONE


def test_ambiguous_cases_are_mixed_not_guessed():
    drifted_and_worse = classify.classify(
        window(feature_psi_max="breach", windowed_loss="breach")
    )
    assert drifted_and_worse.verdict is classify.Verdict.MIXED
    assert set(drifted_and_worse.breached) == {
        "feature_psi_max",
        "windowed_loss",
    }
    assert len(drifted_and_worse.categories) == 2
    errors_and_bad_inputs = classify.classify(
        window(server_error_rate="breach", invalid_request_rate="breach")
    )
    assert errors_and_bad_inputs.verdict is classify.Verdict.MIXED
    assert errors_and_bad_inputs.severity == "critical"
    all_three = classify.classify(
        window(
            availability="breach",
            feature_missing_increase="breach",
            windowed_loss="breach",
        )
    )
    assert all_three.verdict is classify.Verdict.MIXED
    assert len(all_three.categories) == 3


def test_caveats_say_what_a_verdict_does_not_establish():
    drift_only = classify.classify(
        window(feature_psi_max="breach", windowed_loss="unseen")
    )
    assert drift_only.verdict is classify.Verdict.DATA_QUALITY
    assert any("drift is not harm" in c for c in drift_only.caveats)
    blind = classify.classify(
        window(windowed_loss="breach", feature_psi_max="unseen")
    )
    assert blind.verdict is classify.Verdict.MODEL_PERFORMANCE
    assert any("data cause is not excluded" in c for c in blind.caveats)
    outage = classify.classify(
        window(server_error_rate="breach", windowed_loss="unseen")
    )
    assert any("model quality was not observed" in c for c in outage.caveats)
    clean = classify.classify(window(windowed_loss="breach"))
    assert not clean.caveats


def test_nothing_judged_is_indeterminate_and_a_partial_view_is_not_healthy():
    all_unseen = classify.classify(
        [
            signals.Signal(
                "x",
                signals.Category.INFRASTRUCTURE,
                signals.Status.INSUFFICIENT_DATA,
                None,
                1.0,
                "max",
                0,
            )
        ]
    )
    assert all_unseen.verdict is classify.Verdict.INDETERMINATE
    partial = classify.classify(window(windowed_loss="unseen"))
    assert partial.verdict is classify.Verdict.NONE
    assert partial.unobserved == ("windowed_loss",)


def test_the_notification_carries_trace_version_and_digest():
    found = classify.classify(window(server_error_rate="breach"))
    body = classify.notification(
        found, "inc-1", "v1", "sha256:" + "a" * 64, "t" * 32, STAMP
    )
    assert body["verdict"] == "infrastructure" and body["trace_id"] == "t" * 32
    assert body["model_version"] == "v1" and body["severity"] == "warning"
    assert (
        json.loads(json.dumps(found.to_dict()))["verdict"] == "infrastructure"
    )


# ---- cost --------------------------------------------------------------


RATES = cost.Rates(0.000024, 0.0000025, 0.40, 0.12, source="declared in test")


def test_cost_is_grouped_by_service_and_version_from_declared_rates():
    usages = [
        cost.Usage.from_allocation("serve", "v1", 1.0, 2.0, 3600, 100000, 1.5),
        cost.Usage.from_allocation("serve", "v2", 1.0, 2.0, 1800, 50000, 0.5),
        cost.Usage.from_allocation("serve", "v1", 1.0, 2.0, 3600),
        cost.Usage.from_allocation("batch", "v1", 4.0, 8.0, 600),
    ]
    out = cost.report(usages, RATES)
    serve = out["services"]["serve"]
    assert set(serve["versions"]) == {"v1", "v2"}
    v1 = serve["versions"]["v1"]
    assert v1["compute"] == pytest.approx(2 * 3600 * 0.000024, abs=1e-6)
    assert v1["requests"] == pytest.approx(40.0)
    assert v1["egress"] == pytest.approx(0.18)
    assert serve["total"] == pytest.approx(
        sum(v["total"] for v in serve["versions"].values())
    )
    assert out["total"] == pytest.approx(
        sum(s["total"] for s in out["services"].values())
    )
    assert "Not a bill" in out["statement"] and out["rates"]["source"]


def test_rates_and_usage_are_validated():
    with pytest.raises(ValueError):
        cost.Rates(-1.0, 0.0, source="x")
    with pytest.raises(ValueError):
        cost.Rates(1.0, 1.0, source="  ")
    with pytest.raises(ValueError):
        cost.Usage("s", "v", float("nan"), 0.0)
    assert cost.report([], RATES)["total"] == 0.0


# ---- the desk: classify, notify, retain, audit -------------------------


def make_desk(tmp_path, script=(200,)):
    pager, ignored, ignored = sender(list(script), "pager")
    ticket, ignored, ignored = sender(list(script), "ticket")
    log = audit.AuditLog.open(tmp_path / "audit.jsonl")
    desk = respond.ResponseDesk(
        tmp_path / "ops",
        log,
        notify.Notifier(pager, ticket),
        "v1",
        "sha256:" + "a" * 64,
        clock=lambda: 2000.0,
    )
    return desk, log


def test_a_breach_is_classified_notified_retained_and_audited(tmp_path):
    desk, log = make_desk(tmp_path)
    result = desk.handle(window(availability="breach"))
    assert result.incident_id.startswith("inc-")
    assert result.classification.verdict is classify.Verdict.INFRASTRUCTURE
    assert [d.kind for d in result.deliveries] == ["pager", "ticket"]
    assert all(d.delivered for d in result.deliveries)
    kept = desk.retained()
    assert len(kept) == 1 and kept[0]["model_version"] == "v1"
    opened = log.rows("admin")[0]["body"]
    assert opened["action"] == "incident_opened"
    assert opened["incident_id"] == result.incident_id
    assert opened["telemetry_digest"] == result.telemetry_digest
    desk.act(result.incident_id, "oncall-1", "acknowledged", "restarting")
    desk.act(result.incident_id, "oncall-1", "resolved", "rolled back")
    actions = [r["body"]["action"] for r in log.rows("admin")]
    assert actions == [
        "incident_opened",
        "incident_acknowledged",
        "incident_resolved",
    ]
    log.ledger.verify()


def test_a_healthy_window_is_retained_but_opens_nothing(tmp_path):
    desk, log = make_desk(tmp_path)
    result = desk.handle(window())
    assert result.incident_id is None and not result.deliveries
    assert len(desk.retained()) == 1 and log.rows("admin") == []


def test_a_failed_page_is_recorded_not_hidden(tmp_path):
    desk, log = make_desk(tmp_path, script=(500,))
    result = desk.handle(window(availability="breach"))
    assert not any(d.delivered for d in result.deliveries)
    recorded = log.rows("admin")[0]["body"]["deliveries"]
    assert [d["delivered"] for d in recorded] == [False, False]


def test_operator_actions_need_a_known_incident_and_a_known_action(tmp_path):
    desk, ignored = make_desk(tmp_path)
    with pytest.raises(ValueError):
        desk.act("inc-nope", "oncall", "resolved", "x")
    result = desk.handle(window(availability="breach"))
    with pytest.raises(ValueError):
        desk.act(result.incident_id, "oncall", "ignored", "x")


def test_retained_telemetry_can_expire(tmp_path):
    desk, ignored = make_desk(tmp_path)
    desk.handle(window())
    assert desk.expire(older_than=1000.0) == 0
    assert desk.expire(older_than=3000.0) == 1 and desk.retained() == []


def test_a_notification_cannot_override_the_senders_kind():
    made, seen, ignored = sender([202], kind="pager")
    made.send({**NOTE, "kind": "ticket"})
    assert json.loads(seen[0].body)["kind"] == "pager"


def test_verification_rejects_a_nan_timestamp_and_non_ascii_signatures():
    made, seen, ignored = sender([202])
    made.send(NOTE)
    request = seen[0]
    body = request.body
    nan_stamp = "nan"
    forged = {
        notify.TIMESTAMP_HEADER: nan_stamp,
        notify.SIGNATURE_HEADER: notify.sign("shared-secret", nan_stamp, body),
    }
    assert not notify.verify_signature("shared-secret", forged, body, 1000.0)
    garbled = {**request.headers, notify.SIGNATURE_HEADER: "sha256=é"}
    assert not notify.verify_signature("shared-secret", garbled, body, 1000.0)


def test_signing_with_an_empty_secret_is_refused():
    with pytest.raises(ValueError, match="secret"):
        notify.sign("", "1000.000", b"{}")
    with pytest.raises(ValueError, match="secret"):
        notify.sign(b"", "1000.000", b"{}")
