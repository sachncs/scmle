"""Targeted tests for release-evaluation behaviours the main suite left open."""

import dataclasses
from fractions import Fraction
import types

import pytest

from scmle.contracts import evaluation
from scmle.contracts import evidence
from scmle.evaluation import release
from tests.unit.test_release_evaluation import make_index
from tests.unit.test_release_evaluation import make_policy
from tests.unit.test_release_evaluation import make_record


def make_finding(**changes):
    fields = {
        "kind": release.FindingKind.METRIC,
        "name": "m",
        "passed": True,
        "blocking": True,
        "reason": "because",
    }
    fields.update(changes)
    return release.ReleaseFinding(**fields)


def make_counts(**changes):
    fields = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "blocking": 0,
        "blocking_failures": 0,
    }
    fields.update(changes)
    return release.FindingCounts(**fields)


def make_outcome(findings=()):
    ordered = tuple(
        sorted(findings, key=lambda item: (item.kind.value, item.name))
    )
    reasons = tuple(
        item.reason for item in ordered if not item.passed and item.blocking
    )
    return release.ReleaseEvaluation("binary", 3, not reasons, ordered, reasons)


def passing_outcome():
    policy = make_policy()
    return release.evaluate_release(
        policy,
        {"roc_auc": 0.81},
        {"metric_uncertainty": {"estimate": 0.81}},
        {},
        make_index(policy),
    )


def sample_summary():
    findings = (
        make_finding(kind=release.FindingKind.METRIC, name="a"),
        make_finding(
            kind=release.FindingKind.FAIRNESS,
            name="b",
            passed=False,
            reason="bad",
        ),
        make_finding(
            kind=release.FindingKind.INPUT,
            name="c",
            passed=False,
            blocking=False,
            reason="soft",
        ),
    )
    return make_outcome(findings).readiness_summary()


def test_finite_real_accepts_only_finite_real_numbers():
    assert release.finite_real(2) == 2.0
    assert release.finite_real(Fraction(1, 2)) == 0.5
    assert release.finite_real(-1.5) == -1.5
    for value in (True, False, "1", None, [1], float("inf"), float("nan")):
        assert release.finite_real(value) is None
    assert release.finite_real(float("-inf")) is None


def test_normalize_measure_names_the_field_in_errors():
    assert release.normalize_measure("f", 3) == 3.0
    assert release.normalize_measure("f", "text") == "text"
    assert release.normalize_measure("f", None) is None
    with pytest.raises(TypeError, match=r"^f must not be bool$"):
        release.normalize_measure("f", True)
    with pytest.raises(ValueError, match=r"^f must be finite$"):
        release.normalize_measure("f", float("nan"))
    with pytest.raises(TypeError, match=r"^f must be numeric, text, or None$"):
        release.normalize_measure("f", [1])


def test_finding_rejects_blank_or_non_text_fields():
    for bad in ("   ", 5, None):
        with pytest.raises(ValueError, match=r"^name must not be empty$"):
            make_finding(name=bad)
        with pytest.raises(ValueError, match=r"^reason must not be empty$"):
            make_finding(reason=bad)
    for bad in ("  ", 7, None):
        with pytest.raises(ValueError, match="non-empty strings"):
            make_finding(evidence=("ok", bad))
    with pytest.raises(ValueError, match="duplicates"):
        make_finding(evidence=("a", "a"))
    with pytest.raises(TypeError, match="evidence must be a tuple"):
        make_finding(evidence=["a"])
    with pytest.raises(TypeError, match=r"^kind must be FindingKind$"):
        make_finding(kind="metric")
    with pytest.raises(TypeError, match=r"^passed must be bool$"):
        make_finding(passed=1)
    with pytest.raises(TypeError, match=r"^blocking must be bool$"):
        make_finding(blocking=0)


def test_finding_names_the_measure_field_that_is_wrong():
    with pytest.raises(TypeError, match=r"^observed must not be bool$"):
        make_finding(observed=True)
    with pytest.raises(ValueError, match=r"^expected must be finite$"):
        make_finding(expected=float("inf"))
    with pytest.raises(TypeError, match=r"^expected must be numeric"):
        make_finding(expected=[1])
    assert make_finding(observed=2, expected="x").observed == 2.0


def test_finding_to_dict_lists_every_field_and_skips_none_measures():
    bare = make_finding(evidence=("e1", "e2"))
    assert bare.to_dict() == {
        "kind": "metric",
        "name": "m",
        "passed": True,
        "blocking": True,
        "reason": "because",
        "evidence": ["e1", "e2"],
    }
    full = make_finding(observed=0.0, expected="text")
    assert full.to_dict() == {
        "kind": "metric",
        "name": "m",
        "passed": True,
        "blocking": True,
        "reason": "because",
        "evidence": [],
        "observed": 0.0,
        "expected": "text",
    }
    only_expected = make_finding(expected=0)
    assert "observed" not in only_expected.to_dict()
    assert only_expected.to_dict()["expected"] == 0.0


def test_counts_serialize_and_name_the_bad_field():
    counts = make_counts(
        total=5, passed=3, failed=2, blocking=4, blocking_failures=1
    )
    assert counts.to_dict() == {
        "total": 5,
        "passed": 3,
        "failed": 2,
        "blocking": 4,
        "blocking_failures": 1,
    }
    for name in make_counts().to_dict():
        with pytest.raises(
            TypeError, match=rf"^{name} must be a non-negative integer$"
        ):
            make_counts(**{name: "x"})
        with pytest.raises(
            TypeError, match=rf"^{name} must be a non-negative integer$"
        ):
            make_counts(**{name: 1.0})
    with pytest.raises(TypeError, match="non-negative integer"):
        make_counts(total=True, passed=1)
    with pytest.raises(ValueError, match=r"^total must be a non-negative"):
        make_counts(total=-1, passed=-1)
    with pytest.raises(ValueError, match=r"^total must equal passed plus"):
        make_counts(total=2, passed=1)
    with pytest.raises(ValueError, match=r"^blocking must not exceed total$"):
        make_counts(blocking=1)
    with pytest.raises(ValueError, match=r"^blocking_failures must not exceed"):
        make_counts(total=2, passed=2, blocking=2, blocking_failures=1)
    with pytest.raises(
        ValueError, match=r"^blocking_failures must not exceed blocking$"
    ):
        make_counts(total=2, failed=2, blocking=1, blocking_failures=2)


def test_check_count_values_rejects_bools_non_integers_and_negatives():
    release.check_count_values({"a": 0, "b": 3})
    with pytest.raises(TypeError, match=r"^a must be a non-negative integer$"):
        release.check_count_values({"a": True})
    with pytest.raises(TypeError, match=r"^a must be a non-negative integer$"):
        release.check_count_values({"a": 1.5})
    with pytest.raises(ValueError, match=r"^b must be a non-negative integer$"):
        release.check_count_values({"a": 1, "b": -1})


def test_check_kind_coverage_requires_every_kind_with_counts():
    full = {kind.value: make_counts() for kind in release.FindingKind}
    release.check_kind_coverage(full)
    with pytest.raises(TypeError, match=r"^counts_by_kind must be a mapping$"):
        release.check_kind_coverage([])
    with pytest.raises(
        ValueError, match=r"^counts_by_kind must contain every finding kind$"
    ):
        release.check_kind_coverage({"metric": make_counts()})
    wrong = dict(full)
    wrong["metric"] = 5
    with pytest.raises(
        TypeError, match=r"^counts_by_kind must map strings to FindingCounts$"
    ):
        release.check_kind_coverage(wrong)


def test_summary_validation_messages():
    summary = sample_summary()
    for bad in ("  ", 5, None):
        with pytest.raises(ValueError, match=r"^task must not be empty$"):
            dataclasses.replace(summary, task=bad)
    for bad in (-1, True, "1", 1.5):
        with pytest.raises(
            ValueError, match=r"^policy_schema_version must be non-negative$"
        ):
            dataclasses.replace(summary, policy_schema_version=bad)
    assert (
        dataclasses.replace(
            summary, policy_schema_version=0
        ).policy_schema_version
        == 0
    )
    with pytest.raises(TypeError, match=r"^status must be ReleaseStatus$"):
        dataclasses.replace(summary, status="blocked")
    for name in (
        "total_findings",
        "passed_findings",
        "failed_findings",
        "blocking_findings",
        "blocking_failures",
        "non_blocking_failures",
    ):
        with pytest.raises(
            TypeError, match=rf"^{name} must be a non-negative integer$"
        ):
            dataclasses.replace(summary, **{name: "x"})
    with pytest.raises(ValueError, match=r"^total_findings must equal passed"):
        dataclasses.replace(summary, total_findings=9)
    with pytest.raises(ValueError, match=r"^failed_findings must equal"):
        dataclasses.replace(summary, non_blocking_failures=5)
    with pytest.raises(ValueError, match=r"^blocking_failures must not exceed"):
        dataclasses.replace(summary, blocking_findings=0, blocking_failures=1)
    with pytest.raises(
        ValueError, match=r"^status does not match blocking_failures$"
    ):
        dataclasses.replace(summary, status=release.ReleaseStatus.READY)
    with pytest.raises(
        ValueError, match=r"^aggregate counts do not match counts_by_kind$"
    ):
        dataclasses.replace(summary, blocking_findings=3)
    assert summary.status is release.ReleaseStatus.BLOCKED


def test_summary_serializes_exactly():
    summary = sample_summary()
    expected_kinds = {
        kind.value: summary.counts_by_kind[kind.value].to_dict()
        for kind in release.FindingKind
    }
    assert summary.to_dict() == {
        "task": "binary",
        "policy_schema_version": 3,
        "status": "blocked",
        "total_findings": 3,
        "passed_findings": 1,
        "failed_findings": 2,
        "blocking_findings": 2,
        "blocking_failures": 1,
        "non_blocking_failures": 1,
        "counts_by_kind": expected_kinds,
    }
    assert list(summary.to_dict()["counts_by_kind"]) == sorted(expected_kinds)
    text = summary.canonical_json()
    assert text.startswith('{"blocking_failures":1,"blocking_findings":2,')
    assert text.endswith('"task":"binary","total_findings":3}')
    assert ", " not in text
    assert ": " not in text


def test_evaluation_validation_and_exact_serialization():
    first = make_finding(name="naïve", observed=1.5)
    second = make_finding(
        kind=release.FindingKind.INPUT, name="z", passed=False, reason="badé"
    )
    outcome = make_outcome((first, second))
    assert outcome.blocked is True
    assert outcome.to_dict() == {
        "task": "binary",
        "policy_schema_version": 3,
        "passed": False,
        "blocked": True,
        "blocking_reasons": ["badé"],
        "findings": [item.to_dict() for item in outcome.findings],
        "readiness": outcome.readiness_summary().to_dict(),
    }
    text = outcome.canonical_json()
    assert text.startswith('{"blocked":true,"blocking_reasons":["bad\\u00e9"]')
    assert "\\u00ef" in text
    assert ", " not in text
    assert text.index('"findings"') < text.index('"passed"')
    assert text.index('"passed"') < text.index('"policy_schema_version"')
    assert text.index('"policy_schema_version"') < text.index('"readiness"')

    for bad in ("  ", 5, None):
        with pytest.raises(ValueError, match=r"^task must not be empty$"):
            release.ReleaseEvaluation(bad, 0, True, (), ())
    with pytest.raises(ValueError, match="must not be negative"):
        release.ReleaseEvaluation("t", -1, True, (), ())
    assert release.ReleaseEvaluation("t", 0, True, (), ()).blocked is False


def test_reliability_gate_requires_both_status_and_unblocked_flag():
    outcome = passing_outcome()
    ok = {"execution_status": "passed", "release_blocked": False}
    assert release.apply_reliability_gate(outcome, ok) is outcome
    for plan in (
        {"execution_status": "passed", "release_blocked": True},
        {"execution_status": "failed", "release_blocked": False},
        {"execution_status": "passed"},
    ):
        assert not release.apply_reliability_gate(outcome, plan).passed
    with pytest.raises(TypeError, match=r"^outcome must be ReleaseEvaluation$"):
        release.apply_reliability_gate("x", ok)


def test_reliability_gate_finding_is_exact():
    outcome = make_outcome(
        (make_finding(kind=release.FindingKind.SECURITY, name="zzz"),)
    )
    blocked = release.apply_reliability_gate(
        outcome, {"execution_status": "running"}
    )
    assert [item.name for item in blocked.findings] == [
        "fault_injection",
        "zzz",
    ]
    assert blocked.findings[0].to_dict() == {
        "kind": "reliability",
        "name": "fault_injection",
        "passed": False,
        "blocking": True,
        "reason": (
            "fault-injection recovery evidence is not complete; release blocked"
        ),
        "observed": "running",
        "expected": "execution_status=passed and release_blocked=false",
        "evidence": ["fault_injection_plan.json"],
    }
    assert blocked.blocking_reasons == (blocked.findings[0].reason,)
    assert blocked.task == "binary"
    assert blocked.policy_schema_version == 3
    for plan, observed in ((None, "missing"), ("x", "missing"), ({}, "None")):
        finding = release.apply_reliability_gate(outcome, plan).findings[0]
        assert finding.observed == observed


def test_reliability_gate_keeps_passed_blocking_findings_out_of_reasons():
    outcome = passing_outcome()
    blocked = release.apply_reliability_gate(outcome, None)
    assert len(blocked.blocking_reasons) == 1
    assert len(blocked.findings) == len(outcome.findings) + 1


def attestation_record(**signature):
    base = {"algorithm": "hmac-sha256", "value": "abc"}
    base.update(signature)
    return {"signature": base}


def test_attestation_gate_accepts_only_a_valid_signature_block():
    outcome = passing_outcome()
    assert (
        release.apply_attestation_gate(outcome, attestation_record()) is outcome
    )
    assert (
        release.apply_attestation_gate(outcome, attestation_record(), True)
        is outcome
    )
    for record in (
        attestation_record(algorithm="md5"),
        attestation_record(value="  "),
        attestation_record(value=5),
        {"signature": "text"},
        {"other": 1},
        "text",
        None,
    ):
        assert not release.apply_attestation_gate(outcome, record).passed
    assert not release.apply_attestation_gate(
        outcome, attestation_record(), False
    ).passed
    with pytest.raises(TypeError, match=r"^outcome must be ReleaseEvaluation$"):
        release.apply_attestation_gate(None, attestation_record())


def test_attestation_gate_finding_is_exact():
    outcome = make_outcome(
        (make_finding(kind=release.FindingKind.SECURITY, name="zzz"),)
    )
    blocked = release.apply_attestation_gate(
        outcome, attestation_record(), False
    )
    assert [item.name for item in blocked.findings] == [
        "release_attestation",
        "zzz",
    ]
    assert blocked.findings[0].to_dict() == {
        "kind": "security",
        "name": "release_attestation",
        "passed": False,
        "blocking": True,
        "reason": "release attestation is missing or invalid; release blocked",
        "observed": "invalid",
        "expected": "verified hmac-sha256 attestation signature",
        "evidence": ["release_attestation.json"],
    }
    assert blocked.blocking_reasons == (blocked.findings[0].reason,)
    observed = {}
    for label, record, verified in (
        ("signed_none", {"signature": {"algorithm": "x"}}, None),
        ("signed_true", {"signature": {"algorithm": "x"}}, True),
        ("unsigned", {}, None),
        ("unsigned_false", {}, False),
        ("none_record", None, False),
    ):
        finding = release.apply_attestation_gate(
            outcome, record, verified
        ).findings[0]
        observed[label] = finding.observed
    assert observed == {
        "signed_none": "signed",
        "signed_true": "signed",
        "unsigned": "unsigned",
        "unsigned_false": "unsigned",
        "none_record": "unsigned",
    }
    assert release.apply_attestation_gate(
        passing_outcome(), None
    ).blocking_reasons == (blocked.findings[0].reason,)


def test_evidence_index_gate_observed_states_and_exact_finding():
    outcome = make_outcome(
        (make_finding(kind=release.FindingKind.SECURITY, name="zzz"),)
    )
    signed = evidence.Index(signature="key")
    unsigned = evidence.Index()
    cases = (
        (signed, False, "invalid"),
        (unsigned, False, "unsigned"),
        (unsigned, True, "unsigned"),
        (None, False, "unsigned"),
        (None, True, "unsigned"),
        ("text", True, "unsigned"),
    )
    for index, verified, observed in cases:
        blocked = release.apply_evidence_index_gate(outcome, index, verified)
        assert not blocked.passed
        assert blocked.findings[0].observed == observed
    blocked = release.apply_evidence_index_gate(outcome, None, False)
    assert [item.name for item in blocked.findings] == [
        "release_evidence",
        "zzz",
    ]
    assert blocked.findings[0].to_dict() == {
        "kind": "security",
        "name": "release_evidence",
        "passed": False,
        "blocking": True,
        "reason": (
            "release evidence index is missing or has an invalid signature; "
            "release blocked"
        ),
        "observed": "unsigned",
        "expected": "verified hmac-sha256 evidence index signature",
        "evidence": ["release_evidence.json"],
    }
    assert blocked.blocking_reasons == (blocked.findings[0].reason,)
    good = passing_outcome()
    assert release.apply_evidence_index_gate(good, signed, True) is good
    assert len(
        release.apply_evidence_index_gate(good, None, True).findings
    ) == (len(good.findings) + 1)
    with pytest.raises(TypeError, match=r"^outcome must be ReleaseEvaluation$"):
        release.apply_evidence_index_gate(1, signed, True)


def test_blocked_input_finding_is_a_failed_blocking_input():
    finding = release.blocked_input_finding("n", "why")
    assert finding.to_dict() == {
        "kind": "input",
        "name": "n",
        "passed": False,
        "blocking": True,
        "reason": "why",
        "evidence": [],
    }


def test_fairness_metric_values_filters_and_counts_shared_nodes_once():
    shared = {"m": 0.5}
    report = {
        "a": shared,
        "b": shared,
        "c": [{"m": 0.25}, {"m": True}, {"m": "x"}, {"m": float("nan")}],
        "m": 0.75,
        "d": {"m": {"m": 9.0}},
        "e": ({"m": 1},),
    }
    assert release.fairness_metric_values(report, "m") == (
        0.25,
        0.5,
        0.75,
        1.0,
    )
    listed = [shared, shared]
    assert release.fairness_metric_values(listed, "m") == (0.5,)
    assert release.fairness_metric_values("text", "m") == ()


def test_explanation_quality_value_accepts_only_unit_interval_numbers():
    def report(value):
        return {"quality": {"score": value}}

    for value, expected in ((0, 0.0), (1, 1.0), (0.4, 0.4)):
        assert release.explanation_quality_value(report(value), "score") == (
            expected
        )
    for value in (True, "0.5", None, 1.5, -0.5, float("nan"), float("inf")):
        assert release.explanation_quality_value(report(value), "score") is None
    assert release.explanation_quality_value([], "score") is None
    assert release.explanation_quality_value({"quality": 1}, "score") is None
    assert release.explanation_quality_value({"quality": {}}, "score") is None


def test_metric_gate_finding_outcomes():
    maximize = evaluation.MetricGate(
        "auc", evaluation.MetricDirection.MAXIMIZE, 0.5
    )
    minimize = evaluation.MetricGate(
        "loss", evaluation.MetricDirection.MINIMIZE, 0.5
    )
    declared = {"auc", "loss"}

    def run(gate, value, names=declared):
        return release.metric_gate_finding(gate, names, {gate.metric: value})

    at_bound = run(maximize, 0.5).to_dict()
    assert at_bound == {
        "kind": "metric",
        "name": "auc",
        "passed": True,
        "blocking": True,
        "reason": "metric 'auc' passed: 0.5 >= 0.5",
        "evidence": [],
        "observed": 0.5,
        "expected": 0.5,
    }
    assert run(maximize, 0.25).to_dict()["reason"] == (
        "metric 'auc' failed: 0.25 >= 0.5"
    )
    assert run(minimize, 0.5).passed is True
    assert run(minimize, 0.5).reason == "metric 'loss' passed: 0.5 <= 0.5"
    assert run(minimize, 0.75).reason == "metric 'loss' failed: 0.75 <= 0.5"
    for value in (True, "0.5", None):
        finding = run(maximize, value)
        assert finding.reason == "metric 'auc' has an unsupported value"
        assert finding.expected == 0.5
        assert finding.observed is None
    with pytest.raises(ValueError, match=r"^observed must be finite$"):
        run(maximize, float("inf"))
    assert run(maximize, 0.9, names={"other"}).reason == (
        "metric 'auc' is unsupported or undeclared"
    )
    missing = release.metric_gate_finding(maximize, declared, {})
    assert missing.reason == "metric 'auc' is missing"
    assert missing.expected == 0.5


def test_metric_gate_with_a_raw_direction_fails_closed():
    gate = types.SimpleNamespace(metric="auc", direction="max", threshold=0.5)
    finding = release.metric_gate_finding(gate, {"auc"}, {"auc": 0.9})
    assert finding.passed is False
    assert finding.reason == "metric 'auc' failed: 0.9 ? 0.5"
    assert finding.observed == 0.9
    assert finding.expected == 0.5


def test_diagnostic_verdict_reads_the_first_verdict_key():
    assert release.diagnostic_verdict({"passed": True}) is True
    assert release.diagnostic_verdict({"passed": False}) is False
    assert release.diagnostic_verdict({"pass": False}) is False
    assert release.diagnostic_verdict({"ok": True}) is True
    assert release.diagnostic_verdict({"failed": True}) is False
    assert release.diagnostic_verdict({"failed": False}) is True
    assert release.diagnostic_verdict({"pass": 1, "ok": 2}) == 1
    assert release.diagnostic_verdict({"ok": 2, "failed": True}) == 2
    assert release.diagnostic_verdict({"x": 1}) is None
    for text in ("pass", "PASSED", "Ok", "success"):
        assert release.diagnostic_verdict({"status": text}) is True
    assert release.diagnostic_verdict({"status": "fail"}) is False
    assert release.diagnostic_verdict({"status": 5}) == 5
    assert (
        release.diagnostic_verdict({"passed": True, "status": "fail"}) is True
    )
    assert release.diagnostic_verdict({"failed": False, "status": "x"}) is True


def make_gate(action=evaluation.DiagnosticAction.BLOCK):
    return evaluation.DiagnosticGate("diag", action, ("e1",))


def diagnose(report, action=evaluation.DiagnosticAction.BLOCK):
    reports = {} if report is None else {"diag": report}
    return release.diagnostic_gate_finding(make_gate(action), reports)


def test_diagnostic_gate_missing_and_unsupported_reports():
    warn = evaluation.DiagnosticAction.WARN
    missing = diagnose(None, warn)
    assert missing.to_dict() == {
        "kind": "diagnostic",
        "name": "diag",
        "passed": False,
        "blocking": False,
        "reason": "diagnostic 'diag' is missing",
        "evidence": ["e1"],
        "expected": "report without error",
    }
    for report in ({}, [1], "text", 0):
        finding = diagnose(report, warn)
        assert finding.to_dict() == {
            "kind": "diagnostic",
            "name": "diag",
            "passed": False,
            "blocking": True,
            "reason": "diagnostic 'diag' has an unsupported report",
            "evidence": ["e1"],
            "expected": "non-empty mapping",
        }


def test_diagnostic_gate_judges_errors_availability_and_verdicts():
    warn = evaluation.DiagnosticAction.WARN
    present = diagnose({"note": 1}, warn)
    assert present.to_dict() == {
        "kind": "diagnostic",
        "name": "diag",
        "passed": True,
        "blocking": False,
        "reason": "diagnostic 'diag' is present",
        "evidence": ["e1"],
        "expected": "report without error",
    }
    assert diagnose({"error": "boom"}, warn).reason == (
        "diagnostic 'diag' reported an error"
    )
    assert diagnose({"error": True}).passed is False
    for quiet in ("", None, False):
        assert diagnose({"error": quiet}).reason == (
            "diagnostic 'diag' is present"
        )
    assert diagnose({"available": False}).reason == (
        "diagnostic 'diag' is unavailable"
    )
    assert diagnose({"available": True}).passed is True
    assert diagnose({"available": True}).reason == (
        "diagnostic 'diag' is present"
    )
    assert diagnose({"passed": False}).reason == (
        "diagnostic 'diag' reported failure"
    )
    assert diagnose({"passed": True}).reason == "diagnostic 'diag' is present"
    assert diagnose({"passed": True}).passed is True
    odd = diagnose({"passed": "yes"}, warn)
    assert odd.reason == "diagnostic 'diag' has an unsupported verdict"
    assert odd.blocking is True
    assert odd.expected == "boolean pass verdict"
    assert odd.evidence == ("e1",)
    failing = diagnose({"passed": False})
    assert failing.blocking is True
    assert failing.passed is False
    assert diagnose({"passed": False}, warn).blocking is False


def make_policy_namespace(**fields):
    explanations = types.SimpleNamespace(
        required_methods=("shap",),
        require_stability=True,
        stability_threshold=0.5,
        require_fidelity=True,
        fidelity_threshold=0.75,
    )
    for name, value in fields.items():
        setattr(explanations, name, value)
    return types.SimpleNamespace(explanations=explanations)


def test_explanation_findings_report_each_required_quality():
    policy = make_policy_namespace()
    reports = {"shap": {"quality": {"stability_score": 0.5}}}
    found = release.explanation_findings(policy, reports)
    assert [item.name for item in found] == ["shap.stability", "shap.fidelity"]
    assert found[0].to_dict() == {
        "kind": "explanation",
        "name": "shap.stability",
        "passed": True,
        "blocking": True,
        "reason": "explanation 'shap' stability passed: 0.5 >= 0.5",
        "evidence": [],
        "observed": 0.5,
        "expected": 0.5,
    }
    assert found[1].to_dict() == {
        "kind": "explanation",
        "name": "shap.fidelity",
        "passed": False,
        "blocking": True,
        "reason": "explanation 'shap' has no valid fidelity score",
        "evidence": [],
        "expected": 0.75,
    }


def test_explanation_findings_fail_a_score_below_the_threshold():
    policy = make_policy_namespace(require_stability=False)
    reports = {"shap": {"quality": {"fidelity_score": 0.5}}}
    found = release.explanation_findings(policy, reports)
    assert [item.name for item in found] == ["shap.fidelity"]
    assert found[0].passed is False
    assert found[0].reason == "explanation 'shap' fidelity failed: 0.5 >= 0.75"
    assert found[0].observed == 0.5
    assert found[0].expected == 0.75
    assert (
        release.explanation_findings(
            make_policy_namespace(
                require_stability=False, require_fidelity=False
            ),
            reports,
        )
        == []
    )


def test_explanation_findings_need_a_threshold_for_each_requirement():
    with pytest.raises(
        ValueError, match=r"^explanation 'shap' has no stability threshold$"
    ):
        release.explanation_findings(
            make_policy_namespace(stability_threshold=None), {}
        )
    with pytest.raises(
        ValueError, match=r"^explanation 'shap' has no fidelity threshold$"
    ):
        release.explanation_findings(
            make_policy_namespace(
                fidelity_threshold=None, require_stability=False
            ),
            {},
        )


def fairness_gate(**bounds):
    return evaluation.FairnessGate("gap", required_evidence=("fe",), **bounds)


def test_fairness_gate_maximum_uses_the_worst_value():
    gate = fairness_gate(maximum=0.5)
    reports = {"a": {"gap": 0.25}, "b": {"x": {"gap": 0.5}}}
    finding = release.fairness_gate_finding(gate, reports)
    assert finding.to_dict() == {
        "kind": "fairness",
        "name": "gap",
        "passed": True,
        "blocking": True,
        "reason": "fairness metric 'gap' passed: worst value 0.5 <= 0.5",
        "evidence": ["fe"],
        "observed": 0.5,
        "expected": 0.5,
    }
    worse = release.fairness_gate_finding(gate, {"a": {"gap": 0.75}})
    assert worse.passed is False
    assert worse.reason == (
        "fairness metric 'gap' failed: worst value 0.75 <= 0.5"
    )


def test_fairness_gate_minimum_uses_the_lowest_value():
    gate = fairness_gate(minimum=0.5)
    finding = release.fairness_gate_finding(
        gate, {"a": {"gap": 0.5}, "b": {"gap": 0.75}}
    )
    assert finding.passed is True
    assert finding.reason == (
        "fairness metric 'gap' passed: worst value 0.5 >= 0.5"
    )
    assert finding.observed == 0.5
    assert finding.expected == 0.5
    low = release.fairness_gate_finding(gate, {"a": {"gap": 0.25}})
    assert low.passed is False
    assert low.reason == "fairness metric 'gap' failed: worst value 0.25 >= 0.5"


def test_fairness_gate_reads_a_direct_report_and_mixed_report_keys():
    gate = fairness_gate(maximum=0.5)
    direct = {"gap": 0.25, "other": {"x": 1.0}}
    finding = release.fairness_gate_finding(gate, direct)
    assert finding.observed == 0.25
    mixed = {1: {"gap": 0.25}, "z": {"gap": 0.5}}
    assert release.fairness_gate_finding(gate, mixed).observed == 0.5
    nested = {"report": {"gap": 0.25}}
    assert release.fairness_gate_finding(gate, nested).observed == 0.25


def test_fairness_gate_without_values_or_bound_fails_closed():
    maximum = release.fairness_gate_finding(fairness_gate(maximum=0.5), {})
    assert maximum.to_dict() == {
        "kind": "fairness",
        "name": "gap",
        "passed": False,
        "blocking": True,
        "reason": "fairness metric 'gap' is missing or unsupported",
        "evidence": ["fe"],
        "expected": 0.5,
    }
    minimum = release.fairness_gate_finding(fairness_gate(minimum=0.25), {})
    assert minimum.expected == 0.25
    unbounded = types.SimpleNamespace(
        metric="gap", maximum=None, minimum=None, required_evidence=("fe",)
    )
    finding = release.fairness_gate_finding(unbounded, {"a": {"gap": 1.0}})
    assert finding.to_dict() == {
        "kind": "fairness",
        "name": "gap",
        "passed": False,
        "blocking": True,
        "reason": "fairness metric 'gap' has no bound",
        "evidence": ["fe"],
    }


def test_evidence_reference_finding_states():
    index = evidence.Index(
        [
            make_record("good"),
            make_record("bad", result=evidence.Outcome.FAIL),
            make_record("held", decision=evidence.Disposition.BLOCKED),
            make_record("waived", result=evidence.Outcome.WAIVED),
        ]
    )
    good = release.evidence_reference_finding("good", index)
    assert good.to_dict() == {
        "kind": "evidence",
        "name": "good",
        "passed": True,
        "blocking": True,
        "reason": "required evidence 'good' is accepted",
        "evidence": ["good"],
        "observed": "pass:accepted",
        "expected": "pass:accepted",
    }
    bad = release.evidence_reference_finding("bad", index)
    assert bad.passed is False
    assert (
        bad.reason == "required evidence 'bad' is not accepted: fail:accepted"
    )
    assert bad.observed == "fail:accepted"
    held = release.evidence_reference_finding("held", index)
    assert held.passed is False
    assert held.observed == "pass:blocked"
    assert release.evidence_reference_finding("waived", index).observed == (
        "waived:accepted"
    )
    for lookup in (index, None, "text"):
        absent = release.evidence_reference_finding("nope", lookup)
        assert absent.to_dict() == {
            "kind": "evidence",
            "name": "nope",
            "passed": False,
            "blocking": True,
            "reason": "required evidence 'nope' is missing",
            "evidence": ["nope"],
            "expected": "pass:accepted",
        }


def test_evaluate_release_names_foreign_metric_keys_as_text():
    policy = make_policy()
    result = release.evaluate_release(
        policy,
        {1: 2.0, "roc_auc": 0.81, "zz": 1},
        {"metric_uncertainty": {"estimate": 0.81}},
        {},
        make_index(policy),
    )
    reasons = set(result.blocking_reasons)
    assert "metric '1' is not declared by the evaluation policy" in reasons
    assert "metric 'zz' is not declared by the evaluation policy" in reasons
    names = [
        item.name
        for item in result.findings
        if item.kind is release.FindingKind.INPUT
    ]
    assert names == ["1", "zz"]


def test_finding_validation_messages_are_exact():
    with pytest.raises(
        TypeError, match=r"^evidence must be a tuple of strings$"
    ):
        make_finding(evidence=["a"])
    with pytest.raises(
        ValueError, match=r"^evidence must contain non-empty strings$"
    ):
        make_finding(evidence=("a", " "))
    with pytest.raises(
        ValueError, match=r"^evidence must not contain duplicates$"
    ):
        make_finding(evidence=("a", "a"))
    with pytest.raises(
        ValueError, match=r"^policy_schema_version must not be negative$"
    ):
        release.ReleaseEvaluation("t", -1, True, (), ())


def test_summary_canonical_json_escapes_non_ascii_text():
    summary = make_outcome((make_finding(),)).readiness_summary()
    named = dataclasses.replace(summary, task="naïve")
    assert '"task":"na\\u00efve"' in named.canonical_json()


def test_gate_findings_for_unusable_metrics_are_failed_and_blocking():
    gate = evaluation.MetricGate(
        "auc", evaluation.MetricDirection.MAXIMIZE, 0.5
    )
    expected = {
        "kind": "metric",
        "name": "auc",
        "passed": False,
        "blocking": True,
        "evidence": [],
        "expected": 0.5,
    }
    undeclared = release.metric_gate_finding(gate, set(), {"auc": 1.0})
    assert undeclared.to_dict() == dict(
        expected, reason="metric 'auc' is unsupported or undeclared"
    )
    missing = release.metric_gate_finding(gate, {"auc"}, {})
    assert missing.to_dict() == dict(expected, reason="metric 'auc' is missing")
    odd = release.metric_gate_finding(gate, {"auc"}, {"auc": "x"})
    assert odd.to_dict() == dict(
        expected, reason="metric 'auc' has an unsupported value"
    )
    with pytest.raises(ValueError, match=r"^observed must be finite$"):
        release.metric_gate_finding(gate, {"auc"}, {"auc": float("nan")})


def test_diagnostic_failures_are_not_passed():
    for report in (
        {"error": "boom"},
        {"available": False},
        {"passed": False},
        {"passed": 5},
    ):
        for action in (
            evaluation.DiagnosticAction.BLOCK,
            evaluation.DiagnosticAction.WARN,
        ):
            assert diagnose(report, action).passed is False
    assert diagnose({"passed": True}).passed is True
    assert diagnose({"error": "boom"}).blocking is True


def test_every_missing_explanation_score_is_reported():
    policy = make_policy_namespace()
    found = release.explanation_findings(policy, {"shap": {"quality": {}}})
    assert [item.name for item in found] == ["shap.stability", "shap.fidelity"]
    assert [item.passed for item in found] == [False, False]
    only_fidelity = {"shap": {"quality": {"fidelity_score": 0.75}}}
    found = release.explanation_findings(policy, only_fidelity)
    assert [item.passed for item in found] == [False, True]
    assert found[1].reason == "explanation 'shap' fidelity passed: 0.75 >= 0.75"


def test_fairness_metric_values_reads_mappings_with_mixed_key_types():
    report = {1: {"m": 0.5}, "m": 0.25, "z": [{"m": 0.75}]}
    assert release.fairness_metric_values(report, "m") == (0.25, 0.5, 0.75)
