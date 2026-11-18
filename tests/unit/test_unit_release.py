"""Release findings, tallies, verdicts and the independent gates."""

import json
import math

import pytest

from scmle.contracts import errors
from scmle.contracts import evidence
from scmle.evaluation.release import finding
from scmle.evaluation.release import guard
from scmle.evaluation.release import verdict


def make(
    kind=finding.Kind.METRIC, name="auc", passed=True, blocking=True, **extra
):
    return finding.Finding(
        kind, name, passed, blocking, f"{name} reason", **extra
    )


def clean():
    return verdict.Verdict("binary", 1, True, (make(),), ())


def test_the_kinds_and_readiness_values_are_stable():
    assert [k.value for k in finding.Kind] == [
        "diagnostic",
        "evidence",
        "explanation",
        "fairness",
        "input",
        "metric",
        "reliability",
        "security",
    ]
    assert finding.Readiness.READY == "ready"
    assert finding.Readiness.BLOCKED == "blocked"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3, 3.0),
        (2.5, 2.5),
        (True, None),
        ("1", None),
        (math.inf, None),
        (math.nan, None),
        (None, None),
        (0, 0.0),
    ],
)
def test_a_finite_real_reads_only_finite_non_bool_numbers(value, expected):
    assert finding.finite_real(value) == expected


def test_a_quantify_converts_numbers_and_keeps_text_and_none():
    assert finding.quantify("x", 2) == 2.0
    assert isinstance(finding.quantify("x", 2), float)
    assert finding.quantify("x", "text") == "text"
    assert finding.quantify("x", None) is None


def test_a_quantify_rejects_bools_infinities_and_other_types():
    with pytest.raises(errors.KindError, match=r"^field must not be bool$"):
        finding.quantify("field", False)
    with pytest.raises(errors.InputError, match=r"^field must be finite$"):
        finding.quantify("field", math.inf)
    with pytest.raises(
        errors.KindError, match=r"^field must be numeric, text, or None$"
    ):
        finding.quantify("field", [1])


def test_a_finding_serialises_only_present_values():
    item = make(observed=1, expected="x", evidence=("e1", "e2"))
    assert item.observed == 1.0
    assert item.to_dict() == {
        "kind": "metric",
        "name": "auc",
        "passed": True,
        "blocking": True,
        "reason": "auc reason",
        "evidence": ["e1", "e2"],
        "observed": 1.0,
        "expected": "x",
    }
    assert "observed" not in make().to_dict()
    assert "expected" not in make().to_dict()


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        (
            ("metric", "n", True, True, "r"),
            errors.KindError,
            "kind must be Kind",
        ),
        (
            (finding.Kind.METRIC, " ", True, True, "r"),
            errors.InputError,
            "name must not be empty",
        ),
        (
            (finding.Kind.METRIC, 3, True, True, "r"),
            errors.InputError,
            "name must not be empty",
        ),
        (
            (finding.Kind.METRIC, "n", 1, True, "r"),
            errors.KindError,
            "passed must be bool",
        ),
        (
            (finding.Kind.METRIC, "n", True, 0, "r"),
            errors.KindError,
            "blocking must be bool",
        ),
        (
            (finding.Kind.METRIC, "n", True, True, " "),
            errors.InputError,
            "reason must not be empty",
        ),
        (
            (finding.Kind.METRIC, "n", True, True, None),
            errors.InputError,
            "reason must not be empty",
        ),
    ],
)
def test_a_malformed_finding_is_rejected(arguments, error, message):
    with pytest.raises(error, match=rf"^{message}$"):
        finding.Finding(*arguments)


def test_a_finding_rejects_malformed_evidence():
    with pytest.raises(errors.KindError, match=r"evidence must be a tuple"):
        make(evidence=["a"])
    with pytest.raises(errors.InputError, match=r"non-empty strings"):
        make(evidence=("a", " "))
    with pytest.raises(errors.InputError, match=r"non-empty strings"):
        make(evidence=(1,))
    with pytest.raises(errors.InputError, match=r"must not contain duplicates"):
        make(evidence=("a", "a"))
    with pytest.raises(errors.InputError, match=r"^observed must be finite$"):
        make(observed=math.nan)
    with pytest.raises(errors.KindError, match=r"^expected must not be bool$"):
        make(expected=True)


def test_a_tally_serialises_blocking_failures_as_blockers():
    assert finding.Tally(3, 2, 1, 2, 1).to_dict() == {
        "total": 3,
        "passed": 2,
        "failed": 1,
        "blocking": 2,
        "blockers": 1,
    }


@pytest.mark.parametrize(
    ("counts", "error", "message"),
    [
        (
            (1, 1, 0, 0, True),
            errors.KindError,
            "blocking_failures must be a non-negative integer",
        ),
        (
            (1.0, 1, 0, 0, 0),
            errors.KindError,
            "total must be a non-negative integer",
        ),
        (
            (0, 1, -1, 0, 0),
            errors.InputError,
            "failed must be a non-negative integer",
        ),
        (
            (2, 1, 0, 0, 0),
            errors.InputError,
            "total must equal passed plus failed",
        ),
        ((1, 1, 0, 2, 0), errors.InputError, "blocking must not exceed total"),
        (
            (2, 1, 1, 2, 2),
            errors.InputError,
            "blocking_failures must not exceed failed",
        ),
        (
            (2, 0, 2, 1, 2),
            errors.InputError,
            "blocking_failures must not exceed blocking",
        ),
    ],
)
def test_an_inconsistent_tally_is_rejected(counts, error, message):
    with pytest.raises(error, match=rf"^{message}$"):
        finding.Tally(*counts)


def test_a_tally_accepts_the_boundary_counts():
    assert finding.Tally(2, 0, 2, 2, 2).blocking_failures == 2
    assert finding.Tally(0, 0, 0, 0, 0).total == 0


def test_tallies_reject_bools_non_integers_and_negatives():
    finding.tallies({"a": 0, "b": 5})
    with pytest.raises(
        errors.KindError, match=r"^a must be a non-negative integer$"
    ):
        finding.tallies({"a": False})
    with pytest.raises(
        errors.KindError, match=r"^a must be a non-negative integer$"
    ):
        finding.tallies({"a": "1"})
    with pytest.raises(
        errors.InputError, match=r"^b must be a non-negative integer$"
    ):
        finding.tallies({"a": 0, "b": -1})


def every_kind():
    return {kind.value: finding.Tally(0, 0, 0, 0, 0) for kind in finding.Kind}


def test_an_exhaustive_mapping_must_cover_every_kind_with_tallies():
    finding.exhaustive(every_kind())
    with pytest.raises(errors.KindError, match=r"must be a mapping"):
        finding.exhaustive([])
    partial = every_kind()
    partial.pop("metric")
    with pytest.raises(
        errors.InputError, match=r"must contain every finding kind"
    ):
        finding.exhaustive(partial)
    wrong = every_kind()
    wrong["metric"] = {}
    with pytest.raises(errors.KindError, match=r"must map strings to Tally"):
        finding.exhaustive(wrong)


def mixed():
    blocker = make(
        name="auc", passed=False, observed=0.5, expected=0.7, evidence=("a",)
    )
    warning = make(finding.Kind.FAIRNESS, "gap", passed=False, blocking=False)
    fine = make(finding.Kind.INPUT, "rows", passed=True, blocking=False)
    return verdict.Verdict(
        "binary", 2, False, (blocker, warning, fine), ("auc reason",)
    )


def test_a_verdict_serialises_to_the_current_layout():
    record = mixed().to_dict()
    assert record["task"] == "binary"
    assert record["policy"] == 2
    assert record["passed"] is False
    assert record["blocked"] is True
    assert record["blockers"] == ["auc reason"]
    assert [f["name"] for f in record["findings"]] == ["auc", "gap", "rows"]
    readiness = record["readiness"]
    assert readiness["status"] == "blocked"
    assert readiness["blockers"] == 1
    assert readiness["warnings"] == 1
    assert readiness["findings"] == {
        "total": 3,
        "passed": 1,
        "failed": 2,
        "blocking": 1,
    }
    assert readiness["kinds"]["metric"] == {
        "total": 1,
        "passed": 0,
        "failed": 1,
        "blocking": 1,
        "blockers": 1,
    }
    assert readiness["kinds"]["fairness"] == {
        "total": 1,
        "passed": 0,
        "failed": 1,
        "blocking": 0,
        "blockers": 0,
    }
    assert readiness["kinds"]["input"] == {
        "total": 1,
        "passed": 1,
        "failed": 0,
        "blocking": 0,
        "blockers": 0,
    }
    assert list(readiness["kinds"]) == sorted(k.value for k in finding.Kind)


def test_a_verdict_canonical_form_is_compact_sorted_json():
    outcome = mixed()
    text = outcome.canonize()
    assert json.loads(text) == outcome.to_dict()
    assert ", " not in text and ": " not in text
    assert text == json.dumps(
        outcome.to_dict(), sort_keys=True, separators=(",", ":")
    )
    card = outcome.tally()
    assert json.loads(card.canonize()) == card.to_dict()
    assert card.canonize().startswith('{"blockers":1,')


def test_a_tally_of_a_clean_verdict_is_ready():
    card = clean().tally()
    assert card.status is finding.Readiness.READY
    assert card.total_findings == 1
    assert card.passed_findings == 1
    assert card.blocking_findings == 1
    assert card.blocking_failures == 0
    assert card.non_blocking_failures == 0
    with pytest.raises(TypeError):
        card.counts_by_kind["metric"] = None
    assert not clean().blocked


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ((" ", 1, True, (), ()), errors.InputError, "task must not be empty"),
        (
            ("t", -1, True, (), ()),
            errors.InputError,
            "policy_schema_version must not be negative",
        ),
        (("t", 1, 1, (), ()), errors.KindError, "passed must be bool"),
        (("t", 1, True, [], ()), errors.KindError, "findings must be a tuple"),
        (
            ("t", 1, True, ("x",), ()),
            errors.KindError,
            "findings must contain Finding values",
        ),
        (
            ("t", 1, True, (), []),
            errors.KindError,
            "blocking_reasons must be a tuple",
        ),
        (
            ("t", 1, True, (), ("r",)),
            errors.InputError,
            "blocking_reasons do not match findings",
        ),
        (
            ("t", 1, False, (), ()),
            errors.InputError,
            "passed does not match blocking findings",
        ),
    ],
)
def test_an_inconsistent_verdict_is_rejected(arguments, error, message):
    with pytest.raises(error, match=rf"^{message}$"):
        verdict.Verdict(*arguments)


def scorecard(**changes):
    values = {
        "task": "t",
        "policy_schema_version": 0,
        "status": finding.Readiness.READY,
        "total_findings": 0,
        "passed_findings": 0,
        "failed_findings": 0,
        "blocking_findings": 0,
        "blocking_failures": 0,
        "non_blocking_failures": 0,
        "counts_by_kind": every_kind(),
    }
    values.update(changes)
    return verdict.Scorecard(**values)


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"task": ""}, errors.InputError, "task must not be empty"),
        (
            {"policy_schema_version": True},
            errors.InputError,
            "policy_schema_version must be non-negative",
        ),
        (
            {"policy_schema_version": -1},
            errors.InputError,
            "policy_schema_version must be non-negative",
        ),
        ({"status": "ready"}, errors.KindError, "status must be Readiness"),
        (
            {"total_findings": 1},
            errors.InputError,
            "total_findings must equal passed plus failed",
        ),
        (
            {"total_findings": 1, "failed_findings": 1},
            errors.InputError,
            "failed_findings must equal blocking plus non-blocking failures",
        ),
        (
            {
                "total_findings": 1,
                "failed_findings": 1,
                "blocking_failures": 1,
                "status": finding.Readiness.BLOCKED,
            },
            errors.InputError,
            "blocking_failures must not exceed blocking_findings",
        ),
        (
            {"status": finding.Readiness.BLOCKED},
            errors.InputError,
            "status does not match blocking_failures",
        ),
        (
            {"total_findings": 1, "passed_findings": 1},
            errors.InputError,
            "aggregate counts do not match counts_by_kind",
        ),
    ],
)
def test_an_inconsistent_scorecard_is_rejected(changes, error, message):
    with pytest.raises(error, match=rf"^{message}$"):
        scorecard(**changes)


def test_a_scorecard_requires_every_kind():
    with pytest.raises(errors.InputError, match=r"every finding kind"):
        scorecard(counts_by_kind={})


def test_the_kinds_step_renames_each_kinds_counts_both_ways():
    layout = {"metric": {"blocking_failures": 2, "total": 3}}
    upgraded = verdict.KINDS.upgrade(layout)
    assert upgraded == {"metric": {"blockers": 2, "total": 3}}
    assert verdict.KINDS.downgrade(upgraded) == layout
    step = verdict.recount(finding.Tally.migrations[0])
    assert step.upgrade({"a": {"blocking_failures": 1}}) == {
        "a": {"blockers": 1}
    }


def test_a_fault_gate_passes_only_a_completed_unblocked_plan():
    outcome = clean()
    plan = {"execution_status": "passed", "release_blocked": False}
    assert guard.faults(outcome, plan) is outcome


@pytest.mark.parametrize(
    ("plan", "observed"),
    [
        (None, "missing"),
        ({"execution_status": "passed", "release_blocked": 0}, "passed"),
        ({"execution_status": "failed", "release_blocked": False}, "failed"),
        ({}, "None"),
    ],
)
def test_a_fault_gate_blocks_incomplete_evidence(plan, observed):
    blocked = guard.faults(clean(), plan)
    assert blocked.passed is False
    added = next(f for f in blocked.findings if f.name == "fault_injection")
    assert added.kind is finding.Kind.RELIABILITY
    assert added.blocking is True and added.passed is False
    assert added.observed == observed
    assert added.expected == "execution_status=passed and release_blocked=false"
    assert added.evidence == ("fault_injection_plan.json",)
    assert blocked.blocking_reasons == (
        "fault-injection recovery evidence is not complete; release blocked",
    )
    assert [f.kind for f in blocked.findings] == [
        finding.Kind.METRIC,
        finding.Kind.RELIABILITY,
    ]


def test_the_gates_reject_a_non_verdict():
    for call in (
        lambda: guard.faults({}, {}),
        lambda: guard.certify({}, {}, True),
        lambda: guard.countersign({}, None, True),
    ):
        with pytest.raises(
            errors.KindError, match=r"^outcome must be Verdict$"
        ):
            call()


def signed(value="abc", algorithm="hmac-sha256"):
    return {"signature": {"algorithm": algorithm, "value": value}}


def test_a_certificate_gate_passes_a_verified_signature():
    outcome = clean()
    assert guard.certify(outcome, signed(), True) is outcome


@pytest.mark.parametrize(
    ("record", "verified", "observed"),
    [
        (signed(), False, "invalid"),
        (signed(), 1, "signed"),
        (signed(value=" "), True, "signed"),
        (signed(value=5), True, "signed"),
        (signed(algorithm="sha1"), True, "signed"),
        ({}, True, "unsigned"),
        (None, True, "unsigned"),
    ],
)
def test_a_certificate_gate_blocks_a_bad_signature(record, verified, observed):
    blocked = guard.certify(clean(), record, verified)
    added = next(f for f in blocked.findings if f.name == "release_attestation")
    assert blocked.blocked
    assert added.kind is finding.Kind.SECURITY
    assert added.observed == observed
    assert added.expected == "verified hmac-sha256 attestation signature"
    assert added.evidence == ("release_attestation.json",)
    assert blocked.blocking_reasons == (
        "release attestation is missing or invalid; release blocked",
    )


def test_a_countersign_gate_passes_a_verified_signed_index():
    outcome = clean()
    index = evidence.Index(signature="sig")
    assert guard.countersign(outcome, index, True) is outcome


@pytest.mark.parametrize(
    ("index", "verified", "observed"),
    [
        (evidence.Index(signature="sig"), False, "invalid"),
        (evidence.Index(), True, "unsigned"),
        (None, True, "unsigned"),
    ],
)
def test_a_countersign_gate_blocks_a_bad_index(index, verified, observed):
    blocked = guard.countersign(clean(), index, verified)
    added = next(f for f in blocked.findings if f.name == "release_evidence")
    assert blocked.blocked
    assert added.observed == observed
    assert added.expected == "verified hmac-sha256 evidence index signature"
    assert added.evidence == ("release_evidence.json",)
    assert blocked.policy_schema_version == 1
    assert blocked.task == "binary"


def test_the_gates_sort_findings_by_kind_name_and_reason():
    outcome = verdict.Verdict(
        "binary",
        1,
        True,
        (
            make(finding.Kind.SECURITY, "zz"),
            make(finding.Kind.DIAGNOSTIC, "aa"),
        ),
        (),
    )
    blocked = guard.certify(outcome, None, True)
    assert [(f.kind.value, f.name) for f in blocked.findings] == [
        ("diagnostic", "aa"),
        ("security", "release_attestation"),
        ("security", "zz"),
    ]
