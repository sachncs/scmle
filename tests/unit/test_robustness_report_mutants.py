"""Exact documents and validator messages of the robustness reports."""

import types

import pytest

from scmle.data.detectors import certificates
from scmle.data.robustness import report
from scmle.data.robustness import shift
from scmle.data.robustness import stability


class Dumped:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return self.payload


def found_stability(**overrides):
    values = {
        "grid": None,
        "findings": (),
        "stability_note": "no grid",
        "rule": Dumped({"s_min": 0.5}),
        "applicable": (),
        "notes": (),
        "stability_gate": certificates.GateStatus.GATING,
        "store_note": "",
        "rule_note": "",
        "fragile": (),
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def make_finding(name, status):
    return types.SimpleNamespace(
        name=name, status=status, to_dict=lambda: {"name": name}
    )


def test_a_report_without_a_grid_or_findings_is_exact():
    document = report.stability_report(found_stability())

    assert document["schema_version"] == 1
    assert document["status"] == "not_evaluated"
    assert document["reason"] == "no grid"
    assert document["rule"] == {"s_min": 0.5}
    assert document["grid"] == {
        "declaration_sha256": None,
        "total": None,
        "run": None,
        "subset": None,
        "choices": [],
        "not_run": [],
        "inclusion_note": "first_half and second_half refer to the stored row "
        "order",
    }
    assert document["counts"] == {
        "stable": 0,
        "fragile": 0,
        "not_evaluable": 0,
    }
    assert document["findings"] == []
    assert document["fragile"] == []
    assert set(document) == set(report.STABILITY_KEYS)


def test_the_estimator_and_guarantee_texts_are_stated():
    document = report.stability_report(found_stability())

    assert document["estimator"].startswith("Spearman rank correlation")
    assert (
        "bootstrap resamples whole groups or time blocks"
        in (document["estimator"])
    )
    assert document["guarantee"].startswith(
        "Stability is relative to the declared specification set and is not "
        "a confirmation."
    )
    assert document["guarantee"].endswith(
        "single test on the confirmation set."
    )


def test_a_grid_and_findings_are_summarised():
    grid = types.SimpleNamespace(
        declaration_sha256="abc", total=9, specs=(1, 2, 3), subset=True
    )
    findings = (
        make_finding("a", stability.STABLE),
        make_finding("b", stability.FRAGILE),
        make_finding("c", stability.FRAGILE),
        make_finding("d", stability.NOT_EVALUABLE),
    )
    document = report.stability_report(
        found_stability(
            grid=grid,
            findings=findings,
            applicable=(Dumped({"choice": 1}), Dumped({"choice": 2})),
            notes=("n1", "n2"),
            fragile=("b", "c"),
        )
    )

    assert document["status"] == "evaluated"
    assert document["reason"] == (
        "survivors re-run over the declared specification grid"
    )
    assert document["grid"]["declaration_sha256"] == "abc"
    assert document["grid"]["total"] == 9
    assert document["grid"]["run"] == 3
    assert document["grid"]["subset"] is True
    assert document["grid"]["choices"] == [{"choice": 1}, {"choice": 2}]
    assert document["grid"]["not_run"] == ["n1", "n2"]
    assert document["counts"] == {
        "stable": 1,
        "fragile": 2,
        "not_evaluable": 1,
    }
    assert document["fragile"] == ["b", "c"]
    assert document["findings"] == [
        {"name": "a"},
        {"name": "b"},
        {"name": "c"},
        {"name": "d"},
    ]


def test_the_certificate_note_prefers_the_store_then_the_rule_note():
    both = report.stability_report(
        found_stability(store_note="store", rule_note="rule")
    )
    rule_only = report.stability_report(found_stability(rule_note="rule"))

    assert both["certificate"] == {"gate": "gating", "note": "store"}
    assert rule_only["certificate"]["note"] == "rule"


def test_the_default_certificate_note_depends_on_the_gate():
    gating = report.stability_report(found_stability())
    descriptive = report.stability_report(
        found_stability(stability_gate=certificates.GateStatus.DESCRIPTIVE)
    )

    assert gating["certificate"]["note"] == (
        "a current passing certificate lets a fragile verdict quarantine"
    )
    assert descriptive["certificate"]["gate"] == "descriptive"
    assert descriptive["certificate"]["note"] == (
        "no current passing certificate: a fragile verdict is a warning"
    )


def test_the_confirmation_filter_text_defaults_and_is_overridden():
    default = report.stability_report(found_stability())
    given = report.stability_report(found_stability(), "withdrew 2")

    assert default["confirmation_filter"].startswith(
        "not applied: the charter does not ask for fragile survivors to be "
        "withdrawn"
    )
    assert default["confirmation_filter"].endswith("on their own merits")
    assert given["confirmation_filter"] == "withdrew 2"


def found_shift(**overrides):
    values = {
        "shift": Dumped(
            {
                "status": "pass",
                "reason": "r",
                "agreement": 0.5,
                "c2st": {"a": 1},
                "mmd": {"b": 2},
                "features": [{"column": "x"}],
            }
        ),
        "c2st_gate": certificates.GateStatus.GATING,
        "mmd_gate": certificates.GateStatus.DESCRIPTIVE,
        "coverage": (Dumped({"level": "l1"}), Dumped({"level": "l2"})),
        "forward": Dumped({"status": "skipped"}),
        "store_note": "note",
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_the_shift_report_carries_each_part():
    document = report.shift_report(found_shift())

    assert document["schema_version"] == 1
    assert document["status"] == "pass"
    assert document["reason"] == "r"
    assert document["agreement"] == 0.5
    assert document["c2st"] == {"a": 1}
    assert document["mmd"] == {"b": 2}
    assert document["features"] == [{"column": "x"}]
    assert document["coverage"] == [{"level": "l1"}, {"level": "l2"}]
    assert document["forward"] == {"status": "skipped"}
    assert document["certificate"] == {
        "gates": {shift.C2ST_ID: "gating", shift.MMD_ID: "descriptive"},
        "note": "note",
    }
    assert document["periods"] == {
        "status": "not_run",
        "reason": "per-period drift is not implemented",
    }
    assert set(document) == set(report.SHIFT_KEYS)
    assert document["assumptions"] == [
        "Selection into the deployment sample is assumed random.",
        "Two-sample tests detect covariate shift, not label or concept shift.",
        "A pass means no shift was detected, not that none exists.",
        (
            "Exact only under exchangeability of the sides; clustered or "
            "temporal data need block-matched samples."
        ),
    ]


def test_the_key_tuples_are_exact():
    assert report.SCHEMA_VERSION == 1
    assert report.STABILITY_STATUSES == ("evaluated", "not_evaluated")
    assert report.FINDING_STATUSES == ("stable", "fragile", "not_evaluable")
    assert report.FINDING_KEYS == (
        "name",
        "status",
        "reason",
        "specs",
        "attribution",
    )
    assert report.STABILITY_KEYS == (
        "schema_version",
        "status",
        "reason",
        "estimator",
        "rule",
        "grid",
        "certificate",
        "counts",
        "fragile",
        "findings",
        "confirmation_filter",
        "guarantee",
    )
    assert report.SHIFT_KEYS == (
        "schema_version",
        "status",
        "reason",
        "agreement",
        "c2st",
        "mmd",
        "features",
        "coverage",
        "periods",
        "forward",
        "certificate",
        "assumptions",
    )


def test_require_keys_names_the_document_and_the_missing_keys():
    with pytest.raises(ValueError, match=r"^thing is missing \['b', 'c'\]$"):
        report.require_keys(
            {"schema_version": 1, "a": 1}, ("a", "b", "c"), "thing"
        )
    with pytest.raises(
        ValueError, match=r"^thing has an unsupported schema_version$"
    ):
        report.require_keys({"schema_version": 2, "a": 1}, ("a",), "thing")
    report.require_keys({"schema_version": 1, "a": 1}, ("a",), "thing")


def valid_stability():
    document = report.stability_report(found_stability())
    document["findings"] = [
        {
            "name": "x",
            "status": "stable",
            "reason": "",
            "specs": [],
            "attribution": {},
        }
    ]
    return document


def test_validate_stability_messages_and_findings():
    good = valid_stability()
    report.validate_stability(good)

    with pytest.raises(
        ValueError, match=r"^stability report has an unknown status$"
    ):
        report.validate_stability({**good, "status": "bogus"})
    with pytest.raises(ValueError, match=r"^a finding has an unknown status$"):
        report.validate_stability(
            {**good, "findings": [{**good["findings"][0], "status": "bogus"}]}
        )
    lacking = {k: v for k, v in good["findings"][0].items() if k != "specs"}
    with pytest.raises(ValueError, match=r"^finding is missing \['specs'\]$"):
        report.validate_stability({**good, "findings": [lacking]})
    for status in ("stable", "fragile", "not_evaluable"):
        report.validate_stability(
            {**good, "findings": [{**good["findings"][0], "status": status}]}
        )
    report.validate_stability({**good, "status": "not_evaluated"})


def test_validate_shift_messages_and_statuses():
    document = report.shift_report(found_shift())
    document["status"] = "unverifiable"
    report.validate_shift(document)
    report.validate_shift({**document, "status": "fail"})
    report.validate_shift({**document, "status": "pass"})

    with pytest.raises(
        ValueError, match=r"^shift report has an unknown status$"
    ):
        report.validate_shift({**document, "status": "bogus"})
    with pytest.raises(
        ValueError,
        match=r"^a pass needs a deployment sample and a test$",
    ):
        report.validate_shift({**document, "status": "pass", "c2st": None})
    report.validate_shift({**document, "status": "fail", "c2st": None})
    with pytest.raises(ValueError, match=r"^shift report is missing "):
        report.validate_shift({"schema_version": 1})


def test_validate_stability_names_the_report_in_its_errors():
    with pytest.raises(ValueError, match=r"^stability report is missing \["):
        report.validate_stability({"schema_version": 1})
    good = valid_stability()
    with pytest.raises(
        ValueError,
        match=r"^stability report has an unsupported schema_version$",
    ):
        report.validate_stability({**good, "schema_version": 2})
