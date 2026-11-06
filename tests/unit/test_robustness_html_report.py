"""The static HTML report written from the stability and shift reports."""

import re

from scmle.data.robustness import html_report
from scmle.data.robustness import report


def finding(name="income", status="stable", effects=(0.3, 0.35, 0.28)):
    return {
        "name": name,
        "status": status,
        "reason": "",
        "headline": {"median": 0.3},
        "numbers": {},
        "specs": [
            {
                "sha256": "x" * 8 + str(i),
                "choices": {"imputation": "median"},
                "effect": e,
                "low": e - 0.1,
                "high": e + 0.1,
            }
            for i, e in enumerate(effects)
        ],
        "attribution": {},
    }


def stability(findings):
    return {
        "schema_version": 1,
        "status": "evaluated",
        "reason": "survivors re-run over the declared specification grid",
        "estimator": "Spearman",
        "rule": {"band": 0.1},
        "grid": {"total": 3, "run": 3, "not_run": ["a note"]},
        "certificate": {"gate": "gating", "note": "n"},
        "counts": {"stable": 1, "fragile": 0, "not_evaluable": 0},
        "fragile": [],
        "findings": findings,
        "confirmation_filter": "not applied",
        "guarantee": "Stability is relative to the declared specification set.",
    }


def shift(status="unverifiable"):
    return {
        "schema_version": 1,
        "status": status,
        "reason": "no reference sample",
        "agreement": None,
        "c2st": None,
        "mmd": None,
        "features": [],
        "coverage": [],
        "periods": {"status": "not_run", "reason": "x"},
        "forward": {"status": "skipped", "reason": "F is the final test set"},
        "certificate": {"gates": {}, "note": None},
        "assumptions": ["Selection into the deployment sample is assumed."],
    }


def test_the_report_is_one_self_contained_page_without_network_or_script():
    page = html_report.render(stability([finding()]), shift())
    assert page.startswith("<!doctype html>")
    assert "<script" not in page.lower()
    assert not re.search(r"(https?:)?//[a-z]", page)
    assert "income" in page and "unverifiable" in page
    assert "<svg" in page


def test_text_from_the_documents_is_escaped():
    page = html_report.render(
        stability([finding(name="<script>alert(1)</script>")]), shift()
    )
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page


def test_a_run_without_findings_still_renders_both_sections():
    empty = stability([])
    empty["status"] = "not_evaluated"
    page = html_report.render(empty, shift("pass"))
    assert "not_evaluated" in page and "no findings" in page.lower()
    assert "forward" in page.lower()


def test_the_inputs_pass_the_report_validators():
    report.validate_stability(stability([finding()]))
    report.validate_shift(shift())
