"""Tests for the explicit pytest skip-reason release gate."""

from pathlib import Path

from scripts import check_pytest_skips


def write_report(path: Path, message: str | None) -> None:
    """Write one minimal JUnit report with an optional skip message."""
    attribute = "" if message is None else f' message="{message}"'
    path.write_text(
        '<testsuite><testcase classname="suite" name="case">'
        f"<skipped{attribute}/></testcase></testsuite>",
        encoding="utf-8",
    )


def write_xfail_report(path: Path, outcome: str) -> None:
    """Write one JUnit report containing an xfail or xpass outcome."""
    path.write_text(
        '<testsuite><testcase classname="suite" name="case">'
        f'<skipped type="pytest.{outcome}" message="known issue"/>'
        "</testcase></testsuite>",
        encoding="utf-8",
    )


def test_skip_audit_accepts_an_explicit_reason(tmp_path: Path):
    """A skip with a preserved pytest reason passes the gate."""
    report = tmp_path / "results.xml"
    write_report(report, "Skipped: optional dependency is not installed")

    assert check_pytest_skips.main([str(report)]) == 0
    assert check_pytest_skips.parse_skips(report) == (
        ("suite::case", "optional dependency is not installed"),
    )


def test_skip_audit_reports_the_reason_category(tmp_path: Path, capsys):
    """Approved dependency skips are machine-readable in the audit output."""
    report = tmp_path / "results.xml"
    write_report(report, "Skipped: torch is not installed")

    assert check_pytest_skips.main([str(report)]) == 0

    output = capsys.readouterr().out
    assert '"categories": {"dependency": 1}' in output
    assert '"category": "dependency"' in output
    assert '"unapproved": []' in output


def test_skip_audit_accepts_a_missing_optional_extra(tmp_path: Path):
    """Optional capability jobs may explicitly report an absent extra."""
    report = tmp_path / "results.xml"
    write_report(
        report, "Skipped: the optional deeplearning extra is not installed"
    )

    assert check_pytest_skips.main([str(report)]) == 0


def test_skip_audit_accepts_an_environment_reason(tmp_path: Path):
    """Environment-dependent skips remain valid release evidence."""
    report = tmp_path / "results.xml"
    write_report(report, "Skipped: OpenML unavailable: network disabled")

    assert check_pytest_skips.main([str(report)]) == 0


def test_skip_audit_rejects_an_empty_reason(tmp_path: Path):
    """A skip without a reason fails closed."""
    report = tmp_path / "results.xml"
    write_report(report, None)

    assert check_pytest_skips.main([str(report)]) == 1


def test_skip_audit_rejects_an_unapproved_reason(tmp_path: Path, capsys):
    """A new skip category must be registered before it reaches release CI."""
    report = tmp_path / "results.xml"
    write_report(report, "Skipped: temporary workaround")

    assert check_pytest_skips.main([str(report)]) == 1

    assert (
        '"unapproved": [{"reason": "temporary workaround", "test_id": "suite::case"}]'
        in capsys.readouterr().out
    )


def test_skip_audit_rejects_xfail_outcomes(tmp_path: Path):
    """An expected failure cannot become release evidence silently."""
    report = tmp_path / "xfail.xml"
    write_xfail_report(report, "xfail")

    assert check_pytest_skips.main([str(report)]) == 1
    assert check_pytest_skips.parse_xfails(report) == (
        ("suite::case", "known issue"),
    )
