"""Human-readable HTML run report."""

import html

import pandas as pd

from mml.api import results


class HtmlReport:
    """Renders a self-contained HTML summary of a run."""

    def render(self, result: results.RunResult, leaderboard: pd.DataFrame) -> str:
        """Builds the report.

        Args:
            result: The completed run.
            leaderboard: Ranked trials.

        Returns:
            An HTML document as a string.
        """
        escape = html.escape
        metric_rows = "".join(
            f"<tr><td>{escape(name)}</td><td>{value:.4f}</td></tr>"
            for name, value in result.evaluation["test_metrics"].items()
        )
        findings = (
            *result.profile.quality_findings,
            *result.profile.risk_findings,
        )
        items = "".join(f"<li>{escape(f)}</li>" for f in findings) or "<li>none</li>"
        summary = escape(result.plan.summary()).replace("\n", "<br>")
        return (
            f"<!doctype html><meta charset=utf-8>"
            f"<title>{escape(result.run_id)}</title>"
            f"<h1>{escape(result.run_id)}</h1><p>{summary}</p>"
            f"<h2>Leaderboard</h2>{leaderboard.to_html(index=False, border=0)}"
            f"<h2>Final test metrics ({escape(result.selected_model)})</h2>"
            f"<table>{metric_rows}</table>"
            f"<h2>Findings</h2><ul>{items}</ul>"
        )
