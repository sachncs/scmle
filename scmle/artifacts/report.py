"""Human-readable HTML run report."""

import html

import pandas as pd

from scmle.api import results


class HtmlReport:
    """Renders a self-contained HTML summary of a run."""

    def render(
        self, result: results.RunResult, leaderboard: pd.DataFrame
    ) -> str:
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
        items = (
            "".join(f"<li>{escape(f)}</li>" for f in findings)
            or "<li>none</li>"
        )
        summary = escape(result.plan.summary()).replace("\n", "<br>")
        importance = result.feature_importance().head(10)
        importance_html = (
            importance.to_html(
                index=False, border=0, float_format=lambda v: f"{v:.4f}"
            )
            if len(importance)
            else "<p>unavailable</p>"
        )
        calibration = result.diagnostics.get("calibration", {})
        calibration_html = (
            f"<p>Brier score {calibration['brier_score']:.4f}, expected "
            f"calibration error {calibration['expected_calibration_error']:.4f}"
            "</p>"
            if "brier_score" in calibration
            else ""
        )
        reliability = self.reliability(result)
        explanation_records = result.evaluation.get("explanation_evidence", {})
        explanation_links = (
            "".join(
                f'<li><a href="explanations/{escape(name)}.html">'
                f"{escape(name)}</a></li>"
                for name in sorted(explanation_records)
            )
            or "<li>unavailable</li>"
        )
        return (
            f"<!doctype html><meta charset=utf-8>"
            f"<title>{escape(result.run_id)}</title>"
            f"<h1>{escape(result.run_id)}</h1><p>{summary}</p>"
            f"<h2>Leaderboard</h2>{leaderboard.to_html(index=False, border=0)}"
            f"<h2>Final test metrics ({escape(result.selected_model)})</h2>"
            f"<table>{metric_rows}</table>"
            f"<h2>Feature importance (permutation)</h2>{importance_html}"
            f"{calibration_html}"
            f"<h2>Reliability checks</h2>{reliability}"
            f"<h2>Explanation evidence</h2><ul>{explanation_links}</ul>"
            f"<h2>Findings</h2><ul>{items}</ul>"
        )

    @staticmethod
    def reliability(result: results.RunResult) -> str:
        """Summarizes fit health, shift, robustness and weak subgroups."""
        escape = html.escape
        lines = []
        health = result.evaluation.get("fit_health")
        if health:
            lines.append(
                f"Fit health: <b>{escape(health['verdict'])}</b> "
                f"({escape(health['note'])})"
            )
        curve = result.evaluation.get("learning_curve")
        if curve:
            lines.append(
                f"Learning curve: <b>{escape(curve['verdict'])}</b> "
                f"(last step {curve['last_gain']:+.4f}, whole curve "
                f"{curve['total_gain']:+.4f})"
            )
        shift = result.diagnostics.get("adversarial_validation", {})
        if shift.get("available"):
            lines.append(
                f"Development versus test rows: {escape(shift['verdict'])} "
                f"(separability AUC {shift['auc']:.2f})"
            )
        robust = result.diagnostics.get("robustness", {})
        if robust.get("scenarios"):
            fragile = robust["fragile_under"]
            lines.append(
                "Robustness: fragile under "
                + (", ".join(map(escape, fragile)) if fragile else "none")
            )
        worst = result.diagnostics.get("slice_performance", {}).get("worst")
        if worst:
            top = worst[0]
            lines.append(
                f"Weakest slice: {escape(top['feature'])} = "
                f"{escape(top['slice'])} ({top['rows']} rows, "
                f"{top['gap']:+.3f} versus overall)"
            )
        fair = result.diagnostics.get("fairness", {})
        for column in fair.get("flagged", []):
            flags = "; ".join(fair["columns"][column]["flags"])
            lines.append(f"Fairness ({escape(column)}): {escape(flags)}")
        return (
            "".join(f"<p>{line}</p>" for line in lines)
            or "<p>no checks were available</p>"
        )
