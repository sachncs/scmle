"""Deterministic exploratory report and experiment comparison.

Both are computed from retained records only, with no random draw and no
clock, so recomputing them from the same records gives the same bytes and the
same ``sha256``. A reviewer who doubts a report recomputes it.
"""

from collections import Counter
from pathlib import Path
from typing import Any

from scmle.trail import records

BAR_WIDTH = 30


def with_hash(body: dict[str, Any]) -> dict[str, Any]:
    """Adds the digest of ``body`` under ``sha256``.

    Args:
        body: The report content.

    Returns:
        A copy of the body with a `sha256` entry.
    """
    return {**body, "sha256": records.digest(body)}


def exploratory(directory: Path) -> dict[str, Any]:
    """Reports what the data looked like and what exploration was allowed to do.

    Reads the run's curation report, dataset manifest, feasibility verdict,
    hypothesis ledger and claims. The report states that only the exploration
    rows were profiled.

    Args:
        directory: The trail folder.

    Returns:
        The report with its digest: dataset identity, profiled rows, feature
        and dropped columns, quality and risk findings, feasibility, ledger row
        counts by kind and the confirmation outcome with its checks by tier.
    """
    base = directory / records.RETAINED / "run"
    curation = records.read_json(base / "curation_report.json")
    manifest = records.read_json(base / "dataset_manifest.json")
    feasibility = records.read_json(base / "feasibility.json")
    claims = records.read_json(base / "confirmed_claims.json")
    ledger = records.read_lines(base / "hypothesis_ledger.jsonl")
    kinds = Counter(row["kind"] for row in ledger)
    tiers = Counter(check["tier"] for check in claims["checks"])
    findings = curation.get("quality_findings", []) + curation.get(
        "risk_findings", []
    )
    body = {
        "kind": "exploratory_report",
        "dataset": {
            "data_hash": manifest["data_hash"],
            "schema_hash": manifest["schema_hash"],
            "rows": manifest["row_count"],
            "columns": manifest["column_count"],
            "target": manifest["target"],
        },
        "profiled_rows": curation["profiled_rows"],
        "profiled_set": "exploration rows only",
        "feature_columns": sorted(curation["feature_columns"]),
        "dropped_columns": curation["dropped_columns"],
        "findings": [
            {
                "severity": str(item.get("severity", "")),
                "message": str(item.get("evidence", {}).get("message", "")),
                "disposition": str(item.get("disposition", "")),
            }
            for item in findings
        ],
        "feasibility": {
            key: feasibility[key]
            for key in sorted(feasibility)
            if key != "reason"
        }
        | {"reason": feasibility.get("reason", "")},
        "ledger_rows": dict(sorted(kinds.items())),
        "confirmation": {
            "outcome": claims["outcome"],
            "checks_by_tier": dict(sorted(tiers.items())),
        },
    }
    return with_hash(body)


def bar(score: float, top: float) -> str:
    """Draws a text bar.

    Args:
        score: The score; negative values count as zero.
        top: The score that fills the bar.

    Returns:
        A string of `#` up to `BAR_WIDTH` long, empty if `top` is not positive.
    """
    filled = round(BAR_WIDTH * max(score, 0.0) / top) if top > 0 else 0
    return "#" * filled


def experiments(directory: Path) -> dict[str, Any]:
    """Ranks the scored candidates the search tried, with their spread.

    The comparison states whether the selected model's lead over the best
    other scored candidate is larger than the larger of the two candidates'
    cross-validation score standard deviations. Candidates that are not `ok`
    or have no score are listed separately.

    Args:
        directory: The trail folder.

    Returns:
        The report with its digest: the scored trials ranked by descending
        score, the trials that were not scored, the lead over the runner-up
        and whether it exceeds the spread (both None unless the selected trial
        and another trial are scored), the score of the best ``dummy`` model
        or None, the cross-validation interval, the test metrics and a
        reading.
    """
    base = directory / records.RETAINED / "run"
    evaluation = records.read_json(base / "evaluation.json")
    rows = []
    for path in sorted((base / "trials").glob("trial_*.json")):
        trial = records.read_json(path)
        rows.append(
            {
                "trial": trial["trial_id"],
                "model": trial["model_name"],
                "strategy": trial["strategy"],
                "status": trial["status"],
                "score": trial["score_mean"],
                "spread": trial["score_std"],
                "folds": len(trial.get("fold_scores", [])),
            }
        )
    done = [r for r in rows if r["status"] == "ok" and r["score"] is not None]
    done.sort(key=lambda r: (-r["score"], r["trial"]))
    top = done[0]["score"] if done else 1.0
    for rank, row in enumerate(done, start=1):
        row["rank"] = rank
        row["bar"] = bar(row["score"], top)
    selected = evaluation["selected_trial_id"]
    chosen = next((r for r in done if r["trial"] == selected), None)
    others = [r for r in done if r["trial"] != selected]
    lead = None
    separated = None
    if chosen is not None and others:
        runner = others[0]
        lead = chosen["score"] - runner["score"]
        separated = lead > max(chosen["spread"], runner["spread"])
    dummy = [r for r in done if r["model"] == "dummy"]
    body = {
        "kind": "experiment_comparison",
        "metric": evaluation["metric"],
        "selection_policy": evaluation["selection_policy"],
        "selected_trial": selected,
        "selected_model": evaluation["selected_model"],
        "ranked": done,
        "not_scored": [
            {"trial": r["trial"], "model": r["model"], "status": r["status"]}
            for r in rows
            if r not in done
        ],
        "lead_over_runner_up": lead,
        "lead_exceeds_spread": separated,
        "baseline_score": dummy[0]["score"] if dummy else None,
        "cv_interval": evaluation["cv_confidence"].get("interval"),
        "test_metrics": evaluation["test_metrics"],
        "reading": (
            "Cross-validation scores overlap across the top candidates: "
            "the lead of the selected model is "
            + ("larger than" if separated else "not larger than")
            + " the larger fold spread. The test metrics were opened once "
            "after selection was frozen."
        ),
    }
    return with_hash(body)
