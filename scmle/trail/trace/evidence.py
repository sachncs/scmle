"""The evidence steps of a trail: reports, quality, selection and outcomes."""

from typing import Any

from scmle.artifacts import claims_audit
from scmle.artifacts import results
from scmle.trail import inputs
from scmle.trail import outcomes
from scmle.trail import records
from scmle.trail import reports
from scmle.trail.trace import finding
from scmle.trail.trace import reader

MINIMUM_CASES = 100
CUT_TOLERANCE = 1e-9  # the served threshold must equal the chartered cut


class Evidence(reader.Reader):
    """Derives the rebuilt reports, quality, selection and outcome steps."""

    def exploration(self) -> None:
        """Derives the exploratory report."""
        name = "exploration"
        base = "run/"
        files = [
            base + "curation_report.json",
            base + "dataset_manifest.json",
            base + "feasibility.json",
            base + "hypothesis_ledger.jsonl",
            base + "confirmed_claims.json",
        ]
        try:
            fresh = reports.exploratory(self.directory)
        except (OSError, ValueError, KeyError):
            # handled: recorded as trace problem
            self.problem(
                finding.MISSING_LINK,
                name,
                "the exploratory report cannot be rebuilt",
            )
            self.add(name, files, {}, ["question"])
            return
        self.derived["exploratory"] = fresh
        self.recompute(name, records.EXPLORATORY, fresh, files)

    def experiments(self) -> None:
        """Derives the experiment comparison."""
        name = "experiments"
        files = ["run/evaluation.json"]
        trials = sorted((self.base / "run" / "trials").glob("trial_*.json"))
        files += [f"run/trials/{p.name}" for p in trials]
        try:
            fresh = reports.experiments(self.directory)
        except (OSError, ValueError, KeyError):
            # handled: recorded as trace problem
            self.problem(
                finding.MISSING_LINK, name, "the comparison cannot be rebuilt"
            )
            self.add(name, files, {}, ["question"])
            return
        self.derived["experiments"] = fresh
        self.recompute(name, records.EXPERIMENTS, fresh, files)

    def quality(self) -> None:
        """Derives the domain thresholds, which the reviewer signs."""
        name = "quality"
        problem = inputs.diagnose(
            self.inputs, "reviewer", self.keys.get("reviewer")
        )
        if problem:
            self.problem(finding.UNSIGNED, name, problem)
        section = self.inputs.get("reviewer", {})
        results = self.evaluate(section.get("thresholds", []), name)
        for item in results:
            if not item["passed"]:
                self.problem(
                    finding.NOT_MET,
                    name,
                    f"threshold {item['id']}: {item['path']} {item['op']} "
                    f"{item['value']} fails (observed {item['observed']})",
                )
        reviewed = section.get("evidence", {}).get("reviewed", {})
        for relative, held in sorted(reviewed.items()):
            path = self.base / relative
            if not path.is_file():
                self.problem(
                    finding.MISSING_LINK,
                    name,
                    f"reviewed record {relative} is absent",
                )
            elif records.checksum(path) != held:
                self.problem(
                    finding.TAMPERED,
                    name,
                    f"{relative} changed after the reviewer signed",
                )
        self.add(
            name,
            [records.INPUTS, *sorted(reviewed)],
            {
                "reviewer": section.get("name"),
                "thresholds": results,
                "statement": section.get("evidence", {}).get("statement"),
                "signed": problem is None,
            },
            ["stakeholder", "exploration", "experiments"],
        )

    def selection(self) -> None:
        """Derives the selected model, its claims and the bundle identity."""
        name = "selection"
        evaluation = results.evaluation(
            self.json("run/evaluation.json", name) or {}
        )
        manifest = self.need("run/model/model_manifest.json", name)
        facts: dict[str, Any] = {
            "run_id": evaluation.get("run"),
            "selected_model": evaluation["selected"].get("model"),
            "metric": evaluation.get("metric"),
            "cv_score": evaluation["selected"].get("score"),
            "cv_interval": (evaluation.get("confidence") or {}).get("interval"),
            "test_metrics": evaluation.get("metrics"),
            "n_test": evaluation["rows"].get("test"),
        }
        self.run = str(evaluation.get("run", ""))
        if manifest is not None:
            self.content = records.checksum(manifest)
            facts["content_id"] = self.content
        run_dir = self.base / "run"
        if run_dir.is_dir():
            verdict = claims_audit.verify(run_dir, self.keys.get("analyst"))
            facts["claims_steps"] = [
                [f.name, f.status] for f in verdict.findings
            ]
            for found in verdict.findings:
                if found.status == claims_audit.FAIL:
                    kind = (
                        finding.BROKEN_CHAIN
                        if found.name.startswith("ledger chain")
                        else finding.MISMATCH
                    )
                    self.problem(
                        kind, name, f"claims: {found.name}: {found.detail}"
                    )
        claims = self.json("run/confirmed_claims.json", name) or {}
        facts["claims_outcome"] = claims.get("outcome")
        facts["confirmation_tiers"] = sorted(
            {c["tier"] for c in claims.get("checks", [])}
        )
        files = [
            "run/evaluation.json",
            "run/model/model_manifest.json",
            "run/confirmed_claims.json",
            "run/hypothesis_ledger.jsonl",
            "run/custody_log.jsonl",
            "run/split_manifest.json",
        ]
        self.add(name, files, facts, ["question", "experiments", "quality"])

    def observe(self) -> None:
        """Tracks the observed cost per case against the expectation."""
        name = "outcomes"
        feed = self.json("deployment/observed_outcomes.json", name)
        cost = self.json("framing/decision_cost_model.json", name)
        expectation = self.inputs.get("stakeholder", {}).get("expectation", {})
        tracking: dict[str, Any] = {"status": outcomes.ABSENT, "cases": 0}
        if isinstance(feed, dict) and isinstance(cost, dict) and expectation:
            try:
                tracking = outcomes.track(
                    feed, cost, expectation, MINIMUM_CASES
                )
            except (KeyError, ValueError, TypeError):
                # handled: recorded as trace problem
                self.problem(
                    finding.MISMATCH,
                    name,
                    "the outcome feed cannot be evaluated",
                )
            expected_cut = outcomes.optimise(cost)
            if (
                abs(float(feed.get("threshold", -1)) - expected_cut)
                > CUT_TOLERANCE
            ):
                self.problem(
                    finding.IDENTITY,
                    name,
                    "the served decision rule is not the chartered cut",
                )
        tracking = reports.stamp({"kind": "outcome_tracking", **tracking})
        self.derived["outcomes"] = tracking
        self.recompute(
            name,
            records.OUTCOME_TRACKING,
            tracking,
            [
                "deployment/observed_outcomes.json",
                "framing/decision_cost_model.json",
            ],
        )
        self.steps[name]["facts"].update(
            status=tracking["status"],
            cases=tracking.get("cases"),
            mean_cost=tracking.get("mean_cost"),
            interval=tracking.get("interval"),
            approved_range=tracking.get("approved_range"),
        )
        self.steps[name]["requires"] = ["stakeholder", "deployment"]
        status = tracking["status"]
        if status in (outcomes.ABOVE, outcomes.BELOW):
            self.problem(
                finding.OUTSIDE,
                name,
                f"observed cost per case {tracking['mean_cost']:.3f} "
                f"(95% interval {tracking['interval'][0]:.3f} to "
                f"{tracking['interval'][1]:.3f}) is {status} the approved "
                f"range {tracking['approved_range']}",
            )
        elif status in (outcomes.INSUFFICIENT, outcomes.ABSENT):
            self.problem(
                finding.MISSING_LINK,
                name,
                f"outcome status is {status}: no observed outcome can be "
                "compared with the expectation",
            )
