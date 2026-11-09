"""Derives the steps of a trail from the retained records alone.

`derive` reads ``retained/`` (and ``inputs.json``) of a trail folder and
returns the ten steps, each with the records it rests on, the facts read
from them, and every problem found. Building a trail and verifying one both
call it, so a fact is never trusted from the file that states it.
"""

from collections.abc import Mapping
import dataclasses
import hmac
from pathlib import Path
from typing import Any

from scmle.artifacts import claims_audit
from scmle.contracts import errors
from scmle.contracts import signing
from scmle.governance import deployment_ledger
from scmle.governance import registry
from scmle.trail import inputs as inputs_module
from scmle.trail import outcomes
from scmle.trail import records
from scmle.trail import reports

MISSING_LINK = "missing_link"
BROKEN_CHAIN = "broken_chain"
UNSIGNED = "unsigned_step"
UNAPPROVED = "unapproved_step"
OUTSIDE = "outcome_outside_expectation"
TAMPERED = "record_tampered"
MISMATCH = "fact_mismatch"
NOT_MET = "criterion_not_met"
IDENTITY = "identity_mismatch"
KINDS = (
    "question",
    "framing",
    "stakeholder",
    "exploration",
    "experiments",
    "quality",
    "selection",
    "approval",
    "deployment",
    "outcomes",
)
TITLES = {
    "question": "The business question and the signed charter",
    "framing": "Problem framing and the decision-cost model",
    "stakeholder": "KPIs, acceptance criteria and the approved expectation",
    "exploration": "Exploratory report on the exploration rows",
    "experiments": "Experiment comparison",
    "quality": "Domain quality thresholds with reviewer evidence",
    "selection": "Model selection, claims and the bundle identity",
    "approval": "Deployment approval and release state",
    "deployment": "Production decisions in the deployment ledger",
    "outcomes": "Observed outcomes against the approved expectation",
}
MINIMUM_CASES = 100


@dataclasses.dataclass(frozen=True, slots=True)
class Finding:
    """One problem the trail has.

    Attributes:
        kind: One of the module constants (``missing_link`` and so on).
        step: The step kind it belongs to.
        detail: One sentence.
    """

    kind: str
    step: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        """Builds a JSON-ready record.

        Returns:
            A dict with `kind`, `step` and `detail`.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class Derived:
    """What `derive` found.

    Attributes:
        steps: The ten steps in order (without chain fields).
        findings: Every problem.
        reports: The exploratory report, the experiment comparison and the
            outcome tracking, by file name under ``reports/``.
    """

    steps: tuple[dict[str, Any], ...]
    findings: tuple[Finding, ...]
    reports: Mapping[str, Any]


class Tracer:
    """One derivation over one trail folder.

    Attributes:
        directory: The trail folder.
        base: The folder's ``retained/`` directory.
        keys: Verification keys by role.
        findings: Problems found so far.
        steps: Steps derived so far, by kind.
        cache: Retained JSON records already read, by relative path.
        charter_hash: Digest of the chartered question, once derived.
        content_id: Digest of the model manifest, once derived.
        run_id: Identifier of the run that produced the model, once derived.
        inputs: The parsed ``inputs.json``.
        derived: Derived reports by relative path or pseudo-file name.
    """

    def __init__(
        self, directory: Path, keys: Mapping[str, str | bytes | None]
    ) -> None:
        """Binds the folder and the verification keys by role.

        Args:
            directory: The trail folder.
            keys: Verification keys by role.
        """
        self.directory = directory
        self.base = directory / records.RETAINED
        self.keys = dict(keys)
        self.findings: list[Finding] = []
        self.steps: dict[str, dict[str, Any]] = {}
        self.cache: dict[str, Any] = {}
        self.charter_hash = ""
        self.content_id = ""
        self.run_id = ""
        self.inputs: dict[str, Any] = {}
        self.derived: dict[str, Any] = {}

    def problem(self, kind: str, step: str, detail: str) -> None:
        """Records a finding.

        Args:
            kind: One of the finding kinds defined in this module.
            step: The step kind it belongs to.
            detail: One sentence.
        """
        self.findings.append(Finding(kind, step, detail))

    def need(self, relative: str, step: str) -> Path | None:
        """Finds a retained record, recording a finding when it is absent.

        Args:
            relative: Path relative to ``retained/``.
            step: The step that needs the record.

        Returns:
            The path, or None if the record is not a file.
        """
        path = self.base / relative
        if not path.is_file():
            self.problem(MISSING_LINK, step, f"retained/{relative} is absent")
            return None
        return path

    def json(self, relative: str, step: str) -> Any:
        """Reads a retained JSON record, or None with a finding.

        Results are cached, including a failure.

        Args:
            relative: Path relative to ``retained/``.
            step: The step that needs the record.

        Returns:
            The parsed record, or None if it is absent or not valid JSON.
        """
        if relative in self.cache:
            return self.cache[relative]
        path = self.need(relative, step)
        value = None
        if path is not None:
            try:
                value = records.read_json(path)
            except ValueError:
                self.problem(
                    TAMPERED, step, f"retained/{relative} is not valid JSON"
                )
        self.cache[relative] = value
        return value

    def rows(self, relative: str, step: str) -> list[dict[str, Any]] | None:
        """Reads a retained JSON-lines record, or None with a finding.

        Args:
            relative: Path relative to ``retained/``.
            step: The step that needs the record.

        Returns:
            The parsed rows, or None if the record is absent or not valid JSON
            lines.
        """
        path = self.need(relative, step)
        if path is None:
            return None
        try:
            return records.read_lines(path)
        except ValueError:
            self.problem(
                TAMPERED, step, f"retained/{relative} is not valid JSON lines"
            )
            return None

    def add(
        self,
        kind: str,
        files: list[str],
        facts: dict[str, Any],
        requires: list[str],
    ) -> None:
        """Stores a step with the digest of each record it rests on.

        Args:
            kind: The step kind.
            files: Records the step rests on, relative to ``retained/`` (or
                ``inputs.json``); those that do not exist are left out.
            facts: The facts read from the records.
            requires: Kinds of the steps this one builds on.
        """
        held = {}
        for relative in files:
            path = (
                self.directory / relative
                if relative == records.INPUTS
                else self.base / relative
            )
            if path.is_file():
                held[relative] = records.sha256_file(path)
        self.steps[kind] = {
            "kind": kind,
            "title": TITLES[kind],
            "records": dict(sorted(held.items())),
            "facts": facts,
            "requires": requires,
        }

    # -- the steps -----------------------------------------------------------

    def question(self) -> None:
        """Derives the chartered question; the charter must be signed."""
        name = "question"
        wrapper = self.json("run/analysis_charter.json", name)
        facts: dict[str, Any] = {}
        if isinstance(wrapper, dict) and "charter" in wrapper:
            charter = wrapper["charter"]
            self.charter_hash = records.digest(charter)
            if self.charter_hash != wrapper.get("charter_hash"):
                self.problem(
                    TAMPERED, name, "the charter hash does not match its text"
                )
            estimand = charter["estimand"]
            facts = {
                "charter_hash": self.charter_hash,
                "question": charter["question"],
                "target": estimand["target"],
                "unit": charter["unit"],
                "prediction_moment": estimand["prediction_moment"],
                "population": estimand["population"],
                "summary_measure": estimand["summary_measure"],
                "alpha_confirm": charter["error_budget"]["confirmation"][
                    "alpha_confirm"
                ],
            }
            self.check_charter_signature(wrapper, charter, name)
        self.add(name, ["run/analysis_charter.json"], facts, [])

    def check_charter_signature(
        self, wrapper: Mapping[str, Any], charter: Any, step: str
    ) -> None:
        """Requires the charter's HMAC to verify under the analyst key.

        Args:
            wrapper: The stored charter with its signature and hash.
            charter: The charter text that was signed.
            step: The step kind findings are recorded against.
        """
        signature = wrapper.get("signature")
        key = self.keys.get("analyst")
        if not signature:
            self.problem(UNSIGNED, step, "the charter carries no signature")
        elif key is None:
            self.problem(
                UNSIGNED, step, "no analyst key supplied to check the charter"
            )
        else:
            expected = signing.sign_payload(charter, key)
            if not hmac.compare_digest(str(signature), expected):
                self.problem(
                    UNSIGNED, step, "the charter signature does not verify"
                )

    def framing(self) -> None:
        """Derives the framing draft, its questions and the cost model."""
        name = "framing"
        files = [f"framing/{n}" for n in records.FRAMING_FILES]
        draft = self.json(files[0], name)
        questions = self.json(files[1], name)
        cost = self.json(files[2], name)
        facts: dict[str, Any] = {}
        if isinstance(draft, dict):
            if draft.get("charter_hash") != self.charter_hash:
                self.problem(
                    IDENTITY,
                    name,
                    "the framing draft was written for another charter",
                )
            sources = sorted(set(draft.get("fields", {}).values()))
            facts.update(
                framing_status=draft.get("status"),
                field_sources=sources,
                llm_filled=sorted(
                    k for k, v in draft.get("fields", {}).items() if v == "llm"
                ),
                open_items=list(draft.get("open_items", [])),
                data_checks_passed=all(
                    c.get("ok") for c in draft.get("checks", [])
                ),
            )
        if isinstance(questions, dict):
            facts["open_questions"] = len(questions.get("questions", []))
        if isinstance(cost, dict):
            wrapper = self.json("run/analysis_charter.json", name) or {}
            chartered = wrapper.get("charter", {}).get("cost_model")
            if chartered != cost:
                self.problem(
                    IDENTITY,
                    name,
                    "the cost model on file is not the one in the charter",
                )
            facts["costs"] = {k: cost[k] for k in sorted(cost) if k[:2] == "c_"}
            facts["cost_sources"] = list(cost.get("sources", []))
            facts["threshold"] = outcomes.optimal_threshold(cost)
        self.add(name, files, facts, ["question"])

    def stakeholder(self) -> None:
        """Derives the KPIs, acceptance criteria and approved expectation."""
        name = "stakeholder"
        found = inputs_module.structure_problems(self.inputs)
        for detail in found:
            self.problem(MISSING_LINK, name, f"inputs.json: {detail}")
        problem = inputs_module.signature_problem(
            self.inputs, "stakeholder", self.keys.get("stakeholder")
        )
        if problem:
            self.problem(UNSIGNED, name, problem)
        if self.inputs.get("charter_hash") != self.charter_hash:
            self.problem(
                IDENTITY, name, "inputs.json names another charter hash"
            )
        section = self.inputs.get("stakeholder", {})
        results = self.evaluate(section.get("acceptance", []), name)
        for item in results:
            if not item["passed"]:
                self.problem(
                    NOT_MET,
                    name,
                    f"acceptance {item['id']}: {item['path']} "
                    f"{item['op']} {item['value']} is not met "
                    f"(observed {item['observed']})",
                )
        self.add(
            name,
            [records.INPUTS],
            {
                "stakeholder": section.get("name"),
                "kpis": section.get("kpis", []),
                "acceptance": results,
                "expectation": section.get("expectation", {}),
                "signed": problem is None,
            },
            ["question", "framing"],
        )

    def evaluate(
        self, criteria: list[dict[str, Any]], step: str
    ) -> list[dict[str, Any]]:
        """Checks each criterion against the retained records.

        Args:
            criteria: Criteria with `id`, `path`, `op` and `value`.
            step: The step kind findings are recorded against.

        Returns:
            One result per criterion with the observed value and whether it
            passed. A criterion whose path cannot be resolved is recorded as a
            finding and counts as not passed.
        """
        results = []
        for item in criteria:
            observed: Any = None
            passed = False
            try:
                observed = records.resolve(
                    self.directory, item["path"], self.derived
                )
                passed = records.compare(observed, item["op"], item["value"])
            except (KeyError, FileNotFoundError, ValueError, TypeError):
                self.problem(
                    MISSING_LINK,
                    step,
                    f"criterion {item.get('id')} reads {item.get('path')}, "
                    "which cannot be resolved",
                )
            results.append(
                {
                    "id": item.get("id"),
                    "path": item.get("path"),
                    "op": item.get("op"),
                    "value": item.get("value"),
                    "observed": observed,
                    "passed": passed,
                    "reason": item.get("reason", ""),
                }
            )
        return results

    def report_step(
        self, name: str, relative: str, fresh: dict[str, Any], files: list[str]
    ) -> None:
        """Derives a report step: the stored copy must equal the recomputation.

        Args:
            name: The step kind.
            relative: Path of the stored report, relative to the trail folder.
            fresh: The report recomputed from the retained records.
            files: Records the step rests on.
        """
        stored_path = self.directory / relative
        if not stored_path.is_file():
            self.problem(MISSING_LINK, name, f"{relative} is absent")
        else:
            try:
                stored = records.read_json(stored_path)
            except ValueError:
                stored = None
            if stored != fresh:
                self.problem(
                    MISMATCH,
                    name,
                    f"{relative} is not what the retained records give",
                )
        self.derived[relative] = fresh
        self.add(
            name,
            files,
            {"report": relative, "sha256": fresh.get("sha256")},
            ["question"],
        )

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
            self.problem(
                MISSING_LINK, name, "the exploratory report cannot be rebuilt"
            )
            self.add(name, files, {}, ["question"])
            return
        self.derived["exploratory"] = fresh
        self.report_step(name, records.EXPLORATORY, fresh, files)

    def experiments(self) -> None:
        """Derives the experiment comparison."""
        name = "experiments"
        files = ["run/evaluation.json"]
        trials = sorted((self.base / "run" / "trials").glob("trial_*.json"))
        files += [f"run/trials/{p.name}" for p in trials]
        try:
            fresh = reports.experiments(self.directory)
        except (OSError, ValueError, KeyError):
            self.problem(MISSING_LINK, name, "the comparison cannot be rebuilt")
            self.add(name, files, {}, ["question"])
            return
        self.derived["experiments"] = fresh
        self.report_step(name, records.EXPERIMENTS, fresh, files)

    def quality(self) -> None:
        """Derives the domain thresholds, which the reviewer signs."""
        name = "quality"
        problem = inputs_module.signature_problem(
            self.inputs, "reviewer", self.keys.get("reviewer")
        )
        if problem:
            self.problem(UNSIGNED, name, problem)
        section = self.inputs.get("reviewer", {})
        results = self.evaluate(section.get("thresholds", []), name)
        for item in results:
            if not item["passed"]:
                self.problem(
                    NOT_MET,
                    name,
                    f"threshold {item['id']}: {item['path']} {item['op']} "
                    f"{item['value']} fails (observed {item['observed']})",
                )
        reviewed = section.get("evidence", {}).get("reviewed", {})
        for relative, held in sorted(reviewed.items()):
            path = self.base / relative
            if not path.is_file():
                self.problem(
                    MISSING_LINK, name, f"reviewed record {relative} is absent"
                )
            elif records.sha256_file(path) != held:
                self.problem(
                    TAMPERED,
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
        evaluation = self.json("run/evaluation.json", name) or {}
        manifest = self.need("run/model/model_manifest.json", name)
        facts: dict[str, Any] = {
            "run_id": evaluation.get("run_id"),
            "selected_model": evaluation.get("selected_model"),
            "metric": evaluation.get("metric"),
            "cv_score": evaluation.get("selected_cv_score"),
            "cv_interval": evaluation.get("cv_confidence", {}).get("interval"),
            "test_metrics": evaluation.get("test_metrics"),
            "n_test": evaluation.get("n_test"),
        }
        self.run_id = str(evaluation.get("run_id", ""))
        if manifest is not None:
            self.content_id = records.sha256_file(manifest)
            facts["content_id"] = self.content_id
        run_dir = self.base / "run"
        if run_dir.is_dir():
            verdict = claims_audit.verify(run_dir, self.keys.get("analyst"))
            facts["claims_steps"] = [
                [f.name, f.status] for f in verdict.findings
            ]
            for found in verdict.findings:
                if found.status == claims_audit.FAIL:
                    kind = (
                        BROKEN_CHAIN
                        if found.name.startswith("ledger chain")
                        else MISMATCH
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

    def approval(self) -> None:
        """Derives the registered version, its approval and release state."""
        name = "approval"
        facts: dict[str, Any] = {}
        index = self.json("registry/registry.json", name)
        audit = self.rows("registry/audit.jsonl", name)
        model, version = self.inputs.get("model"), self.inputs.get("version")
        entry = None
        if isinstance(index, dict):
            entry = (
                index.get("models", {})
                .get(str(model), {})
                .get("versions", {})
                .get(str(version))
            )
            if entry is None:
                self.problem(
                    MISSING_LINK,
                    name,
                    f"{model} v{version} is not in the retained registry",
                )
        if entry is not None:
            facts.update(
                model=model,
                version=version,
                registered_content_id=entry["content_id"],
                bundle_signed=entry["signed"],
                run_id=entry["run_id"],
            )
            if entry["content_id"] != self.content_id:
                self.problem(
                    IDENTITY,
                    name,
                    "the registered bundle is not the bundle the run wrote",
                )
            if entry["run_id"] != self.run_id:
                self.problem(IDENTITY, name, "the registry names another run")
            alias = index["models"][str(model)]["aliases"].get("champion")
            facts["champion_version"] = alias
        approval = self.promotion(audit or [], model, version, name)
        if approval is not None:
            facts["approval"] = {
                k: approval[k]
                for k in ("reviewer", "decision", "independent", "approved_at")
            }
            facts["approval"]["evidence"] = list(approval["evidence"])
        release = self.json("registry/release_state.json", name)
        if isinstance(release, dict):
            held = (
                release.get("models", {})
                .get(str(model), {})
                .get("versions", {})
                .get(str(version), {})
            )
            facts["release_state"] = held.get("state")
        self.add(
            name,
            [f"registry/{n}" for n in records.REGISTRY_FILES],
            facts,
            ["selection"],
        )

    def promotion(
        self,
        audit: list[dict[str, Any]],
        model: Any,
        version: Any,
        step: str,
    ) -> dict[str, Any] | None:
        """Checks the approval row of the champion promotion.

        Args:
            audit: Rows of the registry's audit log.
            model: Model name.
            version: Model version.
            step: The step kind findings are recorded against.

        Returns:
            The approval record of the latest champion promotion of this
            version, or None if there is none or it cannot be used. Findings are
            recorded for a missing, unsigned or unverifiable approval, for an
            approval that is not valid for this model version and bundle, and
            for an approval by the person who built the model.
        """
        rows = [
            r
            for r in audit
            if r.get("action") == "promote"
            and r.get("alias") == "champion"
            and r.get("model") == model
            and r.get("version") == version
        ]
        if not rows or not rows[-1].get("approval_record"):
            self.problem(
                UNAPPROVED,
                step,
                "no signed approval promoted this version to champion",
            )
            return None
        record = dict(rows[-1]["approval_record"])
        key = self.keys.get("reviewer")
        try:
            held = registry.PromotionApproval(
                model=record["model"],
                version=record["version"],
                content_id=record["content_id"],
                reviewer=record["reviewer"],
                decision=record["decision"],
                reason=record["reason"],
                evidence=tuple(record["evidence"]),
                approved_at=record["approved_at"],
                independent=record["independent"],
                signature=record.get("signature"),
            )
        except (KeyError, ValueError, TypeError) as error:
            self.problem(UNAPPROVED, step, f"approval record unusable: {error}")
            return None
        if not record.get("signature"):
            self.problem(UNSIGNED, step, "the approval carries no signature")
        elif key is None or not held.verify(key):
            self.problem(
                UNSIGNED, step, "the approval signature does not verify"
            )
        try:
            held.validate_target(str(model), int(version), self.content_id, key)
        except (ValueError, TypeError) as error:
            self.problem(UNAPPROVED, step, str(error))
        if record["reviewer"] == self.inputs.get("builder"):
            self.problem(
                UNAPPROVED,
                step,
                "the approver is the person who built the model",
            )
        return record

    def deployment(self) -> None:
        """Derives the deployment ledger: chain, charter, model and feed."""
        name = "deployment"
        path = self.need("deployment/deployment_ledger.jsonl", name)
        facts: dict[str, Any] = {}
        ledger = None
        if path is not None:
            try:
                ledger = deployment_ledger.DeploymentLedger.load(path)
            except errors.ScmleError as error:
                self.problem(BROKEN_CHAIN, name, f"deployment ledger: {error}")
        if ledger is not None:
            header = ledger.header
            if header["charter_hash"] != self.charter_hash:
                self.problem(
                    IDENTITY, name, "the ledger belongs to another charter"
                )
            scope = str(header["policy"].get("scope", ""))
            if not scope.startswith(self.content_id):
                self.problem(
                    IDENTITY,
                    name,
                    "the loop's scope does not name the approved bundle",
                )
            summaries = [
                r["body"]
                for r in ledger.rows
                if r["kind"] == deployment_ledger.OBSERVE_SUMMARY
            ]
            counts: dict[str, int] = {}
            for row in ledger.rows:
                counts[row["kind"]] = counts.get(row["kind"], 0) + 1
            facts.update(
                ledger_head=ledger.head,
                ledger_rows=len(ledger.rows),
                row_kinds=dict(sorted(counts.items())),
                batches=len(summaries),
                cases=sum(int(s["cases"]) for s in summaries),
                escalated=sum(int(s["escalated"]) for s in summaries),
                answered=sum(int(s["answered"]) for s in summaries),
                halted=any(r["kind"] == "halt" for r in ledger.rows),
                scope=scope,
                charter_hash=header["charter_hash"],
            )
            self.reconcile_feed(summaries, name)
        self.add(
            name,
            [f"deployment/{n}" for n in records.DEPLOYMENT_FILES],
            facts,
            ["approval"],
        )

    def reconcile_feed(
        self, summaries: list[dict[str, Any]], step: str
    ) -> None:
        """Requires the outcome feed to cover the cases the ledger saw.

        Args:
            summaries: The ledger's observation summaries.
            step: The step kind findings are recorded against.
        """
        feed = self.json("deployment/observed_outcomes.json", step)
        if not isinstance(feed, dict):
            return
        sizes = [len(b["label"]) for b in feed.get("batches", [])]
        seen = [int(s["cases"]) for s in summaries]
        if sizes != seen:
            self.problem(
                MISMATCH,
                step,
                "the outcome feed does not match the cases the ledger "
                f"recorded (feed {sizes}, ledger {seen})",
            )

    def outcomes_step(self) -> None:
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
                self.problem(
                    MISMATCH, name, "the outcome feed cannot be evaluated"
                )
            expected_cut = outcomes.optimal_threshold(cost)
            if abs(float(feed.get("threshold", -1)) - expected_cut) > 1e-9:
                self.problem(
                    IDENTITY,
                    name,
                    "the served decision rule is not the chartered cut",
                )
        tracking = reports.with_hash({"kind": "outcome_tracking", **tracking})
        self.derived["outcomes"] = tracking
        self.report_step(
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
                OUTSIDE,
                name,
                f"observed cost per case {tracking['mean_cost']:.3f} "
                f"(95% interval {tracking['interval'][0]:.3f} to "
                f"{tracking['interval'][1]:.3f}) is {status} the approved "
                f"range {tracking['approved_range']}",
            )
        elif status in (outcomes.INSUFFICIENT, outcomes.ABSENT):
            self.problem(
                MISSING_LINK,
                name,
                f"outcome status is {status}: no observed outcome can be "
                "compared with the expectation",
            )


def derive(
    directory: str | Path, keys: Mapping[str, str | bytes | None]
) -> Derived:
    """Derives every step and finding from a trail folder's retained records.

    Args:
        directory: The trail folder.
        keys: Verification keys by role: ``analyst``, ``stakeholder``,
            ``reviewer``. A missing key makes the signature it covers
            unverifiable, which is reported as unsigned.

    Returns:
        The steps, findings and derived reports.
    """
    folder = Path(directory)
    tracer = Tracer(folder, keys)
    path = folder / records.INPUTS
    if path.is_file():
        try:
            tracer.inputs = records.read_json(path)
        except ValueError:
            tracer.problem(TAMPERED, "stakeholder", "inputs.json is not JSON")
    else:
        tracer.problem(MISSING_LINK, "stakeholder", "inputs.json is absent")
    tracer.question()
    tracer.framing()
    tracer.exploration()
    tracer.experiments()
    tracer.selection()
    tracer.outcomes_step()
    tracer.stakeholder()
    tracer.quality()
    tracer.approval()
    tracer.deployment()
    ordered = tuple(tracer.steps[k] for k in KINDS)
    return Derived(
        ordered,
        tuple(tracer.findings),
        {
            "exploratory_report.json": tracer.derived.get(records.EXPLORATORY),
            "experiment_comparison.json": tracer.derived.get(
                records.EXPERIMENTS
            ),
            "outcome_tracking.json": tracer.derived.get(
                records.OUTCOME_TRACKING
            ),
        },
    )
