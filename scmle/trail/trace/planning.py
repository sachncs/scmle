"""The planning steps of a trail: question, framing and stakeholder."""

from collections.abc import Mapping
import hmac
from typing import Any

from scmle.contracts import signing
from scmle.trail import inputs
from scmle.trail import outcomes
from scmle.trail import records
from scmle.trail.trace import finding
from scmle.trail.trace import reader


class Planning(reader.Reader):
    """Derives the chartered question, its framing and the stakeholder step."""

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
                    finding.TAMPERED,
                    name,
                    "the charter hash does not match its text",
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
            self.verify(wrapper, charter, name)
        self.add(name, ["run/analysis_charter.json"], facts, [])

    def verify(
        self, wrapper: Mapping[str, Any], charter: Any, step: str
    ) -> None:
        """Records a finding unless the charter's HMAC verifies.

        A finding of kind `unsigned_step` is recorded if the wrapper has no
        signature, no analyst key was supplied, or the signature does not
        match the analyst key.

        Args:
            wrapper: The stored charter with its signature and hash.
            charter: The charter text that was signed.
            step: The step kind findings are recorded against.
        """
        signature = wrapper.get("signature")
        key = self.keys.get("analyst")
        if not signature:
            self.problem(
                finding.UNSIGNED, step, "the charter carries no signature"
            )
        elif key is None:
            self.problem(
                finding.UNSIGNED,
                step,
                "no analyst key supplied to check the charter",
            )
        else:
            expected = signing.mac(charter, key)
            if not hmac.compare_digest(str(signature), expected):
                self.problem(
                    finding.UNSIGNED,
                    step,
                    "the charter signature does not verify",
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
                    finding.IDENTITY,
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
                open_items=list(
                    draft.get("pending") or draft.get("open_items") or []
                ),
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
                    finding.IDENTITY,
                    name,
                    "the cost model on file is not the one in the charter",
                )
            facts["costs"] = {k: cost[k] for k in sorted(cost) if k[:2] == "c_"}
            facts["cost_sources"] = list(cost.get("sources", []))
            facts["threshold"] = outcomes.optimise(cost)
        self.add(name, files, facts, ["question"])

    def stakeholder(self) -> None:
        """Derives the KPIs, acceptance criteria and approved expectation."""
        name = "stakeholder"
        found = inputs.lint(self.inputs)
        for detail in found:
            self.problem(finding.MISSING_LINK, name, f"inputs.json: {detail}")
        problem = inputs.diagnose(
            self.inputs, "stakeholder", self.keys.get("stakeholder")
        )
        if problem:
            self.problem(finding.UNSIGNED, name, problem)
        if self.inputs.get("charter_hash") != self.charter_hash:
            self.problem(
                finding.IDENTITY, name, "inputs.json names another charter hash"
            )
        section = self.inputs.get("stakeholder", {})
        results = self.evaluate(section.get("acceptance", []), name)
        for item in results:
            if not item["passed"]:
                self.problem(
                    finding.NOT_MET,
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
