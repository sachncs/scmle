#!/usr/bin/env python3
"""Walks one deployed model's trail using only the retained records.

    python scripts/trace_model.py TRAIL_DIR [--keys keys.json]

This is the reviewer's tool. It imports nothing from ``scmle`` and reads
nothing outside ``TRAIL_DIR``: not the code that built the model, not the
machine that ran it. It re-computes from the retained files, in the order a
reviewer would ask, whether

1. every retained file is the one that was sealed and the steps chain,
2. the charter says what the business question was and its signature holds,
3. the cost model, the stakeholder's criteria and the reviewer's thresholds
   are signed and are met by the retained numbers,
4. the model that was evaluated is the model that was registered, approved
   by a signed record, and the one the deployment ledger names,
5. the deployment ledger and the claim ledger are unbroken hash chains,
6. the observed outcomes, costed with the chartered cost model, sit inside
   the range the stakeholder approved, with their uncertainty.

Exit codes: 0 every check passed with keys supplied; 1 a check failed; 2 no
keys were supplied, so signatures were not checked.

A signature made with an HMAC key proves the holder of that key made it, not
who that is. A passing walk does not show the data are right or the model is
good; it shows the records agree and were approved by the named keys.
"""

import argparse
import hashlib
import hmac
import json
import math
from pathlib import Path
import sys
from typing import Any

GENESIS = "0" * 64
Z95 = 1.959964
OPERATORS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
}


def canonical(value: Any) -> bytes:
    """The bytes a digest or a signature covers."""
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def digest(value: Any) -> str:
    """SHA-256 of the canonical JSON of a value."""
    return hashlib.sha256(canonical(value)).hexdigest()


def sign(value: Any, key: str) -> str:
    """HMAC-SHA256 over the canonical JSON of a value."""
    return hmac.new(key.encode(), canonical(value), hashlib.sha256).hexdigest()


def file_digest(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> Any:
    """Reads a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path: Path) -> list[dict[str, Any]]:
    """Reads a JSON-lines file."""
    return [
        json.loads(text)
        for text in path.read_text(encoding="utf-8").splitlines()
        if text.strip()
    ]


def dotted(document: Any, path: str) -> Any:
    """Follows ``a.b.0`` through mappings and lists."""
    for part in path.split("."):
        document = (
            document[int(part)]
            if isinstance(document, list)
            else document[part]
        )
    return document


class Walk:
    """Collects the checks of one walk and prints them as it goes."""

    def __init__(self, directory: Path, keys: dict[str, str]) -> None:
        """Binds the trail folder and the keys by role."""
        self.directory = directory
        self.base = directory / "retained"
        self.keys = keys
        self.failed = 0
        self.unchecked = 0

    def say(self, text: str = "") -> None:
        """Prints a line."""
        print(text)

    def check(self, passed: bool, text: str) -> bool:
        """Prints a check and counts a failure."""
        self.say(f"  [{'ok' if passed else 'FAIL'}] {text}")
        if not passed:
            self.failed += 1
        return passed

    def signed(
        self, payload: Any, signature: str | None, role: str, text: str
    ) -> bool:
        """Checks an HMAC with the role's key, or notes it is unchecked."""
        key = self.keys.get(role)
        if not signature:
            return self.check(False, f"{text}: no signature")
        if key is None:
            self.unchecked += 1
            self.say(f"  [--] {text}: not checked, no {role} key supplied")
            return False
        return self.check(
            hmac.compare_digest(signature, sign(payload, key)),
            f"{text}: signature made with the {role} key",
        )

    def criteria(
        self, items: list[dict[str, Any]], derived: dict[str, Any]
    ) -> None:
        """Evaluates ``file#path op value`` criteria on the records."""
        for item in items:
            name, _, path = item["path"].partition("#")
            document = (
                derived[name] if name in derived else load(self.base / name)
            )
            observed = dotted(document, path)
            met = OPERATORS[item["op"]](observed, item["value"])
            self.check(
                met,
                f"{item['id']}: {item['path']} {item['op']} {item['value']}"
                f" (observed {observed}); {item.get('reason', '')}",
            )

    def files_and_chain(self) -> dict[str, Any]:
        """Step 1: sealed files and the step chain."""
        self.say("1. Sealed records and the step chain")
        trail = load(self.directory / "trail.json")
        for relative, held in sorted(trail["retained"].items()):
            path = self.directory / relative
            self.check(
                path.is_file() and file_digest(path) == held,
                f"{relative} is the sealed file",
            )
        previous = GENESIS
        intact = True
        for step in trail["steps"]:
            body = {k: v for k, v in step.items() if k != "hash"}
            intact &= step["prev"] == previous and step["hash"] == digest(body)
            previous = step["hash"]
        self.check(intact and trail["head"] == previous, "steps chain to head")
        self.signed(
            {"head": trail["head"], "retained": digest(trail["retained"])},
            trail["builder_signature"],
            "analyst",
            "the trail file",
        )
        return trail

    def question(self) -> dict[str, Any]:
        """Step 2: the chartered question."""
        self.say("2. The business question")
        wrapper = load(self.base / "run/analysis_charter.json")
        charter = wrapper["charter"]
        self.say(f"  question: {charter['question']}")
        self.say(
            f"  target {charter['estimand']['target']}, unit "
            f"{charter['unit']}, moment "
            f"{charter['estimand']['prediction_moment']}"
        )
        self.check(
            digest(charter) == wrapper["charter_hash"], "charter hash matches"
        )
        self.signed(charter, wrapper["signature"], "analyst", "the charter")
        return wrapper

    def framing(
        self, charter_hash: str, charter: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        """Step 3: the framing and the cost model."""
        self.say("3. Framing and the decision-cost model")
        draft = load(self.base / "framing/framing_draft.json")
        costs = load(self.base / "framing/decision_cost_model.json")
        self.check(
            draft["charter_hash"] == charter_hash,
            "the framing draft is for this charter",
        )
        self.check(
            charter["cost_model"] == costs,
            "the cost model is the chartered one",
        )
        self.say(f"  open items left to a person: {draft['open_items']}")
        cut = (costs["c_10"] - costs["c_00"]) / (
            costs["c_10"] - costs["c_00"] + costs["c_01"] - costs["c_11"]
        )
        self.say(
            f"  costs { ({k: v for k, v in costs.items() if k[:2] == 'c_'}) }"
        )
        self.say(f"  cost-minimising cut on the probability: {cut:.4f}")
        return costs, cut

    def outcomes(
        self, costs: dict[str, Any], cut: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Costs the observed outcomes and compares them with the range."""
        feed = load(self.base / "deployment/observed_outcomes.json")
        self.check(
            abs(feed["threshold"] - cut) < 1e-9,
            "the served cut is the chartered cut",
        )
        spent = []
        for batch in feed["batches"]:
            for p_value, label in zip(
                batch["probability"], batch["label"], strict=True
            ):
                action = 1 if p_value >= feed["threshold"] else 0
                spent.append(costs[f"c_{action}{int(label)}"])
        n = len(spent)
        mean = math.fsum(spent) / n
        spread = math.fsum((v - mean) ** 2 for v in spent) / (n - 1)
        half = Z95 * math.sqrt(spread / n)
        return feed, {
            "cases": n,
            "mean_cost": mean,
            "interval": [mean - half, mean + half],
        }

    def walk(self) -> None:
        """Runs every step."""
        trail = self.files_and_chain()
        wrapper = self.question()
        costs, cut = self.framing(trail["charter_hash"], wrapper["charter"])
        inputs = load(self.directory / "inputs.json")
        self.say("4. KPIs, acceptance and the approved expectation")
        feed, observed = self.outcomes(costs, cut)
        derived = {"outcomes": observed}
        for kpi in inputs["stakeholder"]["kpis"]:
            self.say(f"  KPI '{kpi['kpi']}' -> {kpi['metric']}")
        self.check(
            inputs["charter_hash"] == trail["charter_hash"],
            "inputs name this charter",
        )
        for role in ("stakeholder", "reviewer"):
            section = {
                k: v for k, v in inputs[role].items() if k != "signature"
            }
            payload = {
                "role": role,
                "model": inputs["model"],
                "version": inputs["version"],
                "charter_hash": inputs["charter_hash"],
                "body": section,
            }
            self.signed(payload, inputs[role].get("signature"), role, role)
        self.criteria(inputs["stakeholder"]["acceptance"], derived)
        self.say("5. Domain quality thresholds and reviewer evidence")
        self.criteria(inputs["reviewer"]["thresholds"], derived)
        for relative, held in inputs["reviewer"]["evidence"][
            "reviewed"
        ].items():
            self.check(
                file_digest(self.base / relative) == held,
                f"{relative} is unchanged since the reviewer signed",
            )
        self.say(f"  statement: {inputs['reviewer']['evidence']['statement']}")
        self.selection_and_approval(inputs, trail)
        self.deployment(trail)
        self.say("8. Observed outcomes against the approved expectation")
        expectation = inputs["stakeholder"]["expectation"]
        low, high = observed["interval"]
        self.say(
            f"  {observed['cases']} cases; "
            f"mean cost {observed['mean_cost']:.4f}; "
            f"95% interval {low:.4f} to {high:.4f}"
        )
        self.say(
            f"  approved range {expectation['low']} to {expectation['high']}"
        )
        self.check(
            not (low > expectation["high"] or high < expectation["low"]),
            "the observed interval overlaps the approved range",
        )
        self.say(f"  source of the feed: {feed['source']}")
        self.say(
            "  This is a normal-approximation interval, not a forecast; "
            "it does not show the model caused the outcome."
        )

    def selection_and_approval(
        self, inputs: dict[str, Any], trail: dict[str, Any]
    ) -> None:
        """Step 6: model identity and the signed approval."""
        self.say("6. Selection, registry and the signed approval")
        evaluation = load(self.base / "run/evaluation.json")
        content_id = file_digest(self.base / "run/model/model_manifest.json")
        self.say(
            f"  selected {evaluation['selected_model']}, test "
            f"{evaluation['test_metrics']['roc_auc']:.4f} ROC AUC"
        )
        index = load(self.base / "registry/registry.json")
        entry = index["models"][inputs["model"]]["versions"][
            str(inputs["version"])
        ]
        self.check(
            entry["content_id"] == content_id,
            "the registered bundle is the bundle the run wrote",
        )
        self.check(
            entry["run_id"] == evaluation["run_id"], "registry names this run"
        )
        promoted = [
            row
            for row in lines(self.base / "registry/audit.jsonl")
            if row.get("action") == "promote" and row.get("alias") == "champion"
        ]
        approval = promoted[-1]["approval_record"]
        body = {k: v for k, v in approval.items() if k != "signature"}
        self.signed(body, approval["signature"], "reviewer", "the approval")
        self.check(
            approval["decision"] == "approved" and approval["independent"],
            "the approval is an independent approval",
        )
        self.check(
            approval["content_id"] == content_id,
            "the approval is bound to this bundle",
        )
        self.check(
            approval["reviewer"] != inputs["builder"],
            f"the approver ({approval['reviewer']}) is not the builder "
            f"({inputs['builder']})",
        )

    def deployment(self, trail: dict[str, Any]) -> None:
        """Step 7: the deployment ledger."""
        self.say("7. The deployment ledger")
        rows = lines(self.base / "deployment/deployment_ledger.jsonl")
        previous, intact = GENESIS, True
        for index, row in enumerate(rows):
            content = {k: row[k] for k in ("seq", "kind", "body", "prev")}
            intact &= (
                row["seq"] == index
                and row["prev"] == previous
                and row["hash"] == digest(content)
            )
            previous = row["hash"]
        self.check(intact, f"{len(rows)} ledger rows chain")
        header = rows[0]["body"]
        self.check(
            header["charter_hash"] == trail["charter_hash"],
            "the ledger belongs to this charter",
        )
        content_id = file_digest(self.base / "run/model/model_manifest.json")
        self.check(
            header["policy"]["scope"].startswith(content_id),
            "the loop's scope names the approved bundle",
        )
        cases = sum(
            r["body"]["cases"] for r in rows if r["kind"] == "observe_summary"
        )
        self.say(f"  the loop saw {cases} cases in the ledger")
        claims = lines(self.base / "run/hypothesis_ledger.jsonl")
        previous, intact = GENESIS, True
        for row in claims:
            content = {k: row[k] for k in ("seq", "kind", "body", "prev")}
            intact &= row["prev"] == previous and row["hash"] == digest(content)
            previous = row["hash"]
        self.check(intact, f"{len(claims)} claim-ledger rows chain")


def main(arguments: list[str]) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("trail_dir", type=Path)
    parser.add_argument("--keys", type=Path, help="JSON: role -> key")
    options = parser.parse_args(arguments)
    keys = load(options.keys) if options.keys else {}
    walk = Walk(options.trail_dir, keys)
    try:
        walk.walk()
    except (OSError, KeyError, ValueError, IndexError) as error:
        print(
            f"  [FAIL] a record needed for the walk is missing or "
            f"malformed: {error!r}"
        )
        return 1
    print()
    if walk.failed:
        print(f"{walk.failed} check(s) failed")
        return 1
    if walk.unchecked:
        print(f"{walk.unchecked} signature(s) were not checked (no key)")
        return 2
    print("every check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
