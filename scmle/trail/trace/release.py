"""The release steps of a trail: approval and deployment."""

from collections.abc import Mapping
from typing import Any

from scmle.contracts import errors
from scmle.governance import deployment_ledger
from scmle.governance import registry
from scmle.trail import records
from scmle.trail.trace import finding
from scmle.trail.trace import reader


def sanction(record: Mapping[str, Any]) -> registry.Sanction:
    """Rebuilds a stored approval, requiring every signed field.

    Args:
        record: The approval in its version-1 layout (`content_id`,
            `approved_at`); the written layout is moved down first.

    Returns:
        The approval.

    Raises:
        KeyError: If a signed field is missing.
        ValueError: If a field is invalid.
    """
    return registry.Sanction(
        model=record["model"],
        version=record["version"],
        content=record["content_id"],
        reviewer=record["reviewer"],
        decision=record["decision"],
        reason=record["reason"],
        evidence=tuple(record["evidence"]),
        approved=record["approved_at"],
        independent=record["independent"],
        signature=record.get("signature"),
    )


class Release(reader.Reader):
    """Derives the approved release and the production deployment."""

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
                    finding.MISSING_LINK,
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
            if entry["content_id"] != self.content:
                self.problem(
                    finding.IDENTITY,
                    name,
                    "the registered bundle is not the bundle the run wrote",
                )
            if entry["run_id"] != self.run:
                self.problem(
                    finding.IDENTITY, name, "the registry names another run"
                )
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
            The stored approval record of the last audit row promoting this
            version to champion, or None if there is no such row, it carries
            no approval record, or the record cannot be read. The record is
            returned even when findings are recorded against it: an unsigned
            or unverifiable signature, an approval that names another bundle,
            was rejected or was not independent, and an approver who is the
            person who built the model.
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
                finding.UNAPPROVED,
                step,
                "no signed approval promoted this version to champion",
            )
            return None
        key = self.keys.get("reviewer")
        record = dict(rows[-1]["approval_record"])
        try:
            record = registry.Sanction.KEYS.downgrade(record)
            held = sanction(record)
        except (KeyError, ValueError, TypeError) as error:
            # handled: recorded as trace problem
            self.problem(
                finding.UNAPPROVED, step, f"approval record unusable: {error}"
            )
            return None
        if not record.get("signature"):
            self.problem(
                finding.UNSIGNED, step, "the approval carries no signature"
            )
        elif key is None or not held.verify(key):
            self.problem(
                finding.UNSIGNED, step, "the approval signature does not verify"
            )
        try:
            held.validate(str(model), int(version), self.content, key)
        except (ValueError, TypeError) as error:
            # handled: recorded as trace problem
            self.problem(finding.UNAPPROVED, step, str(error))
        if record["reviewer"] == self.inputs.get("builder"):
            self.problem(
                finding.UNAPPROVED,
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
                ledger = deployment_ledger.Ledger.load(path)
            except errors.ScmleError as error:
                # handled: recorded as trace problem
                self.problem(
                    finding.BROKEN_CHAIN, name, f"deployment ledger: {error}"
                )
        if ledger is not None:
            header = ledger.header
            if header["charter_hash"] != self.charter:
                self.problem(
                    finding.IDENTITY,
                    name,
                    "the ledger belongs to another charter",
                )
            scope = str(header["policy"].get("scope", ""))
            if not scope.startswith(self.content):
                self.problem(
                    finding.IDENTITY,
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
            self.reconcile(summaries, name)
        self.add(
            name,
            [f"deployment/{n}" for n in records.DEPLOYMENT_FILES],
            facts,
            ["approval"],
        )

    def reconcile(self, summaries: list[dict[str, Any]], step: str) -> None:
        """Records a mismatch unless the feed matches the ledger's cases.

        The feed's per-batch case counts (length of each `label` list) must
        equal the ledger's per-observation `cases`, in order. Nothing is
        checked when the feed is absent or not an object; reading it has
        already recorded a finding.

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
                finding.MISMATCH,
                step,
                "the outcome feed does not match the cases the ledger "
                f"recorded (feed {sizes}, ledger {seen})",
            )
