"""Approval steps: signed decisions, their checks, and settling a step."""

from collections.abc import Mapping
import hmac
from typing import Any

from scmle.contracts import signing
from scmle.workflow import definition
from scmle.workflow.executor import failures
from scmle.workflow.executor import status
from scmle.workflow.executor import steps


class Approvals(steps.Steps):
    """Records signed decisions and settles approval steps with them."""

    def admit(
        self,
        step: definition.Step,
        states: Mapping[str, str],
        group_use: Mapping[str, int],
    ) -> bool:
        """Checks that the step's parents succeeded and its group has room.

        Args:
            step: The pending step.
            states: Step state value by step name.
            group_use: Running steps per concurrency group.

        Returns:
            True when every step it depends on succeeded and its
            concurrency group (if any) is below its limit.
        """
        if any(states[d] != status.Phase.SUCCEEDED for d in step.needs):
            return False
        group = step.concurrency
        return group is None or group_use.get(group, 0) < self.workflow.quota(
            group
        )

    def adjudicate(
        self,
        run_id: str,
        step: definition.Step,
        states: dict[str, str],
        outputs: dict[str, Mapping[str, Any]],
    ) -> bool:
        """Settles an approval step and records its outputs when approved.

        Args:
            run_id: The run.
            step: The approval step.
            states: Step state value by step name, updated in place.
            outputs: Step outputs by step name, updated in place.

        Returns:
            True when the step's state changed from waiting.
        """
        settled = self.settle(run_id, step)
        states[step.name] = settled.value
        if settled is status.Phase.SUCCEEDED:
            saved = self.step(run_id, step.name) or {}
            outputs[step.name] = saved.get("outputs", {})
        return settled is not status.Phase.WAITING

    def compose(
        self, run_id: str, step: str, approver: str, decision: str, at: float
    ) -> dict[str, Any]:
        """Builds the content an approver signs.

        Args:
            run_id: The run.
            step: The approval step name.
            approver: The approver's identity.
            decision: `approve` or `reject`.
            at: Decision time in epoch seconds.

        Returns:
            A mapping that also names the workflow digest.
        """
        return {
            "run_id": run_id,
            "workflow_digest": self.workflow.digest(),
            "step": step,
            "approver": approver,
            "decision": decision,
            "at": at,
        }

    def decide(
        self,
        run_id: str,
        step: str,
        approver: str,
        decision: str,
        key: str,
        at: float | None = None,
    ) -> dict[str, Any]:
        """Builds a signed decision (what an approver's tool does).

        Args:
            run_id: The run.
            step: The approval step name.
            approver: The approver's identity.
            decision: `approve` or `reject`.
            key: The approver's key.
            at: Decision time in epoch seconds; the current time if None.

        Returns:
            The decision payload (run, workflow digest, step, approver,
            decision and time) with an added `signature` entry.
        """
        moment = self.clock() if at is None else at
        payload = self.compose(run_id, step, approver, decision, moment)
        return {**payload, "signature": signing.mac(payload, key)}

    def ratify(self, decision: Mapping[str, Any]) -> None:
        """Records a signed decision after checking it.

        Args:
            decision: A decision produced by `decide`. A decision that
                names a step the workflow does not have raises `ConfigError`
                (from `Workflow.step`).

        Raises:
            ApprovalError: If the approver is unknown or not listed for the
                step, is the person who requested the run, the signature does
                not verify, the decision is not approve or reject, a decision is
                already recorded, or the run does not exist.
        """
        run_id, name = str(decision["run_id"]), str(decision["step"])
        record = self.record(run_id)
        if record is None:
            raise failures.ApprovalError("no such run")
        step = self.workflow.step(name)
        approver = str(decision["approver"])
        if (
            step.kind is not definition.Kind.APPROVAL
            or approver not in step.approvers
        ):
            raise failures.ApprovalError("approver is not listed for this step")
        if approver == record["requested_by"]:
            raise failures.ApprovalError(
                "the requester cannot approve their own run"
            )
        key = self.approvers.get(approver)
        payload = self.compose(
            run_id,
            name,
            approver,
            str(decision["decision"]),
            float(decision["at"]),
        )
        if key is None or not hmac.compare_digest(
            signing.mac(payload, key),
            str(decision.get("signature", "")),
        ):
            raise failures.ApprovalError(
                "the approval signature does not verify"
            )
        if decision["decision"] not in ("approve", "reject"):
            raise failures.ApprovalError("decision must be approve or reject")
        bucket = self.bucket(run_id)
        if bucket.fetch(f"approvals/{name}.json"):
            raise failures.ApprovalError("a decision is already recorded")
        bucket.keep(f"approvals/{name}.json", decision)

    def settle(self, run_id: str, step: definition.Step) -> status.Phase:
        """Determines the approval step's state now.

        A recorded decision settles the step to succeeded (approve) or rejected;
        otherwise the step waits, and expires once it has waited longer than its
        expiry.

        Args:
            run_id: The run.
            step: The approval step.

        Returns:
            The step's new state.
        """
        decision = self.bucket(run_id).fetch(f"approvals/{step.name}.json")
        if decision is not None:
            if decision["decision"] == "approve":
                self.stash(
                    run_id,
                    step,
                    status.Phase.SUCCEEDED,
                    outputs={"approver": decision["approver"]},
                    cost=0.0,
                    idempotency_key="approval",
                )
                return status.Phase.SUCCEEDED
            self.stash(
                run_id,
                step,
                status.Phase.REJECTED,
                approver=decision["approver"],
            )
            return status.Phase.REJECTED
        waiting = self.step(run_id, step.name)
        since = (
            waiting["waiting_since"]
            if waiting and waiting["state"] == status.Phase.WAITING
            else self.clock()
        )
        if self.clock() - since > step.expiry:
            self.stash(run_id, step, status.Phase.EXPIRED, waiting_since=since)
            return status.Phase.EXPIRED
        self.stash(run_id, step, status.Phase.WAITING, waiting_since=since)
        return status.Phase.WAITING
