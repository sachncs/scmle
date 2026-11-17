"""The closed loop of one deployment: opening it and each step."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import datetime
from pathlib import Path
import types
from typing import Any

from scmle.contracts import signing
from scmle.governance import deployment_ledger
from scmle.governance import never_list
from scmle.governance.loop import alarms
from scmle.governance.loop import guard
from scmle.governance.loop import intake
from scmle.governance.loop import terms
from scmle.monitoring import labels


class Loop(intake.Intake, alarms.Alarms, guard.Gatekeeper):
    """One deployment's closed loop (see the package docstring)."""

    __module__ = "scmle.governance.loop"  # the path pinned signatures name

    @classmethod
    def open(
        cls,
        deploy_dir: str | Path,
        charter: Any,
        policy: terms.Preregistration,
        *,
        hooks: terms.Hooks | None = None,
        verification_keys: Mapping[str, str | bytes] | None = None,
        system_keys: Sequence[str | bytes] = (),
        today: Callable[[], datetime.date] = datetime.date.today,
        single_use_attestations: bool = False,
    ) -> "Loop":
        """Opens the loop of a deployment, creating its ledger on first use.

        Creates the directory if needed, loads the saved state (halting if it
        does not match the ledger) and saves the state again.

        Args:
            deploy_dir: Directory of `deployment_ledger.jsonl` and
                `loop_state.json`.
            charter: Anything with a `charter_hash` (the analysis charter).
            policy: The pre-registered inputs.
            hooks: Optional callbacks.
            verification_keys: Map from key id (`thumbprint`) to the
                key that verifies a person's attestation. The system must not
                hold the signing side of these keys.
            system_keys: Keys the system itself holds; attestations signed
                with them (or with `SCMLE_SIGNING_KEY`) are refused. Their
                fingerprints are recorded only when the ledger is created; for
                an existing ledger the ones it already records apply and
                this argument has no effect.
            today: Clock, replaceable in tests.
            single_use_attestations: Spend an attestation on the first action
                it allows (an `attestation_used` row) instead of leaving it
                valid until it expires.

        Returns:
            The open loop, with its saved state loaded.

        Raises:
            MisuseError: If an existing ledger fails verification
                or belongs to another charter or policy.
        """
        directory = Path(deploy_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / deployment_ledger.FILE_NAME
        fingerprints = [deployment_ledger.thumbprint(k) for k in system_keys]
        environment = signing.keying(None)
        if environment is not None:
            fingerprints.append(deployment_ledger.thumbprint(environment))
        keys = dict(verification_keys or {})
        if path.exists():
            ledger = deployment_ledger.DeploymentLedger.load(path)
            header = ledger.header
            if header["charter_hash"] != charter.charter_hash:
                raise deployment_ledger.MisuseError(
                    "ledger belongs to a different charter"
                )
            if terms.amend(ledger) != terms.jsonify(policy.to_dict()):
                raise deployment_ledger.MisuseError(
                    "ledger was opened with a different policy"
                )
        else:
            ledger = deployment_ledger.DeploymentLedger.create(
                path,
                charter_hash=charter.charter_hash,
                policy=terms.jsonify(policy.to_dict()),
                alpha_promo=policy.alpha_promo,
                system_keys=sorted(set(fingerprints)),
            )
        guard = never_list.Guard(
            ledger, today, keys, single_use=single_use_attestations
        )
        actions = hooks or terms.Hooks()
        loop = cls(directory, ledger, policy, guard, actions, keys, today)
        loop.restore()
        loop.save_state()
        return loop

    @classmethod
    def resume(
        cls,
        deploy_dir: str | Path,
        *,
        hooks: terms.Hooks | None = None,
        verification_keys: Mapping[str, str | bytes] | None = None,
        system_keys: Sequence[str | bytes] = (),
        today: Callable[[], datetime.date] = datetime.date.today,
        single_use_attestations: bool = False,
    ) -> "Loop":
        """Opens an existing deployment from what its ledger records.

        The charter hash and the policy (with its recorded amendments) are
        read from the ledger, so a command line needs only the directory. A
        missing ledger, or one that fails verification, is refused by the
        ledger with a `MisuseError`. `system_keys` has no effect
        here, because the ledger already exists (see `open`).

        Args:
            deploy_dir: Directory of the deployment ledger.
            hooks: Optional callbacks.
            verification_keys: Map from key id to the key that verifies a
                person's attestation.
            system_keys: Keys the system itself holds; ignored, as the
                ledger records its own.
            today: Clock, replaceable in tests.
            single_use_attestations: Spend an attestation on the first action
                it allows.

        Returns:
            The open loop.
        """
        path = Path(deploy_dir) / deployment_ledger.FILE_NAME
        ledger = deployment_ledger.DeploymentLedger.load(path)
        policy = terms.Preregistration.from_dict(terms.amend(ledger))
        charter = types.SimpleNamespace(
            charter_hash=ledger.header["charter_hash"]
        )
        return cls.open(
            deploy_dir,
            charter,
            policy,
            hooks=hooks,
            verification_keys=verification_keys,
            system_keys=system_keys,
            today=today,
            single_use_attestations=single_use_attestations,
        )

    def step(self) -> terms.Outcome:
        """Processes matured batches and applies the decision map.

        A batch is processed once ``outcome_window`` batches have passed since
        it was observed; younger batches stay pending. If the ledger fails
        verification the step stops at once and writes nothing to the ledger.

        Returns:
            The result. Its status is `halted` if the loop is halted (also
            when the ledger fails verification, with the single event
            ``("halt", reason)``), else `ok` if at least one matured batch
            had enough labels, else `insufficient_labels`. Its events, newest
            escalation and mechanism are described by `Outcome`.
        """
        if not self.recheck():
            self.save_state()
            return terms.Outcome(
                terms.HALTED,
                (("halt", terms.HALT_LEDGER),),
                None,
                self.mechanism,
            )
        first_row = len(self.ledger.rows)
        now = self.batch - 1
        keep: list[dict[str, Any]] = []
        used = False
        for record in self.pending_batches:
            window = self.policy.outcome_window
            if labels.is_matured(record["index"], now, window):
                used |= self.feed(record)
            else:
                keep.append(record)
        self.pending_batches = keep
        newest = self.decide()
        newest = self.expire() or newest
        events = tuple(
            (
                row["kind"],
                str(row["body"].get("reason", row["body"].get("rule", ""))),
            )
            for row in self.ledger.rows[first_row:]
            if row["kind"] != deployment_ledger.OBSERVE_SUMMARY
        )
        self.write_label_mechanism()
        self.save_state()
        status = (
            terms.HALTED
            if self.halted
            else (terms.OK if used else terms.INSUFFICIENT_LABELS)
        )
        return terms.Outcome(status, events, newest, self.mechanism)

    def decide(self) -> terms.Escalation | None:
        """Applies the decision map to what the monitors now say.

        Acts on a naive alarm under an unverified mechanism (halt and
        escalate), an L1 alarm (alarm row, then an investigation escalation or
        a retrain request, by ``alarm_gates_retrain``), a selective-risk alarm
        (halt and escalate) and a crossed coverage or novelty test
        (escalate). Each crossed monitor is replaced by a fresh one. When the
        mechanism permits action it also decides rollback proposals and the
        challenger queue.

        Returns:
            The newest escalation opened by the decisions, or None if none was
            opened.
        """
        batch = self.batch - 1
        verified = labels.actionable(self.mechanism)
        newest = self.alert(batch, verified)
        newest = self.screen(batch) or newest
        if verified:
            newest = self.propose() or newest
            newest = self.drain() or newest
        return newest
