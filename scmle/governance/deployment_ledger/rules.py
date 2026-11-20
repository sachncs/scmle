"""The state machines each ledger row kind is checked and applied by."""

from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from scmle.assurance.ledger import ledger
from scmle.governance.deployment_ledger import grants
from scmle.governance.deployment_ledger import kinds
from scmle.monitoring import sequential_tests


class Oversight:
    """A ledger's replayed state and the rules of the loop's own rows.

    Challenger registration and replay, escalations, attestations and halts.
    """

    def __init__(self, path: Path | None) -> None:
        """Creates an empty, unopened ledger; use `create` or `load`.

        Args:
            path: File to persist rows to, or None for an in-memory ledger.
        """
        self.path = path
        self.rows: list[dict[str, Any]] = []
        self.header: dict[str, Any] = {}
        self.keys: frozenset[str] = frozenset()
        self.escalations: dict[str, dict[str, Any]] = {}
        self.closed: dict[str, str] = {}
        self.attestations: dict[str, dict[str, Any]] = {}
        self.used: dict[str, dict[str, Any]] = {}
        self.promotion: sequential_tests.Sequence | None = None
        self.registered: list[str] = []
        self.decided: dict[str, dict[str, Any]] = {}
        self.halted = False
        self.experiments: dict[str, dict[str, Any]] = {}
        self.envelopes: dict[str, dict[str, Any]] = {}
        self.uses: dict[str, dict[str, int]] = {}
        self.revoked: dict[str, dict[str, Any]] = {}
        self.intents: dict[str, dict[str, Any]] = {}
        self.consumed: dict[str, dict[str, Any]] = {}
        self.rejected: dict[str, dict[str, Any]] = {}

    @property
    def head(self) -> str:
        """Hash of the last row (genesis for an empty ledger)."""
        return str(self.rows[-1]["hash"]) if self.rows else ledger.GENESIS

    def accept(self, body: Mapping[str, Any]) -> None:
        """Rows without a state machine are chained as they are.

        Args:
            body: The row content; not interpreted.
        """

    def apply(self, body: Mapping[str, Any]) -> None:
        """Reads the header and builds the e-LOND sequence.

        Args:
            body: The `open` row content.

        Raises:
            MisuseError: If the ledger version is unsupported or the
                policy does not match its hash.
        """
        if body["ledger_version"] != kinds.LEDGER_VERSION:
            raise kinds.MisuseError("unsupported ledger version")
        if body["policy_hash"] != ledger.digest(body["policy"]):
            raise kinds.MisuseError("policy does not match its hash")
        self.header = dict(body)
        self.keys = frozenset(body["system_keys"])
        self.promotion = sequential_tests.Sequence(body["alpha_promo"])

    def elond(self) -> sequential_tests.Sequence:
        """The e-LOND sequence, which exists once `open` is in.

        Returns:
            The e-LOND sequence.

        Raises:
            MisuseError: If there is no `open` row yet.
        """
        if self.promotion is None:
            raise kinds.MisuseError("ledger has no open row")
        return self.promotion

    def replay(self, body: Mapping[str, Any]) -> None:
        """Registration precedes the decision; the decision is replayed.

        Args:
            body: The challenger-result row content.

        Raises:
            MisuseError: If the challenger is registered twice, the
                phase is neither registered nor decided, a decision names an
                unregistered challenger, one already decided or one that is
                not next in registration order, or the e-value is invalid, or
                the replay of the e-LOND step differs from what was
                recorded.
        """
        name = str(body["name"])
        sequence = self.elond()
        if body["phase"] == kinds.REGISTERED:
            if name in self.registered:
                raise kinds.MisuseError(f"{name!r} is already registered")
            sequence.register(name)
            self.registered.append(name)
            return
        if body["phase"] != kinds.DECIDED:
            raise kinds.MisuseError("phase must be registered or decided")
        if name not in self.registered:
            raise kinds.MisuseError(
                f"{name!r} was not registered before it was scored"
            )
        try:
            step = sequence.decide(name, float(body["e_value"]))
        except ValueError as error:
            raise kinds.MisuseError(str(error)) from error
        recorded = (body["level"], body["rejected"], body["wealth_after"])
        replayed = (str(step.level), step.rejected, str(step.wealth_after))
        if recorded != replayed:
            raise kinds.MisuseError(
                f"replay of e-LOND differs for {name!r}: recorded "
                f"{recorded}, replayed {replayed}"
            )
        self.decided[name] = dict(body)

    def escalate(self, body: Mapping[str, Any]) -> None:
        """Opens an escalation or closes it with the safe default.

        Args:
            body: The escalation row content.

        Raises:
            MisuseError: If an opening row's question does not match
                its hash, its id is not ``esc-`` followed by its row number
                or it lacks evidence or a proposed default, the phase is
                neither open nor timeout, or a timeout names an escalation
                that is not pending.
        """
        number = len(self.rows)
        if body["phase"] == kinds.PHASE_OPEN:
            question = body["question"]
            if body["question_hash"] != ledger.digest(question):
                raise kinds.MisuseError(
                    "escalation question does not match its hash"
                )
            if body["id"] != f"esc-{number}":
                raise kinds.MisuseError("escalation id is not its row")
            for name in ("evidence", "proposed_default"):
                if name not in body:
                    raise kinds.MisuseError(f"escalation needs {name}")
            self.escalations[body["id"]] = dict(body)
            return
        if body["phase"] != kinds.PHASE_TIMEOUT:
            raise kinds.MisuseError("phase must be open or timeout")
        identity = body["id"]
        if identity not in self.escalations or identity in self.closed:
            raise kinds.MisuseError(f"{identity!r} is not pending")
        self.closed[identity] = kinds.PHASE_TIMEOUT

    def attest(self, body: Mapping[str, Any]) -> None:
        """Binds a reply to one pending escalation and its question hash.

        Args:
            body: The attestation row content.

        Raises:
            MisuseError: If the row is not a valid reply, it answers
                no escalation, the escalation is already closed, the reply is
                bound to a different question, or it is signed with a key
                the system holds. The signature and expiry are not checked.
        """
        reply = grants.Attestation.from_dict(body)
        escalation = self.escalations.get(reply.escalation)
        if escalation is None:
            raise kinds.MisuseError("attestation answers no escalation")
        if reply.escalation in self.closed:
            raise kinds.MisuseError("escalation is already closed")
        if reply.digests["question"] != escalation["question_hash"]:
            raise kinds.MisuseError(
                "attestation is bound to a different question"
            )
        if reply.key_id in self.keys:
            raise kinds.MisuseError(
                "attestation is signed with a key the system holds"
            )
        self.closed[reply.escalation] = kinds.ATTESTATION
        self.attestations[reply.escalation] = dict(body)

    def spend(self, body: Mapping[str, Any]) -> None:
        """Marks an answered escalation's attestation as spent, once.

        Args:
            body: The attestation-used row content.

        Raises:
            MisuseError: If the attestation was not recorded or is
                already spent.
        """
        identity = str(body["escalation_id"])
        if identity not in self.attestations:
            raise kinds.MisuseError(
                "an attestation can only be spent after it was recorded"
            )
        if identity in self.used:
            raise kinds.MisuseError("the attestation is already spent")
        self.used[identity] = dict(body)

    def halt(self, body: Mapping[str, Any]) -> None:
        """Enters or leaves the halted state.

        Args:
            body: The halt row content.

        Raises:
            MisuseError: If a release occurs without a halt or
                without a recalibration entry in the row (its content is
                not examined), or the phase is neither enter nor release.
        """
        if body["phase"] == kinds.ENTER:
            self.halted = True
        elif body["phase"] == kinds.RELEASE:
            if not self.halted:
                raise kinds.MisuseError("release without a halt")
            if not body.get("recalibration"):
                raise kinds.MisuseError(
                    "a halt is released only on a passing recalibration"
                )
            self.halted = False
        else:
            raise kinds.MisuseError("phase must be enter or release")

    def pending(self) -> list[dict[str, Any]]:
        """Lists the escalations with no reply and no timeout, oldest first.

        Returns:
            The bodies of the escalation rows that opened them (not the whole
            chained rows) and have no reply and no timeout, oldest first.
        """
        return [
            row
            for identity, row in self.escalations.items()
            if identity not in self.closed
        ]


class Automation(Oversight):
    """The rules of unattended action: envelopes, controller and rollouts."""

    def issue(self, body: Mapping[str, Any]) -> None:
        """Records a granted envelope once, never under a system key.

        Args:
            body: The envelope row content.

        Raises:
            MisuseError: If the row is not a valid envelope, the
                envelope was already granted, or it is signed with a key the
                system holds. The signature and expiry are not checked.
        """
        grant = grants.Envelope.from_dict(body)
        if grant.identifier in self.envelopes:
            raise kinds.MisuseError(
                f"envelope {grant.identifier!r} was already granted"
            )
        if grant.key_id in self.keys:
            raise kinds.MisuseError(
                "envelope is signed with a key the system holds"
            )
        self.envelopes[grant.identifier] = grant.to_dict()
        self.uses[grant.identifier] = {}

    def consume(self, body: Mapping[str, Any]) -> None:
        """Counts one use, refusing what the envelope does not allow.

        The envelope's expiry is not checked here.

        Args:
            body: The envelope-use row content.

        Raises:
            MisuseError: If the envelope is unknown or revoked, does
                not cover the action, the scope differs or its uses are spent.
        """
        identity = str(body["envelope_id"])
        action = str(body["action"])
        grant = self.envelopes.get(identity)
        if grant is None:
            raise kinds.MisuseError(f"envelope {identity!r} is unknown")
        if identity in self.revoked:
            raise kinds.MisuseError(f"envelope {identity!r} was revoked")
        limit = grant["limits"].get(action)
        if limit is None:
            raise kinds.MisuseError(
                f"envelope {identity!r} does not cover {action!r}"
            )
        if body["scope"] != grant["scope"]:
            raise kinds.MisuseError("use is outside the envelope's scope")
        used = self.uses[identity].get(action, 0)
        if used >= limit:
            raise kinds.MisuseError(
                f"envelope {identity!r} has spent its {limit} use(s) of "
                f"{action!r}"
            )
        self.uses[identity][action] = used + 1

    def revoke(self, body: Mapping[str, Any]) -> None:
        """Removes an envelope's authority, once.

        Args:
            body: The revocation row content.

        Raises:
            MisuseError: If the envelope is unknown or already
                revoked.
        """
        identity = str(body["envelope_id"])
        if identity not in self.envelopes:
            raise kinds.MisuseError(f"envelope {identity!r} is unknown")
        if identity in self.revoked:
            raise kinds.MisuseError(f"envelope {identity!r} is revoked")
        self.revoked[identity] = dict(body)

    def journal(self, body: Mapping[str, Any]) -> None:
        """Keeps the controller's journal: each input digest is applied once.

        A `tick` row is accepted without any check and changes nothing.

        Args:
            body: The controller row content.

        Raises:
            MisuseError: If the phase or input kind is not allowed,
                the input was already applied, an intent repeats, or an input is
                applied without an intent.
        """
        phase = body["phase"]
        if phase not in kinds.CONTROLLER_PHASES:
            raise kinds.MisuseError(
                "controller phase must be one of "
                f"{list(kinds.CONTROLLER_PHASES)}"
            )
        if phase == kinds.TICK:
            return
        digest = str(body["digest"])
        if body["what"] not in kinds.CONTROLLER_INPUTS:
            raise kinds.MisuseError(
                "controller input must be one of "
                f"{list(kinds.CONTROLLER_INPUTS)}"
            )
        if digest in self.consumed:
            raise kinds.MisuseError(f"input {digest[:12]} was applied")
        if phase == kinds.INTENT:
            if digest in self.intents:
                raise kinds.MisuseError(
                    f"input {digest[:12]} already has an intent"
                )
            self.intents[digest] = dict(body)
        elif phase == kinds.APPLIED:
            if digest not in self.intents:
                raise kinds.MisuseError(
                    f"input {digest[:12]} was applied without an intent"
                )
            self.consumed[digest] = dict(body)
        else:
            self.rejected[digest] = dict(body)

    def start(self, body: Mapping[str, Any]) -> None:
        """Starts an experiment once, at a level strictly between 0 and 1.

        Args:
            body: The experiment row content.

        Raises:
            MisuseError: If the experiment already started or its
                level is not in (0, 1).
        """
        name = str(body["experiment_id"])
        if name in self.experiments:
            raise kinds.MisuseError(f"experiment {name!r} already started")
        if not 0 < float(Fraction(body["level"])) < 1:
            raise kinds.MisuseError("an experiment level lies in (0, 1)")
        self.experiments[name] = dict(body)

    def assign(self, body: Mapping[str, Any]) -> None:
        """Checks a batch of propensity-logged assignments.

        Args:
            body: The assignment row content.

        Raises:
            MisuseError: If the experiment was not started, the lists
                differ in length, an arm is not 0 or 1, or a probability is not
                strictly in (0, 1).
        """
        name = str(body["experiment_id"])
        if name not in self.experiments:
            raise kinds.MisuseError(
                f"assignment for {name!r}, which was not started"
            )
        units, arms, odds = body["units"], body["arm"], body["probability"]
        if not len(units) == len(arms) == len(odds):
            raise kinds.MisuseError("assignment lists differ in length")
        if any(a not in (0, 1) for a in arms) or any(
            not 0 < float(p) < 1 for p in odds
        ):
            raise kinds.MisuseError(
                "arms are 0 or 1 and probabilities lie strictly in (0, 1)"
            )

    def confirm(self, body: Mapping[str, Any]) -> None:
        """Requires the experiment a rollout decision names to have started.

        Nothing is recorded; a rollback decision also names its experiment.

        Args:
            body: The decision row content.

        Raises:
            MisuseError: If the experiment was not started.
        """
        name = str(body["experiment_id"])
        if name not in self.experiments:
            raise kinds.MisuseError(
                f"decision for {name!r}, which was not started"
            )
