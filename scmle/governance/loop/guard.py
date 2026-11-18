"""Never-list actions, each run only with a person's attestation."""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
from typing import Any

from scmle.contracts import errors
from scmle.contracts import records
from scmle.governance import deployment_ledger
from scmle.governance import never_list
from scmle.governance.loop import desk
from scmle.governance.loop import terms


class Gatekeeper(desk.Desk):
    """The never-list actions of a loop, gated by attestations."""

    def intercept(self, registry: Any) -> None:
        """Routes every champion move of `registry` through this guard.

        `elevate`, `revertible` and `enlist(promote=True)`
        of a `Maintainer` over this registry then need an attestation,
        as `move_champion` does. Moving the alias by editing the registry's
        files is not stopped.

        Args:
            registry: The model registry to guard.
        """
        registry.attach(self.guard, self.policy.scope)

    def move_champion(
        self,
        registry: Any,
        model: str,
        version: int,
        approval: Any,
        signing_key: str | bytes | None = None,
    ) -> Any:
        """Promotes a version to champion, with an attestation for that move.

        The registry still checks its own signed approval underneath.

        Args:
            registry: The model registry.
            model: Model name.
            version: Version to make champion.
            approval: The registry approval of that version.
            signing_key: Key that verifies the approval.

        Returns:
            Whatever the registry's `promote` returns.
        """

        def move() -> Any:
            return registry.promote(
                model, version, approval=approval, signing_key=signing_key
            )

        if getattr(registry, "guard", None) is self.guard:
            try:
                return move()
            finally:
                self.save_state()
        return self.perform(
            never_list.Forbidden.MOVE_CHAMPION,
            {"model": model, "version": version},
            move,
        )

    def revert(self, do: Callable[[], Any], detail: Mapping[str, Any]) -> Any:
        """Runs a rollback, with an attestation for the move-champion action.

        Args:
            do: The rollback to perform.
            detail: Facts that identify it.

        Returns:
            Whatever `do` returns.
        """
        return self.perform(
            never_list.Forbidden.MOVE_CHAMPION, {"rollback": dict(detail)}, do
        )

    def reopen_sealed(
        self, do: Callable[[], Any], detail: Mapping[str, Any]
    ) -> Any:
        """Runs `do()` that reopens `C` or `F` or touches a sealed set.

        It runs only with an attestation for that exact detail.

        Args:
            do: The action to perform.
            detail: Facts that identify it.

        Returns:
            Whatever `do` returns.
        """
        return self.perform(
            never_list.Forbidden.REOPEN_SEALED, dict(detail), do
        )

    def rewrite(self, do: Callable[[], Any], detail: Mapping[str, Any]) -> Any:
        """Runs `do()` that edits, deletes, reorders or truncates ledger rows.

        It runs only with an attestation for that exact detail.

        Args:
            do: The action to perform.
            detail: Facts that identify it.

        Returns:
            Whatever `do` returns.
        """
        return self.perform(never_list.Forbidden.EDIT_LEDGER, dict(detail), do)

    def act_on_unverified(self, kind: str, do: Callable[[], Any]) -> Any:
        """Runs `do()` that retrains, promotes or rolls back unverified.

        It runs only with an attestation naming the kind of action.

        Args:
            kind: The kind of action.
            do: The action to perform.

        Returns:
            Whatever `do` returns.
        """
        return self.perform(
            never_list.Forbidden.ACT_ON_UNVERIFIED, {"kind": kind}, do
        )

    def silence(self, name: str, do: Callable[[], Any]) -> Any:
        """Runs `do()` that suppresses, silences or reroutes an alert.

        It runs only with an attestation naming the alert.

        Args:
            name: Name of the alert.
            do: The action to perform.

        Returns:
            Whatever `do` returns.
        """
        return self.perform(
            never_list.Forbidden.SUPPRESS_ALERT, {"alert": name}, do
        )

    def change_label_policy(
        self, change: Mapping[str, Any], do: Callable[[], Any]
    ) -> Any:
        """Runs `do()` that changes the labelling or randomization policy.

        It runs only with an attestation for exactly this change.

        Args:
            change: The change, as recorded.
            do: The action to perform.

        Returns:
            Whatever `do` returns.
        """
        return self.perform(
            never_list.Forbidden.CHANGE_LABEL_POLICY, dict(change), do
        )

    def delete_or_send(self, what: str, do: Callable[[], Any]) -> Any:
        """Runs `do()` that deletes data, models or artifacts, or sends out.

        It runs only with an attestation naming what is deleted or sent.

        Args:
            what: What is deleted or sent.
            do: The action to perform.

        Returns:
            Whatever `do` returns.
        """
        return self.perform(
            never_list.Forbidden.DELETE_OR_EXTERNAL, {"what": what}, do
        )

    def amend_policy(self, **changes: Any) -> terms.Preregistration:
        """Changes the policy: tightening is free, anything else is gated.

        A field in ``NUMERIC_ORIENTATION`` (for example ``target`` or
        ``tolerance``, or ``promotion.delta`` inside an object) moved to a
        stricter or equal value needs no attestation. A looser value needs
        one (the relax-threshold action), and so does any other field, or
        moving such a field to or from None, as turning a test off or on
        (the change-charter action: fallbacks, high-impact actions, scope).
        A change of `promotion` or `rollback` names only the keys it changes.
        The change is recorded in the ledger and restarts the monitors with
        fresh state. A gated change without an attestation is refused with
        `AttestationRequired`, and nothing is changed.

        Args:
            **changes: New values of policy fields, by field name.

        Returns:
            The updated policy.

        Raises:
            ValueError: For an unknown field, or a new value the policy's
                range checks reject.
        """
        names = {f.name for f in dataclasses.fields(terms.Preregistration)}
        unknown = set(changes) - names
        if unknown:
            raise errors.InputError(f"unknown policy fields: {sorted(unknown)}")
        changes = {
            field: (
                {**getattr(self.policy, field), **new}
                if field in terms.GROUPS
                else new
            )
            for field, new in changes.items()
        }
        moves: dict[str, tuple[Any, Any]] = {}
        for field, new in changes.items():
            old = getattr(self.policy, field)
            if field in terms.GROUPS:
                moves.update(
                    {f"{field}.{k}": (old.get(k), v) for k, v in new.items()}
                )
            else:
                moves[field] = (old, new)
        for field, (old, new) in moves.items():
            if old == new and field not in changes:
                continue
            orientation = terms.NUMERIC_ORIENTATION.get(field)
            if orientation is not None and None not in (old, new):
                if (new - old) * orientation >= 0:
                    continue
                action = never_list.Forbidden.RELAX_THRESHOLD
            else:
                action = never_list.Forbidden.CHANGE_CHARTER
            self.guard.require(
                action,
                {
                    "field": field,
                    "from": terms.jsonify(old),
                    "to": terms.jsonify(new),
                },
                self.policy.scope,
            )
        updated = dataclasses.replace(self.policy, **changes)
        self.ledger.commit(
            deployment_ledger.OBSERVE_SUMMARY,
            {
                "policy_amendment": {
                    records.KEY: terms.VERSION,
                    **terms.jsonify(changes),
                },
                "batch": self.batch,
            },
        )
        self.policy = updated
        self.arm()
        self.save_state()
        return updated
