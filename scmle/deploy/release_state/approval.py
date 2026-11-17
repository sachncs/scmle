"""States set without the runtime: by hand, by approval, and refusals."""

from scmle.contracts import errors
from scmle.deploy import release
from scmle.deploy.release_state import records
from scmle.deploy.release_state import status
from scmle.governance import registry


class Approval(records.Records):
    """Hand transitions, approvals and why a version may not be released."""

    def transition(
        self, model: str, version: int, target: status.ReleaseState, reason: str
    ) -> None:
        """Moves a version to ``challenger`` or ``retired``.

        ``approved`` needs `approve`; ``production`` needs `promote` or
        `rollback`, because only the runtime can confirm it.

        Args:
            model: Model name.
            version: Registered version number.
            target: The state to move to.
            reason: Why the version moves.

        Raises:
            ArtifactError: If the version is unknown, `target` is a state
                that cannot be set by hand, or the version's current state
                does not allow the move.
        """
        if target not in status.MANUAL:
            raise errors.ArtifactError(
                f"{target.value} cannot be set by hand: use approve, "
                "promote or rollback"
            )
        with self.store.lock:
            current = self.lookup(model, version)
            if target not in status.ALLOWED[current]:
                raise errors.ArtifactError(
                    f"{model} v{version}: {current.value} -> {target.value} "
                    "is not allowed"
                )
            state = self.read()
            self.move(
                self.entry(state, model), version, target, reason, current
            )
            self.write(state)
            self.store.audit(
                "release_state",
                model=model,
                version=version,
                state=target.value,
                reason=reason,
            )

    def approve(
        self,
        model: str,
        version: int,
        approval: registry.Sanction,
        signing_key: str | bytes | None = None,
    ) -> None:
        """Marks a version approved after checking its approval.

        Args:
            model: Model name.
            version: Registered version number.
            approval: The independent approval of this exact bundle.
            signing_key: Key that verifies the approval's signature.

        Raises:
            ArtifactError: If the version is unknown, the bundle is unsigned,
                the approval is rejected, not independent, not signed under
                the key or for another bundle, or the version's state does
                not allow approval (only candidate and challenger do).
        """
        record = self.store.resolve(model, version)
        if not record.signed:
            raise errors.ArtifactError("an unsigned bundle cannot be approved")
        try:
            approval.validate(model, version, record.content_id, signing_key)
        except (TypeError, ValueError) as error:
            raise errors.ArtifactError(str(error)) from error
        with self.store.lock:
            current = self.lookup(model, version)
            if status.ReleaseState.APPROVED not in status.ALLOWED[current]:
                raise errors.ArtifactError(
                    f"{model} v{version} is {current.value}"
                )
            state = self.read()
            entry = self.entry(state, model)
            self.move(
                entry,
                version,
                status.ReleaseState.APPROVED,
                "approved",
                current,
            )
            entry["versions"][str(version)]["approval"] = approval.to_dict()
            self.write(state)
            self.store.audit(
                "release_state",
                model=model,
                version=version,
                state="approved",
                reviewer=approval.reviewer,
            )

    def refusal(
        self, model: str, version: int, signing_key: str | bytes | None
    ) -> list[str]:
        """Lists the reasons the version may not be released.

        An unknown model or version raises `ArtifactError`.

        Args:
            model: Model name.
            version: Registered version number.
            signing_key: Key that verifies the stored approval.

        Returns:
            One message for each unmet condition (not approved, unsigned
            bundle, no stored approval, or an invalid approval); empty if the
            version may be released.
        """
        record = self.store.resolve(model, version)
        reasons = []
        current = self.lookup(model, version)
        if current is not status.ReleaseState.APPROVED:
            reasons.append(
                f"{model} v{version} is {current.value}, not approved"
            )
        if not record.signed:
            reasons.append("the bundle is unsigned")
        stored = (
            self.read()["models"]
            .get(model, {})
            .get("versions", {})
            .get(str(version), {})
            .get("approval")
        )
        if stored is None:
            reasons.append("the version has no stored approval")
        else:
            try:
                release.revive(stored).validate(
                    model, version, record.content_id, signing_key
                )
            except (errors.ArtifactError, TypeError, ValueError) as error:
                # handled: appended to reasons
                reasons.append(f"the approval is not valid: {error}")
        return reasons
