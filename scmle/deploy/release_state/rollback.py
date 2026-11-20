"""Rollback to the previous known-good revision, once the runtime confirms."""

from typing import Any

from scmle.contracts import errors
from scmle.contracts import events
from scmle.deploy import control
from scmle.deploy import release
from scmle.deploy.release_state import rollout
from scmle.deploy.release_state import status
from scmle.governance import registry


class Rollback(rollout.Rollout):
    """Restoring the previous known-good revision."""

    def rollback(
        self,
        model: str,
        environment: str,
        plane: control.Platform,
        signing_key: str | bytes | None = None,
    ) -> status.Outcome:
        """Restores the previous known-good revision.

        The registry changes only after the platform confirms the previous
        revision runs its recorded digest with all traffic. With no recorded
        known-good revision, it returns an unsuccessful `Outcome` without
        touching the platform or recording an event.

        Args:
            model: Model name.
            environment: Environment name.
            plane: The platform.
            signing_key: Key that verifies the stored approval of the version
                returned to.

        Returns:
            An `Outcome`; ``done=False`` leaves the registry as it was and
            puts the revision that was serving back.
        """
        current = self.environment(model, environment)
        if current is None or not current.get("known_good"):
            return status.Outcome(
                done=False,
                reasons=("there is no previous known-good revision",),
            )
        target = current["known_good"][-1]
        try:
            plane.route({target["revision"]: status.FULL_ROLLOUT})
            control.confirm(
                plane,
                control.expecting(
                    target["revision"], target["image_digest"], 100
                ),
                self.policy,
            )
            self.rewind(model, environment, current, target, signing_key)
        except (
            control.UnconfirmedError,
            control.Unsupported,
            errors.ScmleError,
        ) as error:
            # ERR-20: this also catches InputError, which was a ValueError
            # that escaped `rewind` with traffic already on the target. Kept
            # on purpose: `rollback` documents that done=False puts the
            # serving revision back, and only this handler does that.
            # handled: recorded event; Outcome done False with reasons
            restored = self.restore(plane, current)
            return self.settle(
                "rollback_rejected",
                model,
                target["version"],
                environment,
                plane,
                target["revision"],
                target["image_digest"],
                [str(error)],
                restored,
            )
        return self.settle(
            "rolled_back",
            model,
            target["version"],
            environment,
            plane,
            target["revision"],
            target["image_digest"],
        )

    def rewind(
        self,
        model: str,
        environment: str,
        current: dict[str, Any],
        target: dict[str, Any],
        signing_key: str | bytes | None,
    ) -> None:
        """Changes the registry after the runtime confirmed the rollback.

        In the production environment the champion alias moves back with the
        restored version's stored approval; if that version has no stored
        approval the alias is not moved. The registry's refusal (approval or
        guard) propagates before anything is written.

        Args:
            model: Model name.
            environment: Environment name.
            current: The record currently in the environment.
            target: The known-good record being restored.
            signing_key: Key that verifies the stored approval of the
                restored version.
        """
        with self.store.lock:
            state = self.read()
            entry = self.entry(state, model)
            if environment == status.PRODUCTION_ENVIRONMENT:
                stored = (
                    entry["versions"]
                    .get(str(target["version"]), {})
                    .get("approval")
                )
                if stored is not None:
                    self.store.promote(
                        model,
                        target["version"],
                        registry.CHAMPION,
                        release.revive(stored),
                        signing_key,
                        guard_detail={
                            "rollback": {
                                "model": model,
                                "from_version": current["version"],
                                "to_version": target["version"],
                            }
                        },
                    )
                self.move(
                    entry,
                    current["version"],
                    status.State.RETIRED,
                    "rolled back",
                    status.State.PRODUCTION,
                )
                self.move(
                    entry,
                    target["version"],
                    status.State.PRODUCTION,
                    "rollback confirmed by the runtime",
                    status.State.RETIRED,
                )
            entry["environments"][environment] = {
                "version": target["version"],
                "revision": target["revision"],
                "image_digest": target["image_digest"],
                "confirmed_at": events.stamp(),
                "known_good": current["known_good"][:-1],
            }
            self.write(state)
            self.store.audit(
                "release_rolled_back",
                model=model,
                environment=environment,
                to_version=target["version"],
            )
