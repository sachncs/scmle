"""Promotion: a staged rollout the runtime confirms, then the registry."""

from collections.abc import Mapping
from typing import Any

from scmle.contracts import errors
from scmle.contracts import events
from scmle.deploy import contract
from scmle.deploy import control
from scmle.deploy import release
from scmle.deploy.release_state import approval
from scmle.deploy.release_state import rollout
from scmle.deploy.release_state import status
from scmle.governance import registry


class Promotion(approval.Approval, rollout.Rollout):
    """Releasing an approved version through the control plane."""

    def promote(
        self,
        model: str,
        version: int,
        plane: control.ControlPlane,
        environment: str,
        image_digest: str,
        steps: tuple[int, ...] = (100,),
        health: status.Health | None = None,
        signing_key: str | bytes | None = None,
    ) -> status.Outcome:
        """Releases an approved version through the control plane.

        A refusal or an unconfirmed step does not raise: it returns an
        unsuccessful `Outcome`. An unknown model or version raises
        `ArtifactError` before anything is asked of the platform.

        Args:
            model: Model name.
            version: An approved registry version.
            plane: The platform.
            environment: Environment name; ``production`` also moves the
                champion alias.
            image_digest: Digest of the image that serves this version.
            steps: Strictly increasing traffic shares, in percent, to pass
                through, ending in 100. A share below 100 splits traffic with
                the revision recorded for the environment, which must exist.
            health: Called after each step; non-empty reasons stop the
                rollout.
            signing_key: Key that verifies the stored approval.

        Returns:
            An `Outcome`. A refusal returns ``done=False`` with the reasons
            and no call to the platform. An unconfirmed step returns
            ``done=False`` with the registry unchanged and the previous
            revision restored, where there is one (see `Outcome.restored`).

        Raises:
            RuntimeUnconfirmed: Raised inside the rollout when a health check
                stops it. It is caught and reported as ``done=False``, so it
                does not reach the caller; neither do the platform's other
                confirmation or support errors.
            ConfigError: If ``steps`` is not increasing and ending in 100, or
                the digest is not a digest. (A staged first step with no
                previous revision raises it inside the rollout, where it is
                caught and reported.)
        """
        contract.require_digest(image_digest, "image_digest")
        if (
            not steps
            or steps[-1] != status.FULL_ROLLOUT
            or list(steps) != sorted(set(steps))
        ):
            raise errors.ConfigError("steps must increase and end in 100")
        refusal = self.refusal(model, version, signing_key)
        if refusal:
            return self.settle(
                "rejected",
                model,
                version,
                environment,
                plane,
                None,
                image_digest,
                refusal,
            )
        return self.deliver(
            model,
            version,
            plane,
            environment,
            image_digest,
            steps,
            health,
            signing_key,
        )

    def deliver(
        self,
        model: str,
        version: int,
        plane: control.ControlPlane,
        environment: str,
        image_digest: str,
        steps: tuple[int, ...],
        health: status.Health | None,
        signing_key: str | bytes | None,
    ) -> status.Outcome:
        """Deploys, rolls out and finalizes a version `promote` admitted.

        Args:
            model: Model name.
            version: The approved version.
            plane: The platform.
            environment: Environment name.
            image_digest: Digest of the image that serves this version.
            steps: The checked traffic shares, in percent.
            health: Called after each step; reasons stop the rollout.
            signing_key: Key that verifies the stored approval.

        Returns:
            The `Outcome` that `promote` documents.
        """
        previous = self.environment(model, environment)
        revision: str | None = None
        try:
            revision = plane.deploy(image_digest)
            control.confirm(
                plane, self.present(revision, image_digest), self.policy
            )
            self.shift(plane, revision, image_digest, steps, health, previous)
            self.finalize(
                model,
                version,
                environment,
                revision,
                image_digest,
                previous,
                signing_key,
            )
        except (
            control.RuntimeUnconfirmed,
            control.Unsupported,
            errors.ScmleError,
        ) as error:
            # handled: recorded rejected event with reasons, restore
            restored = self.restore(plane, previous)
            return self.settle(
                "rejected",
                model,
                version,
                environment,
                plane,
                revision,
                image_digest,
                [str(error)],
                restored,
            )
        return self.settle(
            "promoted",
            model,
            version,
            environment,
            plane,
            revision,
            image_digest,
        )

    def finalize(
        self,
        model: str,
        version: int,
        environment: str,
        revision: str,
        image_digest: str,
        previous: dict[str, Any] | None,
        signing_key: str | bytes | None,
    ) -> None:
        """Changes the registry after the runtime confirmed (under the lock).

        In the production environment the champion alias moves first, with
        the version's stored approval, and the previous version is retired;
        nothing is written to the state file until the alias has moved. If
        the registry refuses the alias, the state file is left as it was and
        the error propagates, so the caller restores the platform. The
        environment record keeps at most the last ten known-good revisions.

        Args:
            model: Model name.
            version: The promoted version.
            environment: Environment name.
            revision: The platform revision now serving.
            image_digest: Digest of the image it runs.
            previous: The record previously in the environment, or None; it
                joins the known-good history.
            signing_key: Key that verifies the stored approval.
        """
        with self.store.lock:
            state = self.read()
            entry = self.entry(state, model)
            production = environment == status.PRODUCTION_ENVIRONMENT
            if production:
                stored = entry["versions"][str(version)]["approval"]
                self.store.promote(
                    model,
                    version,
                    registry.CHAMPION,
                    release.revive(stored),
                    signing_key,
                )
            known_good = self.history(entry, environment, previous)
            if production:
                if previous is not None and previous["version"] != version:
                    self.move(
                        entry,
                        previous["version"],
                        status.ReleaseState.RETIRED,
                        f"superseded by v{version}",
                        status.ReleaseState.PRODUCTION,
                    )
                self.move(
                    entry,
                    version,
                    status.ReleaseState.PRODUCTION,
                    f"confirmed by the runtime in {environment}",
                    status.ReleaseState.APPROVED,
                )
            entry["environments"][environment] = {
                "version": version,
                "revision": revision,
                "image_digest": image_digest,
                "confirmed_at": events.stamp(),
                "known_good": known_good[-10:],
            }
            self.write(state)
            self.store.audit(
                "release_confirmed",
                model=model,
                version=version,
                environment=environment,
                revision=revision,
            )

    @staticmethod
    def history(
        entry: Mapping[str, Any],
        environment: str,
        previous: Mapping[str, Any] | None,
    ) -> list[Mapping[str, Any]]:
        """Returns the environment's known-good revisions, `previous` last.

        Args:
            entry: The model's state entry.
            environment: Environment name.
            previous: The record previously in the environment, or None.

        Returns:
            A new list of the recorded known-good revisions, followed by the
            version, revision and image digest of `previous` when there is
            one.
        """
        known_good = list(
            (entry["environments"].get(environment) or {}).get("known_good", [])
        )
        if previous is not None:
            known_good.append(
                {
                    "version": previous["version"],
                    "revision": previous["revision"],
                    "image_digest": previous["image_digest"],
                }
            )
        return known_good
