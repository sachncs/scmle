"""Talking to the platform: moving traffic and confirming each move."""

from collections.abc import Callable
from collections.abc import Mapping
from typing import Any

from scmle.contracts import errors
from scmle.deploy import control
from scmle.deploy import observation
from scmle.deploy.release_state import records
from scmle.deploy.release_state import status


class Rollout(records.Records):
    """Traffic moves on the platform, each confirmed before it counts."""

    def restore(
        self, plane: control.Platform, previous: dict[str, Any] | None
    ) -> bool | None:
        """Puts the previous revision back and confirms it (best effort).

        Args:
            plane: The platform.
            previous: The record of the revision to restore, or None.

        Returns:
            None if there is nothing to restore, True if the previous revision
            was restored and confirmed with all traffic, False if the platform
            did not confirm it.
        """
        if previous is None:
            return None
        try:
            plane.route({previous["revision"]: 100})
            control.confirm(
                plane,
                control.expecting(
                    previous["revision"], previous["image_digest"], 100
                ),
                self.policy,
            )
        except (
            control.UnconfirmedError,
            control.Unsupported,
            errors.ScmleError,
        ):
            # handled: documented False: platform did not confirm
            return False
        return True

    @staticmethod
    def present(
        revision: str, image_digest: str
    ) -> Callable[[observation.Observation], list[str]]:
        """Builds a check that the revision exists and runs the digest.

        The check ignores the share of traffic and readiness.

        Args:
            revision: Revision name expected on the platform.
            image_digest: Digest the revision must run.

        Returns:
            A function taking a platform observation and returning the reasons
            the check fails; empty if the revision is reported and runs the
            digest.
        """

        def check(seen: observation.Observation) -> list[str]:
            digest = seen.distill(revision)
            if digest is None:
                return [f"revision {revision} is not reported"]
            if digest != image_digest:
                return [f"revision {revision} runs {digest}"]
            return []

        return check

    def shift(
        self,
        plane: control.Platform,
        revision: str,
        image_digest: str,
        steps: tuple[int, ...],
        health: status.Health | None,
        previous: Mapping[str, Any] | None,
    ) -> None:
        """Moves traffic to the revision step by step, confirming each step.

        Args:
            plane: The platform.
            revision: The deployed revision.
            image_digest: Digest the revision must run.
            steps: Traffic shares, in percent, ending in 100.
            health: Called after each step; reasons stop the rollout.
            previous: The revision a share below 100 splits traffic with.
        """
        for percent in steps:
            allocation = {revision: percent}
            if percent < status.FULL_ROLLOUT:
                allocation[status.predecessor(previous)["revision"]] = (
                    status.FULL_ROLLOUT - percent
                )
            plane.route(allocation)
            seen = control.confirm(
                plane,
                control.expecting(revision, image_digest, percent),
                self.policy,
            )
            if health is not None:
                stop = health(seen, percent)
                if stop:
                    status.halt(percent, stop)
