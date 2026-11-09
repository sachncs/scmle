"""Registry state bound to what the runtime says is running.

The registry (`registry.py`) knows versions and aliases; it does not know what
a cloud is serving. This module adds, beside it, the release state of each
version and a record of what each environment runs, and it moves them only
after the platform confirms.

States, all distinct:

``candidate``
    Registered, nobody has judged it.
``challenger``
    Under comparison with the production model.
``approved``
    Signed bundle and an independent, signed approval for this exact bundle;
    may be released, is not released.
``production``
    The runtime confirmed it serves all traffic in the environment.
``retired``
    Superseded, rolled back or withdrawn. A retired version returns to
    ``production`` only through a rollback that the runtime confirms.

Rules this module enforces:

* ``promote`` asks a `ControlPlane` to deploy and to move traffic, step by
  step, and reads the platform back after each request. The registry state,
  the champion alias and the `promoted` ledger event change only after the
  last read shows the revision, its image digest, readiness and the full
  share of traffic. If any read disagrees, nothing in the registry changes,
  a `rejected` event is recorded and the previous revision is restored.
* ``rollback`` restores the previous known-good revision and changes the
  registry only after the platform confirms it.
* ``deployed_version`` answers "what does the registry say is deployed" only
  when an observation agrees; otherwise it raises `RuntimeDisagrees`, so an
  alias can never be read as "deployed" against the runtime.
* ``reconcile`` lists every disagreement between the registry, the ledger
  and an observation.

Nothing here talks to a cloud; the `ControlPlane` does. The state file is a
separate file next to ``registry.json`` and does not change its format.
"""

from collections.abc import Callable
import dataclasses
import enum
import json
from typing import Any

from scmle import durability
from scmle.contracts import errors
from scmle.contracts import events
from scmle.deploy import contract
from scmle.deploy import control
from scmle.deploy import observation
from scmle.deploy import release
from scmle.governance import deployment_ledger
from scmle.governance import registry

STATE_FILE = "release_state.json"
STATE_FORMAT = 1
PRODUCTION_ENVIRONMENT = "production"


class ReleaseState(enum.StrEnum):
    """Where a registered version stands in its release."""

    CANDIDATE = "candidate"
    CHALLENGER = "challenger"
    APPROVED = "approved"
    PRODUCTION = "production"
    RETIRED = "retired"


ALLOWED: dict[ReleaseState, frozenset[ReleaseState]] = {
    ReleaseState.CANDIDATE: frozenset(
        {ReleaseState.CHALLENGER, ReleaseState.APPROVED, ReleaseState.RETIRED}
    ),
    ReleaseState.CHALLENGER: frozenset(
        {ReleaseState.APPROVED, ReleaseState.RETIRED}
    ),
    ReleaseState.APPROVED: frozenset(
        {ReleaseState.PRODUCTION, ReleaseState.RETIRED}
    ),
    ReleaseState.PRODUCTION: frozenset({ReleaseState.RETIRED}),
    ReleaseState.RETIRED: frozenset({ReleaseState.PRODUCTION}),
}
"""Transitions. Into ``production`` only `promote` and `rollback` move."""
MANUAL = frozenset(
    {
        ReleaseState.CHALLENGER,
        ReleaseState.RETIRED,
    }
)
"""States `transition` may set by hand (``approved`` needs `approve`)."""


class RuntimeDisagrees(errors.ScmleError):
    """The registry's claim and the platform's report differ."""

    category = "runtime"


@dataclasses.dataclass(frozen=True, slots=True)
class Finding:
    """One disagreement between the registry, the ledger and the runtime."""

    kind: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        """Returns a JSON-ready record."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class ReconcileReport:
    """The result of comparing records with a platform observation."""

    model: str
    environment: str
    findings: tuple[Finding, ...]

    @property
    def consistent(self) -> bool:
        """Whether registry, ledger and runtime agree."""
        return not self.findings

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-ready record."""
        return {
            "model": self.model,
            "environment": self.environment,
            "consistent": self.consistent,
            "findings": [item.to_dict() for item in self.findings],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Outcome:
    """What a promotion or rollback did.

    Attributes:
        done: Whether the platform confirmed and the registry changed.
        reasons: Why not, when it was not done.
        restored: Whether the previous revision was confirmed back in place
            after a failed attempt (None when there was nothing to restore).
        version: The version now serving, if the call succeeded.
        revision: The revision now holding all traffic, if the call
            succeeded.
    """

    done: bool
    reasons: tuple[str, ...]
    restored: bool | None = None
    version: int | None = None
    revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-ready record."""
        return {
            "done": self.done,
            "reasons": list(self.reasons),
            "restored": self.restored,
            "version": self.version,
            "revision": self.revision,
        }


Health = Callable[[observation.Observation, int], list[str]]
"""``health(observation, percent)`` returns reasons to stop a rollout."""


class ReleaseRegistry:
    """Release state, promotion, rollback and reconciliation for a registry."""

    def __init__(
        self,
        store: registry.ModelRegistryStore,
        ledger: deployment_ledger.DeploymentLedger | None = None,
        policy: control.ConfirmPolicy | None = None,
    ) -> None:
        """Binds the release state to a registry.

        Args:
            store: The model registry (its directory also holds the state).
            ledger: Optional deployment ledger that records each event.
            policy: How long to wait for the platform to confirm.
        """
        self.store = store
        self.ledger = ledger
        self.policy = policy or control.ConfirmPolicy()

    # -- the state file --------------------------------------------------

    def read(self) -> dict[str, Any]:
        """Loads the state file (empty when new).

        Returns:
            The state document, with a `models` mapping; a fresh one if the
            file does not exist.

        Raises:
            ArtifactError: If the file's format is not the supported one.
        """
        path = self.store.root / STATE_FILE
        if not path.exists():
            return {"format": STATE_FORMAT, "models": {}}
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("format") != STATE_FORMAT:
            raise errors.ArtifactError("unsupported release state format")
        return dict(state)

    def write(self, state: dict[str, Any]) -> None:
        """Writes the state atomically.

        Args:
            state: The complete state document.
        """
        durability.write_atomically(
            self.store.root / STATE_FILE,
            json.dumps(state, indent=2, sort_keys=True).encode("utf-8"),
        )

    def entry(self, state: dict[str, Any], model: str) -> dict[str, Any]:
        """Returns the model's entry in ``state``, creating it when absent.

        Args:
            state: The state document, changed in place when the entry is
                created.
            model: Model name.

        Returns:
            The entry, holding the model's `versions` and `environments`.
        """
        entry: dict[str, Any] = state["models"].setdefault(
            model, {"versions": {}, "environments": {}}
        )
        return entry

    def state_of(self, model: str, version: int) -> ReleaseState:
        """Returns the release state of a registered version.

        The version must be registered; the store refuses an unknown one with
        an `ArtifactError`.

        Args:
            model: Model name.
            version: Registered version number.

        Returns:
            The recorded state, or `candidate` if none has been recorded.
        """
        self.store.resolve(model, version)
        found = (
            self.read()["models"]
            .get(model, {})
            .get("versions", {})
            .get(str(version))
        )
        return ReleaseState(found["state"]) if found else ReleaseState.CANDIDATE

    def states(self, model: str) -> dict[int, ReleaseState]:
        """Returns the release state of every registered version.

        Args:
            model: Model name.

        Returns:
            A mapping from version number to its release state.
        """
        return {
            v.version: self.state_of(model, v.version)
            for v in self.store.versions(model)
        }

    def environment(
        self, model: str, environment: str
    ) -> dict[str, Any] | None:
        """Returns what the registry records as running in an environment.

        Args:
            model: Model name.
            environment: Environment name.

        Returns:
            A shallow copy of the record (version, revision, image digest,
            confirmation time and known-good history), or None if nothing is
            recorded.
        """
        found = (
            self.read()["models"]
            .get(model, {})
            .get("environments", {})
            .get(environment)
        )
        return dict(found) if found else None

    def move(
        self,
        entry: dict[str, Any],
        version: int,
        target: ReleaseState,
        reason: str,
        current: ReleaseState,
    ) -> None:
        """Sets a version's state and appends to its history (in memory).

        Args:
            entry: The model's state entry, changed in place.
            version: Version number.
            target: The state to set.
            reason: Why the state changes.
            current: The state the version is in, recorded as the origin.
        """
        record = entry["versions"].setdefault(
            str(version), {"state": current.value, "history": []}
        )
        record["history"].append(
            {
                "from": current.value,
                "to": target.value,
                "at": events.utc_timestamp(),
                "reason": reason,
            }
        )
        record["state"] = target.value

    # -- states that need no runtime ---------------------------------------

    def transition(
        self, model: str, version: int, target: ReleaseState, reason: str
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
        if target not in MANUAL:
            raise errors.ArtifactError(
                f"{target.value} cannot be set by hand: use approve, "
                "promote or rollback"
            )
        with self.store.lock:
            current = self.state_of(model, version)
            if target not in ALLOWED[current]:
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
        approval: registry.PromotionApproval,
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
            approval.validate_target(
                model, version, record.content_id, signing_key
            )
        except (TypeError, ValueError) as error:
            raise errors.ArtifactError(str(error)) from error
        with self.store.lock:
            current = self.state_of(model, version)
            if ReleaseState.APPROVED not in ALLOWED[current]:
                raise errors.ArtifactError(
                    f"{model} v{version} is {current.value}"
                )
            state = self.read()
            entry = self.entry(state, model)
            self.move(
                entry, version, ReleaseState.APPROVED, "approved", current
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

    # -- the ledger --------------------------------------------------------

    def record_event(self, body: dict[str, Any]) -> None:
        """Appends a deployment row to the ledger, if there is one.

        The event is always written to the store's audit trail.

        Args:
            body: The event fields, as built by `body`.
        """
        self.store.audit("release_event", **body)
        if self.ledger is not None:
            self.ledger.commit(deployment_ledger.DEPLOYMENT, body)

    # -- promotion ---------------------------------------------------------

    def promote(
        self,
        model: str,
        version: int,
        plane: control.ControlPlane,
        environment: str,
        image_digest: str,
        steps: tuple[int, ...] = (100,),
        health: Health | None = None,
        signing_key: str | bytes | None = None,
    ) -> Outcome:
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
        if not steps or steps[-1] != 100 or list(steps) != sorted(set(steps)):
            raise errors.ConfigError("steps must increase and end in 100")
        refusal = self.refusal(model, version, signing_key)
        if refusal:
            self.record_event(
                self.body(
                    model,
                    version,
                    environment,
                    plane,
                    "rejected",
                    None,
                    image_digest,
                    refusal,
                )
            )
            return Outcome(False, tuple(refusal))
        previous = self.environment(model, environment)
        revision: str | None = None
        try:
            revision = plane.deploy(image_digest)
            control.confirm(
                plane, self.present(revision, image_digest), self.policy
            )
            for percent in steps:
                allocation = {revision: percent}
                if percent < 100:
                    if previous is None:
                        raise errors.ConfigError(
                            "a staged rollout needs a previous revision"
                        )
                    allocation[previous["revision"]] = 100 - percent
                plane.set_traffic(allocation)
                seen = control.confirm(
                    plane,
                    control.expecting(revision, image_digest, percent),
                    self.policy,
                )
                if health is not None:
                    stop = health(seen, percent)
                    if stop:
                        raise control.RuntimeUnconfirmed(
                            f"health stopped the rollout at {percent}%: "
                            + "; ".join(stop)
                        )
            self.commit_promotion(
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
            restored = self.restore(plane, previous)
            reasons = (str(error),)
            self.record_event(
                self.body(
                    model,
                    version,
                    environment,
                    plane,
                    "rejected",
                    revision,
                    image_digest,
                    list(reasons),
                    restored,
                )
            )
            return Outcome(False, reasons, restored)
        self.record_event(
            self.body(
                model,
                version,
                environment,
                plane,
                "promoted",
                revision,
                image_digest,
                [],
            )
        )
        return Outcome(True, (), None, version, revision)

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
        current = self.state_of(model, version)
        if current is not ReleaseState.APPROVED:
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
                release.approval_from_dict(stored).validate_target(
                    model, version, record.content_id, signing_key
                )
            except (errors.ArtifactError, TypeError, ValueError) as error:
                reasons.append(f"the approval is not valid: {error}")
        return reasons

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
            digest = seen.digest_of(revision)
            if digest is None:
                return [f"revision {revision} is not reported"]
            if digest != image_digest:
                return [f"revision {revision} runs {digest}"]
            return []

        return check

    def restore(
        self, plane: control.ControlPlane, previous: dict[str, Any] | None
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
            plane.set_traffic({previous["revision"]: 100})
            control.confirm(
                plane,
                control.expecting(
                    previous["revision"], previous["image_digest"], 100
                ),
                self.policy,
            )
        except (
            control.RuntimeUnconfirmed,
            control.Unsupported,
            errors.ScmleError,
        ):
            return False
        return True

    def body(
        self,
        model: str,
        version: int,
        environment: str,
        plane: control.ControlPlane,
        event: str,
        revision: str | None,
        image_digest: str,
        reasons: list[str],
        restored: bool | None = None,
    ) -> dict[str, Any]:
        """Builds the ledger row for an event.

        Args:
            model: Model name.
            version: Version number.
            environment: Environment name.
            plane: The platform, whose provider is recorded.
            event: The event name (for example `promoted`).
            revision: The platform revision involved, if any.
            image_digest: Digest of the image.
            reasons: Why the event happened or was refused.
            restored: Whether the previous revision was restored; None if not
                applicable.

        Returns:
            The event fields as a dict.
        """
        return {
            "event": event,
            "model": model,
            "version": version,
            "environment": environment,
            "provider": plane.provider,
            "revision": revision,
            "image_digest": image_digest,
            "reasons": reasons,
            "restored": restored,
        }

    def commit_promotion(
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
            production = environment == PRODUCTION_ENVIRONMENT
            if production:
                stored = entry["versions"][str(version)]["approval"]
                self.store.promote(
                    model,
                    version,
                    registry.CHAMPION,
                    release.approval_from_dict(stored),
                    signing_key,
                )
            known_good = list(
                (entry["environments"].get(environment) or {}).get(
                    "known_good", []
                )
            )
            if previous is not None:
                known_good.append(
                    {
                        "version": previous["version"],
                        "revision": previous["revision"],
                        "image_digest": previous["image_digest"],
                    }
                )
            if production:
                if previous is not None and previous["version"] != version:
                    self.move(
                        entry,
                        previous["version"],
                        ReleaseState.RETIRED,
                        f"superseded by v{version}",
                        ReleaseState.PRODUCTION,
                    )
                self.move(
                    entry,
                    version,
                    ReleaseState.PRODUCTION,
                    f"confirmed by the runtime in {environment}",
                    ReleaseState.APPROVED,
                )
            entry["environments"][environment] = {
                "version": version,
                "revision": revision,
                "image_digest": image_digest,
                "confirmed_at": events.utc_timestamp(),
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

    # -- rollback ----------------------------------------------------------

    def rollback(
        self,
        model: str,
        environment: str,
        plane: control.ControlPlane,
        signing_key: str | bytes | None = None,
    ) -> Outcome:
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
            return Outcome(False, ("there is no previous known-good revision",))
        target = current["known_good"][-1]
        try:
            plane.set_traffic({target["revision"]: 100})
            control.confirm(
                plane,
                control.expecting(
                    target["revision"], target["image_digest"], 100
                ),
                self.policy,
            )
        except (
            control.RuntimeUnconfirmed,
            control.Unsupported,
            errors.ScmleError,
        ) as error:
            restored = self.restore(plane, current)
            self.record_event(
                self.body(
                    model,
                    target["version"],
                    environment,
                    plane,
                    "rollback_rejected",
                    target["revision"],
                    target["image_digest"],
                    [str(error)],
                    restored,
                )
            )
            return Outcome(False, (str(error),), restored)
        try:
            self.commit_rollback(
                model, environment, current, target, signing_key
            )
        except errors.ScmleError as error:
            restored = self.restore(plane, current)
            self.record_event(
                self.body(
                    model,
                    target["version"],
                    environment,
                    plane,
                    "rollback_rejected",
                    target["revision"],
                    target["image_digest"],
                    [str(error)],
                    restored,
                )
            )
            return Outcome(False, (str(error),), restored)
        self.record_event(
            self.body(
                model,
                target["version"],
                environment,
                plane,
                "rolled_back",
                target["revision"],
                target["image_digest"],
                [],
            )
        )
        return Outcome(True, (), None, target["version"], target["revision"])

    def commit_rollback(
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
            if environment == PRODUCTION_ENVIRONMENT:
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
                        release.approval_from_dict(stored),
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
                    ReleaseState.RETIRED,
                    "rolled back",
                    ReleaseState.PRODUCTION,
                )
                self.move(
                    entry,
                    target["version"],
                    ReleaseState.PRODUCTION,
                    "rollback confirmed by the runtime",
                    ReleaseState.RETIRED,
                )
            entry["environments"][environment] = {
                "version": target["version"],
                "revision": target["revision"],
                "image_digest": target["image_digest"],
                "confirmed_at": events.utc_timestamp(),
                "known_good": current["known_good"][:-1],
            }
            self.write(state)
            self.store.audit(
                "release_rolled_back",
                model=model,
                environment=environment,
                to_version=target["version"],
            )

    # -- the claim "deployed" ---------------------------------------------

    def deployed_version(
        self,
        model: str,
        environment: str,
        seen: observation.Observation,
    ) -> int:
        """Returns the version the registry says is deployed, if confirmed.

        Args:
            model: Model name.
            environment: Environment name.
            seen: What the platform reports.

        Returns:
            The version recorded for the environment.

        Raises:
            RuntimeDisagrees: If the registry records no deployment or any
                finding of `reconcile` applies.
        """
        report = self.reconcile(model, environment, seen)
        recorded = self.environment(model, environment)
        if recorded is None or not report.consistent:
            raise RuntimeDisagrees(
                f"{model} is not confirmed deployed in {environment}",
                {"findings": [f.to_dict() for f in report.findings]},
            )
        return int(recorded["version"])

    # -- reconcile ---------------------------------------------------------

    def last_ledger_event(
        self, model: str, environment: str
    ) -> dict[str, Any] | None:
        """Returns the newest confirmed ledger event for the environment.

        Args:
            model: Model name.
            environment: Environment name.

        Returns:
            A copy of the newest promoted or rolled-back event for the model
            and environment, or None if there is no ledger or no such event.
        """
        if self.ledger is None:
            return None
        for row in reversed(self.ledger.rows):
            body = row["body"]
            if (
                row["kind"] == deployment_ledger.DEPLOYMENT
                and body.get("model") == model
                and body.get("environment") == environment
                and body.get("event") in ("promoted", "rolled_back")
            ):
                return dict(body)
        return None

    def reconcile(
        self,
        model: str,
        environment: str,
        seen: observation.Observation,
    ) -> ReconcileReport:
        """Compares the registry and the ledger with a platform observation.

        Every disagreement is a finding; none is repaired. The finding kinds
        are `no_registry_record`, `revision_mismatch`, `digest_mismatch`,
        `traffic_split`, `state_mismatch`, `not_ready`, `alias_not_deployed`,
        `ledger_missing` and `ledger_disagrees`.

        Args:
            model: Model name.
            environment: Environment name.
            seen: What the platform reports.

        Returns:
            The report listing every disagreement; consistent if there is none.
        """
        findings: list[Finding] = []
        recorded = self.environment(model, environment)
        if recorded is None:
            findings.append(
                Finding(
                    "no_registry_record",
                    f"the platform runs {seen.image_digest} but the registry "
                    f"records no deployment of {model} in {environment}",
                )
            )
        else:
            if seen.revision != recorded["revision"]:
                findings.append(
                    Finding(
                        "revision_mismatch",
                        f"registry says {recorded['revision']}, the platform "
                        f"serves {seen.revision}",
                    )
                )
            if seen.image_digest != recorded["image_digest"]:
                findings.append(
                    Finding(
                        "digest_mismatch",
                        f"registry says {recorded['image_digest']}, the "
                        f"platform runs {seen.image_digest}",
                    )
                )
            share = seen.share_of(recorded["revision"])
            if share != 100:
                findings.append(
                    Finding(
                        "traffic_split",
                        f"the recorded revision holds {share}% of traffic",
                    )
                )
            states = self.states(model)
            production = environment == PRODUCTION_ENVIRONMENT
            if (
                production
                and states.get(recorded["version"])
                is not ReleaseState.PRODUCTION
            ):
                findings.append(
                    Finding(
                        "state_mismatch",
                        f"v{recorded['version']} is deployed but its state "
                        f"is {states.get(recorded['version'], 'unknown')}",
                    )
                )
            for number, state in states.items():
                if (
                    production
                    and state is ReleaseState.PRODUCTION
                    and number != recorded["version"]
                ):
                    findings.append(
                        Finding(
                            "state_mismatch",
                            f"v{number} is in production state but the "
                            f"environment records v{recorded['version']}",
                        )
                    )
        if not seen.ready:
            findings.append(
                Finding("not_ready", "the platform does not report ready")
            )
        if environment == PRODUCTION_ENVIRONMENT:
            champion = self.store.aliases(model).get(registry.CHAMPION)
            expected = None if recorded is None else recorded["version"]
            if champion is not None and champion != expected:
                findings.append(
                    Finding(
                        "alias_not_deployed",
                        f"the champion alias points at v{champion}, which the "
                        f"runtime has not confirmed (confirmed: {expected})",
                    )
                )
        event = self.last_ledger_event(model, environment)
        if self.ledger is not None:
            if event is None and recorded is not None:
                findings.append(
                    Finding(
                        "ledger_missing",
                        "the ledger has no confirmed deployment event",
                    )
                )
            elif event is not None and (
                recorded is None
                or (event["version"], event["revision"], event["image_digest"])
                != (
                    recorded["version"],
                    recorded["revision"],
                    recorded["image_digest"],
                )
            ):
                findings.append(
                    Finding(
                        "ledger_disagrees",
                        f"the ledger's last event is v{event['version']} "
                        f"{event['revision']}",
                    )
                )
        return ReconcileReport(model, environment, tuple(findings))
