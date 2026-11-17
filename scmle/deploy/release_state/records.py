"""The release state file and the event log every release step writes."""

from collections.abc import Sequence
import json
from typing import Any

from scmle import durability
from scmle.contracts import errors
from scmle.contracts import events
from scmle.deploy import control
from scmle.deploy.release_state import status
from scmle.governance import deployment_ledger
from scmle.governance import registry


class Records:
    """The state file, the event log and restoring a revision."""

    def __init__(
        self,
        store: registry.ModelRegistryStore,
        ledger: deployment_ledger.DeploymentLedger | None = None,
        policy: control.Polling | None = None,
    ) -> None:
        """Binds the release state to a registry.

        Args:
            store: The model registry (its directory also holds the state).
            ledger: Optional deployment ledger that records each event.
            policy: How long to wait for the platform to confirm.
        """
        self.store = store
        self.ledger = ledger
        self.policy = policy or control.Polling()

    # -- the state file --------------------------------------------------

    def read(self) -> dict[str, Any]:
        """Loads the state file (empty when new).

        Returns:
            The state document, with a `models` mapping; a fresh one if the
            file does not exist.

        Raises:
            ArtifactError: If the file's format is not the supported one.
        """
        path = self.store.root / status.STATE_FILE
        if not path.exists():
            return {"format": status.STATE_FORMAT, "models": {}}
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("format") != status.STATE_FORMAT:
            raise errors.ArtifactError("unsupported release state format")
        return dict(state)

    def write(self, state: dict[str, Any]) -> None:
        """Writes the state atomically.

        Args:
            state: The complete state document.
        """
        durability.write_atomically(
            self.store.root / status.STATE_FILE,
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

    def lookup(self, model: str, version: int) -> status.ReleaseState:
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
        return (
            status.ReleaseState(found["state"])
            if found
            else status.ReleaseState.CANDIDATE
        )

    def states(self, model: str) -> dict[int, status.ReleaseState]:
        """Returns the release state of every registered version.

        Args:
            model: Model name.

        Returns:
            A mapping from version number to its release state.
        """
        return {
            v.version: self.lookup(model, v.version)
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
        target: status.ReleaseState,
        reason: str,
        current: status.ReleaseState,
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
                "at": events.stamp(),
                "reason": reason,
            }
        )
        record["state"] = target.value

    def log(self, body: dict[str, Any]) -> None:
        """Appends a deployment row to the ledger, if there is one.

        The event is always written to the store's audit trail.

        Args:
            body: The event fields, as built by `body`.
        """
        self.store.audit("release_event", **body)
        if self.ledger is not None:
            self.ledger.commit(deployment_ledger.DEPLOYMENT, body)

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

    def settle(
        self,
        event: str,
        model: str,
        version: int,
        environment: str,
        plane: control.ControlPlane,
        revision: str | None,
        image_digest: str,
        reasons: Sequence[str] = (),
        restored: bool | None = None,
    ) -> status.Outcome:
        """Logs a release event and returns the outcome it stands for.

        Args:
            event: The event name (for example `promoted`).
            model: Model name.
            version: The version the event is about.
            environment: Environment name.
            plane: The platform, whose provider is recorded.
            revision: The platform revision involved, if any.
            image_digest: Digest of the image.
            reasons: Why the release was refused or failed; empty if it was
                confirmed.
            restored: Whether the previous revision was restored; None if not
                applicable.

        Returns:
            Done, with the version and revision now serving, when there are
            no reasons; otherwise not done, with the reasons and `restored`.
        """
        self.log(
            self.body(
                model,
                version,
                environment,
                plane,
                event,
                revision,
                image_digest,
                list(reasons),
                restored,
            )
        )
        if reasons:
            return status.Outcome(
                done=False, reasons=tuple(reasons), restored=restored
            )
        return status.Outcome(
            done=True,
            reasons=(),
            restored=None,
            version=version,
            revision=revision,
        )
