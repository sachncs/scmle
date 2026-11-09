"""Local, file-based model registry with versions, aliases and an audit log.

Layout under the registry root:

    registry.json            model -> versions -> metadata, aliases
    audit.jsonl              append-only record of every change
    <model>/v<N>/            the model bundle of each version
"""

from collections.abc import Mapping
import dataclasses
import enum
import hmac
import json
import pathlib
import shutil
from typing import Any
from typing import Self

from scmle import durability
from scmle.artifacts import bundle
from scmle.contracts import errors
from scmle.contracts import events
from scmle.governance import attestation
from scmle.governance import never_list

INDEX_FILE = "registry.json"
AUDIT_FILE = "audit.jsonl"
CHAMPION = "champion"
CHALLENGER = "challenger"
PREVIOUS = "previous"


class PromotionDecision(enum.StrEnum):
    """Decision recorded by an independent promotion reviewer."""

    APPROVED = "approved"
    REJECTED = "rejected"


@dataclasses.dataclass(frozen=True, slots=True)
class PromotionApproval:
    """Approval bound to one exact registered model bundle."""

    model: str
    version: int
    content_id: str
    reviewer: str
    decision: PromotionDecision
    reason: str
    evidence: tuple[str, ...]
    approved_at: str
    independent: bool = True
    signature: str | None = None

    def __post_init__(self) -> None:
        """Validates the review record before it can authorize promotion.

        A `decision` given as text is converted to a `PromotionDecision`.

        Raises:
            TypeError: If `independent` is not a bool.
            ValueError: If a required text field is blank, the version is not
                positive, the decision is not a valid `PromotionDecision`
                value, the evidence is empty or has blank references, or the
                signature is blank.
        """
        for name in (
            "model",
            "content_id",
            "reviewer",
            "reason",
            "approved_at",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.version < 1:
            raise ValueError("version must be positive")
        if not isinstance(self.decision, PromotionDecision):
            object.__setattr__(
                self, "decision", PromotionDecision(self.decision)
            )
        if not self.evidence or any(not item.strip() for item in self.evidence):
            raise ValueError("evidence must contain non-empty references")
        if not isinstance(self.independent, bool):
            raise TypeError("independent must be bool")
        if self.signature is not None and (
            not isinstance(self.signature, str) or not self.signature.strip()
        ):
            raise ValueError("signature must be non-empty text")

    def unsigned_dict(self) -> dict[str, Any]:
        """Returns the approval fields covered by its signature."""
        return {
            "model": self.model,
            "version": self.version,
            "content_id": self.content_id,
            "reviewer": self.reviewer,
            "decision": self.decision.value,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "approved_at": self.approved_at,
            "independent": self.independent,
        }

    def sign(self, signing_key: str | bytes) -> Self:
        """Returns this approval with a content-bound HMAC signature.

        The signature covers `unsigned_dict`. An empty key is refused by the
        signing helper with a `ValueError`.

        Args:
            signing_key: The key to sign with.

        Returns:
            A copy of the approval carrying the signature.
        """
        return dataclasses.replace(
            self,
            signature=attestation.sign_payload(
                self.unsigned_dict(), signing_key
            ),
        )

    def verify(self, signing_key: str | bytes | None = None) -> bool:
        """Checks the approval signature against an explicit or env key.

        Args:
            signing_key: The key to verify with, or None for the configured
                default.

        Returns:
            True if the approval is signed, a key is available (explicit, or
            ``$SCMLE_SIGNING_KEY``) and the signature matches the content.
        """
        if self.signature is None:
            return False
        key = attestation.resolve_key(signing_key)
        if key is None:
            return False
        expected = attestation.sign_payload(self.unsigned_dict(), key)
        return hmac.compare_digest(self.signature, expected)

    def validate_target(
        self,
        model: str,
        version: int,
        content_id: str,
        signing_key: str | bytes | None = None,
    ) -> None:
        """Checks exact identity, approval, independence and the signature.

        Args:
            model: Expected model name.
            version: Expected version.
            content_id: Expected content address of the bundle.
            signing_key: The key to verify with, or None for the configured
                default.

        Raises:
            ValueError: If the approval names another bundle, the reviewer
                rejected it, the review was not independent, or the signature
                is not valid.
        """
        if (self.model, self.version, self.content_id) != (
            model,
            version,
            content_id,
        ):
            raise ValueError(
                "promotion approval does not match bundle identity"
            )
        if self.decision is not PromotionDecision.APPROVED:
            raise ValueError("promotion was rejected by the reviewer")
        if not self.independent:
            raise ValueError("champion promotion requires independent review")
        if not self.verify(signing_key):
            raise ValueError("champion promotion requires a valid signature")

    def to_dict(self) -> dict[str, Any]:
        """Returns the complete approval record for durable audit evidence."""
        record = self.unsigned_dict()
        if self.signature is not None:
            record["signature"] = self.signature
        return record


@dataclasses.dataclass(frozen=True)
class VersionRecord:
    """Metadata of one registered version.

    Attributes:
        model: Registered model name.
        version: Sequential version number (1-based).
        registered_at: UTC registration time.
        metrics: Held-out metrics at registration.
        config: Project configuration the model was trained with.
        run_id: Originating run, if any.
        path: Bundle directory relative to the registry root.
        content_id: SHA-256 of the bundle manifest, which embeds the checksum
            of every file, so identical bundles share an id.
        signed: Whether the bundle's HMAC signature verified at registration
            (otherwise the registrant vouched for it with `trusted=True`).
    """

    model: str
    version: int
    registered_at: str
    metrics: dict[str, float]
    config: dict[str, Any]
    run_id: str
    path: str
    content_id: str = ""
    signed: bool = False


def content_id_of(bundle_directory: str | pathlib.Path) -> str:
    """Returns the content address of a bundle.

    It is the hash of the bundle's manifest, which records the checksum of
    every stored file.

    Args:
        bundle_directory: A saved bundle directory.

    Returns:
        The SHA-256 of the bundle's manifest file.
    """
    manifest = pathlib.Path(bundle_directory) / bundle.MANIFEST_FILE
    return bundle.file_sha256(manifest)


class ModelRegistryStore:
    """Versioned model storage with promotion by alias."""

    def __init__(
        self,
        root: str | pathlib.Path,
        guard: never_list.Guard | None = None,
        scope: str = "stream",
    ) -> None:
        """Opens a registry, creating its root directory if needed.

        Changes take a per-directory lock shared by every store object over
        the same root, so threads of one process never interleave a
        read-modify-write. Separate processes are not coordinated.

        A registry created with a `guard` asks it for a human attestation
        (never-list item 1) before the champion alias moves, whatever calls
        `promote`. Without a guard the registry behaves as before and only
        the signed approval protects the alias. The guard sits in front of
        this class: a person or process that edits `registry.json` directly
        passes neither check.

        Args:
            root: Registry directory.
            guard: Optional never-list guard (see `attach_guard`).
            scope: Model version hash and stream id attestations must name.
        """
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = durability.path_lock(self.root)
        self.guard = guard
        self.scope = scope

    def attach_guard(self, guard: never_list.Guard | None, scope: str) -> None:
        """Routes champion moves through `guard` (None removes it).

        Args:
            guard: The never-list guard, or None to remove it.
            scope: The scope attestations must carry.
        """
        self.guard = guard
        self.scope = scope

    def read_index(self) -> dict[str, Any]:
        """Loads the registry index (empty when new).

        Returns:
            The registry index; a fresh one without models if none is stored.
        """
        path = self.root / INDEX_FILE
        if not path.exists():
            return {"models": {}}
        return json.loads(path.read_text(encoding="utf-8"))

    def write_index(self, index: dict[str, Any]) -> None:
        """Writes the index atomically (temp file + rename).

        Args:
            index: The complete registry index.
        """
        durability.write_atomically(
            self.root / INDEX_FILE, json.dumps(index, indent=2).encode("utf-8")
        )

    def audit(self, action: str, **details: Any) -> None:
        """Appends an audit record.

        Args:
            action: The audit action name.
            **details: Further fields of the record.
        """
        record = {
            "timestamp": events.utc_timestamp(),
            "action": action,
            **details,
        }
        durability.append_text(
            self.root / AUDIT_FILE, json.dumps(record, default=str) + "\n"
        )

    def register(
        self,
        model: str,
        bundle_directory: str | pathlib.Path,
        metrics: dict[str, float] | None = None,
        config: dict[str, Any] | None = None,
        run_id: str = "",
        trusted: bool = False,
        signing_key: str | bytes | None = None,
    ) -> VersionRecord:
        """Registers a bundle as the next version of `model`.

        The bundle is verified (checksums, compatibility, and authenticity)
        before it is copied. Bundles are pickle-based, so an unsigned bundle
        is refused unless the registrant vouches for it with `trusted=True`
        (which is itself refused when ``$SCMLE_ENV`` is ``production``).
        A bundle that fails verification, or is unauthenticated and not
        vouched for, is refused with an `ArtifactError`. A copy is made under
        the registry root and an audit record is appended.

        Args:
            model: Model name.
            bundle_directory: A saved bundle directory.
            metrics: Held-out metrics to record.
            config: Training configuration to record.
            run_id: Originating run identifier.
            trusted: Vouch for an unsigned bundle you produced or trust.
            signing_key: HMAC key verifying the bundle's signature (defaults
                to `$SCMLE_SIGNING_KEY`).

        Returns:
            The new version's record, or the existing one when a bundle with
            identical content is already registered (registration is
            idempotent for the same content).
        """
        bundle.Bundle.load(
            bundle_directory, trusted=trusted, signing_key=signing_key
        )
        signed = bundle.verify_signature(
            pathlib.Path(bundle_directory), signing_key
        )
        content_id = content_id_of(bundle_directory)
        with self.lock:
            return self.register_locked(
                model,
                bundle_directory,
                content_id,
                signed,
                metrics,
                config,
                run_id,
            )

    def register_locked(
        self,
        model: str,
        bundle_directory: str | pathlib.Path,
        content_id: str,
        signed: bool,
        metrics: dict[str, float] | None,
        config: dict[str, Any] | None,
        run_id: str,
    ) -> VersionRecord:
        """Assigns the next version and writes it; the caller holds the lock.

        Copies the bundle into the store, updates the index and appends an
        audit record. A model that already has a version with the same
        content id gets no new version.

        Args:
            model: Model name.
            bundle_directory: A saved bundle directory, copied into the store.
            content_id: Content address of the bundle.
            signed: Whether the bundle's signature verified.
            metrics: Held-out metrics to record, or None.
            config: Training configuration to record, or None.
            run_id: Originating run identifier.

        Returns:
            The record of the new version, or of the existing one with the same
            content.
        """
        index = self.read_index()
        entry = index["models"].setdefault(
            model, {"versions": {}, "aliases": {}}
        )
        for known in entry["versions"].values():
            if known.get("content_id") == content_id:
                return VersionRecord(**known)
        version = max((int(v) for v in entry["versions"]), default=0) + 1
        relative = f"{model}/v{version}"
        shutil.copytree(bundle_directory, self.root / relative)
        record = VersionRecord(
            model,
            version,
            events.utc_timestamp(),
            dict(metrics or {}),
            dict(config or {}),
            run_id,
            relative,
            content_id,
            signed,
        )
        entry["versions"][str(version)] = dataclasses.asdict(record)
        self.write_index(index)
        self.audit("register", model=model, version=version, run_id=run_id)
        return record

    def versions(self, model: str) -> list[VersionRecord]:
        """Lists a model's versions, oldest first.

        Args:
            model: Model name.

        Returns:
            The version records in ascending version order.

        Raises:
            ArtifactError: If the model is not registered.
        """
        entry = self.read_index()["models"].get(model)
        if entry is None:
            raise errors.ArtifactError(f"unknown model '{model}'")
        return [
            VersionRecord(**entry["versions"][v])
            for v in sorted(entry["versions"], key=int)
        ]

    def aliases(self, model: str) -> dict[str, int]:
        """Returns alias -> version for a model.

        Args:
            model: Model name.

        Returns:
            A mapping from alias to version; empty if the model has none.
        """
        entry = self.read_index()["models"].get(model, {"aliases": {}})
        return {alias: int(v) for alias, v in entry["aliases"].items()}

    def promote(
        self,
        model: str,
        version: int,
        alias: str = CHAMPION,
        approval: PromotionApproval | None = None,
        signing_key: str | bytes | None = None,
        guard_detail: Mapping[str, Any] | None = None,
    ) -> None:
        """Points `alias` at `version`.

        A missing model or version, or a champion promotion without valid
        independent approval, is refused with an `ArtifactError`. If the
        registry has a guard and the champion move has no valid attestation,
        the guard raises `AttestationRequired` (an escalation is written and
        nothing is moved); the guard is asked after the model, version and
        approval are checked, so a refused request spends no attestation. When
        the champion moves to a different version, the version it replaced
        becomes `previous`. The move is recorded in the audit log.

        Args:
            model: Model name.
            version: Registered version number.
            alias: Alias to move (default `champion`).
            approval: Independent review bound to the exact bundle. Required
                when moving the champion alias; staging aliases do not serve
                traffic and may be moved without approval.
            signing_key: HMAC key for the approval signature. Defaults to
                ``$SCMLE_SIGNING_KEY``.
            guard_detail: What the attestation must name when the registry
                has a guard and `alias` is the champion; defaults to the
                model and version. A rollback names itself here.
        """
        with self.lock:
            self.check_promotion(model, version, alias, approval, signing_key)
            if alias == CHAMPION and self.guard is not None:
                self.guard.require(
                    never_list.NeverAction.MOVE_CHAMPION,
                    dict(guard_detail or {"model": model, "version": version}),
                    self.scope,
                )
            self.promote_locked(model, version, alias, approval, signing_key)

    def check_promotion(
        self,
        model: str,
        version: int,
        alias: str,
        approval: PromotionApproval | None,
        signing_key: str | bytes | None,
        index: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validates a move without changing anything.

        Args:
            model: Model name.
            version: Registered version number.
            alias: Alias to move.
            approval: The independent approval, required for the champion.
            signing_key: HMAC key for the approval signature.
            index: The registry index, read when not given.

        Returns:
            The model's index entry.

        Raises:
            ArtifactError: If the version is not registered, a champion move
                has no approval, or the approval is not valid for the bundle.
        """
        if index is None:
            index = self.read_index()
        entry = index["models"].get(model)
        if entry is None or str(version) not in entry["versions"]:
            raise errors.ArtifactError(f"{model} v{version} is not registered")
        record = VersionRecord(**entry["versions"][str(version)])
        if alias == CHAMPION:
            if approval is None:
                raise errors.ArtifactError(
                    "champion promotion requires independent approval"
                )
            try:
                approval.validate_target(
                    model,
                    version,
                    record.content_id,
                    signing_key,
                )
            except (TypeError, ValueError) as error:
                raise errors.ArtifactError(str(error)) from error
        return entry

    def promote_locked(
        self,
        model: str,
        version: int,
        alias: str,
        approval: PromotionApproval | None,
        signing_key: str | bytes | None,
    ) -> None:
        """Moves the alias; the caller holds the lock.

        Checks the version and, for the champion, the approval, then updates
        the index (setting `previous` when the champion changes version) and
        appends an audit record.

        Args:
            model: Model name.
            version: Registered version number.
            alias: Alias to move.
            approval: The independent approval, required for the champion.
            signing_key: HMAC key for the approval signature.

        Raises:
            ArtifactError: If the version is not registered, a champion move
                has no approval, or the approval is not valid for the bundle.
        """
        index = self.read_index()
        entry = self.check_promotion(
            model, version, alias, approval, signing_key, index
        )
        previous = entry["aliases"].get(alias)
        entry["aliases"][alias] = version
        if alias == CHAMPION and previous is not None and previous != version:
            entry["aliases"][PREVIOUS] = previous
        self.write_index(index)
        self.audit(
            "promote",
            model=model,
            alias=alias,
            version=version,
            previous=previous,
            approval_record=approval.to_dict() if approval else None,
        )

    def resolve(self, model: str, ref: str | int = CHAMPION) -> VersionRecord:
        """Resolves an alias or version number to a record.

        Args:
            model: Model name.
            ref: A version number (or its digits) or an alias.

        Returns:
            The record of the version.

        Raises:
            ArtifactError: If the model is unknown, the alias is not set or
                the version is not registered.
        """
        versions = {v.version: v for v in self.versions(model)}
        if isinstance(ref, int) or str(ref).isdigit():
            number = int(ref)
        else:
            aliases = self.aliases(model)
            if ref not in aliases:
                raise errors.ArtifactError(f"{model} has no alias '{ref}'")
            number = aliases[ref]
        if number not in versions:
            raise errors.ArtifactError(f"{model} v{number} is not registered")
        return versions[number]

    def load(
        self,
        model: str,
        ref: str | int = CHAMPION,
        signing_key: str | bytes | None = None,
    ) -> bundle.Bundle:
        """Loads a registered bundle by alias or version.

        A bundle registered as signed is authenticated again with the key; one
        the registrant vouched for is loaded as vouched (refused when
        ``$SCMLE_ENV`` is ``production``). An unresolvable `ref` or a bundle
        that fails verification raises `ArtifactError`.

        Args:
            model: Model name.
            ref: Alias or version number.
            signing_key: HMAC key for signed bundles.

        Returns:
            The loaded bundle.
        """
        record = self.resolve(model, ref)
        return bundle.Bundle.load(
            self.root / record.path,
            trusted=not record.signed,
            signing_key=signing_key,
        )
