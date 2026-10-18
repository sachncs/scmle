"""Local, file-based model registry with versions, aliases and an audit log.

Layout under the registry root:

    registry.json            model -> versions -> metadata, aliases
    audit.jsonl              append-only record of every change
    <model>/v<N>/            the model bundle of each version
"""

import dataclasses
import json
import pathlib
import shutil
from typing import Any

from scml.artifacts import bundle as bundle_module
from scml.contracts import errors
from scml.contracts import events

INDEX_FILE = "registry.json"
AUDIT_FILE = "audit.jsonl"
CHAMPION = "champion"
PREVIOUS = "previous"


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
    """Content address of a bundle: the hash of its checksummed manifest."""
    manifest = pathlib.Path(bundle_directory) / bundle_module.MANIFEST_FILE
    return bundle_module.file_sha256(manifest)


class ModelRegistryStore:
    """Versioned model storage with promotion by alias."""

    def __init__(self, root: str | pathlib.Path) -> None:
        """Opens (creating if needed) a registry.

        Args:
            root: Registry directory.
        """
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def read_index(self) -> dict[str, Any]:
        """Loads the registry index (empty when new)."""
        path = self.root / INDEX_FILE
        if not path.exists():
            return {"models": {}}
        return json.loads(path.read_text(encoding="utf-8"))

    def write_index(self, index: dict[str, Any]) -> None:
        """Writes the index atomically (temp file + rename)."""
        temporary = self.root / (INDEX_FILE + ".tmp")
        temporary.write_text(
            json.dumps(index, indent=2), encoding="utf-8", newline="\n"
        )
        temporary.replace(self.root / INDEX_FILE)

    def audit(self, action: str, **details: Any) -> None:
        """Appends an audit record."""
        record = {
            "timestamp": events.utc_timestamp(),
            "action": action,
            **details,
        }
        with (self.root / AUDIT_FILE).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

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
        is refused unless the registrant vouches for it with `trusted=True`.

        Args:
            model: Model name.
            bundle_directory: A saved bundle directory.
            metrics: Held-out metrics to record.
            config: Training configuration to record.
            run_id: Originating run identifier.
            trusted: Vouch for an unsigned bundle you produced or trust.
            signing_key: HMAC key verifying the bundle's signature (defaults
                to `$SCML_SIGNING_KEY`).

        Returns:
            The new version's record, or the existing one when a bundle with
            identical content is already registered (registration is
            idempotent for the same content).

        Raises:
            ArtifactError: If the bundle fails verification or is
                unauthenticated and not vouched for.
        """
        bundle_module.ModelBundle.load(
            bundle_directory, trusted=trusted, signing_key=signing_key
        )
        signed = bundle_module.verify_signature(
            pathlib.Path(bundle_directory), signing_key
        )
        content_id = content_id_of(bundle_directory)
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
        """Lists a model's versions, oldest first."""
        entry = self.read_index()["models"].get(model)
        if entry is None:
            raise errors.ArtifactError(f"unknown model '{model}'")
        return [
            VersionRecord(**entry["versions"][v])
            for v in sorted(entry["versions"], key=int)
        ]

    def aliases(self, model: str) -> dict[str, int]:
        """Returns alias -> version for a model."""
        entry = self.read_index()["models"].get(model, {"aliases": {}})
        return {alias: int(v) for alias, v in entry["aliases"].items()}

    def promote(self, model: str, version: int, alias: str = CHAMPION) -> None:
        """Points `alias` at `version`.

        Args:
            model: Model name.
            version: Registered version number.
            alias: Alias to move (default `champion`).

        Raises:
            ArtifactError: If the model or version does not exist.
        """
        index = self.read_index()
        entry = index["models"].get(model)
        if entry is None or str(version) not in entry["versions"]:
            raise errors.ArtifactError(f"{model} v{version} is not registered")
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
        )

    def resolve(self, model: str, ref: str | int = CHAMPION) -> VersionRecord:
        """Resolves an alias or version number to a record.

        Raises:
            ArtifactError: If it cannot be resolved.
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
    ) -> bundle_module.ModelBundle:
        """Loads a registered bundle by alias or version.

        A bundle registered as signed is authenticated again with the key; one
        the registrant vouched for is loaded as the registrant vouched.

        Args:
            model: Model name.
            ref: Alias or version number.
            signing_key: HMAC key for signed bundles.
        """
        record = self.resolve(model, ref)
        return bundle_module.ModelBundle.load(
            self.root / record.path,
            trusted=not record.signed,
            signing_key=signing_key,
        )
