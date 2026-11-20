"""The deployment record: what was released, from what, and who approved it.

One signed record binds the source commit, the model digest, the container
digest, the configuration digest, the environment and the approver, with the
digest of the evidence each live check produced. A record is written only by
the promotion gate (`scmle.deploy.gate`) after every check passed; reading it
back verifies the signature, so a changed field is detected.

The signature is an HMAC with a shared key: it shows the record was not
edited by someone without the key, not who held the key. The approver field
names the person the protected environment recorded; it is as trustworthy as
that environment's reviewer list.
"""

from collections.abc import Mapping
import dataclasses
import hashlib
import hmac
import json
import pathlib
import re
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.contracts import errors
from scmle.contracts import records
from scmle.contracts import signing
from scmle.deploy import contract

LEGACY = "format"  # the version key of a version-1 record
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
ENVIRONMENTS = ("staging", "production")
DIGESTS = {
    "model_digest": "model",
    "container_digest": "container",
    "config_digest": "config",
}
"""Version-1 digest keys of a record and their names under `digests`."""
MIGRATIONS = (
    records.nest(
        {
            "source_commit": "commit",
            "model_version": "version",
            "requested_by": "requester",
            "approved_at": "approved",
            "recorded_at": "recorded",
        },
        groups={"digests": DIGESTS},
    ),
)
"""The deployment record's version steps, oldest first."""
VERSION = records.ORIGIN + len(MIGRATIONS)


@dataclasses.dataclass(frozen=True, slots=True)
class Record:
    """A signed account of one release to one environment.

    A record keeps the version it was stored in: its signed content and its
    digest are those of the stored layout, so a version-1 record read by
    this build still verifies and keeps its ledger digest.

    Attributes:
        commit: The 40-hex commit the image was built from.
        model: Registered model name.
        version: Registry version number.
        digests: `model` (what the service reports as ``artifact_digest``),
            `container` (the image deployed) and `config` (the deployment
            spec).
        environment: ``staging`` or ``production``.
        provider: ``aws`` or ``gcp``.
        requester: Who started the release.
        approver: Who approved it in the protected environment.
        approved: UTC time of the approval.
        evidence: Check name to the digest of the exchange that passed it.
        revision: The runtime revision that was deployed.
        recorded: UTC time the record was written.
        signature: HMAC over every other field.
        schema_version: The layout the record was stored (and signed) in.
        migrations: The record's version steps, oldest first.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = MIGRATIONS

    commit: str
    model: str
    version: int
    digests: Mapping[str, str]
    environment: str
    provider: str
    requester: str
    approver: str
    approved: str
    evidence: tuple[tuple[str, str], ...]
    revision: str
    recorded: str
    signature: str = ""
    schema_version: int = VERSION

    def __post_init__(self) -> None:
        """Validates every field that a reader will rely on.

        Raises:
            ConfigError: If the commit is not 40 hex characters, the environment
                is not staging or production, a required text field is blank,
                the model version is below 1, or there is no evidence. A
                malformed or missing digest is refused with the same error.
        """
        if COMMIT_PATTERN.fullmatch(self.commit) is None:
            raise errors.ConfigError("commit must be 40 hex characters")
        for name in DIGESTS.values():
            contract.require(str(self.digests.get(name)), f"digests.{name}")
        if self.environment not in ENVIRONMENTS:
            raise errors.ConfigError(
                "environment must be staging or production"
            )
        for name in (
            "model",
            "provider",
            "requester",
            "approver",
            "approved",
            "revision",
            "recorded",
        ):
            if not str(getattr(self, name)).strip():
                raise errors.ConfigError(f"{name} is required")
        if self.version < 1:
            raise errors.ConfigError("version must be positive")
        if not self.evidence:
            raise errors.ConfigError("a record needs evidence")
        for check, digest in self.evidence:
            contract.require(digest, f"evidence[{check}]")

    def preimage(self) -> dict[str, Any]:
        """Builds the signed content, in the layout the record is stored in.

        Returns:
            All fields except the signature, with the evidence as a dict; a
            version-1 record has its version-1 keys and `format` 1.
        """
        record = dataclasses.asdict(self)
        del record["signature"]
        record["evidence"] = dict(self.evidence)
        record["digests"] = dict(self.digests)
        if self.schema_version == records.ORIGIN:
            del record[records.KEY]
            record = records.downgrade(self.migrations, record)
            record[LEGACY] = records.ORIGIN
        return record

    def to_dict(self) -> dict[str, Any]:
        """Builds the full record.

        Returns:
            The signed content plus the signature.
        """
        return {**self.preimage(), "signature": self.signature}

    def digest(self) -> str:
        """Digests the signed record, for the ledger and the next release.

        Returns:
            `sha256:` followed by the hex SHA-256 of the record's key-sorted
            JSON.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def signed(self, key: str | bytes) -> Self:
        """Returns this record carrying its signature under ``key``.

        Args:
            key: The HMAC key.

        Returns:
            A copy of the record with `signature` set.
        """
        return dataclasses.replace(
            self, signature=signing.mac(self.preimage(), key)
        )

    def verify(self, key: str | bytes) -> bool:
        """Checks the signature under ``key``.

        Args:
            key: The HMAC key.

        Returns:
            True if the record is signed and the signature matches; False if it
            is unsigned or does not match.
        """
        if not self.signature:
            return False
        expected = signing.mac(self.preimage(), key)
        return hmac.compare_digest(self.signature, expected)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a record from `to_dict`, of this or the first version.

        A version-1 record (`format` 1) is migrated and remembers its
        version, so its signature is checked against its stored layout.

        Args:
            record: A record produced by `to_dict`.

        Returns:
            The record, with whatever signature the dict carried (not verified).

        Raises:
            ConfigError: If fields are missing, extra or of a newer format.
        """
        values = dict(record)
        if records.KEY not in values and LEGACY not in values:
            raise errors.ConfigError("unsupported deployment record format")
        stored = records.detect("deployment record", values, LEGACY)
        if stored > VERSION:
            raise errors.ConfigError("unsupported deployment record format")
        values = records.upgrade(cls.migrations, values, stored)
        values[records.KEY] = stored
        names = {f.name for f in dataclasses.fields(cls)}
        if set(values) != names or set(values["digests"]) != set(
            DIGESTS.values()
        ):
            raise errors.ConfigError("unexpected or missing record fields")
        evidence = values["evidence"]
        if not isinstance(evidence, Mapping):
            raise errors.ConfigError("evidence must be an object")
        values["evidence"] = tuple(sorted(evidence.items()))
        values["digests"] = dict(values["digests"])
        return cls(**values)


def write(path: str | pathlib.Path, record: Record) -> None:
    """Writes the record as JSON.

    Args:
        path: Destination file; its directory is created.
        record: The record to write.
    """
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read(path: str | pathlib.Path) -> Record:
    """Reads a record; the caller still verifies the signature.

    Args:
        path: File written by `write`.

    Returns:
        The record.

    Raises:
        ArtifactError: If the file cannot be read or does not hold a JSON
            object. A malformed record is refused by `from_dict` with
            `ConfigError`.
    """
    try:
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise errors.ArtifactError(f"cannot read record {path}") from error
    if not isinstance(data, dict):
        raise errors.ArtifactError(f"{path} is not a record")
    return Record.from_dict(data)
