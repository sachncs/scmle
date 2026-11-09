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
from typing import Self

from scmle.contracts import errors
from scmle.contracts import signing
from scmle.deploy import contract

RECORD_FORMAT = 1
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
ENVIRONMENTS = ("staging", "production")


@dataclasses.dataclass(frozen=True, slots=True)
class Record:
    """A signed account of one release to one environment.

    Attributes:
        source_commit: The 40-hex commit the image was built from.
        model: Registered model name.
        model_version: Registry version number.
        model_digest: Digest the service reports as ``artifact_digest``.
        container_digest: Image digest that was deployed.
        config_digest: Digest of the deployment configuration (spec).
        environment: ``staging`` or ``production``.
        provider: ``aws`` or ``gcp``.
        requested_by: Who started the release.
        approver: Who approved it in the protected environment.
        approved_at: UTC time of the approval.
        evidence: Check name to the digest of the exchange that passed it.
        revision: The runtime revision that was deployed.
        recorded_at: UTC time the record was written.
        signature: HMAC over every other field.
    """

    source_commit: str
    model: str
    model_version: int
    model_digest: str
    container_digest: str
    config_digest: str
    environment: str
    provider: str
    requested_by: str
    approver: str
    approved_at: str
    evidence: tuple[tuple[str, str], ...]
    revision: str
    recorded_at: str
    signature: str = ""

    def __post_init__(self) -> None:
        """Validates every field that a reader will rely on.

        Raises:
            ConfigError: If the commit is not 40 hex characters, the environment
                is not staging or production, a required text field is blank,
                the model version is below 1, or there is no evidence. A
                malformed digest is refused with the same error.
        """
        if COMMIT_PATTERN.fullmatch(self.source_commit) is None:
            raise errors.ConfigError("source_commit must be 40 hex characters")
        for name in ("model_digest", "container_digest", "config_digest"):
            contract.require_digest(getattr(self, name), name)
        if self.environment not in ENVIRONMENTS:
            raise errors.ConfigError(
                "environment must be staging or production"
            )
        for name in (
            "model",
            "provider",
            "requested_by",
            "approver",
            "approved_at",
            "revision",
            "recorded_at",
        ):
            if not str(getattr(self, name)).strip():
                raise errors.ConfigError(f"{name} is required")
        if self.model_version < 1:
            raise errors.ConfigError("model_version must be positive")
        if not self.evidence:
            raise errors.ConfigError("a record needs evidence")
        for check, digest in self.evidence:
            contract.require_digest(digest, f"evidence[{check}]")

    def unsigned_dict(self) -> dict[str, Any]:
        """Builds the signed content.

        Returns:
            All fields except the signature, with the evidence as a dict and the
            record format number added.
        """
        record = dataclasses.asdict(self)
        del record["signature"]
        record["evidence"] = dict(self.evidence)
        record["format"] = RECORD_FORMAT
        return record

    def to_dict(self) -> dict[str, Any]:
        """Builds the full record.

        Returns:
            The signed content plus the signature.
        """
        return {**self.unsigned_dict(), "signature": self.signature}

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
            self, signature=signing.sign_payload(self.unsigned_dict(), key)
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
        expected = signing.sign_payload(self.unsigned_dict(), key)
        return hmac.compare_digest(self.signature, expected)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a record from `to_dict`.

        Args:
            record: A record produced by `to_dict`.

        Returns:
            The record, with whatever signature the dict carried (not verified).

        Raises:
            ConfigError: If fields are missing, extra or of a newer format.
        """
        values = dict(record)
        if values.pop("format", None) != RECORD_FORMAT:
            raise errors.ConfigError("unsupported deployment record format")
        names = {f.name for f in dataclasses.fields(cls)}
        if set(values) != names:
            raise errors.ConfigError("unexpected or missing record fields")
        evidence = values["evidence"]
        if not isinstance(evidence, Mapping):
            raise errors.ConfigError("evidence must be an object")
        values["evidence"] = tuple(sorted(evidence.items()))
        return cls(**values)


def write_record(path: str | pathlib.Path, record: Record) -> None:
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


def read_record(path: str | pathlib.Path) -> Record:
    """Reads a record; the caller still verifies the signature.

    Args:
        path: File written by `write_record`.

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
