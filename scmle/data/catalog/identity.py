"""Source identity: normalized URIs, digests and governance context."""

from collections.abc import Mapping
import dataclasses
import pathlib
from typing import Any
from typing import ClassVar
from typing import Self
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

from scmle.contracts import errors
from scmle.contracts import records
from scmle.data.catalog import canon

DIGEST_LENGTH = 64


def normalize(uri: str) -> str:
    """Normalizes a source URI without resolving caller-owned remote data.

    Args:
        uri: A URI or a local filesystem path. Surrounding whitespace is
            stripped.

    Returns:
        For a URI with a scheme, the same URI with the scheme lower-cased.
        A single-letter scheme followed by a colon (a Windows drive) is
        treated as a path, as is any value without a scheme: it is expanded
        and resolved to an absolute ``file://`` URI.  Nothing is fetched.
    """
    value = uri.strip()
    parsed = urlsplit(value)
    if parsed.scheme and not (len(parsed.scheme) == 1 and value[1:2] == ":"):
        return urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc,
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )
    return pathlib.Path(value).expanduser().resolve(strict=False).as_uri()


def assay(value: str, field_name: str) -> str:
    """Validates and normalizes a SHA-256 identity.

    Args:
        value: A 64-character hexadecimal digest, optionally prefixed with
            ``"sha256:"``.
        field_name: Name of the field being validated, used in the error
            message.

    Returns:
        The digest as ``"sha256:"`` followed by 64 lowercase hex digits.

    Raises:
        ValueError: If the value (after removing the prefix) is not exactly
            64 characters or is not hexadecimal.
    """
    normalized = value.removeprefix("sha256:")
    if len(normalized) != DIGEST_LENGTH:
        raise errors.InputError(f"{field_name} must be a SHA-256 digest")
    try:
        int(normalized, 16)
    except ValueError as error:
        raise errors.InputError(
            f"{field_name} must be a SHA-256 digest"
        ) from error
    return f"sha256:{normalized.lower()}"


@dataclasses.dataclass(frozen=True, slots=True)
class Source:
    """Identity and governance context for a caller-owned data source.

    Attributes:
        uri: The source URI or path; normalized on construction.
        version: The source's own version label.
        owner: Who owns the source.
        acquisition: How the data was acquired.
        privacy: The privacy classification of the data.
    """

    __module__ = "scmle.data.catalog"

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "acquisition_method": "acquisition",
                "privacy_classification": "privacy",
            }
        ),
        records.Migration(renames={"source_identity": "identity"}),
    )

    uri: str | pathlib.Path = "memory://"
    version: str = ""
    owner: str = ""
    acquisition: str = ""
    privacy: str = ""

    def __post_init__(self) -> None:
        """Validates and normalizes source identity fields.

        Raises:
            ValueError: If ``uri`` is empty or whitespace.
            TypeError: If ``version``, ``owner``, ``acquisition``, or
                ``privacy`` is not a string.
        """
        uri_value = str(self.uri)
        if not uri_value.strip():
            raise errors.InputError("uri must not be empty")
        object.__setattr__(self, "uri", normalize(uri_value))
        for field_name in ("version", "owner", "acquisition", "privacy"):
            field_value = getattr(self, field_name)
            if not isinstance(field_value, str):
                raise errors.KindError(f"{field_name} must be a string")
            object.__setattr__(self, field_name, field_value.strip())

    @property
    def source_identity(self) -> str:
        """The deterministic identity of the declared source.

        This is the ``sha256:`` digest of the URI, version, owner,
        acquisition method, and privacy classification. The digest names
        the source across runs, so it hashes the version-1 keys
        (``acquisition_method``, ``privacy_classification``): renaming the
        fields never changes the identity of a registered source.
        """
        return canon.digest_value(
            {
                "uri": self.uri,
                "version": self.version,
                "owner": self.owner,
                "acquisition_method": self.acquisition,
                "privacy_classification": self.privacy,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native source evidence.

        Returns:
            A dictionary of the five source fields plus the source identity
            under ``identity``.
        """
        return {
            "uri": self.uri,
            "version": self.version,
            "owner": self.owner,
            "acquisition": self.acquisition,
            "privacy": self.privacy,
            "identity": self.source_identity,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a source and verifies its declared identity.

        Args:
            record: A mapping as produced by ``to_dict``.  ``uri`` is required;
                the other fields default to empty strings.  A missing ``uri``
                raises ``KeyError``.

        Returns:
            The reconstructed source.

        Raises:
            ValueError: If the record's identity (``identity``, or
                ``source_identity`` as version 2 wrote it) is absent or
                differs from the identity recomputed from the source fields.
        """
        source = cls(
            uri=str(record["uri"]),
            version=str(record.get("version", "")),
            owner=str(record.get("owner", "")),
            acquisition=str(record.get("acquisition", "")),
            privacy=str(record.get("privacy", "")),
        )
        expected = record.get("identity", record.get("source_identity"))
        if expected != source.source_identity:
            raise errors.InputError(
                "source_identity does not match source fields"
            )
        return source
