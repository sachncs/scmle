"""Pinned references: image and bundle by digest, secrets by name.

The value checks every part of the deployment contract shares live here too.
"""

from collections.abc import Mapping
import dataclasses
import enum
import math
import re
from typing import Any
from typing import Self

from scmle.contracts import errors

DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}")
VARIABLE_PATTERN = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
VERSION_PATTERN = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,63}")
PRIVATE_KEY_MARKER = "-----BEGIN"


class Architecture(enum.StrEnum):
    """CPU architectures an image may be built for."""

    __module__ = "scmle.deploy.contract"  # the path pinned signatures name

    AMD64 = "amd64"
    ARM64 = "arm64"


class SecretSource(enum.StrEnum):
    """Where a secret is read from at run time."""

    ENVIRONMENT = "env"
    FILE = "file"
    STORE = "store"


class Environment(enum.StrEnum):
    """Named environments a spec may target."""

    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


def require_digest(value: object, field_name: str) -> str:
    """Returns `value` if it is a `sha256:<64 hex>` digest.

    Args:
        value: The candidate digest.
        field_name: Name used in the error message.

    Returns:
        `value`, unchanged.

    Raises:
        ConfigError: If `value` is not `sha256:` followed by 64 lowercase hex
            characters (a tag is not a digest).
    """
    if not isinstance(value, str) or DIGEST_PATTERN.fullmatch(value) is None:
        raise errors.ConfigError(
            f"{field_name} must be 'sha256:' followed by 64 lowercase hex "
            "characters (a tag is not a digest)"
        )
    return value


def simple(value: object, field_name: str) -> str:
    """Returns `value` if it is a plain identifier.

    Args:
        value: The candidate name.
        field_name: Name used in the error message.

    Returns:
        `value`, unchanged.

    Raises:
        ConfigError: If `value` does not match `NAME_PATTERN`.
    """
    if not isinstance(value, str) or NAME_PATTERN.fullmatch(value) is None:
        raise errors.ConfigError(f"{field_name} is not a valid name: {value!r}")
    return value


def require_positive(value: object, field_name: str) -> float:
    """Returns `value` if it is a positive finite number.

    Args:
        value: The candidate number (a bool is refused).
        field_name: Name used in the error message.

    Returns:
        `value`, unchanged.

    Raises:
        ConfigError: If `value` is not an int or float, is not finite, or is not
            greater than zero.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise errors.ConfigError(f"{field_name} must be positive and finite")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class ImageRef:
    """A container image pinned by digest, with the platforms it covers.

    Attributes:
        repository: Image repository name.
        digest: Image digest as `sha256:<64 hex>`.
        architectures: CPU architectures the image is built for; non-empty and
            unique.
    """

    __module__ = "scmle.deploy.contract"  # the path pinned signatures name

    repository: str
    digest: str
    architectures: tuple[Architecture, ...] = (
        Architecture.AMD64,
        Architecture.ARM64,
    )

    def __post_init__(self) -> None:
        """Validates the reference.

        Raises:
            ConfigError: If the repository or digest is invalid, or the
                architectures are empty or repeated. An architecture that is
                not a valid `Architecture` value is rejected with a
                `ValueError`.
        """
        simple(self.repository, "image.repository")
        require_digest(self.digest, "image.digest")
        converted = tuple(Architecture(item) for item in self.architectures)
        object.__setattr__(self, "architectures", converted)
        if not converted or len(set(converted)) != len(converted):
            raise errors.ConfigError(
                "image.architectures must be non-empty and unique"
            )

    @property
    def reference(self) -> str:
        """The pull reference `repository@digest`."""
        return f"{self.repository}@{self.digest}"

    def to_dict(self) -> dict[str, Any]:
        """Builds a JSON-compatible record.

        Returns:
            A dict with `repository`, `digest` and the sorted architecture
            names.
        """
        return {
            "repository": self.repository,
            "digest": self.digest,
            "architectures": sorted(item.value for item in self.architectures),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the reference from a record.

        Args:
            record: A record produced by `to_dict`.

        Returns:
            The reference.
        """
        return cls(
            repository=record["repository"],
            digest=record["digest"],
            architectures=tuple(
                Architecture(item) for item in record["architectures"]
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class BundleRef:
    """A model bundle pinned by the digest of its manifest.

    Attributes:
        model: Model name.
        model_version: Model version string.
        digest: Manifest digest as `sha256:<64 hex>`.
        signed: Whether the bundle carries a verified signature.
    """

    __module__ = "scmle.deploy.contract"  # the path pinned signatures name

    model: str
    model_version: str
    digest: str
    signed: bool

    def __post_init__(self) -> None:
        """Validates the reference.

        Raises:
            ConfigError: If the version is not valid or `signed` is not a
                boolean; an invalid model name or digest is refused with the
                same error.
        """
        simple(self.model, "bundle.model")
        if VERSION_PATTERN.fullmatch(str(self.model_version)) is None:
            raise errors.ConfigError("bundle.model_version is not valid")
        require_digest(self.digest, "bundle.digest")
        if not isinstance(self.signed, bool):
            raise errors.ConfigError("bundle.signed must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        """Builds a JSON-compatible record.

        Returns:
            A dict of the four fields.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the reference from a record.

        Args:
            record: A record produced by `to_dict`.

        Returns:
            The reference.
        """
        return cls(
            model=record["model"],
            model_version=record["model_version"],
            digest=record["digest"],
            signed=record["signed"],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class SecretRef:
    """Names a secret and where to read it; never holds the value.

    `name` is the environment variable the service sees. `locator` is an
    environment variable name (`env`), an absolute path (`file`) or an opaque
    identifier in a secret store (`store`).

    Attributes:
        name: Upper-case variable name the service sees.
        source: Where the secret is read from.
        locator: Reference to the secret, interpreted according to `source`.
        required: Whether the service cannot run without the secret.
    """

    __module__ = "scmle.deploy.contract"  # the path pinned signatures name

    name: str
    source: SecretSource
    locator: str
    required: bool = True

    def __post_init__(self) -> None:
        """Validates the reference and refuses anything that looks inline.

        Raises:
            ConfigError: If the name is not an upper-case variable, or the
                locator is blank, padded, multi-line, contains `=` or key
                material, or does not suit its source (a variable name for
                `env`, an absolute path for `file`).
        """
        if VARIABLE_PATTERN.fullmatch(self.name) is None:
            raise errors.ConfigError(
                f"secret name must be an upper-case variable: {self.name!r}"
            )
        object.__setattr__(self, "source", SecretSource(self.source))
        text: object = self.locator  # may be any value from a record
        if (
            not isinstance(text, str)
            or not text.strip()
            or text != text.strip()
            or "\n" in text
            or "=" in text
            or PRIVATE_KEY_MARKER in text
        ):
            raise errors.ConfigError(
                f"secret {self.name} locator must be a plain reference"
            )
        if (
            self.source is SecretSource.ENVIRONMENT
            and VARIABLE_PATTERN.fullmatch(text) is None
        ):
            raise errors.ConfigError(
                f"secret {self.name} env locator must be a variable name"
            )
        if self.source is SecretSource.FILE and not text.startswith("/"):
            raise errors.ConfigError(
                f"secret {self.name} file locator must be an absolute path"
            )
        if self.source is SecretSource.STORE:
            simple(text, f"secret {self.name} locator")

    def to_dict(self) -> dict[str, Any]:
        """Builds a JSON-compatible record.

        Returns:
            A dict with `name`, `source` (its value), `locator` and `required`.
        """
        return {
            "name": self.name,
            "source": self.source.value,
            "locator": self.locator,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the reference from a record.

        Args:
            record: A record produced by `to_dict`; `required` defaults to True.
                A missing `name`, `source` or `locator` raises `KeyError`, and
                an unknown `source` raises `ValueError`.

        Returns:
            The reference.
        """
        return cls(
            name=record["name"],
            source=SecretSource(record["source"]),
            locator=record["locator"],
            required=record.get("required", True),
        )
