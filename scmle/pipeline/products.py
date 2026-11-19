"""Stage products: what one stage made, as a resumed run needs it again.

Each stage names its product (`makes`) and returns it. A product is a
frozen dataclass of what the stage made; the engine applies it to the
run's board, which holds it by type (`apply`), whether the stage ran or a
resumed run restored the product instead of running the stage.

There are two kinds, and every product is one of them:

- `Kept`: saved next to the stage record as an authenticated envelope
  (`envelope.seal`) of its pickled fields. Unpickling executes code, so a
  kept product loads only when its envelope verifies with the signing key
  and its digest is the one the stage record names; anything else, an
  unsigned file included, is not loaded and its stage runs again (SEC-1).
- `Derived`: never written. It holds data that must not be copied into the
  run directory (the source rows, the locked test set, a held time
  reserve), so its stage runs again on resume. Its digest is that of a
  JSON summary of what it was derived from, the same in every process, so
  the stages after it can tell whether their inputs changed.
"""

import abc
import dataclasses
import hashlib
import io
import json
import pathlib
import pickle  # nosec B403 -- for PicklingError only
from typing import Self
from typing import TypeVar

import joblib

from scmle import envelope
from scmle.contracts import errors

T = TypeVar("T")
TIMING = "seconds"
"""The key suffix of a measured duration (`seconds`, `read_seconds`): it
differs between two runs of the same inputs, so it is no part of a digest."""


def untimed(value: object) -> object:
    """Drops every mapping entry whose key ends in `TIMING`, at any depth.

    Args:
        value: JSON-ready data.

    Returns:
        The data without measured durations.
    """
    if isinstance(value, dict):
        return {
            key: untimed(item)
            for key, item in value.items()
            if not str(key).endswith(TIMING)
        }
    if isinstance(value, list | tuple):
        return [untimed(item) for item in value]
    return value


def require(value: T | None, name: str) -> T:
    """Returns a product field, failing loudly if its stage has not run.

    Args:
        value: The possibly unset field.
        name: Field name used in the error message.

    Returns:
        `value`, narrowed to non-None.

    Raises:
        ScmleError: If the producing stage has not executed.
    """
    if value is None:
        raise errors.ScmleError(
            f"'{name}' is unavailable: its stage has not run yet"
        )
    return value


def digest(payload: bytes) -> str:
    """Hashes product bytes.

    Args:
        payload: The bytes.

    Returns:
        The hex SHA-256 digest.
    """
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class Product(abc.ABC):
    """What one stage made; the board holds it by type."""

    def apply(self, board: object) -> None:
        """Holds the product on the board, replacing one of the same type.

        A board that keeps a `made` mapping (`engine.Board`) holds the
        latest product of each type there, which is what a stage reading a
        product (`get`) finds; on another board this does nothing. A
        product whose stage also changes the run's services or journal (the
        governor, the capabilities, the trials) sets them in its own
        `apply`, so a restored product changes them as the stage did.

        Args:
            board: The run's board.
        """
        made = getattr(board, "made", None)
        if isinstance(made, dict):
            made[type(self)] = self

    def close(self) -> None:  # noqa: B027 -- most products hold nothing open
        """Releases what the product holds open; most hold nothing.

        The board closes every product it holds when the run ends
        (`engine.Board.close`).
        """

    @abc.abstractmethod
    def save(self, path: pathlib.Path) -> str | None:
        """Writes the product, if it is kept, and returns its digest.

        Args:
            path: Where a kept product is written.

        Returns:
            The digest the stage record names, or None when the product
            cannot be digested (its stage then always runs again, and so
            does every stage after it).
        """

    @classmethod
    @abc.abstractmethod
    def load(cls, path: pathlib.Path, expected: str) -> Self | None:
        """Reads a product its stage record names.

        Args:
            path: Where a kept product was written.
            expected: The digest the stage record names.

        Returns:
            The product, or None when it is not kept, is missing, does not
            authenticate or is not the one the record names.
        """


@dataclasses.dataclass(frozen=True)
class Kept(Product):
    """A product saved as an authenticated envelope of its pickled fields."""

    def save(self, path: pathlib.Path) -> str | None:
        """Seals the pickled fields next to the stage record.

        Args:
            path: The envelope to write.

        Returns:
            The digest of the pickled fields, or None (and no file) when a
            field cannot be pickled.
        """
        values = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
        }
        buffer = io.BytesIO()
        try:
            joblib.dump(values, buffer)
        except (pickle.PicklingError, TypeError, AttributeError):
            # handled: returns None, so the stage runs again on resume
            path.unlink(missing_ok=True)
            return None
        payload = buffer.getvalue()
        envelope.seal(path, {"product": type(self).__name__}, payload)
        return digest(payload)

    @classmethod
    def load(cls, path: pathlib.Path, expected: str) -> Self | None:
        """Unpickles the fields once the envelope authenticates.

        Args:
            path: The envelope.
            expected: The digest the stage record names.

        Returns:
            The product, or None when the file is missing, unsigned, not
            authentic, of another product or not the recorded one.
        """
        try:
            meta, payload = envelope.read(path)
        except (OSError, errors.InputError, errors.ArtifactError):
            # handled: returns None, so the stage runs again on resume
            return None
        if meta.get("product") != cls.__name__ or digest(payload) != expected:
            return None
        return cls(**joblib.load(io.BytesIO(payload)))


@dataclasses.dataclass(frozen=True)
class Derived(Product):
    """A product its stage makes again on resume; only its digest is kept."""

    @abc.abstractmethod
    def summary(self) -> object:
        """What the product was derived from, as JSON-ready data.

        Entries whose key ends in `TIMING` are left out of the digest.

        Returns:
            Data that is equal in every process exactly when the product's
            inputs are.
        """

    def save(self, path: pathlib.Path) -> str:  # noqa: ARG002 -- Product.save
        """Digests the summary; nothing is written.

        Args:
            path: Unused: a derived product is never written.

        Returns:
            The digest of the canonical JSON of `summary`, timings left out.
        """
        text = json.dumps(
            untimed(self.summary()),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return digest(text.encode("utf-8"))

    @classmethod
    def load(
        cls,
        path: pathlib.Path,  # noqa: ARG003 -- Product.load
        expected: str,  # noqa: ARG003 -- Product.load
    ) -> None:
        """A derived product is never loaded: its stage runs again.

        Args:
            path: Unused.
            expected: Unused.
        """
        return
