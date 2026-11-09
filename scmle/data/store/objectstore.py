"""The object-store interface and its local backend.

Every backend stores opaque bytes under a key and reports the same metadata:
the SHA-256 of the content (computed by this code, not trusted from the
service), the size and the modification time. Backends differ only in how
they reach the storage. The behaviour all of them must show is defined by
the shared conformance suite in `tests/unit/test_object_store_conformance.py`.

Delivery semantics: `put` replaces atomically (a reader sees the old or the
new object, never a mix); `put(..., if_absent=True)` creates only if no
object exists and otherwise raises `AlreadyExistsError`, which is the
building block for locks and for idempotent writers.
"""

import abc
from collections.abc import Callable
import dataclasses
import datetime
import hashlib
import os
import pathlib
import re
import tempfile
import time
from typing import TypeVar

from scmle import durability
from scmle.contracts import errors
from scmle.core import retry

T = TypeVar("T")
MAX_KEY_LENGTH = 1024
KEY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._=+@ -]*")


class StoreError(errors.DataError):
    """An object-store operation failed."""

    category = "store"


class NotFoundError(StoreError):
    """No object exists under the key."""

    category = "store_not_found"


class AlreadyExistsError(StoreError):
    """An object already exists and the write was conditional."""

    category = "store_exists"


class CorruptObjectError(StoreError):
    """The bytes read do not match the recorded SHA-256."""

    category = "store_corrupt"


class TransientStoreError(retry.TransientError, StoreError):
    """Throttling, a timeout or a 5xx: the operation may be retried."""

    category = "store_transient"


@dataclasses.dataclass(frozen=True, slots=True)
class ObjectMeta:
    """What a store knows about one object.

    Attributes:
        key: The key.
        size: Length in bytes.
        sha256: Hex SHA-256 of the content.
        modified: ISO 8601 UTC modification time.
        version: The backend's version or generation id, if it has one.
    """

    key: str
    size: int
    sha256: str
    modified: str
    version: str | None = None


def sha256_of(data: bytes) -> str:
    """Computes the SHA-256 of some bytes.

    Args:
        data: The content to hash.

    Returns:
        The digest as 64 lowercase hexadecimal characters.
    """
    return hashlib.sha256(data).hexdigest()


def check_key(key: str) -> str:
    """Checks that a key is a safe relative key.

    A key is `/`-separated segments; no empty segment, no `.` or `..`, no
    leading or trailing `/`, at most 1024 characters, and a conservative
    character set, so the same key is valid on every backend and cannot
    escape a root directory.

    Args:
        key: The candidate key.

    Returns:
        `key`, unchanged.

    Raises:
        StoreError: If the key is not a non-empty string of at most 1024
            characters, or a segment is empty, `.`, `..` or uses a character
            outside the allowed set.
    """
    if not isinstance(key, str) or not key or len(key) > MAX_KEY_LENGTH:
        raise StoreError("a key must be a non-empty string of at most 1024")
    for segment in key.split("/"):
        if segment in ("", ".", "..") or KEY_PATTERN.fullmatch(segment) is None:
            raise StoreError(f"invalid key segment {segment!r} in {key!r}")
    return key


def is_temporary(name: str) -> bool:
    """Whether a file name is a half-written temp file, not an object.

    Both the exclusive-create path (``.put-*``) and
    `scmle.durability.write_atomically` (``.<name>.<random>.tmp``) write a
    temp file first. A writer killed between the two steps leaves one behind;
    it must not make the store unlistable.

    Args:
        name: A file name, without its directory.

    Returns:
        True if the name is one of the two temp-file shapes, False for a
        name that may be an object.
    """
    return name.startswith(".put-") or (
        name.startswith(".") and name.endswith(".tmp")
    )


def check_prefix(prefix: str) -> str:
    """Checks a listing prefix: empty, or a key optionally ending in `/`.

    Args:
        prefix: The candidate prefix.

    Returns:
        `prefix`, unchanged. `check_key` raises `StoreError` if a non-empty
        prefix, without one trailing `/`, is not a valid key (so `/` alone
        is refused).
    """
    if prefix:
        check_key(prefix.removesuffix("/"))
    return prefix


def iso(moment: float | datetime.datetime) -> str:
    """Formats a moment as an ISO 8601 UTC time, to the second.

    Args:
        moment: Seconds since the Unix epoch, or a datetime. A naive
            datetime is taken to be in local time.

    Returns:
        The time in UTC with a `+00:00` offset, for example
        `2024-05-01T12:30:00+00:00`.
    """
    if isinstance(moment, int | float):
        moment = datetime.datetime.fromtimestamp(moment, datetime.UTC)
    return moment.astimezone(datetime.UTC).isoformat(timespec="seconds")


class ObjectStore(abc.ABC):
    """Bytes under keys, with the same behaviour on every backend.

    Attributes:
        scheme: The backend's URI scheme.
        location: Human-readable root of the store, without a trailing `/`.
    """

    scheme = "memory"

    def __init__(self, location: str) -> None:
        """Initializes the store.

        Args:
            location: Human-readable root, for lineage (`file:///data/prefix`).
        """
        self.location = location.rstrip("/")

    def uri(self, key: str) -> str:
        """Builds the URI of an object, for lineage records.

        Args:
            key: The object key.

        Returns:
            The store's location, `/` and the key. `check_key` raises
            `StoreError` if the key is not valid.
        """
        return f"{self.location}/{check_key(key)}"

    @abc.abstractmethod
    def put(
        self, key: str, data: bytes, *, if_absent: bool = False
    ) -> ObjectMeta:
        """Stores `data`, replacing any object unless `if_absent`.

        Args:
            key: The object key.
            data: The content to store.
            if_absent: Create the object only if none exists.

        Returns:
            The metadata of the stored object.

        Raises:
            AlreadyExistsError: With `if_absent` and an existing object.
            TransientStoreError: On a retryable service failure.
        """

    @abc.abstractmethod
    def get(self, key: str) -> tuple[bytes, ObjectMeta]:
        """Reads an object.

        Args:
            key: The object key.

        Returns:
            The content and the object's metadata.

        Raises:
            NotFoundError: If there is no such object.
            CorruptObjectError: If the bytes do not match the recorded hash.
        """

    @abc.abstractmethod
    def head(self, key: str) -> ObjectMeta:
        """Reads an object's metadata without its content.

        Args:
            key: The object key.

        Returns:
            The object's metadata.

        Raises:
            NotFoundError: If there is no such object.
        """

    @abc.abstractmethod
    def list_objects(self, prefix: str = "") -> list[ObjectMeta]:
        """Lists the objects whose key starts with a prefix.

        Args:
            prefix: Key prefix; empty for every object.

        Returns:
            The metadata of each matching object, sorted by key.
        """

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """Removes an object; removing a missing object is not an error.

        Args:
            key: The object key.
        """

    def exists(self, key: str) -> bool:
        """Checks whether an object exists.

        Args:
            key: The object key.

        Returns:
            True if `head` finds the object, False if it reports it missing.
        """
        try:
            self.head(key)
        except NotFoundError:
            return False
        return True


# Windows error codes for a link onto a path another writer holds: access
# denied, sharing violation, file exists and already exists.
EXISTING_WINERRORS = frozenset({5, 32, 80, 183})
RETRY_ATTEMPTS = 20
EXISTS_ATTEMPTS = 6


def refused_as_existing(error: OSError, path: pathlib.Path) -> bool:
    """Whether a failed exclusive link means the target is already taken.

    Args:
        error: What `os.link` raised.
        path: The target of the link.

    Returns:
        True for FileExistsError. Also True for a PermissionError or an
        error whose Windows code is in `EXISTING_WINERRORS` (access denied,
        sharing violation, file exists), provided the target is seen to
        exist within a few short retries. False for any other error, or if
        the target never appears.
    """
    if isinstance(error, FileExistsError):
        return True
    code = getattr(error, "winerror", None)
    if code not in EXISTING_WINERRORS and not isinstance(
        error, PermissionError
    ):
        return False
    for attempt in range(EXISTS_ATTEMPTS):
        if os.path.lexists(path):
            return True
        time.sleep(0.005 * (attempt + 1))
    return False


def unlink_tolerantly(path: pathlib.Path) -> None:
    """Removes a temp file, retrying while Windows still holds it open.

    Gives up quietly after 20 attempts that fail with PermissionError.

    Args:
        path: The temp file to remove; a missing file is not an error.
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(0.005 * (attempt + 1))


class LocalObjectStore(ObjectStore):
    """Objects as files under a root directory.

    Attributes:
        root: The resolved directory holding the objects.
    """

    scheme = "file"

    def __init__(self, root: str | pathlib.Path) -> None:
        """Initializes the store.

        Args:
            root: Directory holding the objects (created when needed).
        """
        self.root = pathlib.Path(root).resolve()
        super().__init__(self.root.as_uri())

    def path_of(self, key: str) -> pathlib.Path:
        """The file for a key, never outside the root.

        The check is lexical on the normalised absolute path: keys are
        validated by `check_key`, and resolving the target would depend on
        which directories exist at that instant (and on 8.3 short names on
        Windows), which a concurrent writer can change.

        Args:
            key: The object key.

        Returns:
            The absolute path of the file that holds the object.

        Raises:
            StoreError: If the key is not valid or its path leaves the root.
        """
        target = pathlib.Path(os.path.abspath(self.root / check_key(key)))
        if self.root not in target.parents:
            raise StoreError(f"key {key!r} escapes the store root")
        return target

    def meta_of(self, key: str, path: pathlib.Path, data: bytes) -> ObjectMeta:
        """Builds the metadata of a file whose bytes are known.

        Args:
            key: The object key.
            path: The file holding the object.
            data: The file's content; its length and hash are reported.

        Returns:
            The metadata, with the file's modification time and no version.
        """
        stat = path.stat()
        return ObjectMeta(
            key, len(data), sha256_of(data), iso(stat.st_mtime), None
        )

    def put(
        self, key: str, data: bytes, *, if_absent: bool = False
    ) -> ObjectMeta:
        """Stores `data` atomically.

        An unconditional put replaces the file through
        `scmle.durability.write_atomically`; a conditional one links a
        fully written temp file into place only if no object exists, and
        raises `AlreadyExistsError` if one does. An invalid key raises
        `StoreError`.

        Args:
            key: The object key.
            data: The content to store.
            if_absent: Create the object only if none exists.

        Returns:
            The metadata of the stored object.
        """
        path = self.path_of(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if if_absent:
            self.create_exclusive(path, data)
        else:
            durability.write_atomically(path, data)
        return self.meta_of(key, path, data)

    def create_exclusive(self, path: pathlib.Path, data: bytes) -> None:
        """Writes a temp file, then links it into place only if absent.

        Args:
            path: The destination file; its directory must exist.
            data: The content to write.

        Raises:
            AlreadyExistsError: If the destination already exists.
            OSError: If linking fails for a reason other than the
                destination existing.
        """
        descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".put-")
        temporary = pathlib.Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except OSError as error:
                # Windows reports a link onto a file that another writer is
                # creating at that moment as PermissionError or a sharing
                # violation, not FileExistsError; either way the object is
                # not ours.
                if not refused_as_existing(error, path):
                    raise
                raise AlreadyExistsError(
                    f"{path.name} already exists", {"key": path.name}
                ) from error
        finally:
            unlink_tolerantly(temporary)

    def get(self, key: str) -> tuple[bytes, ObjectMeta]:
        """Reads an object.

        An invalid key raises `StoreError`.

        Args:
            key: The object key.

        Returns:
            The file's content and its metadata. The hash is computed from
            the bytes just read, so a changed file is never reported as
            corrupt.

        Raises:
            NotFoundError: If there is no file under the key.
        """
        path = self.path_of(key)
        try:
            data = path.read_bytes()
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            raise NotFoundError(f"no object {key!r}") from None
        return data, self.meta_of(key, path, data)

    def head(self, key: str) -> ObjectMeta:
        """Reads an object's metadata; the file is read to hash it.

        An invalid key raises `StoreError`, and a key with no file raises
        `NotFoundError`.

        Args:
            key: The object key.

        Returns:
            The object's metadata.
        """
        return self.get(key)[1]

    def list_objects(self, prefix: str = "") -> list[ObjectMeta]:
        """Lists the objects under a prefix, sorted by key.

        Temporary files left by interrupted writers are not listed. An
        invalid prefix raises `StoreError`.

        Args:
            prefix: Key prefix; empty for every object.

        Returns:
            The metadata of each matching object, sorted by key.
        """
        check_prefix(prefix)
        found = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or is_temporary(path.name):
                continue
            key = path.relative_to(self.root).as_posix()
            if key.startswith(prefix):
                found.append(self.head(key))
        return sorted(found, key=lambda meta: meta.key)

    def delete(self, key: str) -> None:
        """Removes an object if present.

        An invalid key raises `StoreError`.

        Args:
            key: The object key.
        """
        self.path_of(key).unlink(missing_ok=True)


class RetryingStore(ObjectStore):
    """Wraps a store so transient failures are retried under a policy.

    Attributes:
        inner: The wrapped store.
        policy: Attempts and backoff.
        sleep: Waits between attempts.
    """

    def __init__(
        self,
        inner: ObjectStore,
        policy: retry.Retries,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initializes the wrapper.

        Args:
            inner: The store to protect.
            policy: Attempts and backoff.
            sleep: Waits between attempts.
        """
        super().__init__(inner.location)
        self.scheme = inner.scheme
        self.inner = inner
        self.policy = policy
        self.sleep = sleep

    def run(self, operation: Callable[[], T]) -> T:
        """Runs one operation under the policy.

        Only transient errors are retried; any other error propagates at
        once, and when every attempt fails with a transient error
        `RetryExhaustedError` propagates.

        Args:
            operation: Zero-argument callable that does the store call.

        Returns:
            Whatever the operation returned on the attempt that succeeded.
        """
        return retry.call_with_retries(operation, self.policy, sleep=self.sleep)

    def put(
        self, key: str, data: bytes, *, if_absent: bool = False
    ) -> ObjectMeta:
        """See `ObjectStore.put`.

        A conditional put whose first try landed but was reported as a
        timeout fails on the retry with `AlreadyExistsError`; that is
        reported as it is, not hidden.

        Args:
            key: The object key.
            data: The content to store.
            if_absent: Create the object only if none exists.

        Returns:
            The metadata of the stored object.
        """
        return self.run(lambda: self.inner.put(key, data, if_absent=if_absent))

    def get(self, key: str) -> tuple[bytes, ObjectMeta]:
        """Reads an object, retrying transient failures.

        Args:
            key: The object key.

        Returns:
            The content and metadata, as from `ObjectStore.get`.
        """
        return self.run(lambda: self.inner.get(key))

    def head(self, key: str) -> ObjectMeta:
        """Reads an object's metadata, retrying transient failures.

        Args:
            key: The object key.

        Returns:
            The metadata, as from `ObjectStore.head`.
        """
        return self.run(lambda: self.inner.head(key))

    def list_objects(self, prefix: str = "") -> list[ObjectMeta]:
        """Lists objects under a prefix, retrying transient failures.

        Args:
            prefix: Key prefix; empty for every object.

        Returns:
            The metadata, as from `ObjectStore.list_objects`.
        """
        return self.run(lambda: self.inner.list_objects(prefix))

    def delete(self, key: str) -> None:
        """Removes an object, retrying transient failures.

        Args:
            key: The object key.
        """
        self.run(lambda: self.inner.delete(key))
