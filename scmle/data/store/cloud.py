"""S3 and GCS backends over injected clients.

Neither SDK is imported unless a factory is called. The backends use a small
protocol that boto3's S3 client and `google.cloud.storage.Client` satisfy, so
tests run them against an in-memory fake and a deployment passes the real
client. Credentials, endpoints (MinIO, the GCS emulator) and transport
retries belong to the client the caller builds; `RetryingStore` adds retries
for the transient errors the backends translate.

The content hash is kept as user metadata (`sha256`) at write time and
verified on read, so a changed object is detected even when the service's own
checksum is trusted.
"""

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
import datetime
import importlib
from types import ModuleType
from typing import Any
from typing import Protocol

from scmle.contracts import errors
from scmle.data.store import objectstore

HASH_FIELD = "sha256"
NOT_FOUND_CODES = frozenset({"NoSuchKey", "404", "NotFound", "NoSuchBucket"})
CONFLICT_CODES = frozenset(
    {"PreconditionFailed", "412", "ConditionalRequestConflict"}
)
TRANSIENT_CODES = frozenset(
    {
        "SlowDown",
        "RequestTimeout",
        "InternalError",
        "ServiceUnavailable",
        "500",
        "502",
        "503",
        "504",
        "429",
        "Throttling",
        "ThrottlingException",
    }
)
TRANSIENT_CLASSES = frozenset(
    {
        "ServiceUnavailable",
        "TooManyRequests",
        "InternalServerError",
        "GatewayTimeout",
        "DeadlineExceeded",
        "ConnectionError",
        "TimeoutError",
    }
)


class S3Client(Protocol):
    """The part of a boto3 S3 client the backend calls."""

    def put_object(self, **kwargs: Any) -> Any:
        """Stores an object (`Bucket`, `Key`, `Body`, `Metadata`...)."""

    def get_object(self, **kwargs: Any) -> Any:
        """Returns `Body`, `Metadata`, `LastModified`, `VersionId`."""

    def head_object(self, **kwargs: Any) -> Any:
        """Returns `ContentLength`, `Metadata`, `LastModified`."""

    def list_objects_v2(self, **kwargs: Any) -> Any:
        """Returns `Contents`, `IsTruncated`, `NextContinuationToken`."""

    def delete_object(self, **kwargs: Any) -> Any:
        """Removes an object."""


class GcsBlob(Protocol):
    """The part of a `google.cloud.storage.Blob` the backend calls."""

    name: str
    size: int | None
    updated: datetime.datetime | None
    generation: int | None
    metadata: dict[str, str] | None

    def upload_from_string(self, data: bytes, **kwargs: Any) -> None:
        """Uploads the bytes (`if_generation_match=0` creates only)."""

    def download_as_bytes(self, **kwargs: Any) -> bytes:
        """Returns the content."""

    def delete(self, **kwargs: Any) -> None:
        """Removes the blob."""


class GcsBucket(Protocol):
    """The part of a `google.cloud.storage.Bucket` the backend calls."""

    def blob(self, name: str) -> GcsBlob:
        """A handle for a possibly missing blob."""

    def get_blob(self, name: str) -> GcsBlob | None:
        """The blob with fresh metadata, or None."""

    def list_blobs(self, **kwargs: Any) -> Iterable[GcsBlob]:
        """Blobs under `prefix`."""


class GcsClient(Protocol):
    """The part of a `google.cloud.storage.Client` the backend calls."""

    def bucket(self, name: str) -> GcsBucket:
        """A handle for a bucket."""


def error_code(error: Exception) -> str:
    """The service error code of a botocore `ClientError`, duck-typed."""
    response = getattr(error, "response", None)
    if isinstance(response, Mapping):
        found = response.get("Error", {})
        if isinstance(found, Mapping):
            return str(found.get("Code", ""))
    return ""


def translate(error: Exception, key: str) -> objectstore.StoreError:
    """Maps an SDK exception to the store's error taxonomy."""
    code = error_code(error)
    name = type(error).__name__
    status = str(getattr(error, "code", ""))
    if code in NOT_FOUND_CODES or name == "NotFound" or status == "404":
        return objectstore.NotFoundError(f"no object {key!r}")
    if (
        code in CONFLICT_CODES
        or name == "PreconditionFailed"
        or status == "412"
    ):
        return objectstore.AlreadyExistsError(f"{key!r} already exists")
    if (
        code in TRANSIENT_CODES
        or name in TRANSIENT_CLASSES
        or status in TRANSIENT_CODES
        or isinstance(error, TimeoutError | ConnectionError)
    ):
        return objectstore.TransientStoreError(
            f"transient failure on {key!r}: {name}"
        )
    return objectstore.StoreError(f"{name} on {key!r}")


def scoped_prefix(root: str, prefix: str) -> str:
    """The bucket-level prefix for a store-level prefix."""
    return (root + "/" if root else "") + prefix


def iso_time(value: Any) -> str:
    """ISO time from a datetime, or the epoch when the service gave none."""
    if isinstance(value, datetime.datetime):
        return objectstore.iso(value)
    return objectstore.iso(0.0)


class S3ObjectStore(objectstore.ObjectStore):
    """An S3 (or MinIO) bucket and key prefix."""

    scheme = "s3"

    def __init__(self, client: S3Client, bucket: str, prefix: str = "") -> None:
        """Initializes the store.

        Args:
            client: A boto3 S3 client, or anything with the same methods.
            bucket: Bucket name.
            prefix: Key prefix that scopes this store (no trailing slash).
        """
        self.client = client
        self.bucket = bucket
        self.prefix = objectstore.check_prefix(prefix).strip("/")
        base = f"s3://{bucket}" + (f"/{self.prefix}" if self.prefix else "")
        super().__init__(base)

    def full(self, key: str) -> str:
        """The bucket key for a store key."""
        key = objectstore.check_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def relative(self, full_key: str) -> str:
        """The store key for a bucket key."""
        if not self.prefix:
            return full_key
        return full_key.removeprefix(self.prefix + "/")

    def put(
        self, key: str, data: bytes, *, if_absent: bool = False
    ) -> objectstore.ObjectMeta:
        """Stores an object; `if_absent` sends `IfNoneMatch: *`."""
        arguments: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": self.full(key),
            "Body": data,
            "Metadata": {HASH_FIELD: objectstore.sha256_of(data)},
        }
        if if_absent:
            arguments["IfNoneMatch"] = "*"
        try:
            self.client.put_object(**arguments)
        except Exception as error:
            raise translate(error, key) from error
        return self.head(key)

    def get(self, key: str) -> tuple[bytes, objectstore.ObjectMeta]:
        """Reads an object and verifies its recorded hash."""
        try:
            reply = self.client.get_object(
                Bucket=self.bucket, Key=self.full(key)
            )
            data = reply["Body"].read()
        except Exception as error:
            raise translate(error, key) from error
        digest = objectstore.sha256_of(data)
        recorded = (reply.get("Metadata") or {}).get(HASH_FIELD)
        if recorded is not None and recorded != digest:
            raise objectstore.CorruptObjectError(
                f"object {key!r} does not match its recorded hash"
            )
        return data, objectstore.ObjectMeta(
            key,
            len(data),
            digest,
            iso_time(reply.get("LastModified")),
            reply.get("VersionId"),
        )

    def head(self, key: str) -> objectstore.ObjectMeta:
        """Metadata from `HeadObject`; the hash is user metadata."""
        try:
            reply = self.client.head_object(
                Bucket=self.bucket, Key=self.full(key)
            )
        except Exception as error:
            raise translate(error, key) from error
        recorded = (reply.get("Metadata") or {}).get(HASH_FIELD)
        if recorded is None:
            return self.get(key)[1]
        return objectstore.ObjectMeta(
            key,
            int(reply["ContentLength"]),
            recorded,
            iso_time(reply.get("LastModified")),
            reply.get("VersionId"),
        )

    def list_objects(self, prefix: str = "") -> list[objectstore.ObjectMeta]:
        """Objects under `prefix`, following continuation tokens."""
        objectstore.check_prefix(prefix)
        full_prefix = (
            self.full(prefix.rstrip("/"))
            + ("/" if prefix.endswith("/") else "")
            if prefix
            else (self.prefix + "/" if self.prefix else "")
        )
        keys: list[str] = []
        token: str | None = None
        while True:
            arguments: dict[str, Any] = {
                "Bucket": self.bucket,
                "Prefix": full_prefix,
            }
            if token:
                arguments["ContinuationToken"] = token
            try:
                reply = self.client.list_objects_v2(**arguments)
            except Exception as error:
                raise translate(error, prefix) from error
            keys += [item["Key"] for item in reply.get("Contents", [])]
            if not reply.get("IsTruncated"):
                break
            token = reply.get("NextContinuationToken")
        return sorted(
            (self.head(self.relative(key)) for key in keys),
            key=lambda meta: meta.key,
        )

    def delete(self, key: str) -> None:
        """Removes an object; a missing object is not an error."""
        try:
            self.client.delete_object(Bucket=self.bucket, Key=self.full(key))
        except Exception as error:
            problem = translate(error, key)
            if not isinstance(problem, objectstore.NotFoundError):
                raise problem from error


class GcsObjectStore(objectstore.ObjectStore):
    """A GCS bucket and key prefix."""

    scheme = "gs"

    def __init__(
        self, client: GcsClient, bucket: str, prefix: str = ""
    ) -> None:
        """Initializes the store.

        Args:
            client: A `google.cloud.storage.Client`, or anything with the
                same methods.
            bucket: Bucket name.
            prefix: Key prefix that scopes this store (no trailing slash).
        """
        self.bucket_handle = client.bucket(bucket)
        self.bucket = bucket
        self.prefix = objectstore.check_prefix(prefix).strip("/")
        base = f"gs://{bucket}" + (f"/{self.prefix}" if self.prefix else "")
        super().__init__(base)

    def full(self, key: str) -> str:
        """The bucket key for a store key."""
        key = objectstore.check_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def relative(self, full_key: str) -> str:
        """The store key for a bucket key."""
        if not self.prefix:
            return full_key
        return full_key.removeprefix(self.prefix + "/")

    def meta_of(self, key: str, blob: GcsBlob) -> objectstore.ObjectMeta:
        """Metadata from a blob with fresh properties."""
        recorded = (blob.metadata or {}).get(HASH_FIELD)
        if recorded is None:
            return self.get(key)[1]
        return objectstore.ObjectMeta(
            key,
            int(blob.size or 0),
            recorded,
            iso_time(blob.updated),
            None if blob.generation is None else str(blob.generation),
        )

    def put(
        self, key: str, data: bytes, *, if_absent: bool = False
    ) -> objectstore.ObjectMeta:
        """Stores an object; `if_absent` sends `if_generation_match=0`."""
        blob = self.bucket_handle.blob(self.full(key))
        blob.metadata = {HASH_FIELD: objectstore.sha256_of(data)}
        arguments: dict[str, Any] = {"content_type": "application/octet-stream"}
        if if_absent:
            arguments["if_generation_match"] = 0
        try:
            blob.upload_from_string(data, **arguments)
        except Exception as error:
            raise translate(error, key) from error
        return self.head(key)

    def get(self, key: str) -> tuple[bytes, objectstore.ObjectMeta]:
        """Reads an object and verifies its recorded hash."""
        try:
            blob = self.bucket_handle.get_blob(self.full(key))
            if blob is None:
                raise objectstore.NotFoundError(f"no object {key!r}")
            data = blob.download_as_bytes()
        except objectstore.StoreError:
            raise
        except Exception as error:
            raise translate(error, key) from error
        digest = objectstore.sha256_of(data)
        recorded = (blob.metadata or {}).get(HASH_FIELD)
        if recorded is not None and recorded != digest:
            raise objectstore.CorruptObjectError(
                f"object {key!r} does not match its recorded hash"
            )
        return data, objectstore.ObjectMeta(
            key,
            len(data),
            digest,
            iso_time(blob.updated),
            None if blob.generation is None else str(blob.generation),
        )

    def head(self, key: str) -> objectstore.ObjectMeta:
        """Metadata from the blob's properties."""
        try:
            blob = self.bucket_handle.get_blob(self.full(key))
        except Exception as error:
            raise translate(error, key) from error
        if blob is None:
            raise objectstore.NotFoundError(f"no object {key!r}")
        return self.meta_of(key, blob)

    def list_objects(self, prefix: str = "") -> list[objectstore.ObjectMeta]:
        """Objects under `prefix`."""
        objectstore.check_prefix(prefix)
        full_prefix = scoped_prefix(self.prefix, prefix)
        try:
            blobs = list(self.bucket_handle.list_blobs(prefix=full_prefix))
        except Exception as error:
            raise translate(error, prefix) from error
        return sorted(
            (self.meta_of(self.relative(blob.name), blob) for blob in blobs),
            key=lambda meta: meta.key,
        )

    def delete(self, key: str) -> None:
        """Removes an object; a missing object is not an error."""
        blob = self.bucket_handle.blob(self.full(key))
        try:
            blob.delete()
        except Exception as error:
            problem = translate(error, key)
            if not isinstance(problem, objectstore.NotFoundError):
                raise problem from error


def import_optional(
    name: str,
    extra: str,
    importer: Callable[[str], ModuleType] | None = None,
) -> ModuleType:
    """Imports an optional SDK or explains how to install it.

    Raises:
        ConfigError: If the module is not installed.
    """
    try:
        return (importer or importlib.import_module)(name)
    except ImportError as error:
        raise errors.ConfigError(
            f"{name} is not installed; install {extra} to use this backend, "
            "or pass a client you built"
        ) from error


def s3_store(
    bucket: str,
    prefix: str = "",
    *,
    importer: Callable[[str], ModuleType] | None = None,
    **client_arguments: Any,
) -> S3ObjectStore:
    """An S3 store over a boto3 client built here.

    Args:
        bucket: Bucket name.
        prefix: Key prefix.
        importer: How modules are imported (injectable for tests); the
            default is `importlib.import_module`.
        **client_arguments: Passed to `boto3.client("s3", ...)`, for example
            `endpoint_url` for MinIO.

    Raises:
        ConfigError: If boto3 is not installed.
    """
    boto3 = import_optional("boto3", "boto3", importer)
    return S3ObjectStore(boto3.client("s3", **client_arguments), bucket, prefix)


def gcs_store(
    bucket: str,
    prefix: str = "",
    *,
    importer: Callable[[str], ModuleType] | None = None,
    **client_arguments: Any,
) -> GcsObjectStore:
    """A GCS store over a `google.cloud.storage.Client` built here.

    Raises:
        ConfigError: If google-cloud-storage is not installed.
    """
    storage = import_optional(
        "google.cloud.storage", "google-cloud-storage", importer
    )
    return GcsObjectStore(storage.Client(**client_arguments), bucket, prefix)
