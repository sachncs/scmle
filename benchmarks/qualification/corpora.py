"""Labelled corpora: a manifest and a hash-verifying fetcher.

Nothing large is vendored. `corpora.json` describes each corpus (source,
licence, SHA-256, ground-truth file, who adjudicated it, intended use) and
its status: `candidate` until someone has checked that it exists, may be used
under its licence and matches its hash, then `verified`. Only a verified
corpus can be fetched, a download is hashed while it streams, and a mismatch
aborts and leaves nothing behind.
"""

from collections.abc import Mapping
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
import urllib.parse
import urllib.request

MANIFEST = Path(__file__).with_name("corpora.json")
SCHEMA_VERSION = 1
USES = ("certificate", "backtest", "both")
STATUSES = ("candidate", "verified")
SCHEMES = ("https", "file")
ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
CHUNK_BYTES = 1 << 20
DEFAULT_MAX_BYTES = 1 << 30
TIMEOUT_SECONDS = 60
OFFLINE_VALUES = ("1", "true", "yes")
FIELDS = (
    "id",
    "name",
    "source_url",
    "licence",
    "sha256",
    "size_bytes",
    "ground_truth_file",
    "adjudication_source",
    "intended_use",
    "status",
    "notes",
)
REQUIRED_WHEN_VERIFIED = (
    "source_url",
    "licence",
    "sha256",
    "ground_truth_file",
    "adjudication_source",
)


class CorpusError(ValueError):
    """The manifest or a corpus file is malformed or fails verification."""


class CorpusHashMismatchError(CorpusError):
    """A downloaded or stored file does not match its recorded SHA-256."""


class CorpusUnavailableError(RuntimeError):
    """A corpus cannot be used here (unverified, or offline and absent)."""


@dataclasses.dataclass(frozen=True)
class Corpus:
    """One corpus entry.

    Attributes:
        id: Lowercase slug.
        name: Human description.
        source_url: `https` or `file` location, or None while a candidate.
        licence: Licence the data may be used under.
        sha256: Hash of the file the URL serves.
        size_bytes: Declared size, used as a download cap.
        ground_truth_file: Where the defect labels are, relative to the file.
        adjudication_source: Who established the ground truth.
        intended_use: `certificate`, `backtest` or `both`.
        status: `candidate` or `verified`.
        notes: Free text; for a candidate it says what is still unchecked.
    """

    id: str
    name: str
    source_url: str | None
    licence: str | None
    sha256: str | None
    size_bytes: int | None
    ground_truth_file: str | None
    adjudication_source: str | None
    intended_use: str
    status: str
    notes: str


def text_or_none(record: Mapping[str, Any], key: str) -> str | None:
    """A non-empty string, or None."""
    value = record[key]
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CorpusError(f"{key} must be a non-empty string or null")
    return value


def parse_entry(record: Mapping[str, Any]) -> Corpus:
    """Validates one manifest entry.

    Raises:
        CorpusError: On unknown or missing fields, a malformed value, an
            insecure URL scheme or a verified entry lacking provenance.
    """
    extra = sorted(set(record) - set(FIELDS))
    if extra:
        raise CorpusError(f"unexpected field(s): {extra}")
    missing = sorted(set(FIELDS) - set(record))
    if missing:
        raise CorpusError(f"missing field(s): {missing}")
    slug = record["id"]
    if not isinstance(slug, str) or not ID_PATTERN.fullmatch(slug):
        raise CorpusError("id must be a lowercase slug")
    name = text_or_none(record, "name")
    notes = text_or_none(record, "notes")
    if name is None or notes is None:
        raise CorpusError("name and notes are required")
    url = text_or_none(record, "source_url")
    if url is not None and urllib.parse.urlparse(url).scheme not in SCHEMES:
        raise CorpusError(f"source_url must use one of {SCHEMES}")
    digest = record["sha256"]
    if digest is not None and not (
        isinstance(digest, str) and SHA256_HEX.fullmatch(digest)
    ):
        raise CorpusError("sha256 must be 64 lowercase hex characters")
    size = record["size_bytes"]
    if size is not None and (
        isinstance(size, bool) or not isinstance(size, int) or size < 1
    ):
        raise CorpusError("size_bytes must be a positive integer or null")
    if record["intended_use"] not in USES:
        raise CorpusError(f"intended_use must be one of {USES}")
    if record["status"] not in STATUSES:
        raise CorpusError(f"status must be one of {STATUSES}")
    entry = Corpus(
        id=slug,
        name=name,
        source_url=url,
        licence=text_or_none(record, "licence"),
        sha256=digest,
        size_bytes=size,
        ground_truth_file=text_or_none(record, "ground_truth_file"),
        adjudication_source=text_or_none(record, "adjudication_source"),
        intended_use=record["intended_use"],
        status=record["status"],
        notes=notes,
    )
    if entry.status == "verified":
        absent = [
            k for k in REQUIRED_WHEN_VERIFIED if getattr(entry, k) is None
        ]
        if absent:
            raise CorpusError(f"verified corpus '{slug}' lacks {absent}")
    return entry


def parse_manifest(document: Mapping[str, Any]) -> tuple[Corpus, ...]:
    """Validates a whole manifest document."""
    if document.get("schema_version") != SCHEMA_VERSION:
        raise CorpusError(f"schema_version must be {SCHEMA_VERSION}")
    extra = sorted(set(document) - {"schema_version", "corpora"})
    if extra:
        raise CorpusError(f"unexpected field(s): {extra}")
    entries = tuple(parse_entry(item) for item in document["corpora"])
    ids = [e.id for e in entries]
    if len(set(ids)) != len(ids):
        raise CorpusError("duplicate corpus id")
    return entries


def load_manifest(path: Path = MANIFEST) -> tuple[Corpus, ...]:
    """Reads and validates a manifest file."""
    return parse_manifest(json.loads(path.read_text(encoding="utf-8")))


def find(manifest: tuple[Corpus, ...], corpus_id: str) -> Corpus:
    """The entry with this id.

    Raises:
        CorpusError: If there is none.
    """
    for entry in manifest:
        if entry.id == corpus_id:
            return entry
    raise CorpusError(f"no corpus with id '{corpus_id}'")


def is_offline() -> bool:
    """Whether `SCMLE_OFFLINE` asks for no network access."""
    return os.environ.get("SCMLE_OFFLINE", "").strip().lower() in OFFLINE_VALUES


def corpus_root() -> Path:
    """Folder that holds fetched corpora (`SCMLE_CORPORA_DIR` overrides)."""
    configured = os.environ.get("SCMLE_CORPORA_DIR")
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "scmle" / "corpora"


def destination(entry: Corpus, root: Path) -> Path:
    """Where a corpus file is stored under `root`.

    Raises:
        CorpusUnavailableError: If the entry has no URL to name the file after.
    """
    if entry.source_url is None:
        raise CorpusUnavailableError(f"corpus '{entry.id}' has no source_url")
    name = Path(urllib.parse.urlparse(entry.source_url).path).name
    return root / entry.id / (name or "corpus.bin")


def sha256_of(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def is_available(entry: Corpus, root: Path) -> bool:
    """Whether a verified corpus is present locally and matches its hash."""
    if entry.status != "verified" or entry.sha256 is None:
        return False
    target = destination(entry, root)
    return target.is_file() and sha256_of(target) == entry.sha256


def fetch(
    entry: Corpus,
    root: Path,
    offline: bool | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Path:
    """Returns the local file for a verified corpus, downloading if needed.

    The download streams into a temporary file, is hashed as it arrives,
    and replaces the destination only if the hash matches. On any failure
    the temporary file is removed.

    Args:
        entry: A corpus to fetch.
        root: Folder that holds corpora.
        offline: Forbid network access; defaults to `is_offline()`.
        max_bytes: Upper cap on the download when the entry declares no size.

    Returns:
        Path of the verified file.

    Raises:
        CorpusUnavailableError: If the entry is a candidate, or it is
            offline and the file is absent.
        CorpusHashMismatchError: If the bytes differ from the recorded hash or
            exceed the declared size.
    """
    if entry.status != "verified" or entry.sha256 is None:
        raise CorpusUnavailableError(f"corpus '{entry.id}' is not verified")
    if is_available(entry, root):
        return destination(entry, root)
    if is_offline() if offline is None else offline:
        raise CorpusUnavailableError(
            f"offline: corpus '{entry.id}' is not present locally"
        )
    assert entry.source_url is not None  # nosec B101 - checked by the schema
    target = destination(entry, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    limit = entry.size_bytes or max_bytes
    digest = hashlib.sha256()
    total = 0
    try:
        with (
            urllib.request.urlopen(  # nosec B310 - https/file only, see schema
                entry.source_url, timeout=TIMEOUT_SECONDS
            ) as response,
            temporary.open("wb") as handle,
        ):
            while chunk := response.read(CHUNK_BYTES):
                total += len(chunk)
                if total > limit:
                    raise CorpusHashMismatchError(
                        f"download is larger than the declared {limit} bytes"
                    )
                digest.update(chunk)
                handle.write(chunk)
        if digest.hexdigest() != entry.sha256:
            raise CorpusHashMismatchError(
                f"corpus '{entry.id}' does not match its recorded SHA-256"
            )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
