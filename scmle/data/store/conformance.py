"""One conformance suite for every object-store backend.

`run_conformance(store)` exercises the behaviour the `ObjectStore` interface
promises and returns a report; the same function runs against the local
backend, against the S3 and GCS backends over in-memory fakes in the unit
tests, and (when a deployment points it at them) against MinIO, the GCS
emulator or a real bucket, through `scripts/run_object_store_conformance.py`.
Each check works under a unique prefix and removes what it wrote. A check
passes by returning; it fails by raising.
"""

from collections.abc import Callable
import dataclasses
import functools
import threading
import time
from typing import Any
import uuid

from scmle.contracts import errors
from scmle.data.store import objectstore

BINARY = bytes(range(256)) * 4
LARGE_BYTES = 1024 * 1024
PAGE_OBJECTS = 25
RACERS = 8


class ConformanceFailure(errors.ScmleError):
    """A backend broke a promise of the interface."""

    category = "conformance"


@dataclasses.dataclass(frozen=True, slots=True)
class CheckResult:
    """The outcome of one check.

    Attributes:
        name: Check name.
        passed: Whether it passed.
        detail: The failure, or empty.
        seconds: Wall time.
    """

    name: str
    passed: bool
    detail: str
    seconds: float


@dataclasses.dataclass(frozen=True, slots=True)
class ConformanceReport:
    """All checks against one backend.

    Attributes:
        backend: Store scheme and location.
        results: One entry per check, in order.
    """

    backend: str
    results: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        """Whether every check passed."""
        return all(result.passed for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "backend": self.backend,
            "passed": self.passed,
            "checks": [dataclasses.asdict(r) for r in self.results],
        }


def require(condition: bool, message: str) -> None:
    """Fails the check unless `condition` holds."""
    if not condition:
        raise ConformanceFailure(message)


def raises(
    expected: type[Exception], operation: Callable[[], object], message: str
) -> None:
    """Fails the check unless `operation` raises `expected`."""
    try:
        operation()
    except expected:
        return
    except Exception as error:
        raise ConformanceFailure(
            f"{message}: raised {type(error).__name__}"
        ) from error
    raise ConformanceFailure(f"{message}: nothing was raised")


def check_round_trip(store: objectstore.ObjectStore, base: str) -> None:
    """Bytes come back unchanged with a correct hash and size."""
    data = b"hello object store"
    written = store.put(f"{base}/a.txt", data)
    content, meta = store.get(f"{base}/a.txt")
    require(content == data, "content differs")
    require(meta.sha256 == objectstore.sha256_of(data), "hash is wrong")
    require(meta.size == len(data) == written.size, "size is wrong")
    require(written.sha256 == meta.sha256, "put and get disagree on the hash")
    require(meta.key == f"{base}/a.txt", "key is not reported as given")
    require(bool(meta.modified), "modification time is missing")


def check_binary_and_empty(store: objectstore.ObjectStore, base: str) -> None:
    """Every byte value and the empty object survive."""
    store.put(f"{base}/bin", BINARY)
    store.put(f"{base}/empty", b"")
    require(store.get(f"{base}/bin")[0] == BINARY, "binary data changed")
    require(store.get(f"{base}/empty")[0] == b"", "empty object changed")
    require(store.head(f"{base}/empty").size == 0, "empty size is wrong")


def check_large(store: objectstore.ObjectStore, base: str) -> None:
    """A one-mebibyte object round-trips."""
    data = (BINARY * (LARGE_BYTES // len(BINARY)))[:LARGE_BYTES]
    store.put(f"{base}/large", data)
    require(store.get(f"{base}/large")[0] == data, "large object changed")


def check_overwrite(store: objectstore.ObjectStore, base: str) -> None:
    """An unconditional put replaces the object."""
    store.put(f"{base}/o", b"one")
    store.put(f"{base}/o", b"two!")
    content, meta = store.get(f"{base}/o")
    require(content == b"two!" and meta.size == 4, "overwrite not visible")


def check_missing(store: objectstore.ObjectStore, base: str) -> None:
    """Reading a missing object is `NotFoundError`; `exists` says no."""
    raises(
        objectstore.NotFoundError,
        lambda: store.get(f"{base}/nope"),
        "get of a missing key",
    )
    raises(
        objectstore.NotFoundError,
        lambda: store.head(f"{base}/nope"),
        "head of a missing key",
    )
    require(not store.exists(f"{base}/nope"), "a missing key exists")
    store.put(f"{base}/there", b"x")
    require(store.exists(f"{base}/there"), "a written key does not exist")


def check_head_matches_get(store: objectstore.ObjectStore, base: str) -> None:
    """`head` reports what `get` reports."""
    store.put(f"{base}/h", b"head me")
    require(
        store.head(f"{base}/h").sha256 == store.get(f"{base}/h")[1].sha256,
        "head and get disagree",
    )


def check_if_absent(store: objectstore.ObjectStore, base: str) -> None:
    """A conditional put creates once and then refuses without changes."""
    key = f"{base}/once"
    store.put(key, b"first", if_absent=True)
    raises(
        objectstore.AlreadyExistsError,
        lambda: store.put(key, b"second", if_absent=True),
        "second conditional put",
    )
    require(store.get(key)[0] == b"first", "a refused put changed the object")


def check_if_absent_race(store: objectstore.ObjectStore, base: str) -> None:
    """Of several simultaneous conditional puts exactly one wins."""
    key = f"{base}/race"
    outcomes: list[str] = []
    guard = threading.Lock()
    start = threading.Barrier(RACERS)

    def racer(index: int) -> None:
        start.wait()
        try:
            store.put(key, f"racer-{index}".encode(), if_absent=True)
            result = "won"
        except objectstore.AlreadyExistsError:
            result = "lost"
        except Exception as error:
            result = f"failed {type(error).__name__}: {error}"
        with guard:
            outcomes.append(result)

    threads = [threading.Thread(target=racer, args=(n,)) for n in range(RACERS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    broken = [item for item in outcomes if item.startswith("failed")]
    require(not broken, f"racers failed: {broken}")
    require(len(outcomes) == RACERS, f"only {len(outcomes)} racers finished")
    broken = [item for item in outcomes if item.startswith("failed")]
    require(not broken, f"racers failed: {broken}")
    require(len(outcomes) == RACERS, f"only {len(outcomes)} racers finished")
    require(outcomes.count("won") == 1, f"winners: {outcomes.count('won')}")
    require(outcomes.count("lost") == RACERS - 1, "losers were not refused")


def check_listing(store: objectstore.ObjectStore, base: str) -> None:
    """Listing is by prefix, sorted by key, and sees nested keys."""
    for name in ("b/2", "a/1", "a/3", "ab/1", "c"):
        store.put(f"{base}/{name}", name.encode())
    keys = [m.key for m in store.list_objects(f"{base}/")]
    expected = [f"{base}/{n}" for n in ("a/1", "a/3", "ab/1", "b/2", "c")]
    require(keys == expected, f"listing {keys}")
    nested = [m.key for m in store.list_objects(f"{base}/a/")]
    require(
        nested == [f"{base}/a/1", f"{base}/a/3"], f"prefix a/ gave {nested}"
    )
    require(store.list_objects(f"{base}/zzz") == [], "empty prefix not empty")


def check_pagination(store: objectstore.ObjectStore, base: str) -> None:
    """More objects than one page are all listed once."""
    names = [f"{base}/page/{n:03d}" for n in range(PAGE_OBJECTS)]
    for name in names:
        store.put(name, name.encode())
    require(
        [m.key for m in store.list_objects(f"{base}/page/")] == names,
        "paginated listing lost or repeated keys",
    )


def check_delete(store: objectstore.ObjectStore, base: str) -> None:
    """Delete removes the object and is idempotent."""
    store.put(f"{base}/d", b"x")
    store.delete(f"{base}/d")
    require(not store.exists(f"{base}/d"), "deleted object still exists")
    store.delete(f"{base}/d")
    store.delete(f"{base}/never-existed")


def check_key_rules(store: objectstore.ObjectStore, base: str) -> None:
    """Unsafe keys are refused before reaching the backend."""
    for bad in ("", "/lead", "trail/", "a//b", "a/../b", "..", "a/./b"):
        raises(
            objectstore.StoreError,
            functools.partial(store.put, bad, b"x"),
            f"put of key {bad!r}",
        )
        raises(
            objectstore.StoreError,
            functools.partial(store.get, bad),
            f"get of key {bad!r}",
        )


def check_isolation(store: objectstore.ObjectStore, base: str) -> None:
    """Objects written under one prefix are not listed under a sibling."""
    store.put(f"{base}/left/x", b"l")
    store.put(f"{base}/right/x", b"r")
    require(
        [m.key for m in store.list_objects(f"{base}/left/")]
        == [f"{base}/left/x"],
        "a sibling prefix leaked into the listing",
    )


CHECKS: tuple[Callable[[objectstore.ObjectStore, str], None], ...] = (
    check_round_trip,
    check_binary_and_empty,
    check_large,
    check_overwrite,
    check_missing,
    check_head_matches_get,
    check_if_absent,
    check_if_absent_race,
    check_listing,
    check_pagination,
    check_delete,
    check_key_rules,
    check_isolation,
)


def run_conformance(
    store: objectstore.ObjectStore, run_id: str | None = None
) -> ConformanceReport:
    """Runs every check against `store`.

    Args:
        store: The backend under test; it must be writable.
        run_id: Unique prefix for this run (random by default).

    Returns:
        The report; failed checks are in it, not raised.
    """
    run = run_id or uuid.uuid4().hex[:12]
    results = []
    for check in CHECKS:
        base = f"conformance-{run}/{check.__name__.removeprefix('check_')}"
        started = time.monotonic()
        try:
            check(store, base)
            outcome, detail = True, ""
        except Exception as error:
            outcome, detail = False, f"{type(error).__name__}: {error}"
        finally:
            cleanup(store, base)
        results.append(
            CheckResult(
                check.__name__.removeprefix("check_"),
                outcome,
                detail,
                round(time.monotonic() - started, 4),
            )
        )
    return ConformanceReport(f"{store.scheme}:{store.location}", tuple(results))


def cleanup(store: objectstore.ObjectStore, base: str) -> None:
    """Removes what a check wrote, ignoring cleanup failures."""
    try:
        for meta in store.list_objects(base + "/"):
            store.delete(meta.key)
    except objectstore.StoreError:
        return
