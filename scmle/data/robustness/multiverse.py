"""The declared specification grid and a cached, resumable runner (F6).

The charter declares the defensible analysis choices before any result
exists (`AnalysisCharter.multiverse`); its hash covers them, so adding an
option after exploration changes the charter hash. This module enumerates the
declared grid, picks a seeded subset when the grid is larger than the size
the run can afford, and evaluates a function over the specifications.

Everything is deterministic under a fixed seed: the order of specifications
and the subset depend only on the declaration and the seed. Results are
cached by ``(finding, specification hash, seed)`` so an interrupted run can
resume, and a time budget skips the remaining specifications and records
them, never silently.

Stability is relative to the declared set; it is not a claim about all
defensible analyses.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import hashlib
import itertools
import json
from pathlib import Path
import time

from scmle.contracts import qualification
from scmle.data.charter import contracts as charter_contracts

DEFAULT_MAX_SPECS = 12


def digest(payload: object) -> str:
    """SHA-256 of canonical JSON."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class Specification:
    """One point of the grid: an option for every declared choice.

    Attributes:
        values: (choice, option) pairs sorted by choice name.
    """

    values: tuple[tuple[str, str], ...]

    @property
    def sha256(self) -> str:
        """Hash of the specification."""
        return digest([list(pair) for pair in self.values])

    def option(self, choice: str, default: str) -> str:
        """The option chosen for `choice`, or `default` when not varied."""
        return dict(self.values).get(choice, default)

    def to_dict(self) -> dict[str, str]:
        """Choice to option."""
        return dict(self.values)


def declaration_sha256(choices: Sequence[charter_contracts.Choice]) -> str:
    """Hash of the declared choices, independent of their order."""
    return digest(
        sorted(
            (c.to_dict() for c in choices), key=lambda record: record["name"]
        )
    )


def tier_after_declaration(
    declared_sha256: str, used_sha256: str, tier: qualification.Tier
) -> qualification.Tier:
    """Downgrades the tier when the grid used differs from the declared one.

    Adding a specification after exploration is a charter amendment; the
    result is then `unverified` whatever it was.
    """
    if declared_sha256 == used_sha256:
        return tier
    return qualification.Tier.UNVERIFIED


@dataclasses.dataclass(frozen=True, slots=True)
class Grid:
    """The specifications to run.

    Attributes:
        specs: The default specification first, then the others in a fixed
            seeded order.
        total: Size of the full declared grid.
        declaration_sha256: Hash of the declared choices.
        subset: Whether `specs` is a seeded subset of the full grid.
    """

    specs: tuple[Specification, ...]
    total: int
    declaration_sha256: str
    subset: bool

    @property
    def default(self) -> Specification:
        """The primary analysis: every choice at its default."""
        return self.specs[0]


def enumerate_grid(
    choices: Sequence[charter_contracts.Choice],
    seed: int,
    max_specs: int = DEFAULT_MAX_SPECS,
) -> Grid:
    """Enumerates the declared grid, or a seeded subset of fixed size.

    Args:
        choices: Declared choices (distinct names).
        seed: Seed that orders the subset.
        max_specs: Largest number of specifications to run, at least 1.

    Returns:
        The grid. The default specification is always included.

    Raises:
        ValueError: On repeated choice names or a non-positive size.
    """
    qualification.count(max_specs, "max_specs", minimum=1)
    ordered = sorted(choices, key=lambda c: c.name)
    if len({c.name for c in ordered}) != len(ordered):
        raise ValueError("choice names must be distinct")
    names = [c.name for c in ordered]
    everything = [
        Specification(tuple(zip(names, combo, strict=True)))
        for combo in itertools.product(*(c.options for c in ordered))
    ]
    default = Specification(tuple((c.name, c.default) for c in ordered))
    others = sorted(
        (s for s in everything if s != default),
        key=lambda s: digest([seed, s.sha256]),
    )
    chosen = (default, *others[: max_specs - 1])
    return Grid(
        chosen,
        len(everything),
        declaration_sha256(ordered),
        len(chosen) < len(everything),
    )


class ResultCache:
    """Results keyed by ``(finding, specification hash, seed)``.

    The cache is a plain mapping that can be saved to JSON and loaded again,
    which makes a long multiverse run resumable.
    """

    def __init__(
        self, items: Mapping[str, Mapping[str, float]] | None = None
    ) -> None:
        """Creates a cache, optionally from stored items."""
        self.items: dict[str, dict[str, float]] = {
            key: dict(value) for key, value in (items or {}).items()
        }

    @staticmethod
    def key(finding: str, spec: Specification, seed: int) -> str:
        """The cache key."""
        return f"{finding}|{spec.sha256}|{seed}"

    def get(
        self, finding: str, spec: Specification, seed: int
    ) -> dict[str, float] | None:
        """The stored result, or None."""
        return self.items.get(self.key(finding, spec, seed))

    def put(
        self,
        finding: str,
        spec: Specification,
        seed: int,
        result: Mapping[str, float],
    ) -> None:
        """Stores a result."""
        self.items[self.key(finding, spec, seed)] = dict(result)

    def save(self, path: Path) -> None:
        """Writes the cache as stable JSON."""
        path.write_text(
            json.dumps(self.items, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "ResultCache":
        """Reads a cache written by `save`; a missing file is empty."""
        if not path.is_file():
            return cls()
        return cls(json.loads(path.read_text(encoding="utf-8")))


@dataclasses.dataclass(frozen=True, slots=True)
class GridRun:
    """What a run over the grid produced.

    Attributes:
        results: Specification hash to the evaluated result.
        skipped: (specification hash, reason) for every specification that
            was not evaluated.
        cached: How many results came from the cache.
        seconds: Wall time spent evaluating.
    """

    results: dict[str, dict[str, float]]
    skipped: tuple[tuple[str, str], ...]
    cached: int
    seconds: float


def run_grid(
    grid: Grid,
    evaluate: Callable[[Specification], Mapping[str, float]],
    *,
    finding: str,
    seed: int,
    cache: ResultCache | None = None,
    budget_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> GridRun:
    """Evaluates every specification of the grid.

    Args:
        grid: The specifications.
        evaluate: Maps a specification to numbers. It must be deterministic
            given the specification and the seed.
        finding: Name of the finding, part of the cache key.
        seed: Run seed, part of the cache key.
        cache: Optional cache; hits are not re-evaluated.
        budget_seconds: Time allowed. Once spent, the remaining
            specifications are skipped and recorded.
        clock: Monotonic clock, replaceable in tests.

    Returns:
        The results and the specifications that were skipped.
    """
    start = clock()
    results: dict[str, dict[str, float]] = {}
    skipped: list[tuple[str, str]] = []
    hits = 0
    for spec in grid.specs:
        stored = None if cache is None else cache.get(finding, spec, seed)
        if stored is not None:
            results[spec.sha256] = stored
            hits += 1
            continue
        if budget_seconds is not None and clock() - start >= budget_seconds:
            skipped.append((spec.sha256, "time budget spent"))
            continue
        found = dict(evaluate(spec))
        results[spec.sha256] = found
        if cache is not None:
            cache.put(finding, spec, seed, found)
    return GridRun(results, tuple(skipped), hits, clock() - start)
