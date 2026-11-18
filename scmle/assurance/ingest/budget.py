"""Cost and time budgets for ingestion.

Each stage estimates its cost from catalog row counts and runs only if the
estimate fits what is left. When it does not, the stage stops at a declared
boundary (the end of the current table or lattice level), the boundary is
recorded, and everything that depends on the stage is marked partial and
`unverified`. Nothing is truncated silently and nothing is sampled to save
cost.
"""

from collections.abc import Callable
import dataclasses
import time
from typing import Any
from typing import ClassVar

from scmle.contracts import errors
from scmle.contracts import records


@dataclasses.dataclass(frozen=True, slots=True)
class Budget:
    """Limits for one ingestion run; `None` means unlimited.

    Attributes:
        max_queries: Source queries allowed.
        max_rows_scanned: Rows the source is estimated to scan. Bytes are not
            observable through DB-API, so the unit is rows.
        max_seconds: Wall time allowed (see `Meter` for repeatability).
        max_features: Cap on synthesized features.
        migrations: The written layout's version steps, oldest first;
            version 2 writes `queries`, `scanned`, `seconds` and
            `features`, the charter's ingest spelling.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "max_queries": "queries",
                "max_rows_scanned": "scanned",
                "max_seconds": "seconds",
                "max_features": "features",
            }
        ),
    )

    max_queries: int | None = None
    max_rows_scanned: int | None = None
    max_seconds: float | None = None
    max_features: int = 200

    def __post_init__(self) -> None:
        """Validates the limits.

        Raises:
            ValueError: If any limit is negative.
        """
        for name in ("max_queries", "max_rows_scanned"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise errors.InputError(f"{name} must be non-negative")
        if self.max_seconds is not None and self.max_seconds < 0:
            raise errors.InputError("max_seconds must be non-negative")
        if self.max_features < 0:
            raise errors.InputError("max_features must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the limits.

        Returns:
            A JSON-ready mapping of limit name to value.
        """
        return records.upgrade(
            self.migrations, dataclasses.asdict(self), records.ORIGIN
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Stop:
    """Where a stage stopped because the budget could not cover the next step.

    Attributes:
        stage: Stage name.
        unit: What was about to run (a table, a lattice level).
        reason: Which limit was short: `queries`, `rows` or `seconds`.
    """

    stage: str
    unit: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        """Serializes the boundary.

        Returns:
            A JSON-ready mapping of the stage, unit and reason.
        """
        return dataclasses.asdict(self)


class Meter:
    """Charges estimated cost against an `Budget`.

    Queries and rows are counted from the estimates the stages give before
    they run, so the counts are deterministic: two runs with the same inputs
    stop at the same boundary. Wall time is read from `clock`; a run stopped
    by `max_seconds` repeats only if the clock does (tests pass a counter).
    """

    def __init__(
        self,
        budget: Budget,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Starts the meter at zero spend.

        Args:
            budget: The limits to charge against.
            clock: Source of wall time in seconds; read once now and again
                whenever `seconds` is needed.
        """
        self.budget = budget
        self.clock = clock
        self.started = clock()
        self.queries = 0
        self.rows = 0
        self.stops: list[Stop] = []

    @property
    def seconds(self) -> float:
        """The wall time, in seconds, since the meter started."""
        return self.clock() - self.started

    @property
    def partial(self) -> bool:
        """Whether any stage has stopped at a boundary."""
        return bool(self.stops)

    def shortfall(self, rows: int, queries: int = 1) -> str | None:
        """Finds the limit that more spending would break or has reached.

        Args:
            rows: Rows the next step would scan.
            queries: Queries the next step would run.

        Returns:
            `queries` or `rows` for the first limit the step would exceed,
            or `seconds` when the elapsed time has already reached its
            limit, checked in that order; None when the step fits.
        """
        limit = self.budget
        if (
            limit.max_queries is not None
            and self.queries + queries > limit.max_queries
        ):
            return "queries"
        if (
            limit.max_rows_scanned is not None
            and self.rows + rows > limit.max_rows_scanned
        ):
            return "rows"
        if limit.max_seconds is not None and self.seconds >= limit.max_seconds:
            return "seconds"
        return None

    def admit(self, stage: str, unit: str, rows: int, queries: int = 1) -> bool:
        """Charges the step if it fits; otherwise records the boundary.

        Once a stage has stopped it stays stopped, so a later, cheaper step of
        the same stage does not run past the declared boundary.

        Args:
            stage: Name of the stage taking the step.
            unit: What the step covers (a table, a lattice level).
            rows: Rows the step is estimated to scan.
            queries: Queries the step will run.

        Returns:
            True if the step was charged and may run; False if the stage had
            already stopped or the step would break a limit (the boundary is
            then recorded).
        """
        if any(stop.stage == stage for stop in self.stops):
            return False
        reason = self.shortfall(rows, queries)
        if reason is not None:
            self.stops.append(Stop(stage, unit, reason))
            return False
        self.queries += queries
        self.rows += rows
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serializes spend against limits, for `ingest_budget.json`.

        Returns:
            A JSON-ready mapping with the allowed limits, the spent queries,
            rows and seconds, whether the run is partial, and the stop
            boundaries.
        """
        return {
            "allowed": self.budget.to_dict(),
            "spent": {
                "queries": self.queries,
                "rows_scanned": self.rows,
                "seconds": self.seconds,
            },
            "partial": self.partial,
            "stopped": [stop.to_dict() for stop in self.stops],
        }
