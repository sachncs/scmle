"""The ingest block of the charter (feature F9, format 7).

Everything here is optional: a charter without an `IngestSpec` serialises,
hashes and loads exactly as a format 1 to 6 charter did. The block fixes, before
any data is read, how much the source may be asked (queries, rows scanned,
seconds), how many features the grammar may propose (`B_feat`), the windows
and the lag of the as-of cutoff, the largest recording delay the reconciliation
accepts, and the tolerances of key and foreign-key inference. The tables and
columns are not here; they are in the plan file the data spec names.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import Self

from scmle.contracts import qualification

DEFAULT_WINDOWS = (7.0, 30.0, 90.0)
DEFAULT_FEATURE_BUDGET = 200


def non_negative(value: Any, name: str) -> float:
    """A finite number that is not negative."""
    number = qualification.finite(value, name)
    if number < 0:
        raise ValueError(f"{name} must not be negative")
    return number


@dataclasses.dataclass(frozen=True, slots=True)
class IngestSpec:
    """Budget, windows and tolerances of an ingestion.

    Attributes:
        max_queries: Source queries allowed, or None for no limit.
        max_rows_scanned: Rows the source may be estimated to scan, or None.
        max_seconds: Wall time allowed, or None.
        feature_budget: `B_feat`, the cap on synthesized features.
        windows: Declared windows in days.
        lag: Days between the cutoff and the prediction moment.
        delta_rec: Relative gap in the extract reconciliation that needs no
            explanation.
        eps_key: Largest `g3` accepted as a key.
        eps_null: Largest null rate accepted in a key.
        eps_fk: Largest share of child values without a parent for a link.
        strict: Drop features whose weakest source is `unverified`.
    """

    max_queries: int | None = None
    max_rows_scanned: int | None = None
    max_seconds: float | None = None
    feature_budget: int = DEFAULT_FEATURE_BUDGET
    windows: tuple[float, ...] = DEFAULT_WINDOWS
    lag: float = 0.0
    delta_rec: float = 0.0
    eps_key: float = 0.0
    eps_null: float = 0.0
    eps_fk: float = 0.0
    strict: bool = False

    def __post_init__(self) -> None:
        """Validates the block."""
        for name in ("max_queries", "max_rows_scanned"):
            value = getattr(self, name)
            if value is not None:
                qualification.count(value, name)
        if self.max_seconds is not None:
            non_negative(self.max_seconds, "max_seconds")
        qualification.count(self.feature_budget, "feature_budget", minimum=1)
        qualification.require_tuple(self.windows, "windows")
        if not self.windows or len(set(self.windows)) != len(self.windows):
            raise ValueError("windows must be non-empty and distinct")
        for window in self.windows:
            if qualification.finite(window, "window") <= 0:
                raise ValueError("windows must be positive")
        for name in ("lag", "delta_rec", "eps_key", "eps_null", "eps_fk"):
            non_negative(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        record = dataclasses.asdict(self)
        record["windows"] = list(self.windows)
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the block, rejecting unknown or missing fields."""
        qualification.require_fields(cls, record)
        values = dict(record)
        values["windows"] = tuple(values["windows"])
        return cls(**values)
