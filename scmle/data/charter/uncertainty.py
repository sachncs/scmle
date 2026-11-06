"""The charter's uncertainty block (feature F12, format 10).

Everything here is optional: a charter without an `UncertaintySpec`
serialises, hashes and loads exactly as a format 1 to 9 charter did. The
block fixes, before any data is read, the largest number of bytes of
cross-validation fold models a regression bundle may carry for CV+ intervals
(a larger set is not kept and the run says why, `cv_plus.skipped`; the pooled
out-of-fold scores are served instead) and the columns whose levels the
coverage report describes slice by slice (`slice_columns`; they enter the
registered coverage claim).
"""

from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import Self

from scmle.contracts import qualification


@dataclasses.dataclass(frozen=True, slots=True)
class UncertaintySpec:
    """Fixed choices of the interval and set layer.

    Attributes:
        cv_plus_max_bytes: Largest serialized size of the fold models kept
            for CV+, or None for the library default
            (`scmle.evaluation.cv_plus.MAX_BYTES`).
        slice_columns: Columns of the features whose levels (numeric columns
            in four bands) get a coverage and set-size description.
    """

    cv_plus_max_bytes: int | None = None
    slice_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates the block."""
        qualification.require_tuple(self.slice_columns, "slice_columns")
        if len(set(self.slice_columns)) != len(self.slice_columns) or any(
            not isinstance(c, str) or not c.strip() for c in self.slice_columns
        ):
            raise ValueError("slice_columns are distinct, non-blank names")
        if self.cv_plus_max_bytes is not None:
            qualification.count(
                self.cv_plus_max_bytes, "cv_plus_max_bytes", minimum=1
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record; unset fields are omitted."""
        record: dict[str, Any] = {}
        if self.cv_plus_max_bytes is not None:
            record["cv_plus_max_bytes"] = self.cv_plus_max_bytes
        if self.slice_columns:
            record["slice_columns"] = list(self.slice_columns)
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the block, rejecting unknown fields."""
        unknown = set(record) - {"cv_plus_max_bytes", "slice_columns"}
        if unknown:
            raise ValueError(f"unknown uncertainty fields {sorted(unknown)}")
        return cls(
            record.get("cv_plus_max_bytes"),
            tuple(record.get("slice_columns", ())),
        )
