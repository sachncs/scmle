"""The charter's uncertainty block (format 10).

Everything here is optional: a charter without an `Uncertainty`
serialises, hashes and loads exactly as a format 1 to 9 charter did. The
block fixes, before any data is read, the largest number of bytes of
cross-validation fold models a regression bundle may carry for CV+ intervals
(a larger set is not kept and the run says why, `cv_plus.skipped`; the pooled
out-of-fold scores are served instead) and the columns whose levels the
coverage report describes slice by slice (`slices`; they enter the
registered coverage claim).
"""

from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.contracts import records


@dataclasses.dataclass(frozen=True, slots=True)
class Uncertainty:
    """Fixed choices of the interval and set layer.

    Attributes:
        limit: Largest serialized size, in bytes, of the fold models kept
            for CV+, or None for the library default
            (`scmle.evaluation.cv_plus.MAX_BYTES`).
        slices: Columns of the features whose levels (numeric columns in
            four bands) get a coverage and set-size description.
        confirmatory: Also test the registered coverage claim on the sealed
            confirmation rows `C` (format 11): the claim becomes a
            registered claim of the ledger, `C` stays sealed until the model
            and its calibration are frozen, and the claim is `confirmed` only
            if a one-sided exact binomial test passes on the one opening of
            `C`. False (the default) keeps the claim on the final test rows
            alone.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "cv_plus_max_bytes": "limit",
                "slice_columns": "slices",
                "coverage_on_confirmation": "confirmatory",
            }
        ),
    )

    limit: int | None = None
    slices: tuple[str, ...] = ()
    confirmatory: bool = False

    def __post_init__(self) -> None:
        """Validates the block.

        Raises:
            ValueError: If `confirmatory` is not a bool, `slices` is not a
                tuple, holds a duplicate or a non-string or blank name, or
                `limit` is set but is not an integer of at least 1.
        """
        if not isinstance(self.confirmatory, bool):
            raise errors.InputError("confirmatory must be a bool")
        qualification.tupled(self.slices, "slices")
        columns: Iterable[object] = self.slices
        if len(set(self.slices)) != len(self.slices) or any(
            not isinstance(c, str) or not c.strip() for c in columns
        ):
            raise errors.InputError("slices are distinct, non-blank names")
        if self.limit is not None:
            qualification.count(self.limit, "limit", minimum=1)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the block.

        Returns:
            A JSON-ready dict holding only the fields that are set:
            `limit` when not None, `slices` (as a list) when non-empty, and
            `confirmatory` (as True) when true. A block with defaults gives
            an empty dict.
        """
        record: dict[str, Any] = {}
        if self.limit is not None:
            record["limit"] = self.limit
        if self.slices:
            record["slices"] = list(self.slices)
        if self.confirmatory:
            record["confirmatory"] = True
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the block from a dict.

        Every field is optional and takes its default when absent. A record
        without `schema_version` is read as version 1; its steps only rename
        keys, so a current layout passes unchanged.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt block.

        Raises:
            ValueError: If `record` has a key that is not a field, or a
                value fails the validation of the constructor.
        """
        record = records.load(cls.__name__, cls.migrations, record)
        del record[records.KEY]
        unknown = set(record) - {"limit", "slices", "confirmatory"}
        if unknown:
            raise errors.InputError(
                f"unknown uncertainty fields {sorted(unknown)}"
            )
        return cls(
            record.get("limit"),
            tuple(record.get("slices", ())),
            record.get("confirmatory", False),
        )
