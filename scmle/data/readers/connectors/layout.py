"""Source schemas and the bounded row pages connectors return."""

import dataclasses
from typing import Any

import pandas

from scmle.contracts import errors
from scmle.data.readers.connectors import locator

Row = tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class Field:
    """One source column and its source-reported type name.

    Attributes:
        name: Column name.
        type_name: Type name as reported by the source.
    """

    name: str
    type_name: str

    def __post_init__(self) -> None:
        """Validates column identity.

        Raises:
            ValueError: If `name` or `type_name` is empty.
        """
        if not self.name:
            raise errors.InputError("column name must not be empty")
        if not self.type_name:
            raise errors.InputError("column type_name must not be empty")


@dataclasses.dataclass(frozen=True)
class Columns:
    """Ordered schema metadata returned by a source connector.

    Attributes:
        columns: The columns in source order.
    """

    columns: tuple[Field, ...]

    def __post_init__(self) -> None:
        """Rejects empty and duplicate column declarations.

        Raises:
            ValueError: If there are no columns or two columns share a
                name.
        """
        names = tuple(column.name for column in self.columns)
        if not names:
            raise errors.InputError(
                "source schema must contain at least one column"
            )
        if len(names) != len(set(names)):
            raise errors.InputError(
                "source schema contains duplicate column names"
            )

    def as_dict(self) -> dict[str, Any]:
        """Serializes the schema.

        Returns:
            A JSON-compatible mapping with a ``columns`` list of
            ``name``/``type_name`` entries in source order.
        """
        return {
            "columns": [
                {"name": column.name, "type_name": column.type_name}
                for column in self.columns
            ]
        }

    @classmethod
    def from_frame(cls, frame: pandas.DataFrame) -> "Columns":
        """Builds a source schema from a pandas frame.

        Args:
            frame: The table whose columns and dtypes are described.

        Returns:
            A schema with one column per frame column, using the dtype
            string as the type name.
        """
        return cls(
            tuple(
                Field(str(name), str(dtype))
                for name, dtype in zip(frame.columns, frame.dtypes, strict=True)
            )
        )


@dataclasses.dataclass(frozen=True)
class Page:
    """A bounded row result and its resume checkpoint.

    ``content_sha256`` is populated when the connector can observe the raw
    source bytes without adding a second read. A resumed checkpoint can bind
    to that digest and reject a changed source snapshot.

    Attributes:
        schema: Schema of the batch.
        rows: The rows of the batch, as tuples in schema order.
        checkpoint: Checkpoint from which to read the next batch.
        has_more: Whether more rows remain after this batch.
        content_sha256: Digest of the raw source bytes observed, or None if
            the connector did not compute one.
    """

    schema: Columns
    rows: tuple[Row, ...]
    checkpoint: locator.Bookmark
    has_more: bool
    content_sha256: str | None = None
