"""Explicit column-name policy applied to incoming inference data."""

import dataclasses

import pandas

from scmle.contracts import errors


@dataclasses.dataclass(frozen=True)
class ColumnMapping:
    """Explicit policy for reconciling incoming column names with a schema.

    Nothing is renamed implicitly: the default policy requires exact names.

    Attributes:
        renames: Incoming name -> expected name.
        case_insensitive: Match remaining columns ignoring letter case.
        strip_whitespace: Match remaining columns ignoring surrounding
            whitespace.
    """

    renames: dict[str, str] = dataclasses.field(default_factory=dict)
    case_insensitive: bool = False
    strip_whitespace: bool = False

    def normalize(self, name: str) -> str:
        """Applies the matching normalizations to a column name.

        Args:
            name: The column name.

        Returns:
            The name stripped of surrounding whitespace when
            `strip_whitespace` is set and lower-cased when `case_insensitive`
            is set.
        """
        if self.strip_whitespace:
            name = name.strip()
        return name.lower() if self.case_insensitive else name

    def apply(
        self, frame: pandas.DataFrame, expected: list[str]
    ) -> pandas.DataFrame:
        """Renames incoming columns according to the policy.

        Columns already named as expected are kept. Another column is renamed
        by an explicit entry of `renames` first, then, if either matching
        option is set, by its normalized name; columns that match neither way
        are left unchanged.

        Args:
            frame: Incoming data (not modified).
            expected: Column names the model expects.

        Returns:
            A renamed copy of the frame whose matching columns carry the
            expected names.

        Raises:
            SchemaError: If a rename target is not an expected column, two
                incoming columns map to the same expected column, or a
                mapped name is already a column of the frame.
        """
        unknown = sorted(set(self.renames.values()) - set(expected))
        if unknown:
            raise errors.SchemaError(
                f"rename targets not in the model schema: {unknown}",
                {"unknown": unknown},
            )
        by_key = {self.normalize(name): name for name in expected}
        mapping: dict[str, str] = {}
        for column in map(str, frame.columns):
            if column in expected:
                continue
            target = self.renames.get(column)
            if target is None and (
                self.case_insensitive or self.strip_whitespace
            ):
                target = by_key.get(self.normalize(column))
            if target is not None:
                mapping[column] = target
        targets = list(mapping.values())
        clashes = sorted({t for t in targets if targets.count(t) > 1})
        clashes += [t for t in mapping.values() if t in frame.columns]
        if clashes:
            raise errors.SchemaError(
                f"ambiguous column mapping onto {sorted(set(clashes))}",
                {"ambiguous": sorted(set(clashes))},
            )
        return frame.rename(columns=mapping)
