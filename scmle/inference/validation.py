"""Schema validation of inference input against a bundle's contract."""

from typing import Any

import pandas

from scmle.contracts import artifacts
from scmle.contracts import errors

MAX_REPORTED_UNSEEN = 20


class InputValidator:
    """Checks incoming data against the input schema of a trained model."""

    def __init__(self, schema: artifacts.BundleSchema) -> None:
        """Binds the validator to a schema.

        Args:
            schema: Input contract of the trained model.
        """
        self.schema = schema

    def report(self, frame: pandas.DataFrame) -> artifacts.Conformance:
        """Describes how the data differs from the schema, without raising.

        Args:
            frame: Incoming data with already-normalized column names.

        Returns:
            The required columns that are absent, the extra columns (neither
            a feature nor the target), the null count of each present
            feature that has any, and, per categorical column, up to 20
            sorted values never seen in training.
        """
        wanted = self.schema.features
        present = [c for c in wanted if c in frame.columns]
        unseen = {}
        for column in self.schema.categorical:
            if column in frame.columns:
                known = set(self.schema.known_categories[column])
                new = sorted(set(frame[column].dropna().astype(str)) - known)
                if new:
                    unseen[column] = new[:MAX_REPORTED_UNSEEN]
        return artifacts.Conformance(
            missing=tuple(c for c in wanted if c not in frame.columns),
            extra=tuple(
                c
                for c in frame.columns
                if c not in wanted and c != self.schema.target
            ),
            missing_value_counts={
                c: int(frame[c].isna().sum())
                for c in present
                if frame[c].isna().any()
            },
            unseen_categories=unseen,
        )

    def row_report(self, frame: pandas.DataFrame) -> list[dict[str, Any]]:
        """Describes each row: its null features and its unseen categories.

        Args:
            frame: Incoming data with already-normalized column names.

        Returns:
            One mapping per row with `missing` (feature columns that are null
            in that row) and `unseen` (categorical column -> value never seen
            in training). Columns absent from the frame are a schema error
            reported by `report`, not row facts.
        """
        wanted = [c for c in self.schema.features if c in frame.columns]
        null_mask = frame[wanted].isna() if wanted else frame.iloc[:, :0]
        unseen_values: dict[str, pandas.Series] = {}
        for column in self.schema.categorical:
            if column in frame.columns:
                known = set(self.schema.known_categories[column])
                values = frame[column]
                unseen_values[column] = values.notna() & ~values.astype(
                    str
                ).isin(known)
        return [
            {
                "missing": [
                    c for c in wanted if bool(null_mask.iloc[position][c])
                ],
                "unseen": {
                    column: str(frame[column].iloc[position])
                    for column, flags in unseen_values.items()
                    if bool(flags.iloc[position])
                },
            }
            for position in range(len(frame))
        ]

    def check(self, frame: pandas.DataFrame) -> None:
        """Raises when required columns are missing or numeric ones are bad.

        Unseen categories, nulls and extra columns do not raise.

        Args:
            frame: Incoming data with already-normalized column names.

        Raises:
            SchemaError: If a required column is missing, or a numeric column
                holds a non-null value that is not a number.
        """
        report = self.report(frame)
        if not report.valid:
            raise errors.SchemaError(
                f"input is missing required columns: {list(report.missing)}",
                {"missing": list(report.missing)},
            )
        bad = []
        for column in self.schema.numeric:
            coerced = pandas.to_numeric(frame[column], errors="coerce")
            if (coerced.isna() & frame[column].notna()).any():
                bad.append(column)
        if bad:
            raise errors.SchemaError(
                f"non-numeric values in numeric columns: {bad}",
                {"columns": bad},
            )
