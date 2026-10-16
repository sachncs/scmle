"""Schema validation of inference input against a bundle's contract."""

import pandas as pd

from mml.contracts import artifacts
from mml.contracts import errors

MAX_REPORTED_UNSEEN = 20


class InputValidator:
    """Checks incoming data against the input schema of a trained model."""

    def __init__(self, schema: artifacts.BundleSchema) -> None:
        """Binds the validator to a schema.

        Args:
            schema: Input contract of the trained model.
        """
        self.schema = schema

    def report(self, frame: pd.DataFrame) -> artifacts.SchemaReport:
        """Describes how the data differs from the schema, without raising.

        Args:
            frame: Incoming data with already-normalized column names.

        Returns:
            Missing and extra columns, missing-value counts and categories
            never seen in training.
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
        return artifacts.SchemaReport(
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

    def check(self, frame: pd.DataFrame) -> None:
        """Raises when the data cannot be scored.

        Args:
            frame: Incoming data with already-normalized column names.

        Raises:
            SchemaError: On missing columns or non-numeric values in numeric
                columns.
        """
        report = self.report(frame)
        if not report.valid:
            raise errors.SchemaError(
                f"input is missing required columns: {list(report.missing)}",
                {"missing": list(report.missing)},
            )
        bad = []
        for column in self.schema.numeric:
            coerced = pd.to_numeric(frame[column], errors="coerce")
            if (coerced.isna() & frame[column].notna()).any():
                bad.append(column)
        if bad:
            raise errors.SchemaError(
                f"non-numeric values in numeric columns: {bad}",
                {"columns": bad},
            )
