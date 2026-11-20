"""The geospatial vector kind: features read as rows, split by spatial blocks.

A vector file (`tables.load`) becomes one row per feature: its properties
and its geometry measures (`tables.COLUMNS`). From there the rows take the
shared stages as a table does, so the shared model families, search,
selection, the sealed test set, explanation, packaging and serving apply
unchanged. What the kind adds is its spatial blocks (`Geospatial.places`):
blocks with a side derived from the label's autocorrelation range
(`blocks.variogram`, MM-11), which the shared splitter (`rows.Cohorts`)
keeps on one side of the test split and of every fold, together with any
declared group.

The kind is experimental for the reasons in `REASONS`.
"""

import pathlib

import numpy
import pandas

from scmle.contracts import capabilities
from scmle.contracts import tasks
from scmle.data.modalities.geospatial import blocks
from scmle.data.modalities.geospatial import geometry
from scmle.data.modalities.geospatial import tables
from scmle.data.readers import streaming
from scmle.pipeline import profiler
from scmle.pipeline.kinds import rows

PARSE = 3e-4
"""Seconds to read and measure one feature: observed at 2.3e-5 s for a point
and 1.2e-4 s for a 40-vertex polygon (Apple M-series, one thread), with a
2.5x margin over the polygon; the time grows with the vertices, so much
larger polygons exceed it."""

REASONS = (
    rows.LIFECYCLE,
    (
        "served models take the geometry columns, not raw GeoJSON; the "
        "request decoder that measures a posted feature is not wired into "
        "serving yet"
    ),
    (
        "the held-out test interval resamples rows, not spatial blocks (the "
        "summary clusters only by a declared group column), so it can be "
        "narrower than the block design warrants"
    ),
)
"""Why the kind is experimental rather than supported."""

LIMITATIONS = (
    "vector features only (rasters are a separate kind)",
    "measures in the declared CRS units; no reprojection",
    (
        "the block side is the label's practical range in CRS units; a "
        "geographic CRS measures it in degrees"
    ),
)
"""What a geospatial capability does not cover, besides `REASONS`."""


class Geospatial(rows.Rows):
    """Vector features: GeoJSON, or what `geopandas` reads when installed."""

    name = "geospatial"
    seconds = PARSE
    reasons = REASONS
    failures = (
        "malformed GeoJSON or an unreadable vector file",
        "too few distinct places for the spatial blocks",
        "model-family dependency unavailable",
    )
    evidence = "tests/unit/test_geospatial_kind.py"

    def limits(self) -> tuple[str, ...]:
        """The boundaries the geospatial capability states.

        Returns:
            `LIMITATIONS`.
        """
        return LIMITATIONS

    def read(
        self, spec: tasks.Config
    ) -> tuple[
        pandas.DataFrame, tuple[streaming.Feed, streaming.Totals] | None
    ]:
        """Reads the vector file as rows; the CRS is kept in `attrs["crs"]`.

        Args:
            spec: The run's specification.

        Returns:
            The rows (`tables.tabulate`) and None: a vector file is read
            whole.

        Raises:
            UnsupportedError: If a memory budget is declared and the file
                exceeds it (vector files are not streamed), or its format
                needs a missing optional library.
        """
        data = spec.data
        path = pathlib.Path(data.uri)
        budget = data.memory_budget_gb
        if (
            budget is not None
            and path.is_file()
            and path.stat().st_size > budget * 1e9
        ):
            raise capabilities.UnsupportedError(
                "vector files are read whole and this one exceeds the "
                "memory budget",
                {
                    "status": "unsupported",
                    "reason": "geospatial files are not streamed",
                    "modality": self.name,
                    "uri": data.uri,
                },
            )
        document, crs = tables.load(path)
        frame = tables.tabulate(document)
        frame.attrs["crs"] = crs
        return frame, None

    def notice(self, frame: pandas.DataFrame) -> list[str]:
        """The CRS the measures are in and any mix of geometry types.

        Args:
            frame: The rows.

        Returns:
            The findings.
        """
        crs = frame.attrs.get("crs", "an undeclared CRS")
        found = [
            f"geospatial: geometry measures are in {crs} units, not reprojected"
        ]
        if "geo_type" in frame.columns:
            kinds = sorted(set(frame["geo_type"].dropna()))
            if len(kinds) > 1:
                found.append(
                    f"geospatial: mixed geometry types {kinds} share one set "
                    "of measures; a point has no area or length"
                )
        return found

    def places(
        self,
        spec: tasks.Config,
        task: tasks.Task,
        frame: pandas.DataFrame,
        timed: bool,
    ) -> tuple[list[object] | None, str | None]:
        """The spatial block of each row, sized by the label's range.

        `blocks.design` with the `folds + 2` blocks the group split needs;
        the note says when the blocks are smaller than the range. A
        time-ordered split keeps its order and gets no blocks.

        Args:
            spec: The specification of the split.
            task: The resolved task.
            frame: The rows, with the `geo_x` and `geo_y` centres and the
                target.
            timed: Whether the split is time-ordered.

        Returns:
            The block of each row and the manifest note; for a time-ordered
            split, no blocks and a note saying so.

        Raises:
            GeometryError: If the centre columns are missing.
            BlockedError: If fewer distinct places exist than blocks needed
                (`blocks.design`).
        """
        if timed:
            return None, (
                "time-ordered split, not spatial blocks: rows near each "
                "other can sit on both sides"
            )
        missing = {"geo_x", "geo_y"} - set(frame.columns)
        if missing:
            raise geometry.GeometryError(
                f"spatial blocks need the centre columns {sorted(missing)}",
                {"code": "missing_centres", "columns": sorted(missing)},
            )
        points = frame[["geo_x", "geo_y"]].to_numpy(dtype=numpy.float64)
        gram = blocks.variogram(
            points,
            blocks.encode(
                frame[spec.data.target], categorical=task.classification
            ),
            spec.validation.seed,
        )
        side, places = blocks.design(points, gram, spec.validation.folds + 2)
        note = (
            f"spatial blocks of {side:g} CRS units ({len(set(places))} "
            "blocks, grid at the CRS origin); the label's autocorrelation "
            f"range is {gram.reach:.4g} from {gram.pairs} pairs"
        )
        if side < gram.reach:
            note += (
                "; the blocks are smaller than the range, so held-out "
                "scores may be optimistic"
            )
        return list(places), note

    def cost(
        self, rows: int, columns: int, classes: int, budget: float
    ) -> float:
        """The profiler's bound plus reading and the block design.

        Args:
            rows: Development rows.
            columns: Feature columns.
            classes: Classes of the target.
            budget: The run's time budget in seconds.

        Returns:
            `profiler.estimate`, plus `PARSE` seconds a feature, plus the
            semivariogram's bound scaled by the pairs it takes.
        """
        sample = min(rows, blocks.SAMPLE)
        return (
            profiler.estimate(rows, columns, classes, budget)
            + rows * self.seconds
            + blocks.SECONDS * (sample / blocks.SAMPLE) ** 2
        )


GEOSPATIAL = Geospatial()
"""The geospatial vector kind."""
