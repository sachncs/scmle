"""Scoring reports, exports and the task-specific predictions."""

import pathlib
from typing import Any

import numpy
import pandas

from scmle.artifacts.bundle import core
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.inference import onnx_export
import scmle.inference.mapping


def newest(
    frame: pandas.DataFrame, entity: str | None, order: numpy.ndarray
) -> numpy.ndarray:
    """Finds the latest row of every series.

    Args:
        frame: The history rows.
        entity: The entity column, or None for a single series.
        order: The time of each row, in a sortable form.

    Returns:
        The positions of each series' latest row (the last in row order
        among rows at the same time), ascending.
    """
    positions = numpy.arange(len(frame))
    by_entity = (
        frame[entity].astype(str).to_numpy()
        if entity
        else numpy.zeros(len(frame), dtype=int)
    )
    latest = {}
    for position in positions[numpy.argsort(order, kind="stable")]:
        latest[by_entity[position]] = position
    return numpy.array(sorted(latest.values()))


class Outputs(core.Core):
    """Scored reports, exports and the predictions of particular tasks."""

    def score(
        self,
        data: str | pathlib.Path | pandas.DataFrame,
        mapping: scmle.inference.mapping.Mapper | None = None,
    ) -> dict[str, Any]:
        """Predicts and reports the schema diagnostics in one structured result.

        Required columns that are missing or malformed raise `SchemaError`.

        Args:
            data: DataFrame or path to a supported tabular dataset.
            mapping: Explicit column-name policy; exact names if omitted.

        Returns:
            A dict with `predictions`, `model_version`, `schema_valid` and
            `diagnostics` (ignored columns, missing-value counts, unseen
            categories). Classifiers also get `classes` and `probabilities`,
            and a bundle with a novelty detector gets `out_of_distribution`.
        """
        frame = self.read(data)
        if mapping is not None:
            frame = mapping.apply(frame, [*self.features, self.schema.target])
        report = self.validate(frame)
        predictions = self.predict(frame)
        result: dict[str, Any] = {
            "predictions": numpy.asarray(predictions).tolist(),
            "model_version": self.info.version,
            "schema_valid": report.valid,
            "diagnostics": {
                "ignored_columns": list(report.extra),
                "missing_values": report.missing_value_counts,
                "unseen_categories": report.unseen_categories,
            },
        }
        if self.schema.classes is not None:
            result["classes"] = list(self.schema.classes)
            result["probabilities"] = self.predict_proba(frame).tolist()
        if self.novelty is not None:
            result["out_of_distribution"] = self.atypical(frame)[
                "out_of_distribution"
            ].tolist()
        return result

    def pin(self, path: str | pathlib.Path | None = None) -> str:
        """Returns the build environment as pinned requirements.

        Args:
            path: Where to write it; omit to only return the text.

        Returns:
            Comment lines with the Python version, platform and framework
            version, then every distribution installed when the model was
            built as `name==version`, ready for `pip install -r`.
        """
        text = self.info.lock()
        if path is not None:
            pathlib.Path(path).write_text(text, encoding="utf-8", newline="\n")
        return text

    def export(
        self,
        path: str | pathlib.Path,
        check: pandas.DataFrame | None = None,
        tolerance: float = onnx_export.DEFAULT_TOLERANCE,
        unverified: bool = False,
    ) -> pathlib.Path:
        """Writes the pipeline as an ONNX graph (`onnx` extra), verified.

        Verification is the default: the graph and the pipeline score
        `check`, rows held out from training, and the export is blocked
        when they diverge (`onnx_export.verify`). Too few held-out rows
        are refused before the converter is imported or run. A blocked
        graph is removed. `<path>.manifest.json` records the graph's
        digest and the check's result, or that the check was skipped.

        Args:
            path: Destination file.
            check: At least `onnx_export.ROWS` held-out rows to score with
                both the graph and the pipeline.
            tolerance: Largest allowed probability / scaled value difference;
                a larger value is the gate's override.
            unverified: Skip the check (the manifest says so); `check` must
                then be None.

        Returns:
            The path written.

        Raises:
            ArtifactError: For models that cannot be exported faithfully
                (only binary or multiclass classification and regression
                are supported), an attached probability calibrator, or a
                missing `onnx` extra.
            BlockedError: When the check fails: fewer than
                `onnx_export.ROWS` rows (none given included), or a
                divergence beyond `tolerance`.
            ConfigError: When `check` is given with `unverified`.
        """
        if unverified and check is not None:
            raise errors.ConfigError(
                "pass held-out rows to check, or unverified, not both"
            )
        if not unverified:
            onnx_export.enough(check)
        written = onnx_export.export(self, path)
        report = None
        if not unverified:
            try:
                report = onnx_export.verify(self, written, check, tolerance)
            except errors.BlockedError:
                written.unlink(missing_ok=True)
                raise
        onnx_export.record(self, written, report, tolerance)
        return written

    def predict_survival(
        self,
        data: str | pathlib.Path | pandas.DataFrame,
        times: Any,
        mapping: scmle.inference.mapping.Mapper | None = None,
    ) -> numpy.ndarray:
        """Returns the probability of no event yet at each time (survival).

        Args:
            data: DataFrame or path to a supported tabular dataset.
            times: Times (in the duration column's unit) to read the curve at.
            mapping: Explicit column-name policy; exact names if omitted.

        Returns:
            Survival probabilities as an array of shape `(rows, len(times))`.

        Raises:
            ArtifactError: For models that are not survival models.
        """
        if self.schema.task != tasks.Task.SURVIVAL.value:
            raise errors.ArtifactError("this bundle is not a survival model")
        frame = self.prepare(data, mapping)
        return numpy.asarray(
            self.pipeline.predict_survival(
                frame, numpy.asarray(times, dtype=float)
            )
        )

    def forecast(
        self,
        history: str | pathlib.Path | pandas.DataFrame,
        steps: int | None = None,
        mapping: scmle.inference.mapping.Mapper | None = None,
    ) -> pandas.DataFrame:
        """Forecasts the next values of every series in `history`.

        The history supplies each series' recent values (the lag and rolling
        features are computed from it), so the series need not have been
        seen in training. Each series is forecast from its latest row. The
        forecast time is that row's time plus the step count times the
        median gap between distinct times in `history` (the latest time
        itself when there is only one distinct time).

        Args:
            history: Past rows with the value, time and (if used) entity
                columns.
            steps: How many steps ahead, from 1 to the trained horizon; all
                of the trained horizon if omitted.
            mapping: Explicit column-name policy; exact names if omitted.

        Returns:
            One row per series and step: the entity column (if any), `step`,
            the forecast time (named like the time column) and `forecast`.

        Raises:
            ArtifactError: For models that are not forecasters or a `steps`
                outside 1 to the trained horizon.
        """
        if self.schema.task != tasks.Task.FORECAST.value:
            raise errors.ArtifactError("this bundle is not a forecaster")
        horizon = len(self.schema.targets)
        wanted = horizon if steps is None else steps
        if not 1 <= wanted <= horizon:
            raise errors.ArtifactError(
                f"this model forecasts 1 to {horizon} steps ahead"
            )
        frame = self.read(history)
        if mapping is not None:
            frame = mapping.apply(frame, self.features)
        clock, entity = self.schema.time, self.schema.group
        time_name = clock or "time"
        prepared = self.prepare(frame)
        stamps = pandas.to_datetime(frame[clock], errors="coerce")
        numeric_clock = stamps.isna().any()
        order = frame[clock].to_numpy() if numeric_clock else stamps.to_numpy()
        chosen = newest(frame, entity, order)
        predicted = numpy.asarray(self.pipeline.predict(prepared.iloc[chosen]))[
            :, :wanted
        ]
        distinct = numpy.unique(order)
        gaps = numpy.sort(numpy.diff(distinct))
        step_size = gaps[len(gaps) // 2] if len(gaps) else None
        rows = []
        for row, position in enumerate(chosen):
            for step in range(wanted):
                stamp = order[position]
                if step_size is not None:
                    stamp = stamp + step_size * (step + 1)
                record: dict[str, Any] = (
                    {}
                    if entity is None
                    else {entity: frame[entity].iloc[position]}
                )
                rows.append(
                    {
                        **record,
                        "step": step + 1,
                        time_name: stamp,
                        "forecast": float(predicted[row, step]),
                    }
                )
        return pandas.DataFrame(rows)

    def rank(
        self,
        data: str | pathlib.Path | pandas.DataFrame,
        query_column: str,
        mapping: scmle.inference.mapping.Mapper | None = None,
    ) -> pandas.DataFrame:
        """Scores rows and orders them within each query (ranking tasks).

        Args:
            data: Rows to rank (DataFrame or supported tabular path).
            query_column: Column naming each row's query.
            mapping: Explicit column-name policy; exact names if omitted.

        Returns:
            The input rows with `score` and `rank` (1 = best within its
            query, ties broken by row order), sorted by query, then rank, with
            a fresh index.

        Raises:
            ArtifactError: For models that are not ranking models, or when
                `query_column` is missing from the data.
        """
        if self.schema.task != tasks.Task.RANKING.value:
            raise errors.ArtifactError("this bundle is not a ranking model")
        frame = self.read(data)
        if query_column not in frame.columns:
            raise errors.ArtifactError(
                f"query column '{query_column}' is not in the data"
            )
        ranked = frame.assign(score=self.predict(frame, mapping))
        ranked["rank"] = (
            ranked.groupby(query_column, sort=False)["score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        return ranked.sort_values(
            [query_column, "rank"], kind="stable"
        ).reset_index(drop=True)
