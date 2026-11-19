"""Clustering unlabeled rows into a bundle that assigns new rows to groups.

`cluster` searches methods and cluster counts (`scmle.unsupervised.
clustering.discover`), labels every row, and wraps the chosen model in a
`Bundle` whose task is `clustering`, so `scmle predict` and `Bundle.predict`
return cluster labels. `evaluate` scores a saved bundle
on other rows: the internal indices always, and agreement with known groups
when a column of them is named.
"""

import contextlib
import dataclasses
import json
import math
import pathlib
import shutil
from typing import Any

import numpy
import pandas

from scmle import durability
from scmle.artifacts import bundle
from scmle.contracts import errors
import scmle.contracts.data
from scmle.data.readers import interchange
from scmle.data.readers import tabular
from scmle.unsupervised import clustering
from scmle.unsupervised.clustering import METHODS
from scmle.unsupervised.clustering import TASK

__all__ = [
    "METHODS",
    "REPORT_FILE",
    "TASK",
    "Segmentation",
    "cluster",
    "evaluate",
]

REPORT_FILE = "clustering.json"
METRIC = "silhouette"
STAGE = ".partial"


@dataclasses.dataclass(frozen=True)
class Segmentation:
    """A clustering search, the labels it gives and its bundle.

    Attributes:
        report: Every candidate tried and the chosen one.
        bundle: The chosen model as a bundle of task `TASK`.
        labels: The cluster of every input row, rows with gaps included
            (their gaps take the training medians).
        agreement: Adjusted Rand index and normalized mutual information
            against the known groups, empty when none were named.
    """

    report: clustering.Clustering
    bundle: bundle.Bundle
    labels: numpy.ndarray
    agreement: dict[str, float]

    @property
    def metrics(self) -> dict[str, float]:
        """The internal indices and the stability of the chosen candidate."""
        best = dataclasses.asdict(self.report.best)
        return {
            k: v for k, v in best.items() if k not in {"method", "clusters"}
        }

    def describe(self) -> str:
        """Writes the JSON report saved next to the bundle.

        Returns:
            `Clustering.to_dict` plus `task`, `columns`, `rows`, `metrics`
            and `agreement`, as indented JSON with sorted keys. Infinite
            indices (one-group candidates) are written as null.
        """
        record = {
            **self.report.to_dict(),
            "task": TASK,
            "columns": list(self.bundle.schema.numeric),
            "rows": len(self.labels),
            "metrics": self.metrics,
            "agreement": self.agreement,
        }
        return json.dumps(finite(record), indent=2, sort_keys=True)

    def save(
        self,
        directory: str | pathlib.Path,
        signing_key: str | bytes | None = None,
    ) -> pathlib.Path:
        """Writes the bundle and `REPORT_FILE` into a new directory.

        Everything is written into a sibling staging directory first and
        renamed into place, so an interrupted save never leaves a half-
        written bundle at `directory`; a stage left by an earlier
        interruption is removed first.

        Args:
            directory: Destination; it must not exist or must be empty.
            signing_key: HMAC key; defaults to `$SCMLE_SIGNING_KEY`. An
                unsigned bundle loads only with `trusted=True`.

        Returns:
            The directory.

        Raises:
            InputError: If `directory` exists and is not an empty directory.
        """
        target = pathlib.Path(directory)
        if target.exists():
            if not target.is_dir() or any(target.iterdir()):
                raise errors.InputError(
                    f"{target} already exists; choose a new directory"
                )
            target.rmdir()
        stage = target.with_name(target.name + STAGE)
        if stage.exists():
            shutil.rmtree(stage)
        try:
            self.bundle.save(stage, signing_key)
            durability.write_atomically(
                stage / REPORT_FILE, self.describe().encode("utf-8")
            )
            stage.replace(target)
        finally:
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(stage)
        durability.sync_directory(target.parent)
        return target


def finite(value: Any) -> Any:
    """Replaces infinite and NaN floats in a JSON-ready value with None.

    Args:
        value: Nested dicts, lists and scalars.

    Returns:
        The same structure, valid as strict JSON.
    """
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read(data: str | pathlib.Path | Any) -> pandas.DataFrame:
    """Reads a table from a supported file or converts an in-memory one.

    Args:
        data: A CSV or Parquet path, or a pandas, Polars or Arrow table.

    Returns:
        The table as a pandas frame.
    """
    if isinstance(data, (str, pathlib.Path)):
        return tabular.read(data)
    return interchange.frame(data)


def check(frame: pandas.DataFrame, names: list[str]) -> None:
    """Refuses column names the table does not have.

    Args:
        frame: The table.
        names: Requested column names.

    Raises:
        InputError: If any name is not a column.
    """
    unknown = sorted(set(names) - set(map(str, frame.columns)))
    if unknown:
        raise errors.InputError(f"unknown column(s): {unknown}")


def agree(frame: pandas.DataFrame, truth: str, labels: Any) -> dict[str, float]:
    """Compares cluster labels with a column of known groups.

    Rows without a known group are left out.

    Args:
        frame: The table holding the column.
        truth: The column of known groups.
        labels: The cluster of each row.

    Returns:
        The adjusted Rand index and normalized mutual information.
    """
    known = frame[truth].notna().to_numpy()
    return clustering.compare(
        frame[truth].to_numpy()[known], numpy.asarray(labels)[known]
    )


def cluster(
    data: str | pathlib.Path | Any,
    counts: range | tuple[int, ...] = range(2, 9),
    methods: tuple[str, ...] = METHODS,
    rounds: int = 10,
    seed: int = 0,
    exclude: tuple[str, ...] = (),
    truth: str | None = None,
) -> Segmentation:
    """Finds groups in unlabeled rows and builds a bundle that assigns them.

    The numeric columns, less `exclude` and `truth`, are the features.

    Args:
        data: A CSV or Parquet path, or an in-memory table.
        counts: Cluster counts to try, each at least 2.
        methods: Methods from `METHODS`.
        rounds: Bootstrap refits per candidate for the stability score.
        seed: Random seed.
        exclude: Columns that are not features, such as identifiers.
        truth: Column of known groups to measure agreement against; it is
            never a feature.

    Returns:
        The search, every row's cluster, the agreement and the bundle.

    Raises:
        InputError: For an unknown column or method, fewer than one round,
            no counts, or data `discover` refuses.
    """
    frame = read(data)
    named = [*exclude, *([truth] if truth else [])]
    check(frame, named)
    unknown = sorted(set(methods) - set(METHODS))
    if unknown or not methods:
        raise errors.InputError(f"unknown method(s) {unknown}; use {METHODS}")
    if rounds < 1:
        raise errors.InputError("rounds must be at least 1")
    if not len(counts):
        raise errors.InputError("give at least one cluster count")
    features = frame.drop(columns=named)
    report = clustering.discover(features, counts, methods, rounds, seed)
    columns = [str(c) for c in report.profile.columns]
    labels = numpy.asarray(report.model.predict(features[columns]))
    model = bundle.Bundle.create(
        report.model,
        scmle.contracts.data.Blueprint(tuple(columns), ()),
        features[columns],
        target="",
        task=TASK,
        metric=METRIC,
        model_name=report.best.method,
        model_params={"clusters": report.best.clusters},
        seed=seed,
    )
    agreement = agree(frame, truth, labels) if truth else {}
    return Segmentation(report, model, labels, agreement)


def evaluate(
    directory: str | pathlib.Path,
    data: str | pathlib.Path | Any,
    truth: str | None = None,
    trusted: bool = False,
    signing_key: str | bytes | None = None,
) -> dict[str, float]:
    """Scores a saved clustering bundle on rows.

    Args:
        directory: The bundle directory.
        data: A CSV or Parquet path, or an in-memory table.
        truth: Column of known groups, if any.
        trusted: Load an unsigned bundle you vouch for.
        signing_key: HMAC key verifying the bundle signature.

    Returns:
        `silhouette`, `davies_bouldin` and `calinski_harabasz` of the assigned
        labels in the bundle's standardized space, plus `adjusted_rand` and
        `normalized_mutual_info` when `truth` is given.

    Raises:
        ArtifactError: If the bundle is not a clustering bundle or does not
            load.
        InputError: If `truth` is not a column.
    """
    model = bundle.Bundle.load(
        directory, trusted=trusted, signing_key=signing_key
    )
    if model.schema.task != TASK:
        raise errors.ArtifactError(
            f"{directory} is not a clustering bundle (task "
            f"{model.schema.task!r})"
        )
    frame = read(data)
    if truth:
        check(frame, [truth])
    labels = model.predict(frame)
    matrix = model.pipeline[:-1].transform(model.prepare(frame))
    scores = clustering.score(matrix, labels)
    return scores | (agree(frame, truth, labels) if truth else {})
