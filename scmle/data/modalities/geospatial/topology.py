"""Neighbour topology, spatial autocorrelation and geometry features."""

from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy

from scmle.contracts import errors
from scmle.data.modalities.geospatial.evidence import Autocorrelation
from scmle.data.modalities.geospatial.evidence import Features
from scmle.data.modalities.geospatial.geometry import autocorrelate
from scmle.data.modalities.geospatial.geometry import GeometryError
from scmle.data.modalities.geospatial.geometry import MIN_OBSERVATIONS
from scmle.data.modalities.geospatial.geometry import Placemark
from scmle.data.modalities.geospatial.geometry import unpack

MAX_PERMUTATIONS = 10_000


def permutation_p_value(
    numeric: numpy.ndarray,
    edges: Sequence[tuple[int, int]],
    statistic: float,
    permutations: int,
    seed: int,
) -> float | None:
    """Computes a two-sided permutation p-value for Moran's I.

    Args:
        numeric: The observed values.
        edges: The neighbour edges.
        statistic: The observed statistic.
        permutations: Number of random permutations; zero skips the test.
        seed: Seed of the permutation random generator.

    Returns:
        ``(exceedances + 1) / (permutations + 1)``, where an exceedance is a
        permuted statistic at least as large in absolute value as the
        observed one; None when ``permutations`` is zero.
    """
    if not permutations:
        return None
    randomizer = numpy.random.default_rng(seed)
    exceedances = 0
    permutation_index = 0
    while permutation_index < permutations:
        shuffled = randomizer.permutation(numeric)
        permuted_statistic = autocorrelate(shuffled, edges)
        if abs(permuted_statistic) >= abs(statistic):
            exceedances += 1
        permutation_index += 1
    return (exceedances + 1) / (permutations + 1)


class Topology:
    """Neighbour topology and numeric features of a vector dataset.

    Attributes:
        features: The features of the dataset.
        crs: Declared CRS; may be empty.
    """

    __slots__ = ()

    features: tuple[Placemark, ...]
    crs: str

    def connect(self, neighbor_count: int = 4) -> tuple[tuple[int, int], ...]:
        """Builds deterministic symmetric k-nearest-neighbor edges.

        Distances are measured between envelope centroids in the declared
        source CRS. The method exposes the exact topology used by the
        autocorrelation statistic so a reviewer can reproduce the evidence.

        Args:
            neighbor_count: Number of nearest neighbors per feature, in
                ``[1, feature_count)``.

        Returns:
            Sorted, de-duplicated undirected edges as (lower, higher) feature
            index pairs. Ties in distance break by feature index.

        Raises:
            GeometryError: If the dataset declares no CRS.
            ValueError: If ``neighbor_count`` is outside
                ``[1, feature_count)``.
        """
        if not self.crs.strip():
            raise GeometryError(
                "spatial neighbors require a declared CRS",
                {"code": "missing_crs"},
            )
        if not 1 <= neighbor_count < len(self.features):
            raise errors.InputError(
                "neighbor_count must be in [1, feature_count)"
            )
        centroids = numpy.asarray(
            [feature.centroid for feature in self.features],
            dtype=numpy.float64,
        )
        edges: set[tuple[int, int]] = set()
        for first in range(len(centroids)):
            distances = [
                (
                    float(
                        numpy.sum((centroids[first] - centroids[second]) ** 2)
                    ),
                    second,
                )
                for second in range(len(centroids))
                if second != first
            ]
            nearest = sorted(distances)[:neighbor_count]
            for distance_and_index in nearest:
                second = distance_and_index[1]
                edges.add(
                    (first, second) if first < second else (second, first)
                )
        return tuple(sorted(edges))

    def collect(self, values: Sequence[Real] | None) -> numpy.ndarray:
        """Collects and checks the values that Moran's I is computed on.

        Args:
            values: Numeric values aligned with the feature order, or None
                to use the feature labels.

        Returns:
            The values as a float array.

        Raises:
            GeometryError: If the value count differs from the
                feature count or a value is not numeric.
        """
        if values is None:
            raw_values: list[Any] = [feature.label for feature in self.features]
        else:
            raw_values = list(values)
        if len(raw_values) != len(self.features):
            raise GeometryError(
                "spatial values must match feature count",
                {
                    "code": "value_count_mismatch",
                    "feature_count": len(self.features),
                    "value_count": len(raw_values),
                },
            )
        if any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in raw_values
        ):
            raise GeometryError(
                "spatial autocorrelation requires numeric values",
                {"code": "numeric_values_required"},
            )
        return numpy.asarray(raw_values, dtype=numpy.float64)

    def spatial_autocorrelation(
        self,
        values: Sequence[Real] | None = None,
        neighbor_count: int = 4,
        permutations: int = 99,
        seed: int = 0,
    ) -> Autocorrelation:
        """Measures global Moran's I with optional deterministic permutations.

        When ``values`` is omitted, numeric feature labels are used. This
        method reports association in the observed spatial arrangement; it
        does not establish causality or replace a domain-specific spatial
        model.

        Args:
            values: Numeric values aligned with the feature order. Defaults
                to the feature labels.
            neighbor_count: Requested nearest neighbors per feature, reduced
                to one less than the feature count when larger.
            permutations: Number of random permutations for the p-value, in
                ``[0, 10000]``. Zero skips the p-value.
            seed: Seed of the permutation random generator.

        Returns:
            The statistic, neighbor topology size, and a two-sided
            permutation p-value, which is None when no permutations are run.

        Raises:
            ValueError: If ``permutations`` is outside ``[0, 10000]`` or
                ``neighbor_count`` is below one.
            GeometryError: If there are fewer than three
                features, the value count differs from the feature count,
                a value is not numeric, the dataset has no CRS, or the
                statistic is not estimable.
        """
        if permutations < 0 or permutations > MAX_PERMUTATIONS:
            raise errors.InputError("permutations must be in [0, 10000]")
        if len(self.features) < MIN_OBSERVATIONS:
            raise GeometryError(
                "spatial autocorrelation requires at least three features",
                {"code": "insufficient_features"},
            )
        if neighbor_count < 1:
            raise errors.InputError("neighbor_count must be positive")
        effective_neighbor_count = min(neighbor_count, len(self.features) - 1)
        numeric = self.collect(values)
        edges = self.connect(effective_neighbor_count)
        try:
            statistic = autocorrelate(numeric, edges)
        except ValueError as error:
            raise GeometryError(
                str(error), {"code": "statistic_not_estimable"}
            ) from error

        p_value = permutation_p_value(
            numeric, edges, statistic, permutations, seed
        )
        return Autocorrelation(
            feature_count=len(self.features),
            neighbor_count=effective_neighbor_count,
            edge_count=len(edges),
            statistic=statistic,
            permutation_count=permutations,
            p_value=p_value,
            seed=seed,
        )

    def prepare_features(self) -> Features:
        """Prepares bounded geometry features without changing CRS units.

        The output is a geometry envelope and centroid representation. It does
        not perform reprojection, spatial joins, rasterization, or distance
        calculations; those operations require an explicit, versioned policy.

        Returns:
            One row per feature with its bounds, width, height, centroid, and
            count of coordinate values, in the dataset CRS units.

        Raises:
            GeometryError: If the dataset has no CRS or a
                prepared value is not finite.
        """
        if not self.crs.strip():
            raise GeometryError(
                "geospatial feature preparation requires a declared CRS",
                {"code": "missing_crs"},
            )
        feature_names = (
            "min_x",
            "min_y",
            "max_x",
            "max_y",
            "width",
            "height",
            "centroid_x",
            "centroid_y",
            "coordinate_value_count",
        )
        rows = []
        for feature in self.features:
            minimum_x, minimum_y, maximum_x, maximum_y = feature.bounds
            coordinates = unpack(feature.geometry)
            rows.append(
                (
                    minimum_x,
                    minimum_y,
                    maximum_x,
                    maximum_y,
                    maximum_x - minimum_x,
                    maximum_y - minimum_y,
                    feature.centroid[0],
                    feature.centroid[1],
                    float(len(coordinates)),
                )
            )
        values: numpy.ndarray = numpy.asarray(rows, dtype=numpy.float64)
        if not numpy.isfinite(values).all():
            raise GeometryError(
                "geospatial features must be finite",
                {"code": "non_finite_features"},
            )
        return Features(
            feature_ids=tuple(feature.feature_id for feature in self.features),
            feature_names=feature_names,
            values=tuple(
                tuple(float(value) for value in row) for row in values
            ),
            crs=self.crs,
        )
