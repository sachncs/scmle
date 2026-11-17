"""The vector geospatial preparer."""

from __future__ import annotations

import dataclasses
from typing import Any
from typing import TYPE_CHECKING

import numpy

from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.data.modalities import curation
from scmle.data.modalities import geospatial
from scmle.pipeline.modality_lifecycle import containers
from scmle.pipeline.modality_lifecycle import curators
from scmle.pipeline.modality_lifecycle import scoring

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sklearn.linear_model import LogisticRegression


REMARKS = (
    ("The baseline uses geometry envelopes and centroids in the declared CRS."),
    (
        "It does not claim reprojection, raster, spatial-join, or "
        "causal spatial inference support."
    ),
)
OCCLUSION_LIMITATIONS = (
    (
        "Axis masks are bounded geometry perturbations, not a "
        "map, region, or spatial-semantic attribution."
    ),
    (
        "Geometry validity beyond the declared source contract "
        "is not inferred from score changes."
    ),
    "Score changes are associational and not causal.",
)
LIMITATIONS = (
    ("Coordinates remain in the declared CRS and are not reprojected."),
    ("The result is not a map attribution, spatial join, or causal claim."),
)


class GeospatialPreparer(curators.Preparer):
    """Adapter for CRS-declared vector feature classification.

    Attributes:
        dataset: Geospatial dataset with a declared coordinate reference system.
        block_size: Side of the spatial split blocks, in CRS units.
    """

    modality = modalities.Modality.GEOSPATIAL

    def __init__(
        self, dataset: geospatial.Dataset, block_size: float = 1.0
    ) -> None:
        """Binds a geospatial dataset and an explicit CRS-unit block size.

        Args:
            dataset: Vector features the lifecycle runs on.
            block_size: Side of the spatial split blocks, in CRS units.

        Raises:
            ValueError: If `block_size` is not positive.
        """
        if block_size <= 0:
            raise errors.InputError("block_size must be positive")
        self.dataset = dataset
        self.block_size = block_size

    @property
    def dataset_hash(self) -> str:
        """The vector dataset identity."""
        return self.dataset.content_hash

    @property
    def samples(self) -> tuple[geospatial.Placemark, ...]:
        """The geospatial features in the dataset."""
        return self.dataset.features

    def curate(self) -> curation.Curation:
        """Builds geometry, CRS, bounds, and label evidence.

        Returns:
            The curation report with spatial boundary leakage added to its
            profile, and a warning finding if geometries cross block envelopes.
        """
        report = self.dataset.curate()
        profile = dict(report.profile)
        quality_findings = list(report.quality_findings)
        if not self.dataset.crs.strip():
            profile["spatial_boundary_leakage"] = {
                "status": "blocked_missing_crs",
                "block_size": self.block_size,
            }
        else:
            leakage = self.dataset.straddle(self.block_size)
            profile["spatial_boundary_leakage"] = leakage.to_dict()
            if leakage.crossing_feature_ids:
                quality_findings.append(
                    curation.flaw(
                        "quality.geospatial-boundary-leakage",
                        data.FindingSeverity.WARNING,
                        {
                            "dataset": self.dataset_hash,
                            "block_size": self.block_size,
                        },
                        leakage.to_dict(),
                        "Review geometries crossing spatial block envelopes "
                        "before using centroid-based evaluation.",
                        blocking=False,
                    )
                )
        return dataclasses.replace(
            report,
            profile=profile,
            quality_findings=tuple(quality_findings),
        )

    def split(self, seed: int) -> containers.Partitions:
        """Splits by spatial blocks in declared CRS units.

        Args:
            seed: Random seed for the split.

        Returns:
            Train, validation and test partitions, with the block IDs of each
            under `groups`.
        """
        result = self.dataset.tile(self.block_size, seed=seed)
        return containers.Partitions(
            result.train,
            result.validation,
            result.test,
            result.tiles(),
        )

    def features(
        self, samples: Sequence[geospatial.Placemark]
    ) -> containers.Batch:
        """Prepares bounded geometry envelope and centroid features.

        The dataset is rebuilt with the adapter's CRS and source; a missing
        CRS or a non-finite prepared value raises GeometryError
        from the dataset.

        Args:
            samples: Features to convert.

        Returns:
            Batch with one row per feature, identified by its feature ID.
        """
        dataset = geospatial.Dataset(
            tuple(samples), crs=self.dataset.crs, source=self.dataset.source
        )
        prepared = dataset.prepare_features()
        return containers.Batch(
            tuple(prepared.feature_ids),
            prepared.feature_names,
            prepared.to_numpy(),
        )

    def sample_id(self, sample: geospatial.Placemark) -> str:
        """Gives a feature identity.

        Args:
            sample: Geospatial feature.

        Returns:
            The feature ID.
        """
        return sample.feature_id

    def label(self, sample: geospatial.Placemark) -> object | None:
        """Gives the feature label.

        Args:
            sample: Geospatial feature.

        Returns:
            The feature's label, or None if it has none.
        """
        return sample.label

    def elucidate(
        self,
        sample: Any,
        model: LogisticRegression,
        feature_names: Sequence[str],
        coefficients: numpy.ndarray,
    ) -> dict[str, object]:
        """Builds CRS-bound geometry and feature-contribution evidence.

        Args:
            sample: Geospatial feature to explain; it must be in the dataset.
            model: Fitted classifier.
            feature_names: Names of the model's features.
            coefficients: Linear coefficients for the model's features, one per
                feature name.

        Returns:
            Mapping with the feature's CRS, geometry type and bounds, the ten
            largest feature contributions (value times coefficient), the
            decision score and the score change when each coordinate axis is
            replaced by the bounds midpoint.

        Raises:
            TypeError: If `sample` is not a `Placemark`.
        """
        if not isinstance(sample, geospatial.Placemark):
            raise errors.KindError("sample must be a Placemark")
        feature_index = next(
            index
            for index, item in enumerate(self.dataset.features)
            if item.feature_id == sample.feature_id
        )
        prepared = self.dataset.prepare_features()
        values = numpy.asarray(
            prepared.to_numpy()[feature_index], dtype=numpy.float64
        )
        decision = numpy.asarray(model.decision_function(values[None]))
        anchored = scoring.anchor(model, values[None], decision)
        feature_terms = scoring.rank(
            (name, float(values[index] * coefficients[index]))
            for index, name in enumerate(feature_names)
        )
        return {
            "status": "available",
            "method": "crs_bound_geometry_and_feature_contribution",
            "feature_id": sample.feature_id,
            "crs": self.dataset.crs,
            "geometry_type": sample.geometry_type,
            "bounds": list(sample.bounds),
            "feature_terms": [[name, value] for name, value in feature_terms],
            "decision_score": float(decision.reshape(-1)[0]),
            "spatial_units_preserved": True,
            "raw_input_occlusion": {
                "status": "available",
                "method": "raw_geospatial_coordinate_axis_occlusion",
                "axes": axes(self, sample, model, anchored),
                "crs": self.dataset.crs,
                "baseline": "centroid replacement in source CRS units",
                "feature_space_only": False,
                "limitations": list(OCCLUSION_LIMITATIONS),
            },
            "limitations": list(LIMITATIONS),
        }

    def monitor(self, observed: curators.Preparer) -> dict[str, object]:
        """Builds geospatial identity, CRS, and profile drift evidence.

        Args:
            observed: Adapter over the newly observed features.

        Returns:
            Mapping with both dataset hashes and flags for whether the CRS and
            the bounds differ.

        Raises:
            TypeError: If `observed` is not a `GeospatialPreparer`.
        """
        if not isinstance(observed, GeospatialPreparer):
            raise errors.KindError(
                "observed adapter must be a GeospatialPreparer"
            )
        reference = self.dataset.profile().to_dict()
        current = observed.dataset.profile().to_dict()
        return {
            "reference_hash": self.dataset.content_hash,
            "observed_hash": observed.dataset.content_hash,
            "crs_changed": current["crs"] != reference["crs"],
            "bounds_changed": current["bounds"] != reference["bounds"],
        }

    def remarks(self) -> tuple[str, ...]:
        """Describes geospatial lifecycle limits.

        Returns:
            Plain statements of what the baseline does and does not claim.
        """
        return REMARKS


def axes(
    preparer: GeospatialPreparer,
    sample: geospatial.Placemark,
    model: LogisticRegression,
    anchored: tuple[int, float],
) -> list[dict[str, object]]:
    """Scores the feature with each coordinate axis set to its midpoint.

    Args:
        preparer: Preparer that featurizes the masked feature.
        sample: Geospatial feature to explain.
        model: Fitted classifier.
        anchored: The tracked class and the unmasked feature's score for it.

    Returns:
        For the x and then the y axis, the midpoint used, the score change
        (unmasked minus masked) and the masked geometry's hash.
    """
    selected_class, original_score = anchored
    minimum_x, minimum_y, maximum_x, maximum_y = sample.bounds
    baselines = ("x", "y")
    baseline_values = (
        (minimum_x + maximum_x) / 2,
        (minimum_y + maximum_y) / 2,
    )
    raw_axes: list[dict[str, object]] = []
    for axis, axis_name, baseline in zip(
        (0, 1), baselines, baseline_values, strict=True
    ):
        masked_sample = sample.substituted(axis, baseline)
        masked_values = preparer.features((masked_sample,)).values
        masked_score = scoring.score(model, masked_values, selected_class)
        raw_axes.append(
            {
                "axis": axis_name,
                "baseline_value": baseline,
                "score_delta_from_full": original_score - masked_score,
                "masked_geometry_hash": masked_sample.geometry_hash,
            }
        )
    return raw_axes
