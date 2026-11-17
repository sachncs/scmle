"""Profiling and curation of a geospatial vector dataset."""

from collections.abc import Mapping
from collections.abc import Sequence
from numbers import Real
from typing import Any
from typing import cast
from typing import TYPE_CHECKING

from scmle.contracts import data
from scmle.data.modalities.curation import Curation
from scmle.data.modalities.curation import flaw
from scmle.data.modalities.curation import label_counts
from scmle.data.modalities.geospatial.evidence import Profile
from scmle.data.modalities.geospatial.geometry import MIN_OBSERVATIONS
from scmle.data.modalities.geospatial.geometry import Placemark
from scmle.data.modalities.geospatial.topology import Topology

if TYPE_CHECKING:
    from datetime import datetime


def timeline(features: Sequence[Placemark]) -> dict[str, str | int]:
    """Summarises how completely the features carry observation times.

    Args:
        features: The features of a dataset.

    Returns:
        Status ``"complete"`` with the earliest and latest observation time
        when every feature is timed, ``"mixed"`` with the timed and untimed
        counts when only some are, and ``"absent"`` when none is.
    """
    observation_times = tuple(
        cast("datetime", feature.observation_time)
        for feature in features
        if feature.observation_time is not None
    )
    if observation_times and len(observation_times) == len(features):
        return {
            "status": "complete",
            "observation_time_min": min(observation_times).isoformat(),
            "observation_time_max": max(observation_times).isoformat(),
        }
    if observation_times:
        return {
            "status": "mixed",
            "timed_feature_count": len(observation_times),
            "untimed_feature_count": len(features) - len(observation_times),
        }
    return {"status": "absent"}


def findings(
    survey: "Survey", profile: Mapping[str, Any]
) -> tuple[tuple[data.Finding, ...], tuple[data.Finding, ...]]:
    """Builds the quality and risk findings of a geospatial curation.

    Args:
        survey: The dataset being curated.
        profile: Its curation profile, with ``temporal``, ``geometry_types``
            and ``label_counts``.

    Returns:
        The quality findings (mixed temporal state, missing CRS, mixed
        geometry types) and the risk findings (mixed label state).
    """
    quality_findings = []
    risk_findings = []
    labels = profile["label_counts"]
    if profile["temporal"]["status"] == "mixed":
        quality_findings.append(
            flaw(
                "quality.geospatial-mixed-temporal-state",
                data.FindingSeverity.WARNING,
                {"source": survey.source},
                {"temporal": profile["temporal"]},
                "Supply observation times for every feature or explicitly "
                "use a non-temporal geospatial capability.",
                blocking=False,
            )
        )
    if not survey.crs.strip():
        quality_findings.append(
            flaw(
                "quality.geospatial-missing-crs",
                data.FindingSeverity.ERROR,
                {"source": survey.source},
                {"feature_count": len(survey.features)},
                "Declare the source CRS before spatial transforms or "
                "modelling.",
                blocking=True,
            )
        )
    if len(profile["geometry_types"]) > 1:
        quality_findings.append(
            flaw(
                "quality.geospatial-mixed-geometry",
                data.FindingSeverity.WARNING,
                {"source": survey.source},
                {"geometry_types": list(profile["geometry_types"])},
                "Choose a geometry normalization or task-specific "
                "handling policy.",
                blocking=False,
            )
        )
    if "unlabeled" in labels and len(labels) > 1:
        risk_findings.append(
            flaw(
                "risk.geospatial-mixed-label-state",
                data.FindingSeverity.WARNING,
                {"source": survey.source},
                {"label_counts": labels},
                "Declare whether unlabeled features are valid for the "
                "intended spatial task before training.",
                blocking=False,
            )
        )
    return tuple(quality_findings), tuple(risk_findings)


class Survey(Topology):
    """Profiling and curation of a vector dataset.

    Attributes:
        source: Canonical source identifier.
        content_hash: Content hash of the dataset.
    """

    __slots__ = ()

    source: str
    content_hash: str

    def profile(self) -> Profile:
        """Returns geometry counts, bounds, CRS, and immutable identity.

        Returns:
            A profile with the feature count, sorted geometry types, CRS,
            dataset-wide bounds, and content hash.
        """
        bounds = tuple(
            value for feature in self.features for value in feature.bounds
        )
        return Profile(
            feature_count=len(self.features),
            geometry_types=tuple(
                sorted({feature.geometry_type for feature in self.features})
            ),
            crs=self.crs,
            bounds=(
                min(bounds[0::4]),
                min(bounds[1::4]),
                max(bounds[2::4]),
                max(bounds[3::4]),
            ),
            content_hash=self.content_hash,
        )

    def curate(self) -> Curation:
        """Builds geometry, CRS, and spatial-interpretation evidence.

        The profile includes label counts, temporal completeness, and global
        Moran's I when a CRS is declared, at least three features exist, and
        every label is numeric. Findings cover a missing CRS, mixed geometry
        types, mixed temporal state, and a mix of labeled and unlabeled
        features.

        Returns:
            A curation report for the geospatial modality.
        """
        profile_data = self.profile().to_dict()
        profile_data["label_counts"] = label_counts(
            feature.label for feature in self.features
        )
        profile_data["temporal"] = timeline(self.features)
        numeric_labels = [feature.label for feature in self.features]
        if not self.crs.strip():
            profile_data["spatial_autocorrelation"] = {
                "status": "blocked_missing_crs"
            }
        elif len(self.features) < MIN_OBSERVATIONS:
            profile_data["spatial_autocorrelation"] = {
                "status": "not_estimable",
                "reason": "at least three features are required",
            }
        elif any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in numeric_labels
        ):
            profile_data["spatial_autocorrelation"] = {
                "status": "not_applicable",
                "reason": "numeric values are required for every feature",
            }
        else:
            profile_data["spatial_autocorrelation"] = (
                self.spatial_autocorrelation().to_dict()
            )
        quality_findings, risk_findings = findings(self, profile_data)
        return Curation(
            modality="geospatial",
            dataset_hash=self.content_hash,
            sample_count=len(self.features),
            profile=profile_data,
            quality_findings=quality_findings,
            risk_findings=risk_findings,
        )
