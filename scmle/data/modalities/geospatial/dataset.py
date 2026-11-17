"""The immutable geospatial vector dataset and its reprojection."""

from collections.abc import Mapping
import dataclasses
import hashlib
from typing import Any
from typing import Self

from scmle.contracts import errors
from scmle.data.modalities.geospatial.evidence import Join
from scmle.data.modalities.geospatial.evidence import Provenance
from scmle.data.modalities.geospatial.geometry import canonical_source
from scmle.data.modalities.geospatial.geometry import CoordinateTransformer
from scmle.data.modalities.geospatial.geometry import GeometryError
from scmle.data.modalities.geospatial.geometry import Placemark
from scmle.data.modalities.geospatial.geometry import placemarks
from scmle.data.modalities.geospatial.geometry import SpatialJoinPredicate
from scmle.data.modalities.geospatial.linkage import pairs
from scmle.data.modalities.geospatial.linkage import transform
from scmle.data.modalities.geospatial.splits import Splits


@dataclasses.dataclass(frozen=True, slots=True)
class Dataset(Splits):
    """Immutable GeoJSON-like vector collection with CRS provenance.

    Attributes:
        features: The features of the dataset, with unique IDs.
        crs: Declared CRS, stripped of surrounding whitespace; may be empty.
        source: Canonical source identifier.
        content_hash: SHA-256 hex digest over the CRS and the feature
            identities, independent of feature order.
    """

    features: tuple[Placemark, ...]
    crs: str = ""
    source: str = "memory://"
    content_hash: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Rejects duplicate identities and derive CRS-bound content.

        Raises:
            GeometryError: If there are no features or feature
                IDs are duplicated.
            TypeError: If ``crs`` is not a string.
        """
        normalized = tuple(self.features)
        if not normalized:
            raise GeometryError(
                "geospatial dataset must contain at least one feature",
                {"code": "empty_dataset"},
            )
        if not isinstance(self.crs, str):
            raise errors.KindError("crs must be a string")
        canonical_crs_value = self.crs.strip()
        ids = [feature.feature_id for feature in normalized]
        if len(ids) != len(set(ids)):
            raise GeometryError(
                "geospatial dataset contains duplicate feature IDs",
                {"code": "duplicate_feature_id"},
            )
        feature_identities = sorted(
            f"{feature.feature_id}\0{feature.content_hash}"
            for feature in normalized
        )
        identity_payload = f"crs:{canonical_crs_value}\n" + "\n".join(
            feature_identities
        )
        digest = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()
        object.__setattr__(self, "features", normalized)
        object.__setattr__(self, "crs", canonical_crs_value)
        object.__setattr__(self, "source", canonical_source(self.source))
        object.__setattr__(self, "content_hash", digest)

    @classmethod
    def from_geojson(
        cls,
        document: Mapping[str, Any],
        source: str = "memory://",
        crs: str = "",
    ) -> Self:
        """Builds a dataset from a GeoJSON FeatureCollection mapping.

        Feature IDs come from each feature's ``id`` member (as text), falling
        back to its index. Feature labels and observation times are not read.
        Errors raised by the feature and dataset constructors propagate.

        Args:
            document: A GeoJSON FeatureCollection.
            source: Source identifier recorded on the dataset.
            crs: CRS identifier declared for the dataset.

        Returns:
            A dataset with one feature per GeoJSON feature.

        Raises:
            GeometryError: If the root is not a
                FeatureCollection, ``features`` is not a list, or a member is
                not a Feature with object geometry and properties.
        """
        return cls(placemarks(document), crs=crs, source=source)

    def vet(
        self,
        other: "Dataset",
        predicate: SpatialJoinPredicate,
        predicate_name: str,
        comparison_limit: int,
        match_limit: int,
    ) -> None:
        """Checks the arguments of a spatial join before any comparison.

        Args:
            other: Right-hand dataset.
            predicate: Caller-owned predicate.
            predicate_name: Stable name and version of the predicate.
            comparison_limit: Maximum Cartesian comparisons allowed.
            match_limit: Maximum matches retained in the evidence report.

        Raises:
            TypeError: If ``other`` is not a dataset or ``predicate`` is not
                callable.
            ValueError: If ``predicate_name`` is empty or a limit is not
                positive.
            GeometryError: If either dataset lacks a CRS or the
                CRS values differ.
        """
        if not isinstance(other, Dataset):
            raise errors.KindError("other must be a Dataset")
        if not callable(predicate):
            raise errors.KindError("predicate must be callable")
        if not predicate_name.strip():
            raise errors.InputError("predicate_name must not be empty")
        if comparison_limit < 1 or match_limit < 1:
            raise errors.InputError("join limits must be positive")
        if not self.crs.strip() or not other.crs.strip():
            raise GeometryError(
                "spatial joins require a declared CRS on both datasets",
                {"code": "missing_crs"},
            )
        if self.crs.strip() != other.crs.strip():
            raise GeometryError(
                "spatial joins require matching CRS declarations; reproject "
                "through an explicit caller-owned policy first",
                {
                    "code": "crs_mismatch",
                    "left_crs": self.crs,
                    "right_crs": other.crs,
                },
            )

    def spatial_join(
        self,
        other: "Dataset",
        predicate: SpatialJoinPredicate,
        predicate_name: str,
        comparison_limit: int = 100_000,
        match_limit: int = 100_000,
    ) -> Join:
        """Joins two same-CRS datasets through a bounded caller predicate.

        The core package does not choose a geometry engine, infer CRS
        transformations, or silently compare coordinates in different units.
        A caller supplies a versioned predicate such as an ``intersects/v1``
        implementation. This method owns the identity checks, bounded
        execution, deterministic evidence, and fail-closed limits.

        Args:
            other: Right-hand dataset with the same declared CRS.
            predicate: Caller-owned predicate receiving one left and one right
                feature and returning a Python boolean.
            predicate_name: Stable name and version of the predicate.
            comparison_limit: Maximum Cartesian comparisons allowed.
            match_limit: Maximum matches retained in the evidence report.

        Returns:
            Dataset identities, counts, and matched feature IDs.

        Raises:
            TypeError: If ``other`` is not a dataset or ``predicate`` is not
                callable.
            ValueError: If ``predicate_name`` is empty or a limit is not
                positive.
            GeometryError: If either dataset lacks a CRS, the
                CRS values differ, the number of comparisons exceeds
                ``comparison_limit``, the predicate raises or returns a
                non-boolean, or the matches exceed ``match_limit``.
        """
        self.vet(
            other, predicate, predicate_name, comparison_limit, match_limit
        )
        total_comparisons = len(self.features) * len(other.features)
        if total_comparisons > comparison_limit:
            raise GeometryError(
                "spatial join exceeds the comparison limit; provide a "
                "bounded indexed policy",
                {
                    "code": "comparison_limit_exceeded",
                    "comparison_count": total_comparisons,
                    "comparison_limit": comparison_limit,
                },
            )
        matches = pairs(
            self.features,
            other.features,
            predicate,
            predicate_name,
            match_limit,
        )
        matched_left = {match.left_feature_id for match in matches}
        matched_right = {match.right_feature_id for match in matches}
        return Join(
            left_dataset_hash=self.content_hash,
            right_dataset_hash=other.content_hash,
            crs=self.crs.strip(),
            predicate_name=predicate_name.strip(),
            left_feature_count=len(self.features),
            right_feature_count=len(other.features),
            comparison_count=total_comparisons,
            match_count=len(matches),
            unmatched_left_count=len(
                {feature.feature_id for feature in self.features} - matched_left
            ),
            unmatched_right_count=len(
                {feature.feature_id for feature in other.features}
                - matched_right
            ),
            comparison_limit=comparison_limit,
            match_limit=match_limit,
            matches=tuple(matches),
        )

    def reproject(
        self,
        target_crs: str,
        transformer: CoordinateTransformer,
        transformer_name: str,
    ) -> "Reprojection":
        """Transforms coordinates through an explicit caller-owned policy.

        The core package does not select a CRS database or geometry engine.
        The caller supplies a versioned transformer that changes geometry only;
        feature IDs, properties, labels, and observation times must remain
        stable. The returned report binds the source and target snapshots so a
        transformed dataset cannot be mistaken for the original source.

        Args:
            target_crs: Non-empty target CRS identifier.
            transformer: Caller-owned coordinate transformation.
            transformer_name: Stable name and version of the transformer.

        Returns:
            A transformed dataset and its source/target evidence report.

        Raises:
            GeometryError: If the source CRS is missing, the
                transformer raises, returns a non-feature, or changes a
                feature's ID, properties, label, or observation time.
            ValueError: If ``target_crs`` is empty or equals the source CRS,
                or ``transformer_name`` is empty.
            TypeError: If ``transformer`` is not callable.
        """
        if not self.crs.strip():
            raise GeometryError(
                "reprojection requires a declared source CRS",
                {"code": "missing_crs"},
            )
        candidate_crs: object = target_crs
        if not isinstance(candidate_crs, str) or not candidate_crs.strip():
            raise errors.InputError("target_crs must be a non-empty string")
        canonical_target_crs = target_crs.strip()
        if canonical_target_crs == self.crs:
            raise errors.InputError(
                "target_crs must differ from the source CRS"
            )
        if not callable(transformer):
            raise errors.KindError("transformer must be callable")
        if not transformer_name.strip():
            raise errors.InputError("transformer_name must not be empty")
        transformed_features = transform(
            self.features,
            transformer,
            self.crs,
            canonical_target_crs,
            transformer_name,
        )
        dataset = Dataset(
            transformed_features,
            crs=canonical_target_crs,
            source=self.source,
        )
        report = Provenance(
            source_dataset_hash=self.content_hash,
            target_dataset_hash=dataset.content_hash,
            source_crs=self.crs,
            target_crs=canonical_target_crs,
            transformer_name=transformer_name.strip(),
            feature_count=len(transformed_features),
        )
        return Reprojection(dataset=dataset, report=report)


@dataclasses.dataclass(frozen=True, slots=True)
class Reprojection:
    """A transformed dataset paired with its source-bound evidence.

    Attributes:
        dataset: The transformed dataset.
        report: Evidence binding the source and transformed datasets.
    """

    dataset: Dataset
    report: Provenance

    def __post_init__(self) -> None:
        """Ensures the report describes the returned transformed dataset.

        Raises:
            ValueError: If the report's target hash or CRS differs from the
                dataset's.
        """
        if self.report.target_dataset_hash != self.dataset.content_hash:
            raise errors.InputError(
                "reprojection report must match the transformed dataset"
            )
        if self.report.target_crs != self.dataset.crs:
            raise errors.InputError(
                "reprojection report CRS must match the dataset"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns the transformed identity and its reprojection evidence.

        Returns:
            A mapping with the dataset hash, CRS, and the report as a
            mapping.
        """
        return {
            "dataset_hash": self.dataset.content_hash,
            "crs": self.dataset.crs,
            "report": self.report.to_dict(),
        }
