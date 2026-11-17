"""Caller-owned predicates and transformers applied to features."""

from collections.abc import Sequence

import numpy

from scmle.data.modalities.geospatial.evidence import Match
from scmle.data.modalities.geospatial.geometry import CoordinateTransformer
from scmle.data.modalities.geospatial.geometry import GeometryError
from scmle.data.modalities.geospatial.geometry import Placemark
from scmle.data.modalities.geospatial.geometry import SpatialJoinPredicate


def probe(
    predicate: SpatialJoinPredicate,
    left: Placemark,
    right: Placemark,
    predicate_name: str,
) -> bool | numpy.bool_:
    """Runs a caller-owned spatial join predicate on one pair of features.

    Args:
        predicate: The predicate.
        left: The left feature.
        right: The right feature.
        predicate_name: Stable name of the predicate, recorded in errors.

    Returns:
        The predicate's boolean result.

    Raises:
        GeometryError: If the predicate raises or returns a
            value that is not a boolean.
    """
    try:
        result = predicate(left, right)
    except Exception as error:
        raise GeometryError(
            "caller-owned spatial join predicate failed",
            {
                "code": "predicate_failure",
                "predicate_name": predicate_name,
                "exception": type(error).__name__,
            },
        ) from error
    if not isinstance(result, (bool, numpy.bool_)):
        raise GeometryError(
            "spatial join predicate must return a boolean",
            {
                "code": "invalid_predicate_result",
                "predicate_name": predicate_name,
                "result_type": type(result).__name__,
            },
        )
    return result


def pairs(
    left: Sequence[Placemark],
    right: Sequence[Placemark],
    predicate: SpatialJoinPredicate,
    predicate_name: str,
    match_limit: int,
) -> list[Match]:
    """Applies a join predicate to every left and right pair of features.

    Args:
        left: Features of the left dataset.
        right: Features of the right dataset.
        predicate: Caller-owned predicate.
        predicate_name: Stable name and version of the predicate.
        match_limit: Maximum matches retained.

    Returns:
        The matching pairs, in comparison order.

    Raises:
        GeometryError: If the predicate raises or returns a non-boolean, or
            the matches exceed ``match_limit``.
    """
    matches: list[Match] = []
    for first in left:
        for second in right:
            result = probe(predicate, first, second, predicate_name)
            if bool(result):
                matches.append(Match(first.feature_id, second.feature_id))
                if len(matches) > match_limit:
                    raise GeometryError(
                        "spatial join exceeds the match limit; provide a "
                        "bounded output policy",
                        {
                            "code": "match_limit_exceeded",
                            "match_limit": match_limit,
                        },
                    )
    return matches


def transform(
    features: Sequence[Placemark],
    transformer: CoordinateTransformer,
    source_crs: str,
    target_crs: str,
    transformer_name: str,
) -> tuple[Placemark, ...]:
    """Applies a coordinate transformer to every feature.

    Args:
        features: The features to transform.
        transformer: Caller-owned coordinate transformation.
        source_crs: CRS of the features.
        target_crs: CRS to transform them to.
        transformer_name: Stable name and version of the transformer.

    Returns:
        The transformed features, in order.

    Raises:
        GeometryError: If the transformer raises, returns a non-feature, or
            changes a feature's ID, properties, label, or observation time.
    """
    transformed_features: list[Placemark] = []
    for feature in features:
        try:
            transformed = transformer(feature, source_crs, target_crs)
        except Exception as error:
            raise GeometryError(
                "caller-owned coordinate transformer failed",
                {
                    "code": "transformer_failure",
                    "transformer_name": transformer_name,
                    "exception": type(error).__name__,
                },
            ) from error
        if not isinstance(transformed, Placemark):
            raise GeometryError(
                "coordinate transformer must return a Placemark",
                {
                    "code": "invalid_transformer_result",
                    "transformer_name": transformer_name,
                },
            )
        if (
            transformed.feature_id != feature.feature_id
            or dict(transformed.properties) != dict(feature.properties)
            or transformed.label != feature.label
            or transformed.observation_time != feature.observation_time
        ):
            raise GeometryError(
                "coordinate transformation changed feature metadata",
                {
                    "code": "transformer_metadata_changed",
                    "feature_id": feature.feature_id,
                },
            )
        transformed_features.append(transformed)
    return tuple(transformed_features)
