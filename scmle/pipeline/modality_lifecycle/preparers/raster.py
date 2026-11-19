"""The raster preparer."""

from __future__ import annotations

import math
from typing import Any
from typing import TYPE_CHECKING
from typing import TypedDict

import numpy

from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.data.modalities import curation
from scmle.data.modalities import geospatial
from scmle.pipeline.modality_lifecycle import constants
from scmle.pipeline.modality_lifecycle import containers
from scmle.pipeline.modality_lifecycle import curators
from scmle.pipeline.modality_lifecycle import scoring

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from sklearn.linear_model import LogisticRegression


METADATA_LIMITATIONS = (
    "Metadata-only rasters cannot produce pixel-region evidence.",
    "Raw encoded-payload occlusion is outside this contract.",
)
OCCLUSION_LIMITATIONS = (
    (
        "Occlusion is applied after the caller-owned decoder and "
        "is not evidence about encoded bytes or decoder failure "
        "modes."
    ),
    (
        "Regions are pixel windows, not semantic objects or "
        "validated geospatial causal explanations."
    ),
    "Score changes are associational and not causal.",
)
METADATA_FEATURES = (
    "width",
    "height",
    "bands",
    "byte_size",
    "transform_0",
    "transform_1",
    "transform_2",
    "transform_3",
    "transform_4",
    "transform_5",
)


class Nodata(TypedDict):
    """The nodata part of a raster profile."""

    statuses: list[str]
    declared_values: list[float | int]
    fraction_min: float | None
    fraction_max: float | None


class Pixels(TypedDict, total=False):
    """Decoded pixel statistics, or a metadata-only marker."""

    status: str
    decoder: str | None
    max_pixels: int
    shapes: list[list[int]]
    minimum: float
    maximum: float
    measured_nodata_fraction: list[float]


class Survey(TypedDict):
    """The profile of a raster dataset."""

    dimensions: list[list[int]]
    crs: list[str]
    dtypes: list[str]
    transform_lengths: list[int]
    total_bytes: int
    nodata: Nodata
    label_counts: dict[str, int]
    pixels: Pixels


class Window(TypedDict):
    """One masked pixel window of an occlusion explanation."""

    pixel_window: list[int]
    spatial_bounds: list[float] | None
    score_delta: float


def summary(pixels: numpy.ndarray, band_count: int) -> tuple[float, ...]:
    """Computes finite per-band summary values for decoded pixels.

    Args:
        pixels: Decoded array of shape (rows, columns) for one band or
            (rows, columns, bands).
        band_count: Number of bands the raster declares.

    Returns:
        For each band in order, the mean, standard deviation, minimum and
        maximum of its values.

    Raises:
        GeometryError: If the pixel shape does not match the
            declared band count.
    """
    values = numpy.asarray(pixels)
    if values.ndim == constants.MATRIX_RANK:
        if band_count != 1:
            raise geospatial.GeometryError(
                "single-band decoded pixels must declare one band",
                {"code": "pixel_band_mismatch"},
            )
        channels = values[:, :, None]
    elif (
        values.ndim == constants.CHANNELED_PIXEL_RANK
        and values.shape[2] == band_count
    ):
        channels = values
    else:
        raise geospatial.GeometryError(
            "decoded pixels do not match the declared band count",
            {"code": "pixel_band_mismatch"},
        )
    return tuple(
        statistic_value
        for band in range(band_count)
        for statistic_value in (
            float(channels[:, :, band].mean()),
            float(channels[:, :, band].std()),
            float(channels[:, :, band].min()),
            float(channels[:, :, band].max()),
        )
    )


class Imagery(curators.Preparer):
    """Adapter for labelled raster metadata or bounded decoded pixels.

    Attributes:
        dataset: Raster samples, in the order given.
        labels_by_source: Caller-owned label of each raster, keyed by source.
        decoder: Caller-owned pixel decoder, or None for metadata only.
        decoder_name: Stripped name and version of the decoder, or None.
        max_pixels: Maximum number of pixels decoded per raster.
    """

    modality = modalities.Modality.RASTER

    def __init__(
        self,
        samples: Sequence[geospatial.Tile],
        labels: Mapping[str, object],
        decoder: geospatial.RasterDecoder | None = None,
        decoder_name: str | None = None,
        max_pixels: int = 4_000_000,
    ) -> None:
        """Binds raster identities and caller-owned labels.

        Raster payload decoding remains caller-owned. When supplied, the
        decoder name and pixel budget become part of the representation
        contract. The label sidecar is keyed by the canonical raster source
        and is part of the caller's dataset evidence.

        Args:
            samples: Rasters to run on; at least one, with unique sources.
            labels: Label of each raster, keyed by its source.
            decoder: Callable that decodes a raster payload to pixels, or None
                to use metadata only.
            decoder_name: Name and version that identify the decoder; required
                with a decoder and not allowed without one.
            max_pixels: Maximum number of pixels decoded per raster.

        Raises:
            ValueError: If there are no samples, sources repeat, a source has no
                label, `max_pixels` is below 1, or the decoder and its name are
                not given together.
            TypeError: If `decoder` is not callable.
        """
        self.dataset = admit(samples, labels, decoder, decoder_name, max_pixels)
        self.labels_by_source = dict(labels)
        self.decoder = decoder
        self.decoder_name = decoder_name.strip() if decoder_name else None
        self.max_pixels = max_pixels

    @property
    def digest(self) -> str:
        """The raster identity and label-sidecar digest."""
        identities = (
            f"{sample.source}\0{sample.file_hash}\0"
            f"{sample.nodata_value}\0{sample.nodata_fraction}\0"
            f"{self.labels_by_source[sample.source]}\0"
            f"decoder={self.decoder_name or 'metadata-only'}\0"
            f"max_pixels={self.max_pixels}"
            for sample in self.dataset
        )
        return curation.dataset_hash(identities)

    @property
    def samples(self) -> tuple[geospatial.Tile, ...]:
        """The raster samples in the dataset."""
        return self.dataset

    def curate(self) -> curation.Curation:
        """Builds raster shape, CRS, transform, and label evidence.

        Returns:
            The curation report for modality `raster` (register row MM-4),
            with decoded pixel statistics if a decoder is set (otherwise a
            metadata-only marker) and warnings for mixed shapes, CRS or data
            types and for unknown nodata handling.
        """
        profile = survey(self)
        quality_findings = flaws(self, profile)
        return curation.Curation(
            modality=modalities.serialise(self.modality),
            dataset_hash=self.digest,
            sample_count=len(self.dataset),
            profile=profile,
            quality_findings=quality_findings,
        )

    def split(self, seed: int) -> containers.Partitions:
        """Splits by source identity because the core lacks raster decoding.

        Args:
            seed: Random seed for the shuffle.

        Returns:
            Train, validation and test partitions where validation and test each
            hold a fifth of the rasters (at least one), with the sources of each
            under `groups`.
        """
        return partition(self.dataset, seed)

    def features(self, samples: Sequence[geospatial.Tile]) -> containers.Batch:
        """Prepares raster metadata or bounded decoded pixel features.

        Args:
            samples: Rasters to featurize.

        Returns:
            Batch with one row per raster, identified by its source: per-band
            pixel statistics if a decoder is set, otherwise width, height, band
            count, byte size and six transform values.

        Raises:
            GeometryError: If decoding is on and the rasters do not
                share one band count (an empty sequence counts as not sharing
                one).
        """
        if self.decoder is None:
            return metadata(samples)
        band_counts = {sample.bands for sample in samples}
        if len(band_counts) != 1:
            raise geospatial.GeometryError(
                "decoded raster features require a consistent band count",
                {"code": "pixel_band_mismatch"},
            )
        band_count = next(iter(band_counts))
        names = self.headers(band_count)
        rows = []
        for sample in samples:
            pixels = sample.decode(self.decoder, max_pixels=self.max_pixels)
            rows.append(self.summarise(pixels, band_count))
        return containers.Batch(
            tuple(sample.source for sample in samples),
            names,
            numpy.asarray(rows, dtype=numpy.float64),
        )

    @staticmethod
    def headers(band_count: int) -> tuple[str, ...]:
        """Names the bounded decoded-pixel summary features.

        Args:
            band_count: Number of raster bands.

        Returns:
            For each band in order, the names of its mean, standard deviation,
            minimum and maximum.

        Raises:
            ValueError: If `band_count` is below 1.
        """
        if band_count < 1:
            raise errors.InputError("band_count must be positive")
        return tuple(
            f"pixel_band_{band}_{statistic}"
            for band in range(band_count)
            for statistic in ("mean", "std", "min", "max")
        )

    summarise = staticmethod(summary)

    def sample_id(self, sample: geospatial.Tile) -> str:
        """Gives the canonical raster source identity.

        Args:
            sample: Raster sample.

        Returns:
            The raster's source.
        """
        return sample.source

    def label(self, sample: geospatial.Tile) -> object | None:
        """Gives the caller-owned raster label.

        Args:
            sample: Raster sample.

        Returns:
            The label recorded for the raster's source.
        """
        return self.labels_by_source[sample.source]

    def elucidate(
        self,
        sample: Any,
        model: LogisticRegression,
        feature_names: Sequence[str],
        coefficients: numpy.ndarray,
    ) -> dict[str, object]:
        """Builds bounded occlusion evidence for decoded raster pixels.

        Args:
            sample: Raster sample to explain.
            model: Fitted classifier.
            feature_names: Names of the model's features; they must be the
                raster's pixel feature names.
            coefficients: Linear coefficients, one per feature.

        Returns:
            Mapping with the ten pixel windows (a 4 x 4 grid) whose masking
            changes the decision score most (original minus masked), all
            windows with their spatial bounds, and limitations; or status
            "not_available" if no decoder is set.

        Raises:
            TypeError: If `sample` is not a `Tile`.
            ValueError: If the feature names or coefficients do not match the
                raster's pixel features.
        """
        if not isinstance(sample, geospatial.Tile):
            raise errors.KindError("sample must be a Tile")
        if self.decoder is None:
            return {
                "status": "not_available",
                "modality": modalities.serialise(self.modality),
                "reason": "an explicit named raster decoder is required",
                "limitations": list(METADATA_LIMITATIONS),
            }
        pixels = sample.decode(self.decoder, max_pixels=self.max_pixels)
        original_values = numpy.asarray(
            self.summarise(pixels, sample.bands), dtype=numpy.float64
        )
        if tuple(feature_names) != self.headers(sample.bands):
            raise errors.InputError(
                "raster explanation feature schema does not match"
            )
        if original_values.shape[0] != coefficients.shape[0]:
            raise errors.InputError(
                "raster explanation coefficients do not match features"
            )
        anchored = scoring.anchor(model, original_values.reshape(1, -1))
        terms, windows = regions(self, sample, pixels, model, anchored)
        height, width = pixels.shape[:2]
        return {
            "status": "available",
            "method": "decoded_pixel_region_occlusion",
            "source": sample.source,
            "decoder": self.decoder_name,
            "decoded_dimensions": [int(width), int(height), int(sample.bands)],
            "crs": sample.crs,
            "window_count": len(terms),
            "terms": [
                [name, float(value)] for name, value in scoring.rank(terms)
            ],
            "windows": windows,
            "baseline": "zero-valued decoded raster pixels",
            "feature_space_only": False,
            "limitations": list(OCCLUSION_LIMITATIONS),
        }

    def monitor(self, observed: curators.Preparer) -> dict[str, object]:
        """Builds raster profile drift evidence.

        Args:
            observed: Adapter over the newly observed rasters.

        Returns:
            Mapping with both dataset hashes and whether the profile differs.

        Raises:
            TypeError: If `observed` is not a `Imagery`.
        """
        if not isinstance(observed, Imagery):
            raise errors.KindError("observed adapter must be a Imagery")
        reference = self.curate()
        current = observed.curate()
        return {
            "reference_hash": reference.dataset_hash,
            "observed_hash": current.dataset_hash,
            "profile_changed": current.profile != reference.profile,
        }

    def remarks(self) -> tuple[str, ...]:
        """Describes raster lifecycle limits.

        Returns:
            Plain statements of the representation in use and what it does not
            cover.
        """
        representation = (
            "The baseline uses caller-decoded, bounded pixel summary "
            "features with an explicit decoder policy."
            if self.decoder is not None
            else "The baseline uses raster metadata and a caller-owned label "
            "sidecar; it does not decode pixels."
        )
        return (
            representation,
            (
                "Raster transforms, reprojection, window sampling, and spatial "
                "joins require an explicit codec policy; nodata metadata is "
                "caller-owned unless measured through the decoder."
            ),
        )


def admit(
    samples: Sequence[geospatial.Tile],
    labels: Mapping[str, object],
    decoder: geospatial.RasterDecoder | None,
    decoder_name: str | None,
    max_pixels: int,
) -> tuple[geospatial.Tile, ...]:
    """Checks the arguments of a raster preparer.

    Args:
        samples: Rasters to run on.
        labels: Label of each raster, keyed by its source.
        decoder: Callable that decodes a raster payload to pixels, or None.
        decoder_name: Name and version that identify the decoder, or None.
        max_pixels: Maximum number of pixels decoded per raster.

    Returns:
        The rasters, in the order given.

    Raises:
        ValueError: If there are no samples, sources repeat, a source has no
            label, `max_pixels` is below 1, or the decoder and its name are
            not given together.
        TypeError: If `decoder` is not callable.
    """
    normalized = tuple(samples)
    if not normalized:
        raise errors.InputError("raster samples must not be empty")
    if decoder is not None and not callable(decoder):
        raise errors.KindError("decoder must be callable")
    if decoder is None and decoder_name is not None:
        raise errors.InputError("decoder_name requires a caller-owned decoder")
    if decoder is not None and not decoder_name:
        raise errors.InputError(
            "decoder_name must identify a caller-owned decoder and version"
        )
    if max_pixels < 1:
        raise errors.InputError("max_pixels must be positive")
    sources = tuple(sample.source for sample in normalized)
    if len(set(sources)) != len(sources):
        raise errors.InputError("raster sources must be unique")
    if any(source not in labels for source in sources):
        raise errors.InputError("raster labels must cover every sample source")
    return normalized


def survey(preparer: Imagery) -> Survey:
    """Profiles the shapes, CRS, data types, nodata and labels of rasters.

    Args:
        preparer: Preparer whose rasters, labels and decoder are profiled.

    Returns:
        The profile, with decoded pixel statistics if a decoder is set and a
        metadata-only marker otherwise.
    """
    dimensions = sorted(
        {
            (sample.width, sample.height, sample.bands)
            for sample in preparer.dataset
        }
    )
    crs_values = sorted({sample.crs for sample in preparer.dataset})
    dtypes = sorted({sample.dtype for sample in preparer.dataset})
    fractions = [
        sample.nodata_fraction
        for sample in preparer.dataset
        if sample.nodata_fraction is not None
    ]
    nodata_profile: Nodata = {
        "statuses": sorted(
            {sample.nodata_status for sample in preparer.dataset}
        ),
        "declared_values": sorted(
            {
                sample.nodata_value
                for sample in preparer.dataset
                if sample.nodata_value is not None
            }
        ),
        "fraction_min": min(fractions, default=None),
        "fraction_max": max(fractions, default=None),
    }
    return {
        "dimensions": [list(value) for value in dimensions],
        "crs": crs_values,
        "dtypes": dtypes,
        "transform_lengths": sorted(
            {len(sample.transform) for sample in preparer.dataset}
        ),
        "total_bytes": sum(sample.byte_size for sample in preparer.dataset),
        "nodata": nodata_profile,
        "label_counts": curation.label_counts(
            preparer.labels_by_source.values()
        ),
        "pixels": decoding(preparer),
    }


def decoding(preparer: Imagery) -> Pixels:
    """Profiles decoded pixel values, or marks the rasters metadata-only.

    Each raster is decoded once and dropped before the next, so the
    profile holds one decoded array at a time (register row MM-6).

    Args:
        preparer: Preparer whose rasters and decoder are profiled.

    Returns:
        The decoder, pixel budget, decoded shapes, value range and measured
        nodata fractions; or a metadata-only marker without a decoder.
    """
    decoder = preparer.decoder
    if decoder is None:
        return {
            "status": "metadata_only",
            "decoder": None,
            "max_pixels": preparer.max_pixels,
        }
    shapes: set[tuple[int, ...]] = set()
    low, high = math.inf, -math.inf
    measured = []
    for sample in preparer.dataset:
        values = sample.decode(decoder, max_pixels=preparer.max_pixels)
        shapes.add(tuple(values.shape))
        low = min(low, float(values.min()))
        high = max(high, float(values.max()))
        if sample.nodata_value is not None:
            measured.append(
                float(numpy.equal(values, sample.nodata_value).mean())
            )
    return {
        "status": "decoded",
        "decoder": preparer.decoder_name,
        "max_pixels": preparer.max_pixels,
        "shapes": [list(shape) for shape in sorted(shapes)],
        "minimum": low,
        "maximum": high,
        "measured_nodata_fraction": sorted(measured),
    }


def flaws(preparer: Imagery, profile: Survey) -> tuple[data.Finding, ...]:
    """Flags mixed raster shapes, CRS or data types and unknown nodata.

    Args:
        preparer: Preparer whose rasters were profiled.
        profile: The rasters' profile from `survey`.

    Returns:
        The warning findings, shape drift first.
    """
    dimensions = profile["dimensions"]
    crs_values = profile["crs"]
    dtypes = profile["dtypes"]
    nodata_profile = profile["nodata"]
    quality_findings: list[data.Finding] = []
    if len(crs_values) > 1 or len(dimensions) > 1 or len(dtypes) > 1:
        quality_findings.append(
            curation.flaw(
                "quality.raster-shape-or-crs-drift",
                data.FindingSeverity.WARNING,
                {"dataset": preparer.digest},
                {
                    "dimensions": profile["dimensions"],
                    "crs": crs_values,
                    "dtypes": dtypes,
                },
                "Record an explicit raster normalization policy before "
                "promotion.",
                blocking=False,
            )
        )
    if any(sample.nodata_status == "unknown" for sample in preparer.dataset):
        quality_findings.append(
            curation.flaw(
                "quality.raster-nodata-unknown",
                data.FindingSeverity.WARNING,
                {"dataset": preparer.digest},
                {"statuses": nodata_profile["statuses"]},
                "Declare or measure nodata handling before interpreting "
                "raster values.",
                blocking=False,
            )
        )
    return tuple(quality_findings)


def partition(
    samples: tuple[geospatial.Tile, ...], seed: int
) -> containers.Partitions:
    """Splits rasters at random by source identity.

    Args:
        samples: Rasters to split.
        seed: Random seed for the shuffle.

    Returns:
        Train, validation and test partitions where validation and test each
        hold a fifth of the rasters (at least one), with the sources of each
        under `groups`.
    """
    generator = numpy.random.default_rng(seed)
    order = generator.permutation(len(samples))
    test_count = max(1, len(samples) // 5)
    validation_count = max(1, len(samples) // 5)
    test_indices = {int(value) for value in order[:test_count]}
    validation_indices = {
        int(value)
        for value in order[test_count : test_count + validation_count]
    }
    train = tuple(
        sample
        for index, sample in enumerate(samples)
        if index not in test_indices | validation_indices
    )
    validation = tuple(
        sample
        for index, sample in enumerate(samples)
        if index in validation_indices
    )
    test = tuple(
        sample for index, sample in enumerate(samples) if index in test_indices
    )
    return containers.Partitions(
        train,
        validation,
        test,
        {
            "train": tuple(sample.source for sample in train),
            "validation": tuple(sample.source for sample in validation),
            "test": tuple(sample.source for sample in test),
        },
    )


def metadata(samples: Sequence[geospatial.Tile]) -> containers.Batch:
    """Prepares raster metadata features without decoding pixels.

    Args:
        samples: Rasters to featurize.

    Returns:
        Batch with one row per raster, identified by its source: width,
        height, band count, byte size and six transform values (missing
        transform values are zero).
    """
    rows = []
    for sample in samples:
        transform = tuple(sample.transform) + (0.0,) * 6
        rows.append(
            (
                float(sample.width),
                float(sample.height),
                float(sample.bands),
                float(sample.byte_size),
                *transform[:6],
            )
        )
    return containers.Batch(
        tuple(sample.source for sample in samples),
        METADATA_FEATURES,
        numpy.asarray(rows),
    )


def regions(
    preparer: Imagery,
    sample: geospatial.Tile,
    pixels: numpy.ndarray,
    model: LogisticRegression,
    anchored: tuple[int, float],
) -> tuple[list[tuple[str, float]], list[Window]]:
    """Scores the raster with each window of a 4 x 4 grid masked to zero.

    Args:
        preparer: Preparer that summarises the masked pixels.
        sample: Raster sample to explain.
        pixels: Its decoded pixels.
        model: Fitted classifier.
        anchored: The tracked class and the unmasked raster's score for it.

    Returns:
        Per window, its name and score change (original minus masked), and
        its pixel window, spatial bounds and score change.
    """
    selected_class, original_score = anchored
    height, width = pixels.shape[:2]
    row_step = max(1, int(numpy.ceil(height / 4)))
    column_step = max(1, int(numpy.ceil(width / 4)))
    terms: list[tuple[str, float]] = []
    windows: list[Window] = []
    for row in range(0, height, row_step):
        for column in range(0, width, column_step):
            end_row = min(row + row_step, height)
            end_column = min(column + column_step, width)
            masked_pixels = pixels.copy()
            if masked_pixels.ndim == constants.MATRIX_RANK:
                masked_pixels[row:end_row, column:end_column] = 0.0
            else:
                masked_pixels[row:end_row, column:end_column, :] = 0.0
            masked_features = numpy.asarray(
                preparer.summarise(masked_pixels, sample.bands),
                dtype=numpy.float64,
            ).reshape(1, -1)
            masked_score = scoring.score(model, masked_features, selected_class)
            region_name = f"region[{row}:{end_row},{column}:{end_column}]"
            score_delta = original_score - masked_score
            spatial_bounds = sample.extent(row, end_row, column, end_column)
            terms.append((region_name, score_delta))
            windows.append(
                {
                    "pixel_window": [row, end_row, column, end_column],
                    "spatial_bounds": (
                        list(spatial_bounds)
                        if spatial_bounds is not None
                        else None
                    ),
                    "score_delta": float(score_delta),
                }
            )
    return terms, windows
