"""Numeric video features from probed metadata and decoded frames."""

import numpy

from scmle.contracts import errors
from scmle.data.modalities.video.evidence import Features
from scmle.data.modalities.video.sample import Sample
from scmle.data.modalities.video.sample import VideoError
from scmle.data.modalities.video.sample import VideoFrameDecoder


def measure(
    sample: Sample,
    decoder: VideoFrameDecoder,
    frame_index: int,
    max_frame_pixels: int,
) -> numpy.ndarray:
    """Decodes one frame and checks it against the sample's metadata.

    Args:
        sample: The video sample.
        decoder: Caller-owned frame decoder.
        frame_index: Absolute index of the frame to decode.
        max_frame_pixels: Maximum number of decoded array elements, including
            channels.

    Returns:
        The frame as a finite float64 array.

    Raises:
        VideoError: If decoding fails, or the frame is not 2-D or 3-D, is
            empty or larger than ``max_frame_pixels``, exceeds the probed
            dimensions, or contains non-finite values.
    """
    where = {"source": sample.source, "frame_index": frame_index}
    try:
        frame = numpy.asarray(decoder(sample.encoded_bytes, frame_index))
    except Exception as error:
        raise VideoError(
            "video frame decoding failed",
            {"code": "frame_decode_failed", **where},
        ) from error
    if frame.ndim not in {2, 3}:
        raise VideoError(
            "decoded video frames must be grayscale or channel-last",
            {
                "code": "invalid_frame_shape",
                **where,
                "shape": list(frame.shape),
            },
        )
    if frame.size < 1 or frame.size > max_frame_pixels:
        raise VideoError(
            "decoded video frame exceeds the configured pixel limit",
            {
                "code": "frame_pixel_limit",
                **where,
                "pixel_count": int(frame.size),
                "max_frame_pixels": max_frame_pixels,
            },
        )
    if (
        frame.shape[0] > sample.metadata.height
        or frame.shape[1] > sample.metadata.width
    ):
        raise VideoError(
            "decoded video frame exceeds probed dimensions",
            {
                "code": "frame_dimension_mismatch",
                **where,
                "shape": list(frame.shape),
                "declared": [sample.metadata.height, sample.metadata.width],
            },
        )
    numeric = numpy.asarray(frame, dtype=numpy.float64)
    if not numpy.isfinite(numeric).all():
        raise VideoError(
            "decoded video frame contains non-finite values",
            {"code": "non_finite_frame", **where},
        )
    return numeric


class Frames:
    """Numeric features of a video collection.

    Attributes:
        samples: The samples, in caller order.
    """

    __slots__ = ()

    samples: tuple[Sample, ...]

    def numerize(self) -> Features:
        """Prepares deterministic numeric features from probed metadata.

        This method never decodes or samples frames. Frame-based features
        need an explicit decoder, sampling policy and resource budget; see
        `summarize`.

        Returns:
            One row of ten features per sample: duration in seconds, width,
            height, pixel count, aspect ratio, frame count, frame rate,
            audio track count, byte size and bytes per frame.

        Raises:
            VideoError: If a feature is not finite.
        """
        feature_names = (
            "duration_seconds",
            "width",
            "height",
            "pixel_count",
            "aspect_ratio",
            "frame_count",
            "frames_per_second",
            "audio_tracks",
            "byte_size",
            "bytes_per_frame",
        )
        rows = []
        for sample in self.samples:
            metadata = sample.metadata
            rows.append(
                (
                    metadata.duration_seconds,
                    float(metadata.width),
                    float(metadata.height),
                    float(metadata.width * metadata.height),
                    metadata.width / metadata.height,
                    float(metadata.frame_count),
                    metadata.frames_per_second,
                    float(metadata.audio_tracks),
                    float(sample.byte_size),
                    sample.byte_size / metadata.frame_count,
                )
            )
        values: numpy.ndarray = numpy.asarray(rows, dtype=numpy.float64)
        if not numpy.isfinite(values).all():
            raise VideoError(
                "video metadata features must be finite",
                {"code": "non_finite_features"},
            )
        return Features(
            sources=tuple(sample.source for sample in self.samples),
            feature_names=feature_names,
            values=tuple(
                tuple(float(value) for value in row) for row in values
            ),
        )

    def summarize(
        self,
        decoder: VideoFrameDecoder,
        max_frames: int = 8,
        max_frame_pixels: int = 1_000_000,
    ) -> Features:
        """Prepares bounded temporal statistics through an explicit decoder.

        The core package does not select or import a video codec. The caller
        supplies a decoder that accepts encoded bytes and an absolute frame
        index. The contract samples evenly spaced frames, validates each
        decoded array, and records the selected indices beside the features.
        Each sampled frame contributes mean, standard deviation, minimum,
        maximum, and an observation flag; absent frame slots are zero-filled.

        Args:
            decoder: Caller-owned frame decoder.
            max_frames: Maximum number of frames decoded per sample.
            max_frame_pixels: Maximum number of decoded array elements per
                frame, including channels.

        Returns:
            Numeric frame-statistics features with temporal provenance.

        Raises:
            TypeError: If ``decoder`` is not callable.
            ValueError: If ``max_frames`` or ``max_frame_pixels`` is below
                one.
            VideoError: If a decoded frame violates its metadata
                or cannot be converted to finite numeric values.
        """
        if not callable(decoder):
            raise errors.KindError("decoder must be callable")
        if max_frames < 1 or max_frame_pixels < 1:
            raise errors.InputError("frame bounds must be positive")
        slot_names = ("mean", "std", "min", "max", "observed")
        feature_names = tuple(
            f"frame[{slot}].{name}"
            for slot in range(max_frames)
            for name in slot_names
        )
        rows: list[tuple[float, ...]] = []
        sampled_indices: list[tuple[int, ...]] = []
        for sample in self.samples:
            indices = self.stride(sample, max_frames)
            values = numpy.zeros(
                max_frames * len(slot_names), dtype=numpy.float64
            )
            for slot, frame_index in enumerate(indices):
                numeric = measure(
                    sample, decoder, frame_index, max_frame_pixels
                )
                start = slot * len(slot_names)
                values[start : start + len(slot_names)] = (
                    float(numeric.mean()),
                    float(numeric.std()),
                    float(numpy.min(numeric)),
                    float(numpy.max(numeric)),
                    1.0,
                )
            rows.append(tuple(float(value) for value in values))
            sampled_indices.append(indices)
        return Features(
            sources=tuple(sample.source for sample in self.samples),
            feature_names=feature_names,
            values=tuple(rows),
            sampled_frame_indices=tuple(sampled_indices),
        )

    @staticmethod
    def stride(sample: Sample, max_frames: int) -> tuple[int, ...]:
        """Returns deterministic, evenly spaced frame indices for a sample.

        Args:
            sample: The video sample.
            max_frames: Largest number of indices to return; at least one.

        Returns:
            Ascending frame indices from the first to the last frame,
            numbering ``min(max_frames, frame_count)``.

        Raises:
            ValueError: If ``max_frames`` is below one.
        """
        if max_frames < 1:
            raise errors.InputError("max_frames must be positive")
        count = min(max_frames, sample.metadata.frame_count)
        return tuple(
            int(index)
            for index in numpy.linspace(
                0,
                sample.metadata.frame_count - 1,
                count,
                dtype=int,
            )
        )
