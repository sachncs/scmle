"""The video preparer."""

from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING
from typing import TypedDict

import numpy

from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.data.modalities import curation
from scmle.data.modalities import video
from scmle.models.modality_lifecycle import containers
from scmle.models.modality_lifecycle import curators

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from sklearn.linear_model import LogisticRegression


FRAME_LIMITATIONS = (
    (
        "Raw frame occlusion is bounded decoded-pixel evidence, "
        "not validated motion or semantic saliency."
    ),
    (
        "Audio-video synchronization and temporal event "
        "localization are not established."
    ),
    "Score changes are associational and not causal.",
)
BLANK_LIMITATIONS = (
    ("The caller-owned decoder defines frame pixels and codec semantics."),
    (
        "Frame masks do not establish motion, synchronization, or "
        "semantic localization."
    ),
    "Score changes are associational and not causal.",
)


class Slot(TypedDict):
    """One sampled frame of a feature-row occlusion."""

    frame_index: int
    time_seconds: float
    score_delta: float


class Blank(TypedDict):
    """One sampled frame of a decoded-frame occlusion."""

    frame_index: int
    time_seconds: float
    score_delta_from_full: float


class Footage(curators.Preparer):
    """Adapter for probed video metadata or an explicit frame decoder.

    Attributes:
        dataset: Immutable video dataset the adapter wraps.
        frame_decoder: Caller-supplied frame decoder, or None to use container
            metadata only.
        max_frames: Maximum number of frames sampled per video.
        max_frame_pixels: Maximum number of pixels per decoded frame.
        temporal_cutoff: Canonical observation time that separates training
            from later data, or None to split by group.
    """

    modality = modalities.Modality.VIDEO

    def __init__(
        self,
        dataset: video.Dataset,
        frame_decoder: video.VideoFrameDecoder | None = None,
        max_frames: int = 8,
        max_frame_pixels: int = 1_000_000,
        temporal_cutoff: datetime | str | None = None,
    ) -> None:
        """Binds a dataset, decoder, and optional explicit temporal policy.

        Args:
            dataset: Videos the lifecycle runs on.
            frame_decoder: Decoder that turns encoded bytes and a frame index
                into pixels; if None, only container metadata is used.
            max_frames: Maximum number of frames sampled per video.
            max_frame_pixels: Maximum number of pixels per decoded frame.
            temporal_cutoff: Time that separates training from later data, if
                the split must be temporal.

        Raises:
            ValueError: If `max_frames` or `max_frame_pixels` is below 1.
        """
        if max_frames < 1 or max_frame_pixels < 1:
            raise errors.InputError("frame bounds must be positive")
        self.dataset = dataset
        self.frame_decoder = frame_decoder
        self.max_frames = max_frames
        self.max_frame_pixels = max_frame_pixels
        self.temporal_cutoff = (
            video.canonical_observation_time(temporal_cutoff)
            if temporal_cutoff is not None
            else None
        )

    @property
    def dataset_hash(self) -> str:
        """The video corpus identity."""
        return self.dataset.content_hash

    @property
    def samples(self) -> tuple[video.Sample, ...]:
        """The video samples in the dataset."""
        return self.dataset.samples

    def curate(self) -> curation.Curation:
        """Builds encoding, shape, timing, and label evidence.

        Returns:
            The curation report for the dataset.
        """
        return self.dataset.curate()

    def split(self, seed: int) -> containers.Partitions:
        """Splits by subject/session/camera group, or by time with a cutoff.

        Args:
            seed: Random seed for the group split; unused for a temporal split.

        Returns:
            Train, validation and test partitions with their group IDs. With a
            temporal cutoff, validation is empty and test holds the later data.
        """
        if self.temporal_cutoff is not None:
            temporal_result = self.dataset.cutover(self.temporal_cutoff)
            return containers.Partitions(
                temporal_result.train,
                (),
                temporal_result.future,
                {
                    "train": groups(temporal_result.train),
                    "validation": (),
                    "test": groups(temporal_result.future),
                },
            )
        group_result = self.dataset.split(seed=seed)
        return containers.Partitions(
            group_result.train,
            group_result.validation,
            group_result.test,
            group_result.cohorts(),
        )

    def features(self, samples: Sequence[video.Sample]) -> containers.Batch:
        """Prepares metadata or explicitly decoded temporal features.

        Args:
            samples: Videos to featurize.

        Returns:
            Batch with one row per video, identified by its source; decoded
            frame statistics if a decoder is set, otherwise container metadata.
        """
        dataset = video.Dataset(tuple(samples))
        prepared = (
            dataset.summarize(
                self.frame_decoder,
                max_frames=self.max_frames,
                max_frame_pixels=self.max_frame_pixels,
            )
            if self.frame_decoder is not None
            else dataset.numerize()
        )
        return containers.Batch(
            tuple(prepared.sources), prepared.feature_names, prepared.to_numpy()
        )

    def sample_id(self, sample: video.Sample) -> str:
        """Gives the video source identity.

        Args:
            sample: Video sample.

        Returns:
            The sample's source.
        """
        return sample.source

    def label(self, sample: video.Sample) -> object | None:
        """Gives the video label.

        Args:
            sample: Video sample.

        Returns:
            The sample's label, or None if it has none.
        """
        return sample.label

    def monitor(self, observed: curators.Preparer) -> dict[str, object]:
        """Builds curation identity and metadata drift evidence.

        Args:
            observed: Adapter over the newly observed videos.

        Returns:
            Mapping with both dataset hashes, the sample count change (observed
            minus reference) and, per profile key, whether its value differs.

        Raises:
            TypeError: If `observed` is not a `Footage`.
        """
        if not isinstance(observed, Footage):
            raise errors.KindError("observed adapter must be a Footage")
        reference = self.dataset.curate()
        current = observed.dataset.curate()
        return curators.drift(reference, current)

    def remarks(self) -> tuple[str, ...]:
        """Describes video lifecycle limits.

        Returns:
            Plain statements of the representation in use and what it does not
            cover.
        """
        representation = (
            "The baseline uses caller-decoded, evenly sampled frame "
            "statistics with a bounded pixel budget."
            if self.frame_decoder is not None
            else "The baseline uses probed container metadata and never "
            "silently decodes frames."
        )
        return (
            representation,
            (
                "Temporal, audio-video synchronization, and frame-local "
                "explanations require a decoder policy."
            ),
        )

    def elucidate(
        self,
        sample: Any,
        model: LogisticRegression,
        feature_names: Sequence[str],  # noqa: ARG002 -- overrides Preparer.elucidate
        coefficients: numpy.ndarray,  # noqa: ARG002 -- overrides Preparer.elucidate
    ) -> dict[str, object]:
        """Builds bounded temporal frame-statistics occlusion evidence.

        Args:
            sample: Video sample to explain.
            model: Fitted classifier.
            feature_names: Names of the model's features (unused).
            coefficients: Linear coefficients (unused).

        Returns:
            Mapping with the score change (zeroed minus full) per sampled frame,
            largest absolute change first, plus raw-frame occlusion evidence; or
            status "not_available" if no frame decoder is set.

        Raises:
            TypeError: If `sample` is not a `Sample`.
        """
        if not isinstance(sample, video.Sample):
            raise errors.KindError("sample must be a Sample")
        if self.frame_decoder is None:
            return curators.unavailable(
                "temporal_frame_statistics_occlusion",
                "an explicit frame decoder is required",
            )
        prepared = self.features((sample,))
        frame_indices = self.dataset.stride(sample, self.max_frames)
        full_score, terms = occlusion(
            sample, model, prepared.values[0], frame_indices
        )
        raw_windows = self.blank(sample, model, full_score)
        return {
            "status": "available",
            "method": "temporal_frame_statistics_occlusion",
            "terms": terms,
            "sampled_frame_indices": list(frame_indices),
            "feature_space_only": True,
            "raw_input_occlusion": raw_windows,
            "limitations": list(FRAME_LIMITATIONS),
        }

    def blank(
        self,
        sample: video.Sample,
        model: LogisticRegression,
        original_score: float,
    ) -> dict[str, object]:
        """Scores sampled decoded frames after replacing each with zeros.

        Args:
            sample: Video sample to explain.
            model: Fitted classifier.
            original_score: Decision score with no frame masked.

        Returns:
            Mapping with one entry per sampled frame holding its index, time in
            seconds and score change (masked minus original); or status
            "not_available" if no frame decoder is set.
        """
        if self.frame_decoder is None:
            return curators.unavailable(
                "raw_video_frame_occlusion",
                "an explicit frame decoder is required",
            )
        frame_indices = self.dataset.stride(sample, self.max_frames)
        windows = frames(self, sample, model, frame_indices, original_score)
        return {
            "status": "available",
            "method": "raw_video_frame_occlusion",
            "windows": windows,
            "sampled_frame_indices": list(frame_indices),
            "baseline": "zero-valued decoded source frames",
            "feature_space_only": False,
            "limitations": list(BLANK_LIMITATIONS),
        }


def groups(samples: Sequence[video.Sample]) -> tuple[str, ...]:
    """Names the groups of a temporal partition.

    Args:
        samples: Videos of one partition.

    Returns:
        The sorted group IDs, a video without one counting by its source.
    """
    return tuple(
        sorted({sample.group_id or sample.source for sample in samples})
    )


def occlusion(
    sample: video.Sample,
    model: LogisticRegression,
    values: numpy.ndarray,
    frame_indices: Sequence[int],
) -> tuple[float, list[Slot]]:
    """Scores the feature row with each sampled frame's statistics zeroed.

    Args:
        sample: Video sample to explain.
        model: Fitted classifier.
        values: Feature row of the sample; five statistics per frame slot.
        frame_indices: Indices of the sampled frames, in slot order.

    Returns:
        The decision score of the full row, and per sampled frame its index,
        time in seconds and score change (zeroed minus full), largest
        absolute change first and ties by frame index.
    """
    full_score = float(
        numpy.asarray(model.decision_function(values[None])).reshape(-1)[0]
    )
    terms: list[Slot] = []
    slot_width = 5
    for slot, frame_index in enumerate(frame_indices):
        occluded = values.copy()
        start = slot * slot_width
        occluded[start : start + slot_width] = 0.0
        score = float(
            numpy.asarray(model.decision_function(occluded[None])).reshape(-1)[
                0
            ]
        )
        terms.append(
            {
                "frame_index": frame_index,
                "time_seconds": frame_index / sample.metadata.frames_per_second,
                "score_delta": score - full_score,
            }
        )
    terms.sort(
        key=lambda item: (
            -abs(float(item["score_delta"])),
            int(item["frame_index"]),
        )
    )
    return full_score, terms


def frames(
    preparer: Footage,
    sample: video.Sample,
    model: LogisticRegression,
    frame_indices: Sequence[int],
    original_score: float,
) -> list[Blank]:
    """Scores the sample with each sampled frame decoded as zeros.

    Args:
        preparer: Preparer whose decoder and frame bounds apply.
        sample: Video sample to explain.
        model: Fitted classifier.
        frame_indices: Indices of the sampled frames.
        original_score: Decision score with no frame masked.

    Returns:
        Per sampled frame its index, time in seconds and score change
        (masked minus original).

    Raises:
        RuntimeError: If the preparer has no frame decoder.
    """
    decoder = preparer.frame_decoder
    if decoder is None:
        raise errors.OrderError("an explicit frame decoder is required")
    windows: list[Blank] = []
    for outcomes in frame_indices:

        def blackout(
            payload: bytes,
            frame_index: int,
            target_frame_index: int = outcomes,
        ) -> numpy.ndarray:
            frame = numpy.asarray(decoder(payload, frame_index))
            if frame_index == target_frame_index:
                return numpy.zeros_like(frame)
            return frame

        prepared = video.Dataset((sample,)).summarize(
            blackout,
            max_frames=preparer.max_frames,
            max_frame_pixels=preparer.max_frame_pixels,
        )
        values = prepared.to_numpy()[0]
        score = float(
            numpy.asarray(model.decision_function(values[None])).reshape(-1)[0]
        )
        windows.append(
            {
                "frame_index": outcomes,
                "time_seconds": outcomes / sample.metadata.frames_per_second,
                "score_delta_from_full": score - original_score,
            }
        )
    return windows
