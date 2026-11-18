"""The WAV-capable audio preparer."""

from __future__ import annotations

import io
import math
from typing import Any
from typing import TYPE_CHECKING
from typing import TypedDict
import wave

import numpy

from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.data.modalities import audio
from scmle.data.modalities import curation
from scmle.pipeline.modality_lifecycle import constants
from scmle.pipeline.modality_lifecycle import containers
from scmle.pipeline.modality_lifecycle import curators

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sklearn.linear_model import LogisticRegression


REMARKS = (
    (
        "Signal features are bounded WAV statistics; non-WAV codecs "
        "require an explicit decoder adapter."
    ),
    (
        "The baseline does not claim transcript, speaker, or "
        "time-local explanations."
    ),
)
WINDOW_LIMITATIONS = (
    (
        "Raw waveform windows are bounded occlusion evidence, not "
        "validated waveform saliency."
    ),
    ("The score is associational and does not establish a causal audio event."),
)
NOISE_LIMITATIONS = (
    (
        "Synthetic white noise is a repeatable stressor, not a "
        "model of every recording environment."
    ),
    ("Score changes do not establish perceptual quality or causal robustness."),
    (
        "Clipping is measured after mixing and must be reviewed "
        "before interpreting low-SNR results."
    ),
)
SILENCE_LIMITATIONS = (
    ("The model consumes bounded aggregate signal features after masking."),
    (
        "This does not establish transcript, speaker, event, or "
        "causal attribution."
    ),
)


class Span(TypedDict):
    """One time window of a WAV sample and its score."""

    start_seconds: float
    end_seconds: float
    start_frame: int
    frames_observed: int
    score: float
    score_delta_from_full: float


class AudioPreparer(curators.Preparer):
    """Adapter for WAV-capable audio samples and explicit codec probes.

    Attributes:
        dataset: Immutable audio dataset the adapter wraps.
    """

    modality = modalities.Modality.AUDIO

    def __init__(self, dataset: audio.Dataset) -> None:
        """Binds an immutable audio dataset.

        Args:
            dataset: Audio the lifecycle runs on.
        """
        self.dataset = dataset

    @property
    def digest(self) -> str:
        """The audio corpus identity."""
        return self.dataset.content_hash

    @property
    def samples(self) -> tuple[audio.Sample, ...]:
        """The audio samples in the dataset."""
        return self.dataset.samples

    def curate(self) -> curation.Curation:
        """Builds codec, signal shape, duration, and label evidence.

        Returns:
            The curation report for the dataset.
        """
        return self.dataset.curate()

    def split(self, seed: int) -> containers.Partitions:
        """Splits by speaker/session group.

        Args:
            seed: Random seed for the split.

        Returns:
            Train, validation and test partitions, with the dataset's group IDs
            under `groups`.
        """
        result = self.dataset.split(seed=seed)
        return containers.Partitions(
            result.train, result.validation, result.test, result.cohorts()
        )

    def features(self, samples: Sequence[audio.Sample]) -> containers.Batch:
        """Prepares bounded WAV signal statistics.

        Args:
            samples: Audio samples to featurize.

        Returns:
            Batch with one row per sample, identified by its source.
        """
        prepared = audio.Dataset(tuple(samples)).prepare_features()
        return containers.Batch(
            tuple(prepared.sources), prepared.feature_names, prepared.to_numpy()
        )

    def sample_id(self, sample: audio.Sample) -> str:
        """Gives the audio source identity.

        Args:
            sample: Audio sample.

        Returns:
            The sample's source.
        """
        return sample.source

    def label(self, sample: audio.Sample) -> object | None:
        """Gives the audio label.

        Args:
            sample: Audio sample.

        Returns:
            The sample's label, or None if it has none.
        """
        return sample.label

    def monitor(self, observed: curators.Preparer) -> dict[str, object]:
        """Builds curation identity and profile drift evidence.

        Args:
            observed: Adapter over the newly observed audio.

        Returns:
            Mapping with both dataset hashes, the sample count change (observed
            minus reference) and, per profile key, whether its value differs.

        Raises:
            TypeError: If `observed` is not an `AudioPreparer`.
        """
        if not isinstance(observed, AudioPreparer):
            raise errors.KindError("observed adapter must be an AudioPreparer")
        current = observed.dataset.curate()
        reference = self.dataset.curate()
        return curators.drift(reference, current)

    def remarks(self) -> tuple[str, ...]:
        """Describes audio lifecycle limits.

        Returns:
            Plain statements of what the representation does not claim.
        """
        return REMARKS

    def elucidate(
        self,
        sample: Any,
        model: LogisticRegression,
        feature_names: Sequence[str],  # noqa: ARG002 -- overrides Preparer.elucidate
        coefficients: numpy.ndarray,  # noqa: ARG002 -- overrides Preparer.elucidate
    ) -> dict[str, object]:
        """Builds bounded time-local score evidence for WAV samples.

        Args:
            sample: Audio sample to explain.
            model: Fitted classifier.
            feature_names: Names of the model's features (unused).
            coefficients: Linear coefficients (unused).

        Returns:
            Mapping with the model's decision score for up to eight consecutive
            time windows of the sample, raw-waveform occlusion scores and a
            noise robustness probe; or a mapping with status "not_available" if
            the sample is not WAV.

        Raises:
            TypeError: If `sample` is not an `Sample`.
        """
        if not isinstance(sample, audio.Sample):
            raise errors.KindError("sample must be an Sample")
        if sample.metadata.format.lower() != "wav":
            return curators.unavailable(
                "time_local_wav_window_score",
                "an explicit WAV decoder is required",
            )
        window_frames, full_score, windows = timeline(
            sample, model, self.features((sample,)).values[0]
        )
        raw_windows = self.silence(sample, model, full_score, window_frames)
        noise_robustness = self.background_noise_robustness(
            sample, model, full_score
        )
        return {
            "status": "available",
            "method": "time_local_wav_window_score",
            "window_count": len(windows),
            "windows": windows,
            "feature_space_only": True,
            "raw_input_occlusion": raw_windows,
            "background_noise_robustness": noise_robustness,
            "limitations": list(WINDOW_LIMITATIONS),
        }

    def background_noise_robustness(
        self,
        sample: audio.Sample,
        model: LogisticRegression,
        original_score: float,
        snr_db: tuple[float, ...] = (20.0, 10.0, 0.0),
        seed: int = 0,
    ) -> dict[str, object]:
        """Measures bounded score changes under deterministic additive noise.

        This is a reproducible perturbation probe for the current linear
        baseline. It is not a perceptual speech-quality test, a denoising
        system, or evidence that the model is robust to real recording noise.

        Args:
            sample: WAV audio sample to perturb.
            model: Fitted classifier.
            original_score: Decision score of the unperturbed sample.
            snr_db: Requested signal-to-noise ratios in decibels; one to eight
                finite values in [0, 120].
            seed: Seed of the noise generator.

        Returns:
            Mapping with one scenario per requested ratio holding the observed
            ratio, the score, its change from `original_score` and the share
            of clipped values; or status "not_available" if the sample is not
            WAV or is silent.

        Raises:
            ValueError: If `snr_db` is empty, has more than eight values, or
                holds a value outside [0, 120].
        """
        if sample.metadata.format.lower() != "wav":
            return curators.unavailable(
                "deterministic_additive_background_noise",
                "an explicit WAV decoder is required",
            )
        if not snr_db or len(snr_db) > constants.MAX_SNR_VALUES:
            raise errors.InputError(
                "snr_db must contain between one and eight values"
            )
        if any(
            not numpy.isfinite(value)
            or value < 0.0
            or value > constants.MAX_SNR_DB
            for value in snr_db
        ):
            raise errors.InputError(
                "snr_db values must be finite and in [0, 120]"
            )
        parameters, values = audio.pcm(sample.encoded_bytes)
        signal_rms = float(numpy.sqrt(numpy.mean(numpy.square(values))))
        if signal_rms <= 0.0:
            return curators.unavailable(
                "deterministic_additive_background_noise",
                "source signal is silent",
            )
        randomizer = numpy.random.default_rng(seed)
        scenarios = [
            scenario(
                self,
                sample,
                model,
                (parameters, values, signal_rms),
                (requested_snr, randomizer, original_score),
            )
            for requested_snr in snr_db
        ]
        return {
            "status": "available",
            "method": "deterministic_additive_background_noise",
            "seed": seed,
            "scenarios": scenarios,
            "feature_space_only": False,
            "limitations": list(NOISE_LIMITATIONS),
        }

    def silence(
        self,
        sample: audio.Sample,
        model: LogisticRegression,
        original_score: float,
        window_frames: int,
    ) -> dict[str, object]:
        """Scores source PCM windows after replacing them with silence.

        Args:
            sample: WAV audio sample to explain.
            model: Fitted classifier.
            original_score: Decision score of the unmasked sample.
            window_frames: Number of PCM frames in each window.

        Returns:
            Mapping with one entry per window holding its frame and second
            bounds and the score change (masked score minus original); or
            status "not_available" if the sample width is unknown.
        """
        if sample.metadata.sample_width is None:
            return curators.unavailable(
                "raw_wav_waveform_window_occlusion",
                "sample width is required to mask PCM frames",
            )
        with wave.open(io.BytesIO(sample.encoded_bytes), "rb") as opened:
            parameters = opened.getparams()
            raw_frames = bytearray(
                opened.readframes(sample.metadata.frame_count)
            )
        windows: list[dict[str, object]] = []
        for start_frame in range(0, sample.metadata.frame_count, window_frames):
            end_frame = min(
                sample.metadata.frame_count, start_frame + window_frames
            )
            masked_sample = mute(
                sample, parameters, raw_frames, (start_frame, end_frame)
            )
            features = self.features((masked_sample,)).values
            score = float(
                numpy.asarray(model.decision_function(features)).reshape(-1)[0]
            )
            windows.append(
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_seconds": start_frame / sample.metadata.sample_rate,
                    "end_seconds": end_frame / sample.metadata.sample_rate,
                    "score_delta_from_full": score - original_score,
                }
            )
        return {
            "status": "available",
            "method": "raw_wav_waveform_window_occlusion",
            "windows": windows,
            "sample_rate": sample.metadata.sample_rate,
            "frame_count": sample.metadata.frame_count,
            "baseline": "silence replacement in source PCM frames",
            "feature_space_only": False,
            "limitations": list(SILENCE_LIMITATIONS),
        }


def timeline(
    sample: audio.Sample,
    model: LogisticRegression,
    full_features: numpy.ndarray,
) -> tuple[int, float, list[Span]]:
    """Scores up to eight consecutive time windows of one WAV sample.

    Args:
        sample: WAV audio sample to explain.
        model: Fitted classifier.
        full_features: Feature row of the whole sample.

    Returns:
        The number of frames per window, the decision score of the whole
        sample, and per window its bounds in seconds and frames, its score
        and its score change from the whole sample.
    """
    window_count = min(8, sample.metadata.frame_count)
    window_frames = max(
        1, int(numpy.ceil(sample.metadata.frame_count / window_count))
    )
    full_decision = numpy.asarray(model.decision_function(full_features[None]))
    full_score = float(full_decision.reshape(-1)[0])
    windows: list[Span] = []
    for start_frame in range(0, sample.metadata.frame_count, window_frames):
        prepared = audio.Dataset((sample,)).prepare_features(
            max_frames=window_frames,
            start_frame=start_frame,
        )
        values = prepared.to_numpy()[0]
        decision = numpy.asarray(model.decision_function(values[None]))
        windows.append(
            {
                "start_seconds": start_frame / sample.metadata.sample_rate,
                "end_seconds": min(
                    sample.metadata.frame_count,
                    start_frame + prepared.frames_observed[0],
                )
                / sample.metadata.sample_rate,
                "start_frame": start_frame,
                "frames_observed": prepared.frames_observed[0],
                "score": float(decision.reshape(-1)[0]),
                "score_delta_from_full": float(
                    decision.reshape(-1)[0] - full_score
                ),
            }
        )
    return window_frames, full_score, windows


def scenario(
    preparer: AudioPreparer,
    sample: audio.Sample,
    model: LogisticRegression,
    signal: tuple[Any, numpy.ndarray, float],
    noise: tuple[float, numpy.random.Generator, float],
) -> dict[str, object]:
    """Scores one sample after mixing in white noise at one ratio.

    Args:
        preparer: Preparer that featurizes the noisy sample.
        sample: WAV audio sample to perturb.
        model: Fitted classifier.
        signal: The sample's WAV parameters, PCM values and RMS level.
        noise: The requested signal-to-noise ratio in decibels, the noise
            generator, and the decision score of the clean sample.

    Returns:
        The requested and observed ratios, the score, its change from the
        clean score and the share of clipped values.
    """
    parameters, values, signal_rms = signal
    requested_snr, randomizer, original_score = noise
    noise_rms = signal_rms / (10.0 ** (requested_snr / 20.0))
    mixed = values + randomizer.normal(0.0, noise_rms, size=values.shape)
    clipping_fraction = float(numpy.mean(numpy.abs(mixed) > 1.0))
    payload = audio.wav_payload_from_pcm(parameters, mixed)
    perturbed = audio.Sample.from_bytes(
        payload,
        source=f"{sample.source}#noise-{requested_snr:g}",
        label=sample.label,
        group_id=sample.group_id,
        transcript=sample.transcript,
    )
    features = preparer.features((perturbed,)).values
    score = float(
        numpy.asarray(model.decision_function(features)).reshape(-1)[0]
    )
    observed_noise = mixed - values
    observed_noise_rms = float(
        numpy.sqrt(numpy.mean(numpy.square(observed_noise)))
    )
    observed_snr = 20.0 * math.log10(signal_rms / observed_noise_rms)
    return {
        "requested_snr_db": requested_snr,
        "observed_snr_db": observed_snr,
        "score": score,
        "score_delta_from_clean": score - original_score,
        "clipping_fraction": clipping_fraction,
    }


def mute(
    sample: audio.Sample,
    parameters: Any,
    raw_frames: bytearray,
    window: tuple[int, int],
) -> audio.Sample:
    """Replaces one window of PCM frames with silence.

    Args:
        sample: WAV audio sample to mask.
        parameters: The sample's WAV parameters.
        raw_frames: The sample's PCM frames; left unchanged.
        window: First frame and the frame after the last.

    Returns:
        A sample with the same source, label and group whose frames in the
        window are silent.
    """
    start_frame, end_frame = window
    frame_width = parameters.sampwidth * parameters.nchannels
    silence_value = 128 if parameters.sampwidth == 1 else 0
    masked_frames = raw_frames.copy()
    masked_frames[start_frame * frame_width : end_frame * frame_width] = (
        bytes([silence_value]) * frame_width * (end_frame - start_frame)
    )
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setparams(parameters)
        writer.writeframes(bytes(masked_frames))
    return audio.Sample.from_bytes(
        payload.getvalue(),
        source=sample.source,
        label=sample.label,
        group_id=sample.group_id,
    )
