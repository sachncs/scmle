"""Standard-library WAV decoding, statistics and encoding."""

import io
import wave

import numpy

from scmle.contracts import errors
from scmle.data.modalities.audio.metadata import Metadata
from scmle.data.modalities.audio.metadata import ValidationError

WIDTH_16_BIT = 2
WIDTH_24_BIT = 3
WIDTH_32_BIT = 4
CLIPPING_LEVEL = 0.999
SILENCE_LEVEL = 0.01
MATRIX_DIMENSIONS = 2


def parse(encoded_bytes: bytes) -> Metadata:
    """Reads WAV metadata using the standard library.

    Args:
        encoded_bytes: A complete WAV file.

    Returns:
        The format (``wav``), sample rate, channel count, frame count and
        sample width in bytes.

    Raises:
        ValidationError: If the payload is not a readable WAV file.
    """
    try:
        with wave.open(io.BytesIO(encoded_bytes), "rb") as opened:
            return Metadata(
                format="wav",
                sample_rate=opened.getframerate(),
                channels=opened.getnchannels(),
                frame_count=opened.getnframes(),
                sample_width=opened.getsampwidth(),
            )
    except (EOFError, wave.Error) as error:
        raise ValidationError(
            "audio payload is not a readable WAV file",
            {"code": "invalid_wav"},
        ) from error


def scaled(raw: bytes, sample_width: int) -> tuple[numpy.ndarray, float]:
    """Decodes little-endian PCM bytes to samples scaled to [-1, 1].

    Args:
        raw: Whole frames of PCM bytes.
        sample_width: Bytes per sample.

    Returns:
        The samples as a float64 array and the PCM scale they were divided
        by (1.0 for 8-bit samples, which are offset by 128 and divided by
        128).

    Raises:
        ValidationError: If the sample width is not 1, 2, 3 or 4 bytes.
    """
    if sample_width == 1:
        samples = numpy.frombuffer(raw, dtype=numpy.uint8).astype(numpy.float64)
        samples = (samples - 128.0) / 128.0
        scale = 1.0
    elif sample_width in (2, 4):
        dtype = numpy.dtype("<i2" if sample_width == WIDTH_16_BIT else "<i4")
        samples = numpy.frombuffer(raw, dtype=dtype).astype(numpy.float64)
        scale = float(2 ** (sample_width * 8 - 1))
        samples /= scale
    elif sample_width == WIDTH_24_BIT:
        packed = numpy.frombuffer(raw, dtype=numpy.uint8).reshape(-1, 3)
        unsigned = (
            packed[:, 0].astype(numpy.int64)
            | (packed[:, 1].astype(numpy.int64) << 8)
            | (packed[:, 2].astype(numpy.int64) << 16)
        )
        signed = unsigned - ((unsigned & 0x800000) << 1)
        samples = signed.astype(numpy.float64)
        scale = float(2**23)
        samples /= scale
    else:
        raise ValidationError(
            "WAV sample width is unsupported",
            {"code": "unsupported_sample_width", "sample_width": sample_width},
        )
    return samples, scale


def statistics(
    samples: numpy.ndarray,
    channels: int,
    sample_rate: int,
    frame_count: int,
    scale: float,
) -> tuple[float, ...]:
    """Computes the signal statistics of a window of scaled samples.

    Args:
        samples: Interleaved samples scaled to [-1, 1].
        channels: Number of channels.
        sample_rate: Frames per second.
        frame_count: Number of frames in the window.
        scale: PCM scale the samples were divided by.

    Returns:
        Mean, mean absolute value, RMS, peak absolute value, share of
        samples at or above 0.999 in magnitude, share at or below 0.01,
        window duration in seconds, frame count and PCM scale.

    Raises:
        ValidationError: If a statistic is not finite.
    """
    matrix = samples.reshape(-1, channels)
    absolute = numpy.abs(matrix)
    frame_values = numpy.asarray(
        [
            float(matrix.mean()),
            float(absolute.mean()),
            float(numpy.sqrt(numpy.mean(numpy.square(matrix)))),
            float(absolute.max()),
            float(numpy.mean(absolute >= CLIPPING_LEVEL)),
            float(numpy.mean(absolute <= SILENCE_LEVEL)),
            float(frame_count / sample_rate),
            float(frame_count),
            float(scale),
        ],
        dtype=numpy.float64,
    )
    if not numpy.isfinite(frame_values).all():
        raise ValidationError(
            "WAV signal features must be finite",
            {"code": "non_finite_features"},
        )
    return tuple(float(value) for value in frame_values)


def extract(
    encoded_bytes: bytes,
    max_frames: int,
    start_frame: int = 0,
) -> tuple[tuple[float, ...], int]:
    """Extracts bounded, normalized PCM statistics from one WAV payload.

    The returned statistics describe the bounded window beginning at
    'start_frame'. Callers receive the observed frame count so a truncated
    boundary cannot be mistaken for a full-clip measurement.

    Args:
        encoded_bytes: A complete WAV file.
        max_frames: Largest number of frames to read; at least one.
        start_frame: Index of the first frame to read; not negative.

    Returns:
        A pair. The first item holds nine statistics of the window, with
        samples scaled to [-1, 1]: mean, mean absolute value, RMS, peak
        absolute value, share of samples at or above 0.999 in magnitude,
        share at or below 0.01, window duration in seconds, frame count
        and PCM scale. The second item is the number of frames read.

    Raises:
        ValueError: If ``max_frames`` is below one or ``start_frame`` is
            negative.
        ValidationError: If the payload is not a readable WAV file,
            ``start_frame`` is at or beyond the last frame, the window has
            no complete frame, the sample width is not 1, 2, 3 or 4 bytes,
            or a statistic is not finite.
    """
    if max_frames < 1:
        raise errors.InputError("max_frames must be at least one")
    if start_frame < 0:
        raise errors.InputError("start_frame must not be negative")
    try:
        with wave.open(io.BytesIO(encoded_bytes), "rb") as opened:
            sample_width = opened.getsampwidth()
            channels = opened.getnchannels()
            sample_rate = opened.getframerate()
            if start_frame >= opened.getnframes():
                raise ValidationError(
                    "audio window starts beyond the available frames",
                    {
                        "code": "window_start_out_of_range",
                        "start_frame": start_frame,
                    },
                )
            opened.setpos(start_frame)
            requested_frames = min(
                opened.getnframes() - start_frame,
                max_frames,
            )
            raw = opened.readframes(requested_frames)
    except (EOFError, wave.Error) as error:
        raise ValidationError(
            "audio payload is not a readable WAV file",
            {"code": "invalid_wav"},
        ) from error

    frame_width = sample_width * channels
    complete_bytes = len(raw) - (len(raw) % frame_width)
    raw = raw[:complete_bytes]
    if not raw:
        raise ValidationError(
            "WAV payload contains no complete audio frames",
            {"code": "empty_audio_frames"},
        )
    frame_count = len(raw) // frame_width
    samples, scale = scaled(raw, sample_width)

    return (
        statistics(samples, channels, sample_rate, frame_count, scale),
        frame_count,
    )


def pcm(
    encoded_bytes: bytes,
    limit: int | None = None,
) -> tuple[tuple[int, int, int, int, str, str], numpy.ndarray]:
    """Decodes WAV PCM values and retain the writer parameters.

    Values are normalized to ``[-1, 1]`` and shaped as frames by channels;
    a dependency-light primitive for perturbation probes, not a decoder.

    Args:
        encoded_bytes: A complete WAV file.
        limit: Frames decoded from the first (all when None), so a probe
            scoring a prefix holds only the prefix.

    Returns:
        A pair. The first item is the writer parameters as channels, sample
        width in bytes, frame rate, frame count, compression type and
        compression name. The second is a float64 array of shape
        ``(frames, channels)`` with values in [-1, 1].

    Raises:
        ValidationError: If the payload is not a readable WAV file,
            holds no complete frame, has an unsupported sample width, or
            decodes to non-finite values.
    """
    try:
        with wave.open(io.BytesIO(encoded_bytes), "rb") as opened:
            parameters = (
                opened.getnchannels(),
                opened.getsampwidth(),
                opened.getframerate(),
                opened.getnframes(),
                opened.getcomptype(),
                opened.getcompname(),
            )
            raw = opened.readframes(limit or opened.getnframes())
    except (EOFError, wave.Error) as error:
        raise ValidationError(
            "audio payload is not a readable WAV file",
            {"code": "invalid_wav"},
        ) from error
    channels = parameters[0]
    sample_width = parameters[1]
    frame_width = sample_width * channels
    complete_bytes = len(raw) - (len(raw) % frame_width)
    raw = raw[:complete_bytes]
    if not raw:
        raise ValidationError(
            "WAV payload contains no complete audio frames",
            {"code": "empty_audio_frames"},
        )
    if sample_width == 1:
        samples = numpy.frombuffer(raw, dtype=numpy.uint8).astype(numpy.float64)
        samples = (samples - 128.0) / 128.0
    elif sample_width in (2, 4):
        dtype = numpy.dtype("<i2" if sample_width == WIDTH_16_BIT else "<i4")
        samples = numpy.frombuffer(raw, dtype=dtype).astype(numpy.float64)
        samples /= float(2 ** (sample_width * 8 - 1))
    elif sample_width == WIDTH_24_BIT:
        packed = numpy.frombuffer(raw, dtype=numpy.uint8).reshape(-1, 3)
        unsigned = (
            packed[:, 0].astype(numpy.int64)
            | (packed[:, 1].astype(numpy.int64) << 8)
            | (packed[:, 2].astype(numpy.int64) << 16)
        )
        signed = unsigned - ((unsigned & 0x800000) << 1)
        samples = signed.astype(numpy.float64) / float(2**23)
    else:
        raise ValidationError(
            "WAV sample width is unsupported",
            {"code": "unsupported_sample_width", "sample_width": sample_width},
        )
    values = samples.reshape(-1, channels)
    if not numpy.isfinite(values).all():
        raise ValidationError(
            "WAV PCM values must be finite",
            {"code": "non_finite_pcm"},
        )
    return parameters, values


def encode(
    parameters: tuple[int, int, int, int, str, str], values: numpy.ndarray
) -> bytes:
    """Encodes normalized PCM values with the supplied WAV parameters.

    Values are clipped to [-1, 1] and rounded to the sample width given in
    ``parameters``.

    Args:
        parameters: Writer parameters in the layout `pcm`
            returns: channels, sample width in bytes, frame rate, frame
            count, compression type and compression name.
        values: Frame-by-channel matrix of normalized values.

    Returns:
        The bytes of a WAV file holding the values.

    Raises:
        ValueError: If ``values`` is not a finite two-dimensional matrix.
        ValidationError: If the sample width is not 1, 2, 3 or 4.
    """
    matrix = numpy.asarray(values, dtype=numpy.float64)
    if matrix.ndim != MATRIX_DIMENSIONS or not numpy.isfinite(matrix).all():
        raise errors.InputError(
            "PCM values must be a finite frame-by-channel matrix"
        )
    sample_width = parameters[1]
    clipped = numpy.clip(matrix, -1.0, 1.0)
    if sample_width == 1:
        encoded = numpy.rint(clipped * 127.0 + 128.0).astype(numpy.uint8)
    elif sample_width == WIDTH_16_BIT:
        encoded = numpy.rint(clipped * 32767.0).astype("<i2")
    elif sample_width == WIDTH_24_BIT:
        integer = numpy.rint(clipped * (2**23 - 1)).astype(numpy.int32)
        unsigned = integer.astype(numpy.int64) & 0xFFFFFF
        encoded = numpy.empty((*integer.shape, 3), dtype=numpy.uint8)
        encoded[..., 0] = unsigned & 0xFF
        encoded[..., 1] = (unsigned >> 8) & 0xFF
        encoded[..., 2] = (unsigned >> 16) & 0xFF
    elif sample_width == WIDTH_32_BIT:
        encoded = numpy.rint(clipped * (2**31 - 1)).astype("<i4")
    else:
        raise ValidationError(
            "WAV sample width is unsupported",
            {"code": "unsupported_sample_width", "sample_width": sample_width},
        )
    raw = encoded.tobytes()
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setparams(parameters)
        writer.writeframes(raw)
    return payload.getvalue()
