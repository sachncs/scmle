"""Whole-clip audio description, read in bounded chunks.

`describe` reads a clip once, `CHUNK` frames at a time, so the memory it
holds is bounded by the chunk whatever the clip's length (register row
MM-6), and every frame of the clip counts towards every statistic (MM-13:
nothing stops at a prefix). It returns the clip's features, the digest of
its decoded content (exact duplicates) and its band-energy envelope, whose
distance to another clip's measures how alike the two sound
(near-duplicates).

WAV is read with the standard library. Any other container is read through
the optional `soundfile` library; without it such a clip raises
`DecoderError` naming the library.
"""

from collections.abc import Iterator
import dataclasses
import hashlib
import importlib
import math
import pathlib
from typing import Any
import wave

import numpy

from scmle.data.modalities.audio.metadata import DecoderError
from scmle.data.modalities.audio.metadata import ValidationError
from scmle.data.modalities.audio.wav import CLIPPING_LEVEL
from scmle.data.modalities.audio.wav import scaled
from scmle.data.modalities.audio.wav import SILENCE_LEVEL

CHUNK = 65_536
"""Frames decoded per step. 64 Ki frames of 8 channels as float64 is 4 MiB,
so the working set of one clip is a few MiB at any length (MM-6)."""

WINDOW = 2_048
"""Samples per analysis frame. At 16 kHz that is 128 ms with 7.8 Hz bins,
over an order of magnitude finer than the narrowest band (125 Hz wide)."""

EDGES = (125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0)
"""Band edges in Hz: the octave bands of IEC 61260 from 125 Hz, plus the
bands below the first and above the last edge (eight bands)."""

PRINT = (500.0, 1_000.0, 2_000.0)
"""Envelope band edges in Hz: four bands, split at the octaves that hold
most speech and music energy."""

SEGMENTS = 16
"""Time segments of the envelope, each a sixteenth of the clip."""

CELLS = SEGMENTS * (len(PRINT) + 1)
"""Cells of an envelope: segments times bands."""

NOISE = 20.0
"""Signal-to-noise ratio in dB down to which a noisy copy of a clip is the
same recording. 20 dB is where speech intelligibility is no longer limited
by the noise (ANSI S3.5), so a copy that clean carries the same content."""

FLOOR = 20.0
"""dB below the loudest cell where an envelope is floored, equal to
`NOISE`: quieter cells are where noise at `NOISE` dB could dominate, so
they carry no identity."""

RADIUS = 10.0 * math.log10(1.0 + 10.0 ** ((FLOOR - NOISE) / 10.0))
"""Largest shift in dB that noise `NOISE` dB below the signal can give an
unfloored cell: power adds, so a cell at the floor rises by at most
10 log10(1 + 10^((FLOOR - NOISE) / 10)), 3.01 dB. Two envelopes whose
root-mean-square difference is within it are near-duplicates."""

TINY = 1e-12
"""Power floor that keeps logarithms of silent frames finite."""


@dataclasses.dataclass(frozen=True)
class Header:
    """What a clip's container says about its samples.

    Attributes:
        rate: Frames per second.
        channels: Channels per frame.
        width: Bytes per sample; 0 when the codec does not say.
        frames: Frames in the clip.
    """

    rate: int
    channels: int
    width: int
    frames: int


class Wave:
    """A WAV clip read with the standard library.

    Attributes:
        path: The clip's file.
    """

    def __init__(self, path: pathlib.Path) -> None:
        """Binds the file.

        Args:
            path: The clip's file.
        """
        self.path = path

    def header(self) -> Header:
        """Reads the header.

        Returns:
            The rate, channels, sample width and frame count.

        Raises:
            ValidationError: If the file is not a readable WAV file.
        """
        try:
            with wave.open(str(self.path), "rb") as opened:
                return Header(
                    opened.getframerate(),
                    opened.getnchannels(),
                    opened.getsampwidth(),
                    opened.getnframes(),
                )
        except (EOFError, wave.Error) as error:
            raise ValidationError(
                f"'{self.path}' is not a readable WAV file",
                {"code": "invalid_wav", "source": str(self.path)},
            ) from error

    def blocks(self) -> Iterator[tuple[bytes, numpy.ndarray]]:
        """Decodes the clip `CHUNK` frames at a time.

        Yields:
            The raw bytes of each chunk and its samples scaled to [-1, 1],
            shaped frames by channels.

        Raises:
            ValidationError: If the file is not a readable WAV file or its
                sample width is not 1, 2, 3 or 4 bytes.
        """
        try:
            with wave.open(str(self.path), "rb") as opened:
                channels = opened.getnchannels()
                width = opened.getsampwidth()
                while raw := opened.readframes(CHUNK):
                    whole = len(raw) - len(raw) % (width * channels)
                    if not whole:
                        return
                    samples = scaled(raw[:whole], width)[0]
                    yield raw[:whole], samples.reshape(-1, channels)
        except (EOFError, wave.Error) as error:
            raise ValidationError(
                f"'{self.path}' is not a readable WAV file",
                {"code": "invalid_wav", "source": str(self.path)},
            ) from error


class Codec:
    """A compressed clip read through the optional `soundfile` library.

    Attributes:
        path: The clip's file.
        library: The imported `soundfile` module.
    """

    def __init__(self, path: pathlib.Path) -> None:
        """Binds the file and imports the library.

        Args:
            path: The clip's file.

        Raises:
            DecoderError: If `soundfile` is not installed.
        """
        self.path = path
        try:
            self.library: Any = importlib.import_module("soundfile")
        except ImportError as error:
            raise DecoderError(
                f"'{path.name}' is not WAV; reading it needs the optional "
                "soundfile library (pip install soundfile)",
                {"code": "decoder_missing", "source": str(path)},
            ) from error

    def header(self) -> Header:
        """Reads the header.

        Returns:
            The rate, channels and frame count; the width is 0.

        Raises:
            ValidationError: If the library cannot read the file.
        """
        try:
            found = self.library.info(str(self.path))
        except RuntimeError as error:
            raise ValidationError(
                f"'{self.path}' is not a readable audio file",
                {"code": "invalid_audio", "source": str(self.path)},
            ) from error
        return Header(found.samplerate, found.channels, 0, found.frames)

    def blocks(self) -> Iterator[tuple[bytes, numpy.ndarray]]:
        """Decodes the clip `CHUNK` frames at a time.

        Yields:
            The decoded bytes of each chunk and its samples in [-1, 1],
            shaped frames by channels.
        """
        for block in self.library.blocks(
            str(self.path), blocksize=CHUNK, always_2d=True, dtype="float64"
        ):
            values = numpy.ascontiguousarray(block)
            yield values.tobytes(), values


def reader(path: pathlib.Path) -> Wave | Codec:
    """The reader of a clip's container.

    Args:
        path: The clip's file.

    Returns:
        `Wave` for a RIFF/WAVE file, else `Codec`.
    """
    with path.open("rb") as handle:
        head = handle.read(12)
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return Wave(path)
    return Codec(path)


def masks(rate: int, edges: tuple[float, ...]) -> numpy.ndarray:
    """Assigns each frequency bin of an analysis frame to its band.

    Args:
        rate: Frames per second.
        edges: Band edges in Hz.

    Returns:
        A bins by bands 0/1 matrix.
    """
    frequencies = numpy.fft.rfftfreq(WINDOW, 1.0 / rate)
    band = numpy.searchsorted(numpy.asarray(edges), frequencies, side="right")
    return numpy.eye(len(edges) + 1)[band]


class Tally:
    """Running sums over a clip's chunks.

    Attributes:
        header: The clip's header.
        count: Samples seen (all channels).
        total: Sum of samples.
        magnitude: Sum of absolute samples.
        energy: Sum of squared samples.
        peak: Largest absolute sample.
        clipped: Samples at or above `CLIPPING_LEVEL` in magnitude.
        quiet: Samples at or below `SILENCE_LEVEL` in magnitude.
        crossings: Sign changes of the mono signal.
        last: The last mono sample seen, or None.
        carry: Mono samples waiting for a full analysis frame.
        start: Mono samples already analysed.
        bands: Power per band, summed over analysis frames.
        centroid: Sum of the frames' spectral centroids.
        flatness: Sum of the frames' spectral flatness.
        loudness: Sum of the frames' level in dB.
        spread: Sum of the squared levels.
        frames: Analysis frames seen.
        segments: Envelope band power per time segment; a frame adds to
            each segment in proportion to the samples they share, so a
            clip shorter than `SEGMENTS` frames leaves no segment empty.
        octaves: The bins-to-bands matrix of the feature bands.
        coarse: The bins-to-bands matrix of the envelope bands.
        hasher: The digest of the decoded content.
    """

    def __init__(self, header: Header) -> None:
        """Starts empty sums.

        Args:
            header: The clip's header.
        """
        self.header = header
        self.count = 0
        self.total = 0.0
        self.magnitude = 0.0
        self.energy = 0.0
        self.peak = 0.0
        self.clipped = 0
        self.quiet = 0
        self.crossings = 0
        self.last: float | None = None
        self.carry = numpy.zeros(0)
        self.start = 0
        self.bands = numpy.zeros(len(EDGES) + 1)
        self.centroid = 0.0
        self.flatness = 0.0
        self.loudness = 0.0
        self.spread = 0.0
        self.frames = 0
        self.segments = numpy.zeros((SEGMENTS, len(PRINT) + 1))
        self.octaves = masks(header.rate, EDGES)
        self.coarse = masks(header.rate, PRINT)
        self.hasher = hashlib.sha256(
            f"{header.rate}:{header.channels}:{header.width}".encode()
        )

    def add(self, raw: bytes, samples: numpy.ndarray) -> None:
        """Adds one chunk.

        Args:
            raw: The chunk's bytes, for the content digest.
            samples: Its samples, frames by channels, in [-1, 1].
        """
        self.hasher.update(raw)
        absolute = numpy.abs(samples)
        self.count += samples.size
        self.total += float(samples.sum())
        self.magnitude += float(absolute.sum())
        self.energy += float(numpy.square(samples).sum())
        self.peak = max(self.peak, float(absolute.max(initial=0.0)))
        self.clipped += int((absolute >= CLIPPING_LEVEL).sum())
        self.quiet += int((absolute <= SILENCE_LEVEL).sum())
        mono = samples.mean(axis=1)
        signs = numpy.signbit(mono)
        self.crossings += int((signs[1:] != signs[:-1]).sum())
        if self.last is not None and mono.size:
            self.crossings += int(bool(numpy.signbit(self.last)) != signs[0])
        if mono.size:
            self.last = float(mono[-1])
        pending = numpy.concatenate([self.carry, mono])
        whole = len(pending) - len(pending) % WINDOW
        self.analyse(pending[:whole])
        self.carry = pending[whole:]

    def analyse(self, mono: numpy.ndarray) -> None:
        """Adds the spectra of whole analysis frames.

        Args:
            mono: Mono samples, a multiple of `WINDOW` long.
        """
        if not mono.size:
            return
        frames = mono.reshape(-1, WINDOW)
        power = numpy.square(
            numpy.abs(numpy.fft.rfft(frames * numpy.hanning(WINDOW), axis=1))
        )
        frequencies = numpy.fft.rfftfreq(WINDOW, 1.0 / self.header.rate)
        sums = power.sum(axis=1)
        self.bands += (power @ self.octaves).sum(axis=0)
        self.centroid += float(
            ((power @ frequencies) / numpy.maximum(sums, TINY)).sum()
        )
        self.flatness += float(
            (
                numpy.exp(numpy.log(power + TINY).mean(axis=1))
                / (power.mean(axis=1) + TINY)
            ).sum()
        )
        levels = 10.0 * numpy.log10(numpy.square(frames).mean(axis=1) + TINY)
        self.loudness += float(levels.sum())
        self.spread += float(numpy.square(levels).sum())
        starts = self.start + WINDOW * numpy.arange(len(frames))[:, None]
        bounds = self.header.frames * numpy.arange(SEGMENTS + 1) / SEGMENTS
        shares = numpy.clip(
            numpy.minimum(starts + WINDOW, bounds[1:])
            - numpy.maximum(starts, bounds[:-1]),
            0.0,
            None,
        )
        self.segments += (shares / WINDOW).T @ (power @ self.coarse)
        self.start += len(frames) * WINDOW
        self.frames += len(frames)

    def finish(self) -> "Trace":
        """Analyses the last partial frame and closes the sums.

        Returns:
            The clip's features, content digest and envelope.
        """
        if self.carry.size:
            padded = numpy.zeros(WINDOW)
            padded[: self.carry.size] = self.carry
            self.analyse(padded)
            self.carry = numpy.zeros(0)
        frames = max(self.frames, 1)
        level = self.loudness / frames
        shares = self.bands / max(float(self.bands.sum()), TINY)
        header = self.header
        features = {
            "audio_mean": self.total / self.count,
            "audio_magnitude": self.magnitude / self.count,
            "audio_rms": float(numpy.sqrt(self.energy / self.count)),
            "audio_peak": self.peak,
            "audio_clipping": self.clipped / self.count,
            "audio_silence": self.quiet / self.count,
            "audio_crossings": self.crossings / max(self.count, 1),
            "audio_centroid": self.centroid / frames,
            "audio_flatness": self.flatness / frames,
            "audio_loudness": level,
            "audio_dynamics": float(
                numpy.sqrt(max(self.spread / frames - level * level, 0.0))
            ),
            **{
                f"audio_band{index}": float(share)
                for index, share in enumerate(shares)
            },
            "audio_duration": header.frames / header.rate,
            "audio_rate": float(header.rate),
            "audio_channels": float(header.channels),
            "audio_width": float(header.width),
        }
        return Trace(features, self.hasher.hexdigest(), envelope(self))


def envelope(tally: Tally) -> tuple[float, ...]:
    """The clip's band energy per time segment, in dB below its loudest.

    Each cell is the energy of one fingerprint band in one time segment
    (a band-energy fingerprint after Haitsma and Kalker, 2002), in dB
    relative to the loudest cell and floored `FLOOR` dB below it. Relative
    levels do not change with gain, and segments are fractions of the clip,
    so a louder or quieter copy keeps its envelope; noise `NOISE` dB below
    the signal moves no cell by more than `RADIUS` dB.

    Args:
        tally: The closed sums of a clip.

    Returns:
        `CELLS` levels in dB, segment by segment.
    """
    levels = 10.0 * numpy.log10(tally.segments + TINY)
    levels -= float(numpy.max(levels))
    return tuple(float(value) for value in numpy.maximum(levels, -FLOOR).flat)


@dataclasses.dataclass(frozen=True)
class Trace:
    """What one clip yields.

    Attributes:
        features: Feature name to value, every one finite.
        digest: SHA-256 of the decoded content and its format.
        envelope: The `CELLS` band levels for near-duplicate detection.
    """

    features: dict[str, float]
    digest: str
    envelope: tuple[float, ...]


def describe(path: pathlib.Path) -> Trace:
    """Reads a whole clip in bounded chunks and describes it.

    Args:
        path: The clip's file.

    Returns:
        Its features, content digest and envelope.

    Raises:
        ValidationError: If the clip is unreadable, holds no frame, or a
            feature is not finite.
        DecoderError: If the clip is not WAV and `soundfile` is missing.
    """
    source = reader(path)
    header = source.header()
    if header.frames < 1 or header.rate < 1 or header.channels < 1:
        raise ValidationError(
            f"audio clip '{path}' holds no frames",
            {"code": "empty_audio_frames", "source": str(path)},
        )
    tally = Tally(header)
    for raw, samples in source.blocks():
        tally.add(raw, samples)
    if not tally.count:
        raise ValidationError(
            f"audio clip '{path}' holds no complete frame",
            {"code": "empty_audio_frames", "source": str(path)},
        )
    trace = tally.finish()
    if not numpy.isfinite(list(trace.features.values())).all():
        raise ValidationError(
            f"audio clip '{path}' gave a non-finite feature",
            {"code": "non_finite_features", "source": str(path)},
        )
    return trace
