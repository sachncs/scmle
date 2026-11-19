"""Video files probed and decoded in a stream, never held in memory.

`ffprobe` reads the container metadata and `ffmpeg` decodes `COUNT` evenly
spaced frames, scaled to `descriptor.WORK` pixels a side, straight into a
pipe: the encoded file is never read into this process, so the memory a
video takes is the sampled frames alone (MM-6). Both programs are optional:
without them every operation raises a `VideoError` coded
`missing_dependency`.

A video's descriptor is the image descriptor of its mean sampled frame
(so near-duplicate videos are linked as images are), the brightness of each
sampled frame in time order, the motion between consecutive sampled frames
and the container timing.
"""

import json
import math
import pathlib
import shutil
import subprocess

import numpy

from scmle.data.modalities.image import descriptor
from scmle.data.modalities.video.sample import Metadata
from scmle.data.modalities.video.sample import VideoError

COUNT = 8
"""Frames sampled per video, evenly over its length."""

RATE = 30.0
"""Frames per second the decode timeout assumes: a tenth of the about 300
frames per second one core decodes 1080p H.264 at, so a slow host stays
inside the bound."""

STARTUP = 10.0
"""Seconds the timeout allows a program to start and open the file."""

SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})
"""File suffixes read as video."""


def tool(name: str) -> str:
    """Locates an optional program.

    Args:
        name: `ffprobe` or `ffmpeg`.

    Returns:
        Its path.

    Raises:
        VideoError: If it is not on the PATH (`missing_dependency`).
    """
    found = shutil.which(name)
    if found is None:
        raise VideoError(
            f"video decoding requires the optional '{name}' program on the "
            "PATH; install FFmpeg",
            {"code": "missing_dependency", "dependency": name},
        )
    return found


def execute(arguments: list[str], seconds: float, source: str) -> bytes:
    """Runs a program and returns what it wrote.

    Args:
        arguments: The program and its arguments.
        seconds: Seconds it may take.
        source: The file it reads, named in errors.

    Returns:
        Its standard output.

    Raises:
        VideoError: If it fails or exceeds `seconds`.
    """
    try:
        done = subprocess.run(  # noqa: S603 -- fixed program, no shell
            arguments, capture_output=True, timeout=seconds, check=False
        )
    except subprocess.TimeoutExpired as error:
        raise VideoError(
            f"{pathlib.Path(arguments[0]).name} exceeded {seconds:.0f} s",
            {"code": "timeout", "source": source},
        ) from error
    if done.returncode != 0:
        raise VideoError(
            f"could not decode video: {done.stderr.decode(errors='replace')}",
            {"code": "decode_failure", "source": source},
        )
    return done.stdout


def fraction(text: str) -> float:
    """Reads a number ffprobe writes, plain (`2.5`) or a ratio (`30000/1001`).

    Args:
        text: The text.

    Returns:
        Its value; 0.0 for text that is not a positive number or ratio
        (`N/A`, `0/0`).
    """
    numerator, slash, denominator = text.strip().partition("/")
    parts = [numerator, denominator] if slash else [numerator]
    if not all(part.replace(".", "", 1).isdigit() for part in parts):
        return 0.0
    if not slash:
        return float(numerator)
    divisor = float(denominator)
    return float(numerator) / divisor if divisor > 0 else 0.0


def probe(path: pathlib.Path) -> Metadata:
    """Reads a video's container metadata.

    Args:
        path: The video file.

    Returns:
        Its metadata; the frame count is estimated from the duration and
        rate when the container does not store it.

    Raises:
        VideoError: If the file has no video stream or no usable timing.
    """
    source = str(path)
    output = execute(
        [
            tool("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=width,height,avg_frame_rate,nb_frames,codec_name"
                ":format=duration,format_name"
            ),
            "-of",
            "json",
            source,
        ],
        STARTUP,
        source,
    )
    found = json.loads(output or b"{}")
    streams = found.get("streams") or []
    if not streams:
        raise VideoError(
            "file has no video stream",
            {"code": "no_video_stream", "source": source},
        )
    stream = streams[0]
    rate = fraction(str(stream.get("avg_frame_rate") or "0"))
    duration = fraction(str(found.get("format", {}).get("duration") or "0"))
    count = int(stream.get("nb_frames") or 0) or math.ceil(duration * rate)
    if rate <= 0 or count < 1:
        raise VideoError(
            "video has no usable frame rate or length",
            {"code": "no_timing", "source": source},
        )
    return Metadata(
        format=str(found.get("format", {}).get("format_name") or "unknown"),
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        frame_count=count,
        frames_per_second=rate,
        codec=str(stream.get("codec_name") or "unknown"),
    )


def frames(path: pathlib.Path, metadata: Metadata) -> numpy.ndarray:
    """Decodes `COUNT` evenly spaced frames.

    Args:
        path: The video file.
        metadata: Its metadata (`probe`).

    Returns:
        `COUNT` x `descriptor.WORK` x `descriptor.WORK` x 3 RGB values; when
        the stream holds fewer frames than its metadata promised, the
        decoded frames are repeated in time order to fill the slots.

    Raises:
        VideoError: If no frame decodes.
    """
    side = descriptor.WORK
    step = max(1, metadata.frame_count // COUNT)
    source = str(path)
    output = execute(
        [
            tool("ffmpeg"),
            "-v",
            "error",
            "-i",
            source,
            "-vf",
            f"select='not(mod(n\\,{step}))',scale={side}:{side}",
            "-fps_mode",
            "passthrough",
            "-frames:v",
            str(COUNT),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        STARTUP + metadata.frame_count / RATE,
        source,
    )
    size = side * side * 3
    decoded = len(output) // size
    if decoded < 1:
        raise VideoError(
            "no frame of the video decodes",
            {"code": "no_frames", "source": source},
        )
    raw = numpy.frombuffer(output[: decoded * size], dtype=numpy.uint8)
    slots = numpy.linspace(0, decoded - 1, COUNT).round().astype(int)
    chosen: numpy.ndarray = raw.reshape(decoded, side, side, 3)[slots]
    return chosen


def names() -> tuple[str, ...]:
    """Names the video descriptor columns, in the order `describe` fills them.

    Returns:
        The image descriptor names of the mean frame, then the brightness of
        each sampled frame, the motion statistics and the timing.
    """
    return (
        *descriptor.names(),
        *(f"time_{slot}" for slot in range(COUNT)),
        "motion_mean",
        "motion_spread",
        "motion_peak",
        "timing_seconds",
        "timing_rate",
    )


def describe(path: pathlib.Path) -> numpy.ndarray:
    """Describes one video file.

    Args:
        path: The video file.

    Returns:
        One float row, named by `names`.
    """
    metadata = probe(path)
    sampled = frames(path, metadata).astype(numpy.float64)
    mean = sampled.mean(axis=0)
    brightness = (sampled / 255.0 @ descriptor.LUMA).mean(axis=(1, 2))
    motion = numpy.abs(numpy.diff(sampled, axis=0)).mean(axis=(1, 2, 3)) / 255.0
    return numpy.concatenate(
        [
            descriptor.describe(mean, metadata.width, metadata.height),
            brightness,
            [
                float(motion.mean()),
                float(motion.std()),
                float(numpy.max(motion)),
                metadata.duration_seconds,
                metadata.frames_per_second,
            ],
        ]
    )
