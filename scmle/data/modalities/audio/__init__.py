"""Dependency-light audio data contracts and bounded profiling.

The core package decodes WAV files with the Python standard library. Other
audio containers must provide an explicit decoder/probe function so that a
codec dependency never becomes an implicit trust or reproducibility boundary.
"""

from scmle.data.modalities.audio.dataset import Dataset
from scmle.data.modalities.audio.evidence import Features
from scmle.data.modalities.audio.evidence import Split
from scmle.data.modalities.audio.metadata import DecoderError
from scmle.data.modalities.audio.metadata import Metadata
from scmle.data.modalities.audio.metadata import Profile
from scmle.data.modalities.audio.metadata import Segment
from scmle.data.modalities.audio.metadata import SHA256_HEX_LENGTH
from scmle.data.modalities.audio.metadata import Transcript
from scmle.data.modalities.audio.metadata import ValidationError
from scmle.data.modalities.audio.sample import AudioProbe
from scmle.data.modalities.audio.sample import canonical_source
from scmle.data.modalities.audio.sample import Sample
from scmle.data.modalities.audio.wav import CLIPPING_LEVEL
from scmle.data.modalities.audio.wav import extract
from scmle.data.modalities.audio.wav import MATRIX_DIMENSIONS
from scmle.data.modalities.audio.wav import parse
from scmle.data.modalities.audio.wav import pcm
from scmle.data.modalities.audio.wav import SILENCE_LEVEL
from scmle.data.modalities.audio.wav import wav_payload_from_pcm
from scmle.data.modalities.audio.wav import WIDTH_16_BIT
from scmle.data.modalities.audio.wav import WIDTH_24_BIT
from scmle.data.modalities.audio.wav import WIDTH_32_BIT

__all__ = [
    "CLIPPING_LEVEL",
    "MATRIX_DIMENSIONS",
    "SHA256_HEX_LENGTH",
    "SILENCE_LEVEL",
    "WIDTH_16_BIT",
    "WIDTH_24_BIT",
    "WIDTH_32_BIT",
    "AudioProbe",
    "Dataset",
    "DecoderError",
    "Features",
    "Metadata",
    "Profile",
    "Sample",
    "Segment",
    "Split",
    "Transcript",
    "ValidationError",
    "canonical_source",
    "extract",
    "parse",
    "pcm",
    "wav_payload_from_pcm",
]
