"""Video data contracts with explicit, caller-owned probing."""

from scmle.data.modalities.video.dataset import Dataset
from scmle.data.modalities.video.evidence import Chronology
from scmle.data.modalities.video.evidence import Features
from scmle.data.modalities.video.evidence import Split
from scmle.data.modalities.video.sample import identify
from scmle.data.modalities.video.sample import Metadata
from scmle.data.modalities.video.sample import Profile
from scmle.data.modalities.video.sample import Sample
from scmle.data.modalities.video.sample import SHA256_HEX_LENGTH
from scmle.data.modalities.video.sample import TemporalError
from scmle.data.modalities.video.sample import timestamp
from scmle.data.modalities.video.sample import VideoError
from scmle.data.modalities.video.sample import VideoFrameDecoder
from scmle.data.modalities.video.sample import VideoProbe

__all__ = [
    "SHA256_HEX_LENGTH",
    "Chronology",
    "Dataset",
    "Features",
    "Metadata",
    "Profile",
    "Sample",
    "Split",
    "TemporalError",
    "VideoError",
    "VideoFrameDecoder",
    "VideoProbe",
    "identify",
    "timestamp",
]
