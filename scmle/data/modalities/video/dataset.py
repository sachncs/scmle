"""The immutable video collection."""

import dataclasses
import hashlib

from scmle.data.modalities.video.frames import Frames
from scmle.data.modalities.video.sample import Sample
from scmle.data.modalities.video.sample import VideoError
from scmle.data.modalities.video.splits import Splits
from scmle.data.modalities.video.survey import Survey


@dataclasses.dataclass(frozen=True, slots=True)
class Dataset(Survey, Frames, Splits):
    """Immutable video collection with deterministic source identity.

    Attributes:
        samples: The samples, in caller order.
        content_hash: SHA-256 hex digest of the sources, file hashes and
            observation times, in sample order; set on construction.
    """

    samples: tuple[Sample, ...]
    content_hash: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Rejects an empty or duplicate-source dataset and derives its hash.

        Raises:
            VideoError: If there are no samples or two samples
                share a source.
        """
        normalized = tuple(self.samples)
        if not normalized:
            raise VideoError(
                "video dataset must contain at least one sample",
                {"code": "empty_dataset"},
            )
        sources = [sample.source for sample in normalized]
        if len(sources) != len(set(sources)):
            raise VideoError(
                "video dataset contains duplicate sources",
                {"code": "duplicate_source"},
            )
        digest = hashlib.sha256(
            "\n".join(
                f"{sample.source}\0{sample.file_hash}"
                + (
                    f"\0{sample.observation_time.isoformat()}"
                    if sample.observation_time is not None
                    else ""
                )
                for sample in normalized
            ).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "samples", normalized)
        object.__setattr__(self, "content_hash", digest)
