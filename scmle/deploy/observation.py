"""What a platform says is running, in one shape for every provider.

`Observation` is the answer an adapter (or a test double)
gives to "what is serving traffic right now?": the revision, the image
digest, whether it is ready and how traffic is split between revisions. The
registry, the reconcile command and the promotion gate compare against this
and nothing else; none of them reads a provider response directly.
"""

import dataclasses
import re
from typing import Any

from scmle.contracts import errors
from scmle.deploy import contract

REVISION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")


@dataclasses.dataclass(frozen=True, slots=True)
class Traffic:
    """The share of traffic one revision receives."""

    revision: str
    percent: int

    def __post_init__(self) -> None:
        """Validates the share."""
        if REVISION_PATTERN.fullmatch(self.revision) is None:
            raise errors.ConfigError(f"bad revision {self.revision!r}")
        if isinstance(self.percent, bool) or not 0 <= self.percent <= 100:
            raise errors.ConfigError("percent must be between 0 and 100")


@dataclasses.dataclass(frozen=True, slots=True)
class Observation:
    """The running state of one service.

    Attributes:
        provider: The name of the platform the adapter reads.
        service: The service name or ARN.
        revision: The revision that holds most traffic (the primary).
        image_digest: Digest of the image the primary revision runs.
        ready: Whether the platform reports the service healthy and settled.
        traffic: Revisions and their shares; the shares sum to 100.
        observed_at: UTC time of the read.
        revisions: Image digest of every revision the response named.
    """

    provider: str
    service: str
    revision: str
    image_digest: str
    ready: bool
    traffic: tuple[Traffic, ...]
    observed_at: str
    revisions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Validates digests and the traffic split."""
        if REVISION_PATTERN.fullmatch(self.revision) is None:
            raise errors.ConfigError(f"bad revision {self.revision!r}")
        contract.require_digest(self.image_digest, "observation.image_digest")
        if sum(item.percent for item in self.traffic) != 100:
            raise errors.ConfigError("traffic shares must sum to 100")

    def digest_of(self, revision: str) -> str | None:
        """The image digest the response named for ``revision``."""
        for name, digest in self.revisions:
            if name == revision:
                return digest
        return self.image_digest if revision == self.revision else None

    def share_of(self, revision: str) -> int:
        """Percent of traffic ``revision`` holds."""
        return sum(t.percent for t in self.traffic if t.revision == revision)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-ready record."""
        return {
            "provider": self.provider,
            "service": self.service,
            "revision": self.revision,
            "image_digest": self.image_digest,
            "ready": self.ready,
            "traffic": [dataclasses.asdict(t) for t in self.traffic],
            "revisions": [list(item) for item in self.revisions],
            "observed_at": self.observed_at,
        }
