"""A file-backed spool that keeps telemetry while its destination is down.

Samples are appended to a JSON-lines file before any send is tried, and are
removed only after the destination accepted them. A process that dies, or a
destination that is unreachable, therefore loses nothing it had accepted
into the spool; on recovery `flush` sends what is left, oldest first.

Delivery is at least once: if the process dies after the destination
accepted a batch and before the spool rewrote its file, that batch is sent
again. Consumers should treat samples as idempotent by (name, labels,
timestamp), which the metric formats here already are.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import json
import pathlib

from scmle import durability
from scmle.contracts import observability

Send = Callable[[list[observability.MetricSample]], int]


@dataclasses.dataclass(frozen=True, slots=True)
class FlushResult:
    """What a flush did.

    Attributes:
        sent: Samples the destination accepted.
        remaining: Samples still in the spool.
        error: Why the flush stopped early, or None.
    """

    sent: int
    remaining: int
    error: str | None


class MetricsSpool:
    """Spools samples to a file and drains them to a destination."""

    def __init__(
        self, path: str | pathlib.Path, send: Send, batch_size: int = 100
    ) -> None:
        """Opens the spool at ``path``.

        Args:
            path: The spool file.
            send: Called with a batch of samples; returns how many the
                destination accepted.
            batch_size: Number of samples per send.

        Raises:
            ValueError: If ``batch_size`` is not positive.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.path = pathlib.Path(path)
        self.send = send
        self.batch_size = batch_size

    def lines(self) -> list[str]:
        """Spooled lines, oldest first.

        Returns:
            The non-blank lines of the spool file; empty if it does not exist.
        """
        if not self.path.exists():
            return []
        return [
            line
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def pending(self) -> int:
        """Samples waiting to be sent.

        Returns:
            The number of spooled samples.
        """
        return len(self.lines())

    def add(self, samples: Sequence[observability.MetricSample]) -> int:
        """Spools samples (before anything is sent); returns how many.

        Args:
            samples: The samples to spool.

        Returns:
            The number of samples spooled.
        """
        text = "".join(
            json.dumps(s.to_dict(), sort_keys=True) + "\n" for s in samples
        )
        if text:
            durability.append_text(self.path, text)
        return len(samples)

    def flush(self) -> FlushResult:
        """Sends batches in order until the spool is empty or a send fails.

        Returns:
            How many samples were sent, how many remain and the error that
            stopped the flush, if any.
        """
        sent = 0
        lines = self.lines()
        error: str | None = None
        while lines:
            batch = lines[: self.batch_size]
            samples = [
                observability.MetricSample.from_dict(json.loads(line))
                for line in batch
            ]
            try:
                accepted = self.send(samples)
            except (OSError, ValueError) as failure:
                error = f"{type(failure).__name__}: {failure}"
                break
            if accepted != len(batch):
                error = f"destination accepted {accepted} of {len(batch)}"
                break
            lines = lines[len(batch) :]
            sent += len(batch)
            text = "".join(line + "\n" for line in lines)
            durability.write_atomically(self.path, text.encode("utf-8"))
        return FlushResult(sent, len(lines), error)
