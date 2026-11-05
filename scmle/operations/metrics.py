"""Request, latency, error, resource and availability metrics.

A small registry with counters, gauges and histograms and a text exposition in
the Prometheus format (version 0.0.4). `ServiceMetrics` records what a
serving process sees: requests by route and status, a latency histogram,
resident memory and CPU time. Availability is defined here and nowhere else:
a request is *unavailable* when the service answered 5xx or 429 (it could
not take the work); a 4xx caused by the caller counts as available.

The registry keeps cumulative values, as Prometheus expects. Windowed rates
for alerting are computed from events by `scmle.operations.signals`.
"""

from collections.abc import Mapping
import math
import re
import resource
import sys
import threading

DEFAULT_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)
NAME_PATTERN = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
LABEL_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
UNAVAILABLE_STATUSES = frozenset({429})
Labels = tuple[tuple[str, str], ...]


def freeze(labels: Mapping[str, str] | None) -> Labels:
    """Validates label names and returns them as a sorted tuple."""
    items = tuple(sorted((labels or {}).items()))
    for key, value in items:
        if not LABEL_PATTERN.match(key) or key.startswith("__"):
            raise ValueError(f"invalid label name {key!r}")
        if not isinstance(value, str):
            raise TypeError("label values must be text")
    return items


def escape(value: str) -> str:
    """Escapes a label value for the text exposition."""
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def render_labels(labels: Labels, extra: Labels = ()) -> str:
    """Formats ``{a="b",...}`` (empty text for no labels)."""
    pairs = [*labels, *extra]
    if not pairs:
        return ""
    body = ",".join(f'{key}="{escape(value)}"' for key, value in pairs)
    return "{" + body + "}"


def format_number(value: float) -> str:
    """Formats a sample value the way the exposition format spells it."""
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if math.isnan(value):
        return "NaN"
    return repr(float(value)) if value != int(value) else str(int(value))


class Histogram:
    """Cumulative bucket counts, a sum and a count for one label set."""

    def __init__(self, buckets: tuple[float, ...]) -> None:
        """Creates empty buckets (upper bounds, ascending)."""
        self.buckets = buckets
        self.counts = [0] * len(buckets)
        self.total = 0
        self.sum = 0.0

    def observe(self, value: float) -> None:
        """Adds one observation."""
        self.total += 1
        self.sum += value
        for position, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[position] += 1

    def quantile(self, share: float) -> float | None:
        """Estimates a quantile by linear interpolation inside a bucket.

        This is the estimate Prometheus makes from bucket counts; it is exact
        only at the bucket edges. None when nothing was observed.
        """
        if self.total == 0:
            return None
        rank = share * self.total
        lower = 0.0
        previous = 0
        for bound, count in zip(self.buckets, self.counts, strict=True):
            if count >= rank:
                width = count - previous
                if width == 0:
                    return bound
                return lower + (bound - lower) * (rank - previous) / width
            lower, previous = bound, count
        return self.buckets[-1]


class MetricsRegistry:
    """Named counters, gauges and histograms with a text exposition."""

    def __init__(self, buckets: tuple[float, ...] = DEFAULT_BUCKETS) -> None:
        """Creates an empty registry.

        Args:
            buckets: Histogram upper bounds, strictly ascending.

        Raises:
            ValueError: If the buckets are empty or not ascending.
        """
        if not buckets or list(buckets) != sorted(set(buckets)):
            raise ValueError("buckets must be non-empty and ascending")
        self.buckets = buckets
        self.lock = threading.Lock()
        self.help: dict[str, str] = {}
        self.kinds: dict[str, str] = {}
        self.values: dict[str, dict[Labels, float]] = {}
        self.histograms: dict[str, dict[Labels, Histogram]] = {}

    def declare(self, name: str, kind: str, text: str) -> None:
        """Declares a metric once; a second declaration must agree."""
        if not NAME_PATTERN.match(name):
            raise ValueError(f"invalid metric name {name!r}")
        known = self.kinds.get(name)
        if known is not None and known != kind:
            raise ValueError(f"{name} is already a {known}")
        self.kinds[name] = kind
        self.help.setdefault(name, text)

    def increment(
        self,
        name: str,
        labels: Mapping[str, str] | None = None,
        amount: float = 1.0,
        text: str = "",
    ) -> None:
        """Adds to a counter.

        Raises:
            ValueError: If ``amount`` is negative.
        """
        if amount < 0:
            raise ValueError("a counter only goes up")
        with self.lock:
            self.declare(name, "counter", text)
            series = self.values.setdefault(name, {})
            key = freeze(labels)
            series[key] = series.get(key, 0.0) + amount

    def set(
        self,
        name: str,
        value: float,
        labels: Mapping[str, str] | None = None,
        text: str = "",
    ) -> None:
        """Sets a gauge."""
        if not math.isfinite(value):
            raise ValueError("a gauge value must be finite")
        with self.lock:
            self.declare(name, "gauge", text)
            self.values.setdefault(name, {})[freeze(labels)] = value

    def observe(
        self,
        name: str,
        value: float,
        labels: Mapping[str, str] | None = None,
        text: str = "",
    ) -> None:
        """Adds an observation to a histogram."""
        if not math.isfinite(value) or value < 0:
            raise ValueError("an observation must be finite and not negative")
        with self.lock:
            self.declare(name, "histogram", text)
            series = self.histograms.setdefault(name, {})
            key = freeze(labels)
            if key not in series:
                series[key] = Histogram(self.buckets)
            series[key].observe(value)

    def value(
        self, name: str, labels: Mapping[str, str] | None = None
    ) -> float | None:
        """Current value of one counter or gauge series (None if absent)."""
        with self.lock:
            return self.values.get(name, {}).get(freeze(labels))

    def total(self, name: str, **match: str) -> float:
        """Sum of the series of a counter whose labels contain ``match``."""
        with self.lock:
            return sum(
                value
                for labels, value in self.values.get(name, {}).items()
                if all(dict(labels).get(k) == v for k, v in match.items())
            )

    def histogram(
        self, name: str, labels: Mapping[str, str] | None = None
    ) -> Histogram | None:
        """One histogram series (None if absent)."""
        with self.lock:
            return self.histograms.get(name, {}).get(freeze(labels))

    def series(self, name: str) -> dict[Labels, float]:
        """A copy of every series of a counter or gauge."""
        with self.lock:
            return dict(self.values.get(name, {}))

    def merged_histogram(self, name: str) -> Histogram | None:
        """All series of a histogram added together (None if empty)."""
        with self.lock:
            found = list(self.histograms.get(name, {}).values())
            if not found:
                return None
            merged = Histogram(self.buckets)
            for item in found:
                merged.counts = [
                    a + b
                    for a, b in zip(merged.counts, item.counts, strict=True)
                ]
                merged.total += item.total
                merged.sum += item.sum
            return merged

    def render(self) -> str:
        """The Prometheus text exposition of every series, sorted."""
        lines: list[str] = []
        with self.lock:
            for name in sorted(self.kinds):
                kind = self.kinds[name]
                text = self.help.get(name, "")
                if text:
                    lines.append(f"# HELP {name} {text}")
                lines.append(f"# TYPE {name} {kind}")
                if kind == "histogram":
                    lines.extend(self.render_histogram(name))
                    continue
                for labels, value in sorted(self.values.get(name, {}).items()):
                    lines.append(
                        f"{name}{render_labels(labels)} {format_number(value)}"
                    )
        return "\n".join(lines) + "\n"

    def render_histogram(self, name: str) -> list[str]:
        """Bucket, sum and count lines of one histogram."""
        lines: list[str] = []
        for labels, hist in sorted(self.histograms.get(name, {}).items()):
            for bound, count in zip(hist.buckets, hist.counts, strict=True):
                suffix = render_labels(labels, (("le", format_number(bound)),))
                lines.append(f"{name}_bucket{suffix} {count}")
            inf = render_labels(labels, (("le", "+Inf"),))
            lines.append(f"{name}_bucket{inf} {hist.total}")
            lines.append(
                f"{name}_sum{render_labels(labels)} {format_number(hist.sum)}"
            )
            lines.append(f"{name}_count{render_labels(labels)} {hist.total}")
        return lines


def resident_bytes() -> int:
    """Peak resident memory of this process in bytes (0 if unknown).

    ``ru_maxrss`` is a peak, not the current size; it is bytes on macOS and
    kibibytes on Linux.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def cpu_seconds() -> float:
    """User plus system CPU time of this process, in seconds."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


class ServiceMetrics:
    """What a serving process records, attributed to a model version."""

    REQUESTS = "scmle_http_requests_total"
    DURATION = "scmle_http_request_duration_seconds"
    MEMORY = "scmle_process_resident_memory_peak_bytes"
    CPU = "scmle_process_cpu_seconds"

    def __init__(
        self,
        registry: MetricsRegistry,
        model_version: str,
        artifact_digest: str,
    ) -> None:
        """Binds a registry to the model the process serves.

        Args:
            registry: Where series are kept.
            model_version: Version label attached to every series.
            artifact_digest: Digest of the served artifact; recorded as a
                gauge label so a scrape shows which bundle answered.
        """
        self.registry = registry
        self.model_version = model_version
        self.artifact_digest = artifact_digest
        registry.set(
            "scmle_model_info",
            1.0,
            {
                "model_version": model_version,
                "artifact_digest": artifact_digest,
            },
            "The model and artifact this process serves.",
        )

    def record_request(self, route: str, status: int, seconds: float) -> None:
        """Counts a request and observes its latency."""
        labels = {
            "route": route,
            "status": str(status),
            "model_version": self.model_version,
        }
        self.registry.increment(
            self.REQUESTS, labels, 1.0, "HTTP requests by route and status."
        )
        self.registry.observe(
            self.DURATION,
            seconds,
            {"route": route, "model_version": self.model_version},
            "Request duration in seconds.",
        )

    def record_resources(self) -> None:
        """Samples memory and CPU of this process into gauges."""
        labels = {"model_version": self.model_version}
        self.registry.set(
            self.MEMORY,
            float(resident_bytes()),
            labels,
            "Peak resident memory in bytes.",
        )
        self.registry.set(
            self.CPU, cpu_seconds(), labels, "Process CPU time in seconds."
        )

    def requests(self) -> float:
        """Requests counted so far."""
        return self.registry.total(self.REQUESTS)

    def count_where(self, predicate_statuses: set[int]) -> float:
        """Requests whose status is in the set."""
        return sum(
            self.registry.total(self.REQUESTS, status=str(status))
            for status in predicate_statuses
        )

    def summary(self) -> dict[str, float | None]:
        """Cumulative request count, error rates, availability, latency.

        Rates are None until a request is seen. ``server_error_rate`` is the
        share of 5xx, ``client_error_rate`` the share of 4xx other than 429,
        and ``availability`` the share that was not 5xx or 429.
        """
        total = 0.0
        server = 0.0
        client = 0.0
        unavailable = 0.0
        for labels, count in self.registry.series(self.REQUESTS).items():
            status = int(dict(labels)["status"])
            total += count
            if status >= 500:
                server += count
            elif status >= 400 and status not in UNAVAILABLE_STATUSES:
                client += count
            if status >= 500 or status in UNAVAILABLE_STATUSES:
                unavailable += count
        hist = self.registry.merged_histogram(self.DURATION)
        return {
            "requests": total,
            "server_error_rate": server / total if total else None,
            "client_error_rate": client / total if total else None,
            "availability": 1.0 - unavailable / total if total else None,
            "latency_p50_seconds": hist.quantile(0.5) if hist else None,
            "latency_p95_seconds": hist.quantile(0.95) if hist else None,
            "latency_p99_seconds": hist.quantile(0.99) if hist else None,
        }
