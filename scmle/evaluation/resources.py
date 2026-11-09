"""Inference resource measurement and constraint checking.

Latency and size are measured on the machine running the workflow; the
numbers are only meaningful for that hardware, which is recorded alongside.
"""

import dataclasses
import io
import platform
import time
from typing import Any

import joblib
import numpy
import pandas

BYTES_PER_MB = 1024 * 1024
LATENCY_ROWS = 30


@dataclasses.dataclass(frozen=True)
class ResourceUsage:
    """Measured serving cost of a fitted candidate.

    Attributes:
        model_mb: Joblib-serialized size in megabytes (1024 * 1024 bytes).
        p95_latency_ms: 95th percentile single-row prediction latency.
        rows_timed: Number of single-row predictions timed.
        machine: Architecture, operating system and Python version the
            numbers belong to.
    """

    model_mb: float
    p95_latency_ms: float
    rows_timed: int
    machine: str

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping.

        Returns:
            A mapping of every field of the measurement.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ResourceLimits:
    """Serving constraints a candidate must satisfy.

    Attributes:
        max_model_mb: Largest allowed serialized size.
        max_p95_latency_ms: Largest allowed p95 single-row latency.
        latency_safety_factor: Multiplier on the measured latency, for
            serving hardware slower than this machine.
    """

    max_model_mb: float | None = None
    max_p95_latency_ms: float | None = None
    latency_safety_factor: float = 1.0

    @property
    def active(self) -> bool:
        """Whether any limit is set."""
        return self.max_model_mb is not None or (
            self.max_p95_latency_ms is not None
        )

    def violations(self, usage: ResourceUsage) -> list[str]:
        """Lists the limits a measurement breaks.

        Args:
            usage: The measured model size and latency.

        Returns:
            One message per broken limit; empty when none is broken. Latency
            is compared after multiplying by `latency_safety_factor`.
        """
        broken = []
        if self.max_model_mb is not None and usage.model_mb > self.max_model_mb:
            broken.append(
                f"model size {usage.model_mb:.2f} MB > {self.max_model_mb} MB"
            )
        scaled = usage.p95_latency_ms * self.latency_safety_factor
        if self.max_p95_latency_ms is not None and (
            scaled > self.max_p95_latency_ms
        ):
            broken.append(
                f"p95 latency {scaled:.1f} ms (measured "
                f"{usage.p95_latency_ms:.1f} ms x safety factor "
                f"{self.latency_safety_factor:g}) > "
                f"{self.max_p95_latency_ms} ms"
            )
        return broken


def measure(
    estimator: Any, features: pandas.DataFrame, rows: int = LATENCY_ROWS
) -> ResourceUsage:
    """Measures serialized size and single-row prediction latency.

    Args:
        estimator: Fitted candidate.
        features: Rows to time predictions on; only the first `rows` are
            used.
        rows: Maximum number of single-row predictions to time, after one
            untimed warm-up prediction.

    Returns:
        The measured usage.
    """
    buffer = io.BytesIO()
    joblib.dump(estimator, buffer)
    sample = features.head(rows)
    estimator.predict(sample.head(1))  # Warm-up, excluded from timing.
    timings = []
    for position in range(len(sample)):
        started = time.perf_counter()
        estimator.predict(sample.iloc[[position]])
        timings.append((time.perf_counter() - started) * 1000.0)
    return ResourceUsage(
        model_mb=buffer.getbuffer().nbytes / BYTES_PER_MB,
        p95_latency_ms=float(numpy.percentile(timings, 95)),
        rows_timed=len(timings),
        machine=f"{platform.machine()} {platform.system()} "
        f"python {platform.python_version()}",
    )
