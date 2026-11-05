"""Cost by service and deployed version, from declared rates.

The numbers are an estimate: measured usage multiplied by rates that the
operator declares. They are not a bill and do not know discounts, taxes,
free tiers or shared costs. The report says so, names the rates it used and
keeps services and versions apart so a release can be judged by what it cost.
"""

from collections.abc import Sequence
import dataclasses
import math
from typing import Any

GIB = 1024**3
STATEMENT = (
    "Estimate: measured usage times declared rates. Not a bill; discounts, "
    "taxes, free tiers and shared costs are not modelled."
)


@dataclasses.dataclass(frozen=True, slots=True)
class Rates:
    """Unit prices the operator declares.

    Attributes:
        vcpu_second: Price of one vCPU for one second.
        gib_second: Price of one GiB of memory for one second.
        per_thousand_requests: Price of 1000 requests.
        per_gb_egress: Price of one GB leaving the provider.
        currency: Currency code.
        source: Where the rates came from (a price page and a date).
    """

    vcpu_second: float
    gib_second: float
    per_thousand_requests: float = 0.0
    per_gb_egress: float = 0.0
    currency: str = "USD"
    source: str = ""

    def __post_init__(self) -> None:
        """Rates must be finite and not negative; a source is required."""
        for name in (
            "vcpu_second",
            "gib_second",
            "per_thousand_requests",
            "per_gb_egress",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and not negative")
        if not self.source.strip():
            raise ValueError("say where the rates come from")


@dataclasses.dataclass(frozen=True, slots=True)
class Usage:
    """Measured usage of one deployed version of one service.

    Attributes:
        service: Service name.
        version: Deployed model/application version.
        vcpu_seconds: Allocated vCPUs times seconds the revision ran.
        gib_seconds: Allocated GiB times seconds the revision ran.
        requests: Requests served.
        egress_gb: Data sent out, in GB.
    """

    service: str
    version: str
    vcpu_seconds: float
    gib_seconds: float
    requests: float = 0.0
    egress_gb: float = 0.0

    def __post_init__(self) -> None:
        """Usage must be finite and not negative."""
        for name in ("vcpu_seconds", "gib_seconds", "requests", "egress_gb"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and not negative")

    @classmethod
    def from_allocation(
        cls,
        service: str,
        version: str,
        vcpus: float,
        memory_gib: float,
        seconds: float,
        requests: float = 0.0,
        egress_gb: float = 0.0,
    ) -> "Usage":
        """Usage of a revision that held an allocation for ``seconds``."""
        return cls(
            service,
            version,
            vcpus * seconds,
            memory_gib * seconds,
            requests,
            egress_gb,
        )


def line(usage: Usage, rates: Rates) -> dict[str, float]:
    """The cost of one usage record, by component and in total."""
    parts = {
        "compute": usage.vcpu_seconds * rates.vcpu_second,
        "memory": usage.gib_seconds * rates.gib_second,
        "requests": usage.requests / 1000.0 * rates.per_thousand_requests,
        "egress": usage.egress_gb * rates.per_gb_egress,
    }
    parts["total"] = sum(parts.values())
    return {name: round(value, 6) for name, value in parts.items()}


def report(usages: Sequence[Usage], rates: Rates) -> dict[str, Any]:
    """Cost grouped by service and, inside it, by deployed version.

    Usage records for the same service and version are added together.
    """
    services: dict[str, dict[str, dict[str, float]]] = {}
    for usage in usages:
        versions = services.setdefault(usage.service, {})
        current = versions.setdefault(
            usage.version,
            {"compute": 0.0, "memory": 0.0, "requests": 0.0, "egress": 0.0},
        )
        for name, value in line(usage, rates).items():
            if name != "total":
                current[name] += value
    output: dict[str, Any] = {}
    grand = 0.0
    for service, versions in sorted(services.items()):
        total = 0.0
        shown = {}
        for version, parts in sorted(versions.items()):
            parts["total"] = sum(parts.values())
            shown[version] = {k: round(v, 6) for k, v in parts.items()}
            total += parts["total"]
        output[service] = {"versions": shown, "total": round(total, 6)}
        grand += total
    return {
        "currency": rates.currency,
        "rates": dataclasses.asdict(rates),
        "services": output,
        "total": round(grand, 6),
        "statement": STATEMENT,
    }
