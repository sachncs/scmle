"""OpenTelemetry tracing of requests, scoring and pipeline stages.

Tracing is off by default and costs one attribute read per span while off.
`enable` installs a tracer provider of this package (the global provider of
the application is left alone) that exports through OTLP over HTTP, set up
by the standard `OTEL_EXPORTER_OTLP_*` variables, or through the exporter it
is given. Without the optional `tracing` extra, `enable` reports the
degraded path and returns False, and every span stays a no-op.

The spans are `request` (one HTTP request of the service, joined to the
caller's trace through its `traceparent` header), `predict` (one scoring
call, online or batch), `batch` (one scored file) and `stage <name>` (one
pipeline stage).
"""

from collections.abc import Iterator
from collections.abc import Mapping
import contextlib
import logging
import os
from typing import Any

from scmle.runtime import logs

LOGGER = logging.getLogger(__name__)
# Set (to any non-empty value) to trace, as `scmle --trace`.
SWITCH = "SCMLE_TRACING"
SERVICE = "scmle"


class Tracing:
    """The tracer of this package, or none while tracing is off.

    Attributes:
        provider: The SDK tracer provider, or None while tracing is off.
        tracer: The tracer the spans come from, or None while off.
    """

    def __init__(self) -> None:
        """Starts with tracing off."""
        self.provider: Any = None
        self.tracer: Any = None

    @property
    def active(self) -> bool:
        """Whether spans are recorded."""
        return self.tracer is not None

    def enable(
        self,
        exporter: Any = None,
        environ: Mapping[str, str] | None = None,
    ) -> bool:
        """Starts exporting spans; a second call replaces the first.

        Args:
            exporter: An OpenTelemetry span exporter; by default OTLP over
                HTTP, configured by the `OTEL_EXPORTER_OTLP_*` variables.
            environ: The environment; `OTEL_SERVICE_NAME` names the service
                (`scmle` by default). Defaults to the process environment.

        Returns:
            True when spans are now exported; False when the `tracing` extra
            is not installed, which is logged and warned as `Degraded`.
        """
        try:
            from opentelemetry.sdk import resources  # noqa: PLC0415 -- optional extra
            from opentelemetry.sdk import trace  # noqa: PLC0415 -- optional extra
            from opentelemetry.sdk.trace import export  # noqa: PLC0415 -- optional extra
        except ImportError:
            # handled: reported by `degrade`; spans stay no-ops
            logs.degrade(
                LOGGER,
                "tracing needs the optional 'tracing' extra (install "
                "scmle[tracing]); spans are not exported",
            )
            return False
        if exporter is None:
            from opentelemetry.exporter.otlp.proto.http import trace_exporter  # noqa: PLC0415 -- optional extra

            exporter = trace_exporter.OTLPSpanExporter()
        self.disable()
        variables = os.environ if environ is None else environ
        name = variables.get("OTEL_SERVICE_NAME") or SERVICE
        provider = trace.TracerProvider(
            resource=resources.Resource.create({"service.name": name})
        )
        provider.add_span_processor(export.BatchSpanProcessor(exporter))
        self.provider = provider
        self.tracer = provider.get_tracer(SERVICE)
        LOGGER.info("tracing enabled for service %s", name)
        return True

    def disable(self) -> None:
        """Exports the spans still buffered and turns tracing off."""
        provider = self.provider
        self.provider = self.tracer = None
        if provider is not None:
            provider.shutdown()

    def flush(self) -> None:
        """Exports the spans still buffered, keeping tracing on."""
        if self.provider is not None:
            self.provider.force_flush()

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        attributes: Mapping[str, str | int | float | bool] | None = None,
        carrier: Any = None,
    ) -> Iterator[None]:
        """Records the enclosed block as a span; a no-op while off.

        The span is a child of the span current in this context, or, with a
        `carrier`, of the remote span its `traceparent` names. An exception
        leaving the block is recorded on the span and re-raised.

        Args:
            name: Span name, from a small fixed set.
            attributes: Span attributes.
            carrier: Incoming headers (anything with `get`) to continue a
                caller's trace from.

        Yields:
            None, while the span is current.
        """
        tracer = self.tracer
        if tracer is None:
            yield
            return
        parent = None
        if carrier is not None:
            from opentelemetry import propagate  # noqa: PLC0415 -- optional extra

            parent = propagate.extract(carrier)
        with tracer.start_as_current_span(
            name, context=parent, attributes=dict(attributes or {})
        ):
            yield

    def annotate(self, key: str, value: str | float | bool) -> None:
        """Sets an attribute on the span current in this context.

        Args:
            key: Attribute name.
            value: Attribute value.
        """
        if self.tracer is None:
            return
        from opentelemetry import trace  # noqa: PLC0415 -- optional extra

        trace.get_current_span().set_attribute(key, value)


TRACING = Tracing()
