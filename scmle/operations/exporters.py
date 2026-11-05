"""Payloads for CloudWatch and Cloud Monitoring, built from metric samples.

Three shapes are produced, each a plain dictionary that follows the
documented request:

* CloudWatch embedded metric format (a log line the agent or Lambda turns
  into metrics);
* CloudWatch ``PutMetricData`` (``Namespace`` and ``MetricData``);
* Cloud Monitoring ``projects.timeSeries.create`` (``timeSeries`` with a
  ``metric.type`` under ``custom.googleapis.com/`` and a monitored resource).

The builders are pure. The exporters take a client with the one method the
real SDK has (``put_metric_data(**payload)``, or ``create_time_series(name=,
body=)``) so a test passes a recorder and a deployment passes boto3 or
google-cloud-monitoring. Nothing here has called either service: the shapes
were checked against recorded payloads in ``tests/fixtures/ops``, which were
written from the public API references, not captured from an account.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import datetime
import json
import re
from typing import Any
from typing import Protocol

from scmle.contracts import observability

CLOUDWATCH_UNITS = {
    "s": "Seconds",
    "ms": "Milliseconds",
    "ratio": "None",
    "percent": "Percent",
    "count": "Count",
    "bytes": "Bytes",
}
CLOUDWATCH_UNIT_NAMES = frozenset(CLOUDWATCH_UNITS.values())
MAX_DIMENSIONS = 30
MAX_PUT_BATCH = 1000
MAX_SERIES_BATCH = 200
METRIC_TYPE_PREFIX = "custom.googleapis.com/"
NAMESPACE_PATTERN = re.compile(r"^[A-Za-z0-9._#:/-]{1,255}$")
METRIC_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_./-]{1,200}$")


def epoch_milliseconds(timestamp: str) -> int:
    """Milliseconds since the epoch of an aware ISO-8601 timestamp."""
    parsed = observability.validate_timestamp(timestamp, "timestamp")
    value = datetime.datetime.fromisoformat(parsed.replace("Z", "+00:00"))
    return int(value.timestamp() * 1000)


def rfc3339(timestamp: str) -> str:
    """UTC RFC 3339 text with a ``Z`` suffix."""
    parsed = observability.validate_timestamp(timestamp, "timestamp")
    value = datetime.datetime.fromisoformat(parsed.replace("Z", "+00:00"))
    return (
        value.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        + "Z"
    )


def cloudwatch_unit(unit: str) -> str:
    """The CloudWatch unit for a sample unit (``None`` when unknown)."""
    return CLOUDWATCH_UNITS.get(unit.lower(), "None")


def check_dimensions(labels: Mapping[str, str]) -> None:
    """Raises if there are more dimensions than CloudWatch allows."""
    if len(labels) > MAX_DIMENSIONS:
        raise ValueError(f"at most {MAX_DIMENSIONS} dimensions")


def embedded_metric_format(
    namespace: str, samples: Sequence[observability.MetricSample]
) -> list[dict[str, Any]]:
    """One embedded-metric-format record per timestamp and label set.

    Samples that share a timestamp and the same labels are one record with
    several metrics, as the format intends.
    """
    if not NAMESPACE_PATTERN.match(namespace):
        raise ValueError("invalid CloudWatch namespace")
    groups: dict[tuple[str, tuple[tuple[str, str], ...]], list[Any]] = {}
    for sample in samples:
        check_dimensions(sample.labels)
        key = (sample.timestamp, tuple(sorted(sample.labels.items())))
        groups.setdefault(key, []).append(sample)
    records = []
    for (timestamp, labels), members in sorted(groups.items()):
        names = [name for name, ignored in labels]
        record: dict[str, Any] = {
            "_aws": {
                "Timestamp": epoch_milliseconds(timestamp),
                "CloudWatchMetrics": [
                    {
                        "Namespace": namespace,
                        "Dimensions": [names] if names else [[]],
                        "Metrics": [
                            {
                                "Name": member.name,
                                "Unit": cloudwatch_unit(member.unit),
                            }
                            for member in members
                        ],
                    }
                ],
            }
        }
        record.update(dict(labels))
        for member in members:
            record[member.name] = member.value
        records.append(record)
    return records


def put_metric_data(
    namespace: str, samples: Sequence[observability.MetricSample]
) -> list[dict[str, Any]]:
    """``PutMetricData`` requests, at most ``MAX_PUT_BATCH`` data per call."""
    if not NAMESPACE_PATTERN.match(namespace):
        raise ValueError("invalid CloudWatch namespace")
    data = []
    for sample in samples:
        check_dimensions(sample.labels)
        data.append(
            {
                "MetricName": sample.name,
                "Dimensions": [
                    {"Name": key, "Value": value}
                    for key, value in sorted(sample.labels.items())
                ],
                "Timestamp": rfc3339(sample.timestamp),
                "Value": sample.value,
                "Unit": cloudwatch_unit(sample.unit),
            }
        )
    return [
        {
            "Namespace": namespace,
            "MetricData": data[start : start + MAX_PUT_BATCH],
        }
        for start in range(0, len(data), MAX_PUT_BATCH)
    ]


def time_series(
    resource: Mapping[str, Any],
    samples: Sequence[observability.MetricSample],
    prefix: str = "scmle",
) -> list[dict[str, Any]]:
    """``timeSeries.create`` bodies, at most ``MAX_SERIES_BATCH`` per call.

    Args:
        resource: Monitored resource, e.g. ``{"type": "generic_task",
            "labels": {...}}``.
        samples: Samples to write (one point per series per request).
        prefix: Path under ``custom.googleapis.com/``.
    """
    if "type" not in resource or "labels" not in resource:
        raise ValueError("resource needs a type and labels")
    series = []
    for sample in samples:
        kind = f"{METRIC_TYPE_PREFIX}{prefix}/{sample.name}"
        series.append(
            {
                "metric": {
                    "type": kind,
                    "labels": dict(sorted(sample.labels.items())),
                },
                "resource": {
                    "type": resource["type"],
                    "labels": dict(resource["labels"]),
                },
                "metricKind": "GAUGE",
                "valueType": "DOUBLE",
                "unit": sample.unit if sample.unit != "ratio" else "1",
                "points": [
                    {
                        "interval": {"endTime": rfc3339(sample.timestamp)},
                        "value": {"doubleValue": sample.value},
                    }
                ],
            }
        )
    return [
        {"timeSeries": series[start : start + MAX_SERIES_BATCH]}
        for start in range(0, len(series), MAX_SERIES_BATCH)
    ]


def put_metric_data_problems(payload: Mapping[str, Any]) -> list[str]:
    """Reasons a ``PutMetricData`` request would be rejected (empty = ok)."""
    problems = []
    if not NAMESPACE_PATTERN.match(str(payload.get("Namespace", ""))):
        problems.append("Namespace is missing or invalid")
    data = payload.get("MetricData")
    if not isinstance(data, list) or not data:
        return [*problems, "MetricData must be a non-empty list"]
    if len(data) > MAX_PUT_BATCH:
        problems.append("too many MetricData in one request")
    for position, item in enumerate(data):
        where = f"MetricData[{position}]"
        if not item.get("MetricName"):
            problems.append(f"{where}: MetricName is missing")
        if not isinstance(item.get("Value"), int | float):
            problems.append(f"{where}: Value must be a number")
        if item.get("Unit", "None") not in CLOUDWATCH_UNIT_NAMES:
            problems.append(f"{where}: unknown Unit")
        if len(item.get("Dimensions", [])) > MAX_DIMENSIONS:
            problems.append(f"{where}: too many Dimensions")
        for dimension in item.get("Dimensions", []):
            if not dimension.get("Name") or not dimension.get("Value"):
                problems.append(f"{where}: a Dimension needs Name and Value")
    return problems


def embedded_metric_problems(record: Mapping[str, Any]) -> list[str]:
    """Reasons an embedded-metric-format record would not be extracted."""
    problems = []
    meta = record.get("_aws")
    if not isinstance(meta, Mapping):
        return ["_aws is missing"]
    if not isinstance(meta.get("Timestamp"), int):
        problems.append("_aws.Timestamp must be epoch milliseconds")
    directives = meta.get("CloudWatchMetrics")
    if not isinstance(directives, list) or not directives:
        return [*problems, "_aws.CloudWatchMetrics must be a non-empty list"]
    for directive in directives:
        if not directive.get("Namespace"):
            problems.append("a directive has no Namespace")
        for group in directive.get("Dimensions", []):
            for name in group:
                if name not in record:
                    problems.append(f"dimension {name} has no value")
        for metric in directive.get("Metrics", []):
            if metric.get("Name") not in record:
                problems.append(f"metric {metric.get('Name')} has no value")
    return problems


def time_series_problems(payload: Mapping[str, Any]) -> list[str]:
    """Reasons a ``timeSeries.create`` body would be rejected."""
    series = payload.get("timeSeries")
    if not isinstance(series, list) or not series:
        return ["timeSeries must be a non-empty list"]
    problems = []
    if len(series) > MAX_SERIES_BATCH:
        problems.append("too many time series in one request")
    for position, item in enumerate(series):
        where = f"timeSeries[{position}]"
        kind = str(item.get("metric", {}).get("type", ""))
        if not kind.startswith(METRIC_TYPE_PREFIX):
            problems.append(f"{where}: metric.type must be custom")
        elif not METRIC_NAME_PATTERN.match(kind[len(METRIC_TYPE_PREFIX) :]):
            problems.append(f"{where}: metric.type has invalid characters")
        if not item.get("resource", {}).get("type"):
            problems.append(f"{where}: resource.type is missing")
        points = item.get("points", [])
        if len(points) != 1:
            problems.append(f"{where}: send exactly one point per series")
        for point in points:
            end = point.get("interval", {}).get("endTime", "")
            if not re.fullmatch(r"\d{4}-\d\d-\d\dT[\d:.]+Z", end):
                problems.append(f"{where}: endTime must be RFC 3339 UTC")
            if "doubleValue" not in point.get("value", {}):
                problems.append(f"{where}: value needs doubleValue")
        if item.get("metricKind") != "GAUGE":
            problems.append(f"{where}: only GAUGE is produced here")
    return problems


class CloudWatchClient(Protocol):
    """The one call of the CloudWatch SDK client that is used."""

    def put_metric_data(self, **payload: Any) -> Any:
        """Sends a ``PutMetricData`` request."""


class CloudMonitoringClient(Protocol):
    """The one call of the Cloud Monitoring client that is used."""

    def create_time_series(self, name: str, body: dict[str, Any]) -> Any:
        """Sends a ``projects.timeSeries.create`` request."""


class CloudWatchExporter:
    """Sends samples to CloudWatch through an injected client."""

    def __init__(self, client: CloudWatchClient, namespace: str) -> None:
        """Binds the client and a namespace."""
        self.client = client
        self.namespace = namespace

    def export(self, samples: Sequence[observability.MetricSample]) -> int:
        """Sends the samples; returns how many were sent.

        Raises:
            ValueError: If a request fails its own shape check; nothing is
                sent in that case.
        """
        requests = put_metric_data(self.namespace, samples)
        for request in requests:
            problems = put_metric_data_problems(request)
            if problems:
                raise ValueError("; ".join(problems))
        for request in requests:
            self.client.put_metric_data(**request)
        return sum(len(r["MetricData"]) for r in requests)


class CloudMonitoringExporter:
    """Sends samples to Cloud Monitoring through an injected client."""

    def __init__(
        self,
        client: CloudMonitoringClient,
        project: str,
        resource: Mapping[str, Any],
        prefix: str = "scmle",
    ) -> None:
        """Binds the client, project and monitored resource."""
        self.client = client
        self.project = project
        self.resource = resource
        self.prefix = prefix

    def export(self, samples: Sequence[observability.MetricSample]) -> int:
        """Sends the samples; returns how many were sent."""
        bodies = time_series(self.resource, samples, self.prefix)
        for body in bodies:
            problems = time_series_problems(body)
            if problems:
                raise ValueError("; ".join(problems))
        for body in bodies:
            self.client.create_time_series(
                name=f"projects/{self.project}", body=body
            )
        return sum(len(b["timeSeries"]) for b in bodies)


def write_embedded(
    records: Sequence[Mapping[str, Any]], write: Callable[[str], None]
) -> int:
    """Writes embedded-metric-format records as JSON lines; returns count."""
    for record in records:
        write(json.dumps(record, sort_keys=True))
    return len(records)
