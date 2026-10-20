"""Prediction and outcome events with delayed-label handling.

Predictions are logged when served; outcomes arrive later (or never) and are
joined by event id. Rows without an outcome are carried explicitly as missing
rather than silently dropped, so performance estimates know their coverage.
"""

import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import tempfile
import time
from typing import Any
import uuid

import numpy as np
import pandas as pd

from scml.artifacts import bundle as bundle_module
from scml.contracts import events
from scml.contracts import observability
from scml.monitoring import observability as observability_monitor


@dataclasses.dataclass(frozen=True)
class PredictionEvent:
    """One served prediction.

    Attributes:
        event_id: Identifier used to attach the outcome later.
        model_version: Version label of the model that predicted.
        timestamp: UTC time of the prediction.
        features: The input feature values.
        prediction: The predicted label or value.
        probability: Positive-class probability for binary models.
    """

    event_id: str
    model_version: str
    timestamp: str
    features: dict[str, Any]
    prediction: Any
    probability: float | None = None


@dataclasses.dataclass(frozen=True)
class OutcomeEvent:
    """The true outcome for an earlier prediction.

    Attributes:
        event_id: Identifier of the prediction it belongs to.
        timestamp: UTC time the outcome became known.
        actual: The observed label or value.
    """

    event_id: str
    timestamp: str
    actual: Any


class PredictionLog:
    """Append-only JSONL log of predictions and outcomes."""

    def __init__(
        self,
        path: str | pathlib.Path,
        *,
        observability_policy: observability.ObservabilityPolicy | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initializes the log.

        Args:
            path: JSONL file; created on first write.
            observability_policy: Optional policy enabling a redacted
                observability sidecar beside the prediction log.
            run_id: Stable serving-run identity used in observability events;
                the model version is used when omitted.
        """
        self.path = pathlib.Path(path)
        self.observability_policy = observability_policy
        self.run_id = run_id

    @property
    def observability_path(self) -> pathlib.Path:
        """Path of the redacted sidecar, next to the prediction log."""
        return self.path.with_suffix(".observability.jsonl")

    def append(self, kind: str, record: dict[str, Any]) -> None:
        """Appends one typed record."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"type": kind, **record}, default=str)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def log_predictions(
        self,
        bundle: bundle_module.ModelBundle,
        data: pd.DataFrame,
        event_ids: list[str] | None = None,
    ) -> list[PredictionEvent]:
        """Predicts with a bundle and logs every row.

        Args:
            bundle: Serving model.
            data: Rows to predict.
            event_ids: Optional caller-supplied ids (default: fresh UUIDs).

        Returns:
            The logged events, in row order.
        """
        started = time.perf_counter()
        if event_ids is not None and len(event_ids) != len(data):
            raise ValueError("event_ids must have one id per input row")
        ids = event_ids or [uuid.uuid4().hex for row in range(len(data))]
        try:
            frame = bundle.prepare_frame(data)
            predictions = bundle.predict(data)
            probabilities: list[float | None] = (
                [float(p) for p in bundle.predict_proba(data)[:, 1]]
                if bundle.schema.task == "binary_classification"
                else [None] * len(frame)
            )
        except Exception as error:
            self.log_observability(
                bundle,
                ids[0] if ids else uuid.uuid4().hex,
                started,
                "error",
                input_fingerprint=None,
                error_category=type(error).__name__,
            )
            raise
        try:
            payload = (
                pd.util.hash_pandas_object(frame, index=False)
                .to_numpy(dtype="uint64")
                .tobytes()
            )
        except (TypeError, ValueError):
            payload = frame.to_json(
                orient="split", date_format="iso", default_handler=str
            ).encode("utf-8")
        input_fingerprint = hashlib.sha256(payload).hexdigest()
        for event_id in ids:
            self.log_observability(
                bundle,
                event_id,
                started,
                "success",
                input_fingerprint=input_fingerprint,
            )
        logged = []
        for row, event_id in enumerate(ids):
            event = PredictionEvent(
                event_id=event_id,
                model_version=bundle.info.model_version,
                timestamp=events.utc_timestamp(),
                features={
                    str(k): (None if pd.isna(v) else v)
                    for k, v in frame.iloc[row].to_dict().items()
                },
                prediction=to_plain(predictions[row]),
                probability=probabilities[row],
            )
            self.append("prediction", dataclasses.asdict(event))
            logged.append(event)
        return logged

    def log_observability(
        self,
        bundle: bundle_module.ModelBundle,
        request_id: str,
        started: float,
        outcome: str,
        *,
        input_fingerprint: str | None,
        error_category: str | None = None,
    ) -> None:
        """Writes one redacted inference event when telemetry is enabled."""
        if self.observability_policy is None:
            return
        model_fingerprint = bundle.info.files.get(
            bundle_module.MODEL_FILE, bundle.info.model_version
        )
        event = observability.InferenceEvent(
            request_id=request_id,
            run_id=self.run_id or bundle.info.model_version,
            model_fingerprint=model_fingerprint,
            outcome=outcome,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            timestamp=events.utc_timestamp(),
            input_fingerprint=input_fingerprint,
            error_category=error_category,
            redacted_fields=self.observability_policy.redacted_fields,
        )
        path = self.observability_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"type": "inference", **event.to_dict()},
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

    def enforce_retention(
        self, observed_at: str | None = None
    ) -> dict[str, Any]:
        """Atomically remove redacted events older than the policy window.

        The raw prediction log is intentionally untouched. Retention applies
        only to the redacted observability sidecar, and malformed evidence
        aborts before replacement so an operator never receives a silently
        truncated log.
        """
        if self.observability_policy is None:
            raise ValueError(
                "retention enforcement requires an observability policy"
            )
        observed_text = observed_at or events.utc_timestamp()
        observed_timestamp = observability_monitor.parse_timestamp(
            observed_text, "observed_at"
        )
        cutoff = observed_timestamp - datetime.timedelta(
            days=self.observability_policy.retention_days
        )
        path = self.observability_path
        if not path.exists():
            return {
                "observed_at": observed_text,
                "cutoff": cutoff.isoformat(),
                "retained": 0,
                "removed": 0,
            }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        retained = 0
        removed = 0
        try:
            with os.fdopen(
                descriptor, "w", encoding="utf-8", newline="\n"
            ) as handle:
                with path.open(encoding="utf-8") as source:
                    for line in source:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError(
                                "observability records must be JSON objects"
                            )
                        if record.get("type") != "inference":
                            raise ValueError(
                                "observability records must have type inference"
                            )
                        timestamp = observability_monitor.parse_timestamp(
                            str(record["timestamp"]), "event.timestamp"
                        )
                        if timestamp < cutoff:
                            removed += 1
                            continue
                        handle.write(line)
                        retained += 1
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            pathlib.Path(temporary_name).unlink(missing_ok=True)
            raise
        return {
            "observed_at": observed_text,
            "cutoff": cutoff.isoformat(),
            "retained": retained,
            "removed": removed,
        }

    def record_outcomes(self, outcomes: dict[str, Any]) -> list[OutcomeEvent]:
        """Logs outcomes that became known.

        Args:
            outcomes: Mapping of event id to the observed value.
        """
        logged = []
        for event_id, actual in outcomes.items():
            event = OutcomeEvent(
                event_id, events.utc_timestamp(), to_plain(actual)
            )
            self.append("outcome", dataclasses.asdict(event))
            logged.append(event)
        return logged

    def load(self) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Joins predictions with whatever outcomes have arrived.

        Returns:
            Features per prediction, predictions, and outcomes aligned to the
            same index; outcomes are NaN where the label is still missing.
        """
        predictions: dict[str, dict[str, Any]] = {}
        outcomes: dict[str, Any] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record["type"] == "prediction":
                    predictions[record["event_id"]] = record
                else:
                    outcomes[record["event_id"]] = record["actual"]
        ids = list(predictions)
        features = pd.DataFrame(
            [predictions[i]["features"] for i in ids], index=ids
        )
        predicted = pd.Series(
            [predictions[i]["prediction"] for i in ids], index=ids
        )
        actual = pd.Series([outcomes.get(i, np.nan) for i in ids], index=ids)
        return features, predicted, actual


def to_plain(value: Any) -> Any:
    """Converts numpy scalars to plain Python values."""
    return value.item() if hasattr(value, "item") else value
