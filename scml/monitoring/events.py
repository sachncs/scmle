"""Prediction and outcome events with delayed-label handling.

Predictions are logged when served; outcomes arrive later (or never) and are
joined by event id. Rows without an outcome are carried explicitly as missing
rather than silently dropped, so performance estimates know their coverage.
"""

import dataclasses
import json
import pathlib
from typing import Any
import uuid

import numpy as np
import pandas as pd

from scml.artifacts import bundle as bundle_module
from scml.contracts import events


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

    def __init__(self, path: str | pathlib.Path) -> None:
        """Initializes the log.

        Args:
            path: JSONL file; created on first write.
        """
        self.path = pathlib.Path(path)

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
        frame = bundle.prepare_frame(data)
        predictions = bundle.predict(data)
        probabilities: list[float | None] = (
            [float(p) for p in bundle.predict_proba(data)[:, 1]]
            if bundle.schema.task == "binary_classification"
            else [None] * len(frame)
        )
        ids = event_ids or [uuid.uuid4().hex for row in range(len(frame))]
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
