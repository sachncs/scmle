"""The serving contract: one scoring path for online and batch callers."""

import abc
from collections.abc import Callable
import contextlib
import logging
import threading
from typing import Any

import pandas

import scmle.artifacts.bundle
from scmle.contracts import errors
from scmle.contracts import serving
from scmle.deploy import provenance
from scmle.inference import guard
from scmle.inference import validation
from scmle.runtime import tracing

LOGGER = logging.getLogger("scmle.serving")

MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_REQUEST_RECORDS = 10_000
MAX_CONCURRENT_REQUESTS = 8
SCORING_TIMEOUT_SECONDS = 30.0
BATCH_CHUNK_ROWS = 10_000
Loader = Callable[[guard.Readiness], scmle.artifacts.bundle.Bundle]


class Server(abc.ABC):
    """Exposes a model bundle to callers."""

    def __init__(
        self,
        bundle: scmle.artifacts.bundle.Bundle | None,
        on_null: str = "impute",
        on_unseen: str = "report",
        warmup_records: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initializes the adapter and its readiness tracker.

        With a bundle, the verify and load steps are marked complete at once;
        the warm-up step is left to `warm`.

        Args:
            bundle: Model to serve. `None` defers loading to a subclass that
                drives `readiness` itself (see `HttpServing`).
            on_null: `impute` lets the fitted imputers handle null features
                and reports them per row; `reject` answers 422 instead.
            on_unseen: `report` scores rows with categories never seen in
                training and flags them per row; `reject` answers 422.
            warmup_records: Records for the warm-up call; by default one is
                derived from the schema.

        Raises:
            ConfigError: If a policy name is unknown.
        """
        if on_null not in serving.NULL_POLICIES:
            raise errors.ConfigError(
                f"on_null must be in {serving.NULL_POLICIES}"
            )
        if on_unseen not in serving.UNSEEN_POLICIES:
            raise errors.ConfigError(
                f"on_unseen must be in {serving.UNSEEN_POLICIES}"
            )
        self.loaded: scmle.artifacts.bundle.Bundle | None = bundle
        self.lock = threading.Lock()
        self.on_null = on_null
        self.on_unseen = on_unseen
        self.warmup_records = warmup_records
        self.readiness = guard.Readiness()
        if bundle is not None:
            note = "bundle provided by the caller"
            self.readiness.begin(guard.Phase.VERIFYING)
            self.readiness.complete(guard.Phase.VERIFYING, note)
            self.readiness.begin(guard.Phase.LOADING)
            self.readiness.complete(guard.Phase.LOADING, note)

    @property
    def bundle(self) -> scmle.artifacts.bundle.Bundle:
        """The loaded bundle.

        Raises:
            UnreadyError: Before verification and loading finished.
        """
        if self.loaded is None:
            raise guard.UnreadyError("the model is not loaded")
        return self.loaded

    def serial(self) -> bool:
        """Whether scoring is serialized: the model has series features.

        Returns:
            True when a loaded bundle has series features, whose remembered
            history makes score and remember one step.
        """
        return bool(self.loaded and self.loaded.schema.series)

    @property
    def artifact_digest(self) -> str:
        """Digest of the manifest of the bundle being served."""
        return provenance.manifest_digest(self.bundle.info.to_dict())

    @property
    def schema_digest(self) -> str:
        """Digest of the input schema of the bundle being served."""
        return serving.schema_digest(self.bundle.schema.anchor())

    def identity(self) -> dict[str, Any]:
        """Reports the model version and digest.

        Returns:
            `model_version` and `artifact_digest`; an empty mapping before
            the model is loaded.
        """
        if self.loaded is None:
            return {}
        return {
            "model_version": self.loaded.info.model_version,
            "artifact_digest": self.artifact_digest,
        }

    def synthesise(
        self, bundle: scmle.artifacts.bundle.Bundle | None = None
    ) -> dict[str, Any] | None:
        """Derives a record that is valid for the schema.

        Args:
            bundle: The bundle whose schema is used; the served one when
                None.

        Returns:
            A record with 0.0 for every numeric feature and the first known
            category (an empty string if none is known) for every
            categorical one; None when the schema has series, extra or
            optional inputs.
        """
        schema = (self.bundle if bundle is None else bundle).schema
        if schema.series or schema.extras or schema.optional:
            return None
        record: dict[str, Any] = dict.fromkeys(schema.numeric, 0.0)
        for name in schema.categorical:
            known = schema.categories.get(name) or [""]
            record[name] = known[0]
        return record

    def rehearse(self, bundle: scmle.artifacts.bundle.Bundle) -> str:
        """Scores the warm-up records once on `bundle`.

        Warm-up records come from `warmup_records` or `synthesise`. The
        call never feeds series history.

        Args:
            bundle: The bundle to warm up.

        Returns:
            What was scored, or why nothing was.
        """
        records = self.warmup_records
        note = "explicit warm-up records"
        if records is None:
            record = self.synthesise(bundle)
            records = None if record is None else [record]
            note = "synthetic record from the schema"
        if records is None:
            return "skipped: no synthetic record for series or extra inputs"
        bundle.score(pandas.DataFrame(records))
        return note

    def warm(self) -> None:
        """Scores once so the first request does not pay for initialization.

        Marks the warm-up step complete (see `rehearse`), or marks the
        service failed if scoring raises. A `ScmleError` from
        `Readiness.begin` (warm-up started out of order) is not caught.
        """
        self.readiness.begin(guard.Phase.WARMING)
        try:
            note = self.rehearse(self.bundle)
        except Exception as error:
            LOGGER.exception("warm-up failed")
            self.readiness.fail(f"warm-up failed: {type(error).__name__}")
            return
        self.readiness.complete(guard.Phase.WARMING, note)

    def respond(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Scores records and builds the structured response.

        A row that breaks a policy set to `reject` makes `enforce`
        raise `PolicyError`.

        Args:
            records: Feature dictionaries. Every record must carry every
                required feature (null is allowed, a missing key is not).

        Returns:
            The version 1 response: predictions, probabilities (classifiers),
            model version, artifact digest, schema digest, schema diagnostics,
            `record_count` and one `rows` entry per record.

        Raises:
            SchemaError: If the records do not satisfy the model's schema.
            ValueError: If no records were supplied.
        """
        if not records:
            raise errors.InputError("records must not be empty")
        required = self.bundle.features
        absent = sorted(
            {
                name
                for record in records
                for name in required
                if name not in record
            }
        )
        if absent:
            raise errors.SchemaError(
                f"records are missing required features: {absent}",
                {
                    "missing": absent,
                    "records": [
                        index
                        for index, record in enumerate(records)
                        if any(name not in record for name in required)
                    ][:20],
                },
            )
        return self.infer(pandas.DataFrame(records))

    def infer(
        self, frame: pandas.DataFrame, remember: bool = True
    ) -> dict[str, Any]:
        """Scores a frame; the one path used by online and batch serving.

        Args:
            frame: Rows to score, with the model's feature columns.
            remember: Whether to add the rows to the bundle's remembered
                series history after scoring.

        Calls run concurrently unless the model has series features, whose
        history makes scoring and remembering one serialized step.

        The policies are enforced after scoring and before the rows are
        remembered, so a rejected batch leaves the history unchanged and
        `PolicyError` propagates. Rows of a request whose caller already got
        a 504 are not remembered either (SRV-2).

        Returns:
            The bundle's score response extended with the schema version,
            artifact digest, schema digest, `record_count` and the per-row
            `rows` report.
        """
        bundle = self.bundle
        # A model without series features is read-only, so its calls run in
        # parallel (SRV-1); with series, scoring reads the history the
        # previous request remembered, so score and remember stay one step.
        held = self.lock if bundle.schema.series else contextlib.nullcontext()
        traced = tracing.TRACING.span(
            "predict",
            {
                "scmle.rows": len(frame),
                "scmle.model_version": bundle.info.model_version,
            },
        )
        with traced, held:
            rows = validation.Validator(bundle.schema).describe(frame)
            response = bundle.score(frame)
            self.enforce(rows)
            if remember and not guard.abandoned():
                bundle.remember(frame)
        response.update(
            {
                "schema_version": serving.SCHEMA_VERSION,
                "artifact_digest": self.artifact_digest,
                "schema_digest": self.schema_digest,
                "record_count": len(rows),
                "rows": rows,
            }
        )
        return response

    def enforce(self, rows: list[dict[str, Any]]) -> None:
        """Applies the null and unseen-category policies to scored rows.

        Args:
            rows: Per-row reports with `missing` and `unseen` entries.

        Raises:
            PolicyError: If `on_null` is `reject` and a row has null
                features, or `on_unseen` is `reject` and a row has unseen
                categories; the details list up to 20 record indexes.
        """
        if self.on_null == "reject":
            nulls = [i for i, row in enumerate(rows) if row["missing"]]
            if nulls:
                raise serving.PolicyError(
                    "null features are rejected by policy",
                    {"records": nulls[:20]},
                )
        if self.on_unseen == "reject":
            unseen = [i for i, row in enumerate(rows) if row["unseen"]]
            if unseen:
                raise serving.PolicyError(
                    "unseen categories are rejected by policy",
                    {"records": unseen[:20]},
                )

    @abc.abstractmethod
    def start(self) -> None:
        """Begins serving."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Stops serving and releases resources."""
