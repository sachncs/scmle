"""The ingest stage: reading, validating and admitting the data."""

import datetime
import json
import time
from typing import ClassVar

import pandas

from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.contracts import tasks
import scmle.core.context
from scmle.core.stages import interface
from scmle.data import feature_graph
from scmle.data.readers import streaming
from scmle.data.readers import tabular
import scmle.data.readers.registry
from scmle.preparation import forecasting
import scmle.runtime.state


class IngestStage(interface.Stage):
    """Reads the dataset and fingerprints it."""

    state: ClassVar[scmle.runtime.state.Progress] = (
        scmle.runtime.state.Progress.VALIDATING
    )

    def describe(
        self,
        context: scmle.core.context.Context,  # noqa: ARG002 -- Stage.describe
    ) -> str:
        """Names the stage.

        Args:
            context: The run context; not consulted.

        Returns:
            The text "ingest dataset".
        """
        return "ingest dataset"

    @staticmethod
    def overflows(data: tasks.Dataset) -> bool:
        """Returns whether the file is bigger than the memory budget.

        Args:
            data: The data specification, giving the source and the optional
                memory budget in gigabytes.

        Returns:
            True if a memory budget is set and the source's estimated size
            exceeds it; False if no budget is set.
        """
        if data.memory_budget_gb is None:
            return False
        size = streaming.locate(data.uri).footprint()
        return size > data.memory_budget_gb * 1e9

    def read_stream(
        self, context: scmle.core.context.Context, data: tasks.Dataset
    ) -> pandas.DataFrame:
        """Reads the file in one pass, keeping a sample of both sides.

        A hash of each row's position assigns it to the locked or the
        development side, and a capped sample of each side is kept. The
        streaming source and the pass totals are stored in `context.stream`.

        Args:
            context: The run context, supplying the validation settings.
            data: The data specification naming the source and memory budget.

        Returns:
            The kept development and locked rows together, with a
            `split_side` column marking the side of each row.
        """
        source = streaming.locate(data.uri)
        columns = source.shape()[1]
        development, locked = streaming.budgets(
            columns,
            data.memory_budget_gb or 1.0,
            context.spec.validation.test_fraction,
        )
        frame, totals = streaming.sweep(
            source,
            development,
            locked,
            context.spec.validation.test_fraction,
            context.spec.validation.seed,
        )
        context.stream = (source, totals)
        return frame

    def execute(self, context: scmle.core.context.Context) -> None:
        """Loads the data, records its manifest and applies the feature graph.

        An in-memory input frame is copied; otherwise the file is read whole,
        or in one streaming pass when it exceeds the memory budget. Infinite
        values are treated as missing and noted. The source snapshot and the
        dataset manifest are written. A schema change that is incompatible
        with the stored snapshot, or checkpoints computed on different data,
        raise `ConfigError` (through `enrol` and
        `forbid`).

        Args:
            context: The run context; receives the frame, manifest and cleaning
                notes.

        Raises:
            UnsupportedError: If the data is not tabular and no
                in-memory frame was supplied, since no other modality has an
                ingestion adapter.
        """
        data = context.spec.data
        if data.modality != "tabular" and context.input_frame is None:
            raise capabilities.UnsupportedError(
                "file-backed modality has no registered ingestion adapter",
                {
                    "status": "unsupported",
                    "reason": "file-backed modality has no ingestion adapter",
                    "modality": data.modality,
                    "uri": data.uri,
                    "supported_modalities": ["tabular"],
                },
            )
        started = time.monotonic()
        if context.input_frame is not None:
            frame = context.input_frame.copy(deep=False)
        elif self.overflows(data):
            frame = self.read_stream(context, data)
        else:
            frame = tabular.read_table(data.uri)
        if data.horizon:
            frame = forecasting.augment(frame, data)
        context.read_seconds = time.monotonic() - started
        context.manifest = tabular.manifested(
            frame, data.target, data.uri, data.modality
        )
        self.enrol(context, frame)
        frame, infinite = tabular.sanitize(frame)
        context.cleaning_notes = [
            f"column '{name}': {count} infinite value(s) treated as missing"
            for name, count in infinite.items()
        ]
        if context.stream is not None:
            totals = context.stream[1]
            context.cleaning_notes.append(
                f"streamed {totals.rows} rows; {totals.development_sample} "
                f"development rows kept in memory, {totals.locked} rows locked"
            )
        graph = context.spec.feature_graph
        if graph.nodes:
            feature_graph.validate_graph(graph, set(frame.columns), data)
            ordered = feature_graph.segregate(graph)[0]
            frame = feature_graph.Temporal(
                ordered, data.group_column, data.time_column
            ).apply(frame)
            context.emit(
                "features.graph",
                f"feature graph v{graph.version} {graph.fingerprint()}",
                data={
                    "fingerprint": graph.fingerprint(),
                    "nodes": [n.model_dump(mode="json") for n in graph.nodes],
                },
            )
        context.frame = frame
        self.forbid(context)
        context.write_json("dataset_manifest.json", context.manifest.to_dict())
        context.track(
            dataset_id=context.manifest.dataset_id,
            data_hash=context.manifest.data_hash,
        )

    def enrol(
        self, context: scmle.core.context.Context, frame: pandas.DataFrame
    ) -> None:
        """Records the source identity and refuses incompatible schema changes.

        Adds the snapshot to `source_registry.json` and emits a
        `source.snapshot` event.

        Args:
            context: The run context, holding the manifest and the run store.
            frame: The loaded data, whose column names and dtypes are recorded.

        Raises:
            ConfigError: If the stored source registry holds an earlier snapshot
                of the source whose schema is incompatible with the new one.
        """
        manifest = scmle.core.context.require(context.manifest, "manifest")
        snapshot = scmle.data.readers.registry.Observation.restore(
            manifest,
            {
                str(name): str(dtype)
                for name, dtype in zip(frame.columns, frame.dtypes, strict=True)
            },
            datetime.datetime.now(datetime.UTC).isoformat(),
        )
        registry = scmle.data.readers.registry.Sources()
        if context.store is not None:
            saved = context.store.path("source_registry.json")
            if saved.exists():
                registry = scmle.data.readers.registry.Sources.from_json(
                    saved.read_text(encoding="utf-8")
                )
        compatibility = registry.compatibility(snapshot)
        if (
            compatibility.previous_snapshot_id is not None
            and not compatibility.compatible
        ):
            raise errors.ConfigError(
                "source schema changed incompatibly while resuming: "
                f"{compatibility.to_dict()}"
            )
        registry.register(snapshot)
        record = registry.to_dict()
        record["current_snapshot_id"] = snapshot.snapshot_id
        record["schema_compatibility"] = compatibility.to_dict()
        context.write_json("source_registry.json", record)
        context.emit(
            "source.snapshot",
            "source identity and schema evolution recorded",
            ("source_registry.json",),
            {
                "source_id": snapshot.source_id,
                "snapshot_id": snapshot.snapshot_id,
                "schema_status": compatibility.status,
            },
        )

    def forbid(self, context: scmle.core.context.Context) -> None:
        """Rejects resuming trials computed on different data.

        Args:
            context: The run context, holding the checkpoints and manifest.

        Raises:
            ConfigError: If the run has checkpoints and the data hash differs
                from the one recorded when they were computed.
        """
        if context.store is None or not context.checkpoints:
            return
        saved = context.store.path("dataset_manifest.json")
        if not saved.exists():
            return
        previous = json.loads(saved.read_text(encoding="utf-8")).get(
            "data_hash"
        )
        current = scmle.core.context.require(
            context.manifest, "manifest"
        ).data_hash
        if previous != current:
            raise errors.ConfigError(
                "cannot resume: the data changed since the checkpointed "
                "trials were computed",
                {"saved": previous, "current": current},
            )
