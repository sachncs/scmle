"""The ingest stage: reading, validating and admitting the data."""

import dataclasses
import json
from typing import ClassVar

import pandas

from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.contracts.data import provenance
from scmle.data import feature_graph
from scmle.data.readers import streaming
from scmle.data.readers import tabular
import scmle.data.readers.registry
from scmle.pipeline import products
import scmle.pipeline.context
from scmle.pipeline.stages import interface
from scmle.preparation import forecasting
import scmle.runtime.state


@dataclasses.dataclass(frozen=True)
class Intake(products.Derived):
    """The ingested rows and their identity; read again on resume.

    Attributes:
        frame: The ingested rows.
        manifest: The dataset's identity.
        cleaning_notes: Repairs made while reading.
        read_seconds: Seconds the read took.
        stream: The chunked source and its totals, when streamed.
    """

    frame: pandas.DataFrame
    manifest: provenance.DatasetManifest
    cleaning_notes: list[str]
    read_seconds: float
    stream: tuple[streaming.Feed, streaming.Totals] | None

    def summary(self) -> object:
        """The content and schema digests and the repairs, not the timing.

        Returns:
            The identity of the ingested data.
        """
        return {
            "digests": dict(self.manifest.digests),
            "notes": list(self.cleaning_notes),
            "streamed": self.stream is not None,
        }


class IngestStage(interface.Stage):
    """Reads the dataset and fingerprints it."""

    makes: ClassVar[type[products.Product] | None] = Intake
    state: ClassVar[scmle.runtime.state.Progress] = (
        scmle.runtime.state.Progress.VALIDATING
    )

    def describe(
        self,
        context: scmle.pipeline.context.Context,  # noqa: ARG002 -- Stage.describe
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
        self, context: scmle.pipeline.context.Context, data: tasks.Dataset
    ) -> tuple[pandas.DataFrame, tuple[streaming.Feed, streaming.Totals]]:
        """Reads the file in one pass, keeping a sample of both sides.

        A hash of each row's position assigns it to the locked or the
        development side, and a capped sample of each side is kept.

        Args:
            context: The run context, supplying the validation settings.
            data: The data specification naming the source and memory budget.

        Returns:
            The kept development and locked rows together, with a
            `split_side` column marking the side of each row, and the
            streaming source with the pass totals.
        """
        source = streaming.locate(data.uri)
        columns = source.shape()[1]
        development, locked = streaming.budgets(
            columns,
            data.memory_budget_gb or 1.0,
            context.services.spec.validation.test_fraction,
        )
        frame, totals = streaming.sweep(
            source,
            development,
            locked,
            context.services.spec.validation.test_fraction,
            context.services.spec.validation.seed,
        )
        return frame, (source, totals)

    def execute(self, context: scmle.pipeline.context.Context) -> Intake:
        """Loads the data, records its manifest and applies the feature graph.

        An in-memory input frame is copied; otherwise the file is read whole,
        or in one streaming pass when it exceeds the memory budget. Infinite
        values are treated as missing and noted. The source snapshot and the
        dataset manifest are written. A schema change that is incompatible
        with the stored snapshot, or checkpoints computed on different data,
        raise `ConfigError` (through `enrol` and `forbid`).

        Args:
            context: The run context; the intake is applied to it, so a
                caller that runs the stage outside the engine finds it there.

        Returns:
            The intake: the rows, their manifest and the read's repairs.

        Raises:
            UnsupportedError: If the data is not tabular and no
                in-memory frame was supplied, since no other modality has an
                ingestion adapter.
        """
        data = context.services.spec.data
        if data.modality != "tabular" and context.services.input_frame is None:
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
        started = context.services.clock.monotonic()
        stream = None
        if context.services.input_frame is not None:
            frame = context.services.input_frame.copy(deep=False)
        elif self.overflows(data):
            frame, stream = self.read_stream(context, data)
        else:
            frame = tabular.read_table(data.uri)
        if data.horizon:
            frame = forecasting.augment(frame, data)
        seconds = context.services.clock.monotonic() - started
        manifest = tabular.manifested(
            frame, data.target, data.uri, data.modality
        )
        self.enrol(context, manifest, frame)
        frame, infinite = tabular.sanitize(frame)
        notes = [
            f"column '{name}': {count} infinite value(s) treated as missing"
            for name, count in infinite.items()
        ]
        if stream is not None:
            totals = stream[1]
            notes.append(
                f"streamed {totals.rows} rows; {totals.development_sample} "
                f"development rows kept in memory, {totals.locked} rows locked"
            )
        frame = self.graph(context, frame)
        self.forbid(context, manifest)
        context.journal.write_json("dataset_manifest.json", manifest.to_dict())
        context.journal.track(
            dataset_id=manifest.identifier, data_hash=manifest.digests["data"]
        )
        made = Intake(frame, manifest, notes, seconds, stream)
        made.apply(context)
        return made

    @staticmethod
    def graph(
        context: scmle.pipeline.context.Context, frame: pandas.DataFrame
    ) -> pandas.DataFrame:
        """Applies the ordered nodes of the specification's feature graph.

        Args:
            context: The run context, holding the specification.
            frame: The sanitised rows.

        Returns:
            The rows with the ordered graph columns, or `frame` itself when
            the graph has no nodes.
        """
        data = context.services.spec.data
        graph = context.services.spec.feature_graph
        if not graph.nodes:
            return frame
        feature_graph.validate_graph(graph, set(frame.columns), data)
        ordered = feature_graph.segregate(graph)[0]
        frame = feature_graph.Temporal(
            ordered, data.group_column, data.time_column
        ).apply(frame)
        context.journal.emit(
            "features.graph",
            f"feature graph v{graph.version} {graph.fingerprint()}",
            data={
                "fingerprint": graph.fingerprint(),
                "nodes": [n.model_dump(mode="json") for n in graph.nodes],
            },
        )
        return frame

    def enrol(
        self,
        context: scmle.pipeline.context.Context,
        manifest: provenance.DatasetManifest,
        frame: pandas.DataFrame,
    ) -> None:
        """Records the source identity and refuses incompatible schema changes.

        Adds the snapshot to `source_registry.json` and emits a
        `source.snapshot` event.

        Args:
            context: The run context, holding the run store.
            manifest: The dataset's identity.
            frame: The loaded data, whose column names and dtypes are recorded.

        Raises:
            ConfigError: If the stored source registry holds an earlier snapshot
                of the source whose schema is incompatible with the new one.
        """
        snapshot = scmle.data.readers.registry.Observation.restore(
            manifest,
            {
                str(name): str(dtype)
                for name, dtype in zip(frame.columns, frame.dtypes, strict=True)
            },
            context.services.clock.now().isoformat(),
        )
        registry = scmle.data.readers.registry.Sources()
        if context.services.store is not None:
            saved = context.services.store.path("source_registry.json")
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
        record["current_snapshot_id"] = snapshot.snapshot
        record["schema_compatibility"] = compatibility.to_dict()
        context.journal.write_json("source_registry.json", record)
        context.journal.emit(
            "source.snapshot",
            "source identity and schema evolution recorded",
            ("source_registry.json",),
            {
                "source_id": snapshot.source,
                "snapshot_id": snapshot.snapshot,
                "schema_status": compatibility.status,
            },
        )

    def forbid(
        self,
        context: scmle.pipeline.context.Context,
        manifest: provenance.DatasetManifest,
    ) -> None:
        """Rejects resuming trials computed on different data.

        Args:
            context: The run context, holding the checkpoints.
            manifest: The identity of the data just read.

        Raises:
            ConfigError: If the run has checkpoints and the data hash differs
                from the one recorded when they were computed.
        """
        if context.services.store is None or not context.services.checkpoints:
            return
        saved = context.services.store.path("dataset_manifest.json")
        if not saved.exists():
            return
        previous = provenance.DatasetManifest.from_dict(
            json.loads(saved.read_text(encoding="utf-8"))
        ).digests["data"]
        current = manifest.digests["data"]
        if previous != current:
            raise errors.ConfigError(
                "cannot resume: the data changed since the checkpointed "
                "trials were computed",
                {"saved": previous, "current": current},
            )
