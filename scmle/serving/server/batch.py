"""Batch scoring of files through the serving contract.

The input is streamed a chunk at a time (`Loader.stream`) and each scored
chunk is appended to `<destination>.partial`, so memory holds one chunk
however large the file is. After every chunk a journal,
`<destination>.progress.json`, records the rows and bytes written; a run
that was interrupted resumes after the last recorded chunk when the same
source, bundle and policies are scored again. The finished file replaces
`<destination>` atomically and the journal is removed.
"""

import dataclasses
import gc
import hashlib
import json
import logging
import os
import pathlib
from typing import Any
from typing import BinaryIO

import pandas

from scmle import durability
from scmle.contracts import errors
from scmle.contracts import serving
from scmle.data.readers import tabular
from scmle.runtime import tracing
from scmle.serving.server import contract

LOGGER = logging.getLogger("scmle.serving")
# Version of the progress journal; a journal of another version is ignored.
JOURNAL = 1
# Bytes hashed at a time when a resumed run re-reads its partial output.
BLOCK = 1024 * 1024


@dataclasses.dataclass
class Progress:
    """How much of a scoring's output is written, as its journal records.

    Attributes:
        rows: Rows scored and written.
        size: Bytes of output written.
        flagged: Rows with nulls and rows with unseen categories so far.
    """

    rows: int = 0
    size: int = 0
    flagged: dict[str, int] = dataclasses.field(
        default_factory=lambda: dict.fromkeys(
            ("rows_with_nulls", "rows_with_unseen_categories"), 0
        )
    )

    def encode(self, mark: dict[str, str | int]) -> bytes:
        """The journal of this progress, in its stored format.

        Args:
            mark: The fingerprint of the scoring.

        Returns:
            The JSON journal: version, fingerprint, rows, bytes, flagged.
        """
        record = {
            "version": JOURNAL,
            "fingerprint": mark,
            "rows": self.rows,
            "bytes": self.size,
            "flagged": self.flagged,
        }
        return json.dumps(record, sort_keys=True).encode("utf-8")


class Batcher(contract.Server):
    """Scores CSV / Parquet files through the same path as online serving."""

    def start(self) -> None:
        """Nothing to start for batch scoring."""

    def stop(self) -> None:
        """Nothing to stop for batch scoring."""

    def fingerprint(self, source: pathlib.Path) -> dict[str, str | int]:
        """Identifies one scoring of one input, for resuming it.

        Args:
            source: The input file.

        Returns:
            The input's absolute path, size and modification time, the
            artifact digest of the bundle, and the two policies.
        """
        status = source.stat()
        return {
            "source": str(source.resolve()),
            "size": status.st_size,
            "modified": status.st_mtime_ns,
            "artifact_digest": self.digest,
            "on_null": self.missing,
            "on_unseen": self.unseen,
        }

    def recall(
        self,
        journal: pathlib.Path,
        partial: pathlib.Path,
        mark: dict[str, str | int],
    ) -> Progress:
        """Reads the progress of an interrupted run of the same scoring.

        Args:
            journal: The progress journal.
            partial: The partial output it describes.
            mark: The fingerprint of this scoring.

        Returns:
            The recorded progress, or a fresh start when there is no
            journal, it is unreadable, it describes another scoring, or the
            partial output is shorter than it says.
        """
        if not journal.exists() or not partial.exists():
            return Progress()
        try:
            state = json.loads(journal.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning(
                "unreadable batch journal %s; starting over", journal
            )
            return Progress()
        if (
            not isinstance(state, dict)
            or state.get("version") != JOURNAL
            or state.get("fingerprint") != mark
            or partial.stat().st_size < int(state.get("bytes", 0))
        ):
            LOGGER.warning(
                "batch journal %s is for another input, bundle or policy; "
                "starting over",
                journal,
            )
            return Progress()
        LOGGER.info("resuming after %d scored rows", state["rows"])
        return Progress(
            int(state["rows"]), int(state["bytes"]), dict(state["flagged"])
        )

    def tabulate(self, response: dict[str, Any]) -> pandas.DataFrame:
        """Turns one scoring response into output rows.

        Args:
            response: The response of `infer`.

        Returns:
            A `prediction` column and, for classifiers, one `proba_<class>`
            column per class.
        """
        result = pandas.DataFrame({"prediction": response["predictions"]})
        if "classes" in response:
            probabilities = response["probabilities"]
            for index, label in enumerate(response["classes"]):
                result[f"proba_{label}"] = [row[index] for row in probabilities]
        return result

    def append(
        self, out: BinaryIO, response: dict[str, Any], state: Progress
    ) -> bytes:
        """Writes one scored chunk durably and counts it in `state`.

        Args:
            out: The partial output, positioned at its end.
            response: The chunk's response of `infer`.
            state: The progress, updated in place.

        Returns:
            The bytes written.
        """
        content = self.tabulate(response).to_csv(
            index=False, header=state.size == 0
        )
        payload = content.encode("utf-8")
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())
        state.rows += len(response["rows"])
        state.size += len(payload)
        for key, flag in (
            ("rows_with_nulls", "missing"),
            ("rows_with_unseen_categories", "unseen"),
        ):
            state.flagged[key] += sum(
                1 for row in response["rows"] if row[flag]
            )
        return payload

    def drive(
        self,
        source: pathlib.Path,
        partial: pathlib.Path,
        journal: pathlib.Path,
        chunk_rows: int,
        mark: dict[str, str | int],
    ) -> tuple[Progress, str]:
        """Scores the chunks not yet in `partial`, journaling each one.

        Args:
            source: The input file.
            partial: The partial output.
            journal: The progress journal.
            chunk_rows: Rows read and scored at a time.
            mark: The fingerprint of this scoring.

        Returns:
            The final progress and the SHA-256 of the whole output.
        """
        state = self.recall(journal, partial, mark)
        digest = hashlib.sha256()
        with partial.open("r+b" if state.size else "w+b") as out:
            out.truncate(state.size)
            while block := out.read(BLOCK):
                digest.update(block)
            done, seen = state.rows, 0
            for chunk in tabular.Loader().stream(source, chunk_rows):
                start, seen = seen, seen + len(chunk)
                skipped = min(max(done - start, 0), len(chunk))
                if skipped:
                    self.bundle.remember(chunk.iloc[:skipped])
                if skipped == len(chunk):
                    continue
                rest = chunk.iloc[skipped:].reset_index(drop=True)
                digest.update(self.append(out, self.infer(rest), state))
                durability.write_atomically(journal, state.encode(mark))
                # Scoring leaves reference cycles (scikit-learn routing,
                # joblib) the generational collector frees only after many
                # more allocations; collecting here holds memory at a chunk.
                gc.collect()
        return state, "sha256:" + digest.hexdigest()

    def score(
        self,
        source: str | pathlib.Path,
        destination: str | pathlib.Path,
        chunk_rows: int = contract.BATCH_CHUNK_ROWS,
        resume: bool = True,
    ) -> int:
        """Scores a file chunk by chunk and writes predictions.

        The output CSV has a `prediction` column and, for classifiers, one
        `proba_<class>` column per class. It is accompanied by
        `<destination>.manifest.json`, which carries the model version,
        artifact digest, schema digest, row count, the SHA-256 of the CSV and
        the counts of rows with nulls and with unseen categories. Chunks are
        scored through `infer`, so a policy set to `reject` raises
        `PolicyError` before `destination` is written, and the scored rows
        are added to the bundle's remembered series history (rows skipped
        on resume are remembered without being scored again).

        Args:
            source: Input CSV / Parquet (any format `Loader` reads).
            destination: Output CSV.
            chunk_rows: Rows read and scored at a time.
            resume: Whether to continue an interrupted run of the same
                scoring instead of starting over.

        Returns:
            Number of rows scored.

        Raises:
            ConfigError: If `chunk_rows` is not positive.
            DataError: If `source` does not exist or cannot be read.
        """
        if chunk_rows < 1:
            raise errors.ConfigError("chunk_rows must be positive")
        origin = pathlib.Path(source)
        if not origin.exists():
            raise errors.DataError(f"dataset not found: {origin}")
        target = pathlib.Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        journal = target.with_name(target.name + ".progress.json")
        if not resume:
            journal.unlink(missing_ok=True)
        with tracing.TRACING.span("batch", {"scmle.chunk_rows": chunk_rows}):
            state, digest = self.drive(
                origin, partial, journal, chunk_rows, self.fingerprint(origin)
            )
        durability.swap(partial, target)
        durability.sync_directory(target.parent)
        manifest = {
            "schema_version": serving.SCHEMA_VERSION,
            "model_version": self.bundle.info.model_version,
            "artifact_digest": self.digest,
            "schema_digest": self.layout,
            "rows": state.rows,
            "output_sha256": digest,
            **state.flagged,
        }
        durability.write_atomically(
            target.with_name(target.name + ".manifest.json"),
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        )
        journal.unlink(missing_ok=True)
        return state.rows
