"""Batch scoring of files through the serving contract."""

import json
import pathlib
from typing import Any

import pandas

from scmle import durability
from scmle.contracts import errors
from scmle.contracts import serving
from scmle.deploy import provenance
from scmle.serving.server import contract


class Batcher(contract.Server):
    """Scores CSV / Parquet files through the same path as online serving."""

    def start(self) -> None:
        """Nothing to start for batch scoring."""

    def stop(self) -> None:
        """Nothing to stop for batch scoring."""

    def score_file(
        self,
        source: str | pathlib.Path,
        destination: str | pathlib.Path,
        chunk_rows: int = contract.BATCH_CHUNK_ROWS,
    ) -> int:
        """Scores a file and writes predictions (and probabilities).

        The output CSV has a `prediction` column and, for classifiers, one
        `proba_<class>` column per class. It is accompanied by
        `<destination>.manifest.json`, which carries the model version,
        artifact digest, schema digest, row count, the SHA-256 of the CSV and
        the counts of rows with nulls and with unseen categories. Chunks are
        scored through `infer`, so a policy set to `reject` raises
        `PolicyError` before anything is written, and the scored rows are
        added to the bundle's remembered series history.

        Args:
            source: Input CSV / Parquet.
            destination: Output CSV.
            chunk_rows: Rows scored per call.

        Returns:
            Number of rows scored.

        Raises:
            ConfigError: If `chunk_rows` is not positive.
        """
        if chunk_rows < 1:
            raise errors.ConfigError("chunk_rows must be positive")
        frame = self.bundle.read_frame(source)
        pieces = []
        flagged = {"rows_with_nulls": 0, "rows_with_unseen_categories": 0}
        response: dict[str, Any] = {}
        for start in range(0, max(len(frame), 1), chunk_rows):
            chunk = frame.iloc[start : start + chunk_rows].reset_index(
                drop=True
            )
            response = self.infer(chunk)
            result = pandas.DataFrame({"prediction": response["predictions"]})
            if "classes" in response:
                probabilities = response["probabilities"]
                for index, label in enumerate(response["classes"]):
                    result[f"proba_{label}"] = [
                        row[index] for row in probabilities
                    ]
            pieces.append(result)
            flagged["rows_with_nulls"] += sum(
                1 for row in response["rows"] if row["missing"]
            )
            flagged["rows_with_unseen_categories"] += sum(
                1 for row in response["rows"] if row["unseen"]
            )
        output = pandas.concat(pieces, ignore_index=True)
        content = output.to_csv(index=False).encode("utf-8")
        target = pathlib.Path(destination)
        durability.write_atomically(target, content)
        manifest = {
            "schema_version": serving.SCHEMA_VERSION,
            "model_version": response.get(
                "model_version", self.bundle.info.model_version
            ),
            "artifact_digest": self.artifact_digest,
            "schema_digest": self.schema_digest,
            "rows": len(output),
            "output_sha256": provenance.sha256_digest(content),
            **flagged,
        }
        durability.write_atomically(
            target.with_name(target.name + ".manifest.json"),
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        )
        return len(output)
