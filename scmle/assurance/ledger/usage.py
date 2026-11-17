"""The durable record that a confirmation set was spent.

`C` opens once per run: the ledger refuses a second confirmation of the same
run. A *new* run of the same signed charter on the same data carves the same
confirmation rows again, so without a record outside the run it would test
them a second time and the guarantee ``FDR <= alpha_confirm`` would no longer
hold for the claims that are kept (the user can rerun until a claim passes).

A use is keyed by the charter hash, a fingerprint of the exploration data
(the rows and target of `X`, a function of the data and the charter's split)
and the ids of the confirmation rows. Before `C` is opened, the use is written
as one file per key in the caller's registry folder (the pipeline's folder
of run directories), created with
``O_CREAT | O_EXCL`` so two runs racing for the same key cannot both win. The
run that owns a key may record it again (a resumed run; the ledger then
decides); for any other run `record` says why, and `confirm.confirm` refuses
with `LeakageError` (the `confirm.single_use` gate).

Limits: the record lives with the output directory, so a run written to
another output directory does not see it, and a different split of the same
data (other rows in `X`) is a different key.
"""

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas

import scmle.assurance.ledger.ledger

FOLDER = ".confirmations"


def fingerprint(features: pandas.DataFrame, target: pandas.Series) -> str:
    """Hashes the exploration rows and target, with their index and columns.

    Args:
        features: Exploration features.
        target: Exploration target.

    Returns:
        The SHA-256 hex digest of the row hashes and column names.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps([str(c) for c in features.columns]).encode())
    for part in (features, target):
        rows = pandas.util.hash_pandas_object(part, index=True)
        digest.update(rows.to_numpy().tobytes())
    return digest.hexdigest()


def key(charter: str, data: str, rows: Any) -> str:
    """Digests what identifies one confirmation set.

    Args:
        charter: The charter hash.
        data: The fingerprint of the exploration data.
        rows: The ids of the confirmation rows.

    Returns:
        The hex digest the use is recorded under.
    """
    return str(
        scmle.assurance.ledger.ledger.digest(
            {
                "charter_hash": charter,
                "data": data,
                "rows": [str(i) for i in rows],
            }
        )
    )


class Register:
    """The record, in a registry folder, of one confirmation set."""

    def __init__(
        self,
        folder: str | Path,
        digest: str,
        run: str,
        detail: Mapping[str, str] | None = None,
    ) -> None:
        """Names the record.

        Args:
            folder: The registry folder; the record goes into its `FOLDER`.
            digest: The key from `key`.
            run: Name of the run that would spend the set.
            detail: Further fields written with the record.
        """
        self.path = Path(folder) / FOLDER / f"{digest}.json"
        self.run = run
        self.detail = dict(detail or {})

    def record(self) -> str | None:
        """Writes the use, or accepts it when this run already wrote it.

        Returns:
            None when this run holds the set; otherwise why it is spent
            (another run's record, or a record that cannot be read). The
            caller refuses the confirmation.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps({**self.detail, "run": self.run}, sort_keys=True)
        try:
            handle = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
            )
        except FileExistsError:
            # handled: another run recorded first; check names the spender
            return self.check()
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(body + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return None

    def check(self) -> str | None:
        """Reads who spent the set.

        Returns:
            None when the existing record names this run, otherwise why the
            set is spent.
        """
        try:
            owner = json.loads(self.path.read_text(encoding="utf-8"))["run"]
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
        ):
            # handled: an unreadable record fails closed (the set is spent)
            return (
                f"the confirmation set record {self.path} is unreadable; "
                "the set is treated as spent"
            )
        if owner == self.run:
            return None
        return (
            "this charter already confirmed claims on these confirmation "
            f"rows in run {owner}; a confirmation set is used once "
            f"(record {self.path})"
        )
