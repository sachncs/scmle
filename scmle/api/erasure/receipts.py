"""The signed receipt of an erasure and the journal that makes it resumable.

Under the receipts folder each erasure has:

    <id>/plan.json        the request and every planned step, written once
    <id>/journal.jsonl    one line per finished step, appended and synced
    <id>.json             the signed receipt, written after the last step

Running the same request again reads the stored plan rather than searching
again (half of what it found may already be gone), skips the steps the
journal lists and carries on; once the receipt exists it is returned as is.
"""

from collections.abc import Mapping
import dataclasses
import hmac
import json
import pathlib
from typing import Any
from typing import Self

from scmle import durability
from scmle.contracts import errors
from scmle.contracts import records
from scmle.contracts import signing

FORMAT = 1
PLAN = "plan.json"
JOURNAL = "journal.jsonl"


@dataclasses.dataclass(frozen=True)
class Receipt:
    """What an erasure removed, kept, retired and retrained, signed.

    Attributes:
        identifier: The erasure's id, derived from its request.
        kind: `forget`, `retire` or `delete`.
        reason: Why it was asked for.
        request: What was asked, with a subject's identifiers hashed.
        steps: Each step's action, target, planned evidence and result.
        started: When the plan was written (UTC, ISO 8601).
        finished: When the last step finished.
        format: The receipt layout's version.
        signature: HMAC-SHA256 over every other field.
    """

    identifier: str
    kind: str
    reason: str
    request: records.Layout
    steps: tuple[records.Layout, ...]
    started: str
    finished: str
    format: int = FORMAT
    signature: str = ""

    def preimage(self) -> records.Layout:
        """Returns the fields the signature covers.

        Returns:
            Every field except the signature.
        """
        layout = dataclasses.asdict(self)
        layout.pop("signature")
        layout["steps"] = list(self.steps)
        return layout

    def sign(self, key: str | bytes) -> Self:
        """Returns the receipt signed with a key.

        Args:
            key: The signing key.

        Returns:
            A signed copy.
        """
        return dataclasses.replace(
            self, signature=signing.mac(self.preimage(), key)
        )

    def verify(self, key: str | bytes | None = None) -> bool:
        """Checks the signature.

        Args:
            key: The key, or None for `SCMLE_SIGNING_KEY`.

        Returns:
            Whether the receipt is signed and the signature matches.
        """
        resolved = signing.keying(key)
        if not self.signature or resolved is None:
            return False
        expected = signing.mac(self.preimage(), resolved)
        return hmac.compare_digest(self.signature, expected)

    def tally(self) -> dict[str, int]:
        """Counts the steps by action.

        Returns:
            Action name to the number of steps.
        """
        counts: dict[str, int] = {}
        for step in self.steps:
            counts[step["action"]] = counts.get(step["action"], 0) + 1
        return counts

    def to_dict(self) -> records.Layout:
        """Returns the stored form.

        Returns:
            Every field, the signature included.
        """
        return {**self.preimage(), "signature": self.signature}

    @classmethod
    def load(cls, stored: Mapping[str, Any]) -> Self:
        """Rebuilds a receipt from its stored form.

        Args:
            stored: A record from `to_dict`.

        Returns:
            The receipt.

        Raises:
            ArtifactError: If a field is missing or the format is unknown.
        """
        fields = dict(stored)
        if fields.get("format") != FORMAT:
            raise errors.ArtifactError(
                f"unsupported erasure receipt format {fields.get('format')}"
            )
        try:
            fields["steps"] = tuple(fields["steps"])
            return cls(**fields)
        except (KeyError, TypeError) as error:
            raise errors.ArtifactError(
                f"not an erasure receipt: {error}"
            ) from error

    def write(self, path: pathlib.Path) -> None:
        """Writes the receipt atomically.

        Args:
            path: The file.
        """
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        durability.write(path, text.encode("utf-8"))

    @classmethod
    def read(cls, path: str | pathlib.Path) -> Self:
        """Reads a stored receipt.

        Args:
            path: The file.

        Returns:
            The receipt.

        Raises:
            ArtifactError: If the file cannot be read or is not a receipt.
        """
        try:
            stored = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise errors.ArtifactError(
                f"cannot read the receipt {path}: {error}"
            ) from error
        if not isinstance(stored, dict):
            raise errors.ArtifactError(f"{path} is not an erasure receipt")
        return cls.load(stored)


class Journal:
    """The stored plan of one erasure and the steps it has finished.

    Attributes:
        folder: The erasure's folder under the receipts folder.
    """

    def __init__(self, folder: pathlib.Path) -> None:
        """Binds the journal to its folder (created when the plan is set).

        Args:
            folder: The erasure's folder.
        """
        self.folder = folder

    def plan(self) -> records.Layout | None:
        """Returns the stored plan.

        Returns:
            The plan, or None if none was written.

        Raises:
            ArtifactError: If the plan file exists but cannot be read.
        """
        path = self.folder / PLAN
        if not path.is_file():
            return None
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise errors.ArtifactError(
                f"cannot read the erasure plan {path}: {error}"
            ) from error
        if not isinstance(stored, dict):
            raise errors.ArtifactError(f"{path} is not an erasure plan")
        return stored

    def record(self, plan: records.Layout) -> None:
        """Writes the plan once, before any step runs.

        Args:
            plan: The request, the start time and the steps.
        """
        self.folder.mkdir(parents=True, exist_ok=True)
        text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
        durability.write(self.folder / PLAN, text.encode("utf-8"))

    def done(self) -> dict[int, records.Layout]:
        """Returns the finished steps and their results.

        A last line cut short by a crash is ignored: its step runs again,
        which is safe because every step is idempotent.

        Returns:
            Step index to the step's result.
        """
        path = self.folder / JOURNAL
        if not path.is_file():
            return {}
        finished: dict[int, records.Layout] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # handled: a torn last line; that step runs again
                continue
            finished[int(row["step"])] = row["result"]
        return finished

    def finish(self, index: int, result: records.Layout) -> None:
        """Records a finished step, synced before the next one starts.

        Args:
            index: The step's position in the plan.
            result: What the step did.
        """
        path = self.folder / JOURNAL
        line = json.dumps({"step": index, "result": result}, sort_keys=True)
        torn = path.is_file() and not path.read_bytes().endswith(b"\n")
        durability.append(path, ("\n" if torn else "") + line + "\n")
        durability.sync(path)
