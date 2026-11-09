"""Digests of deterministic outputs, compared across Python versions.

`digests()` runs the deterministic framing drafter on a seeded table and
hashes the draft charter and every artifact it writes. The values are pinned
in `tests/fixtures/replay_golden.json` and the test runs on every interpreter
of the CI matrix, so one stored value holds for 3.11 to 3.14. Regenerate with SCMLE_WRITE_GOLDEN=1 only after a reviewed change.
"""

import hashlib
import pathlib
import tempfile

from scmle.data.framing import draft
from tests.unit import test_framing_draft as base


def file_digest(path: pathlib.Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digests() -> dict[str, str]:
    """Charter hash and artifact digests of the seeded framing draft."""
    frame = base.table()
    result = draft.frame(
        base.request(frame),
        frame,
        cost_model=base.MODEL,
        baseline_column="current_rule",
    )
    out: dict[str, str] = {"charter_hash": result.charter.charter_hash}
    with tempfile.TemporaryDirectory() as folder:
        root = pathlib.Path(folder)
        for name in sorted(draft.write_artifacts(result, root)):
            out[name] = file_digest(root / name)
    return out
