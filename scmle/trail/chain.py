"""Chains the derived steps into ``trail.json`` and verifies a trail.

Each step carries the hash of the one before it and its own hash, and the
file is signed with the analyst key over the head and the digest of every
retained record. A step edited after the fact, a step removed, a record
replaced or a signature that does not verify is a finding. The chain shows
that nothing was changed after sealing; it does not show the first write was
honest, and a party holding every key can reseal a changed set (canon P6).
"""

from collections.abc import Mapping
import dataclasses
import hmac
from pathlib import Path
from typing import Any

from scmle.contracts import signing
from scmle.trail import records
from scmle.trail import trace

GENESIS = "0" * 64
FOLLOW_UP = (
    "A trail is evidence that the records agree and were approved by the "
    "named keys. It does not show that the data are correct, that the model "
    "is right, or that a key holder is who they say."
)


@dataclasses.dataclass(frozen=True, slots=True)
class Report:
    """The result of verifying a trail.

    Attributes:
        directory: The trail folder.
        head: Hash of the last step, or the empty string.
        steps: Number of steps in ``trail.json``.
        findings: Every problem, without duplicates.
    """

    directory: str
    head: str
    steps: int
    findings: tuple[trace.Finding, ...]

    @property
    def ok(self) -> bool:
        """True when nothing was found."""
        return not self.findings

    def kinds(self) -> set[str]:
        """The kinds of finding present."""
        return {f.kind for f in self.findings}

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "directory": self.directory,
            "ok": self.ok,
            "head": self.head,
            "steps": self.steps,
            "findings": [f.to_dict() for f in self.findings],
            "note": FOLLOW_UP,
        }


def link(steps: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """Adds ``prev`` and ``hash`` to each step."""
    chained: list[dict[str, Any]] = []
    previous = GENESIS
    for step in steps:
        body = {**step, "prev": previous}
        body["hash"] = records.digest(body)
        chained.append(body)
        previous = body["hash"]
    return chained


def manifest_of(directory: Path) -> dict[str, str]:
    """Digest of every retained record and of ``inputs.json``."""
    held = {}
    base = directory / records.RETAINED
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        held[path.relative_to(directory).as_posix()] = records.sha256_file(path)
    inputs = directory / records.INPUTS
    if inputs.is_file():
        held[records.INPUTS] = records.sha256_file(inputs)
    return held


def signature_for(head: str, held: Mapping[str, str], key: str | bytes) -> str:
    """The analyst's signature over the head and the record digests."""
    return signing.sign_payload(
        {"head": head, "retained": records.digest(dict(held))}, key
    )


def seal(
    directory: str | Path, keys: Mapping[str, str | bytes | None]
) -> Report:
    """Derives the steps from the retained records and writes the trail.

    Args:
        directory: A trail folder with ``retained/`` and ``inputs.json``.
        keys: Keys by role. ``analyst`` signs the file; every role's key is
            also needed to check that role's signatures.

    Returns:
        The verification of what was just written.
    """
    folder = Path(directory)
    derived = trace.derive(folder, keys)
    for name, report in derived.reports.items():
        if report is not None:
            records.write_json(folder / records.REPORTS / name, report)
    derived = trace.derive(folder, keys)
    steps = link(derived.steps)
    held = manifest_of(folder)
    head = steps[-1]["hash"]
    inputs = {}
    if (folder / records.INPUTS).is_file():
        inputs = records.read_json(folder / records.INPUTS)
    analyst = keys.get("analyst")
    document = {
        "format": records.FORMAT,
        "model": inputs.get("model"),
        "version": inputs.get("version"),
        "charter_hash": steps[0]["facts"].get("charter_hash"),
        "steps": steps,
        "head": head,
        "retained": held,
        "builder_signature": (
            signature_for(head, held, analyst) if analyst else None
        ),
    }
    records.write_json(folder / records.TRAIL_FILE, document)
    return verify(folder, keys)


def check_chain(
    document: Mapping[str, Any], keys: Mapping[str, Any]
) -> list[trace.Finding]:
    """Hash chain, head and signature of ``trail.json``."""
    found: list[trace.Finding] = []
    previous = GENESIS
    for index, step in enumerate(document.get("steps", [])):
        body = {k: v for k, v in step.items() if k != "hash"}
        if step.get("prev") != previous or step.get("hash") != records.digest(
            body
        ):
            found.append(
                trace.Finding(
                    trace.BROKEN_CHAIN,
                    str(step.get("kind", index)),
                    f"step {index} does not continue the hash chain",
                )
            )
        previous = str(step.get("hash"))
    if document.get("head") != previous:
        found.append(
            trace.Finding(
                trace.BROKEN_CHAIN, "trail", "the head is not the last step"
            )
        )
    signature = document.get("builder_signature")
    key = keys.get("analyst")
    if not signature:
        found.append(
            trace.Finding(trace.UNSIGNED, "trail", "trail.json is unsigned")
        )
    elif key is None:
        found.append(
            trace.Finding(
                trace.UNSIGNED, "trail", "no analyst key to check trail.json"
            )
        )
    else:
        expected = signature_for(
            str(document.get("head")), document.get("retained", {}), key
        )
        if not hmac.compare_digest(str(signature), expected):
            found.append(
                trace.Finding(
                    trace.UNSIGNED, "trail", "the trail signature is invalid"
                )
            )
    return found


def check_manifest(
    folder: Path, document: Mapping[str, Any]
) -> list[trace.Finding]:
    """Every listed record exists and has the recorded digest."""
    found: list[trace.Finding] = []
    for relative, expected in sorted(document.get("retained", {}).items()):
        path = folder / relative
        if not path.is_file():
            found.append(
                trace.Finding(
                    trace.MISSING_LINK, "trail", f"{relative} is absent"
                )
            )
        elif records.sha256_file(path) != expected:
            found.append(
                trace.Finding(
                    trace.TAMPERED,
                    "trail",
                    f"{relative} differs from the sealed copy",
                )
            )
    return found


def check_steps(
    document: Mapping[str, Any], derived: trace.Derived
) -> list[trace.Finding]:
    """The stored steps must be what the retained records give now."""
    found: list[trace.Finding] = []
    stored = {s.get("kind"): s for s in document.get("steps", [])}
    for step in derived.steps:
        kind = step["kind"]
        if kind not in stored:
            found.append(
                trace.Finding(
                    trace.MISSING_LINK,
                    kind,
                    "the step is absent from the trail",
                )
            )
            continue
        for field in ("facts", "records", "requires", "title"):
            if stored[kind].get(field) != step[field]:
                found.append(
                    trace.Finding(
                        trace.MISMATCH,
                        kind,
                        f"the stored {field} differ from the retained records",
                    )
                )
    return found


def verify(
    directory: str | Path, keys: Mapping[str, str | bytes | None]
) -> Report:
    """Verifies a trail folder against its own retained records.

    Args:
        directory: The trail folder.
        keys: Verification keys by role.

    Returns:
        Every finding. A trail with a missing link, a broken chain, an
        unsigned or unapproved step or an outcome outside the approved
        expectation reports that kind.
    """
    folder = Path(directory)
    path = folder / records.TRAIL_FILE
    if not path.is_file():
        found = (
            trace.Finding(trace.MISSING_LINK, "trail", "trail.json is absent"),
        )
        return Report(str(folder), "", 0, found)
    try:
        document = records.read_json(path)
    except ValueError:
        found = (
            trace.Finding(trace.BROKEN_CHAIN, "trail", "trail.json is bad"),
        )
        return Report(str(folder), "", 0, found)
    problems = check_chain(document, keys) + check_manifest(folder, document)
    derived = trace.derive(folder, keys)
    problems += list(derived.findings) + check_steps(document, derived)
    unique: list[trace.Finding] = []
    for item in problems:
        if item not in unique:
            unique.append(item)
    return Report(
        str(folder),
        str(document.get("head", "")),
        len(document.get("steps", [])),
        tuple(unique),
    )
