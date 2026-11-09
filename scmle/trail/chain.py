"""Chains the derived steps into ``trail.json`` and verifies a trail.

Each step carries the hash of the one before it and its own hash, and the
file is signed with the analyst key over the head and the digest of every
retained record. A step edited after the fact, a step removed, a record
replaced or a signature that does not verify is a finding. The chain shows
that nothing was changed after sealing; it does not show the first write was
honest, and a party holding every key can reseal a changed set.
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
class Verification:
    """The result of verifying a trail.

    Attributes:
        directory: The trail folder.
        head: Hash of the last step as recorded in ``trail.json``, or the
            empty string if that file could not be read.
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
        """Collects the distinct kinds of finding present.

        Returns:
            The set of finding kinds.
        """
        return {f.kind for f in self.findings}

    def to_dict(self) -> dict[str, Any]:
        """Builds a JSON-ready record.

        Returns:
            A dict with the directory, `ok`, the head, the step count, the
            findings and a note on what a trail does not show.
        """
        return {
            "directory": self.directory,
            "ok": self.ok,
            "head": self.head,
            "steps": self.steps,
            "findings": [f.to_dict() for f in self.findings],
            "note": FOLLOW_UP,
        }


def link(steps: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """Chains the steps by adding ``prev`` and ``hash`` to each.

    Args:
        steps: The derived steps, in order.

    Returns:
        Copies of the steps in which each carries the hash of the previous one
        (`GENESIS` for the first) and its own hash.
    """
    chained: list[dict[str, Any]] = []
    previous = GENESIS
    for step in steps:
        body = {**step, "prev": previous}
        body["hash"] = records.digest(body)
        chained.append(body)
        previous = body["hash"]
    return chained


def manifest_of(directory: Path) -> dict[str, str]:
    """Digests every file under ``retained/`` and ``inputs.json`` if present.

    Args:
        directory: The trail folder.

    Returns:
        A mapping from each file's path relative to the folder (with forward
        slashes, so ``retained/...`` or ``inputs.json``) to its hex SHA-256,
        with the retained paths sorted and `inputs.json` last.
    """
    held = {}
    base = directory / records.RETAINED
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        held[path.relative_to(directory).as_posix()] = records.sha256_file(path)
    inputs = directory / records.INPUTS
    if inputs.is_file():
        held[records.INPUTS] = records.sha256_file(inputs)
    return held


def signature_for(head: str, held: Mapping[str, str], key: str | bytes) -> str:
    """Signs the head and the record digests.

    Args:
        head: Hash of the last step.
        held: Digest of each retained record, by path.
        key: The analyst's key.

    Returns:
        The signature over the head and the digest of the record digests.
    """
    return signing.sign_payload(
        {"head": head, "retained": records.digest(dict(held))}, key
    )


def seal(
    directory: str | Path, keys: Mapping[str, str | bytes | None]
) -> Verification:
    """Derives the steps from the retained records and writes the trail.

    The derived reports are written first and the steps are derived again, so
    the steps see the reports' digests.

    Args:
        directory: A trail folder with ``retained/`` and ``inputs.json``.
        keys: Keys by role. ``analyst`` signs the file (without it
            ``builder_signature`` is null); each role's key is also needed to
            check that role's signature in ``inputs.json``.

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
    """Checks the hash chain, head and signature of ``trail.json``.

    Args:
        document: The parsed ``trail.json``.
        keys: Verification keys by role; the analyst key checks the signature.

    Returns:
        A finding for each step that does not continue the chain, one if the
        head is not the last step's hash, and one if the signature is absent,
        cannot be checked for lack of an analyst key, or is invalid.
    """
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
    """Checks that every listed record exists and has the recorded digest.

    Args:
        folder: The trail folder.
        document: The parsed ``trail.json``.

    Returns:
        A finding for each record that is absent or differs from the sealed
        copy.
    """
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
    """Checks that the stored steps are what the retained records give now.

    Args:
        document: The parsed ``trail.json``.
        derived: The steps and findings derived from the retained records.

    Returns:
        A finding for each derived step that is absent from the trail, and a
        finding for each of the facts, records, requirements and title of a
        stored step that differs from the derived one. Stored steps with no
        derived counterpart are ignored.
    """
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
) -> Verification:
    """Verifies a trail folder against its own retained records.

    Args:
        directory: The trail folder.
        keys: Verification keys by role.

    Returns:
        The report, with the findings of `check_chain`, `check_manifest`,
        `check_steps` and the derivation itself, without duplicates. An absent
        ``trail.json`` gives a single missing-link finding and a file that is
        not JSON a single broken-chain finding, each with an empty head and
        zero steps.
    """
    folder = Path(directory)
    path = folder / records.TRAIL_FILE
    if not path.is_file():
        found = (
            trace.Finding(trace.MISSING_LINK, "trail", "trail.json is absent"),
        )
        return Verification(str(folder), "", 0, found)
    try:
        document = records.read_json(path)
    except ValueError:
        found = (
            trace.Finding(trace.BROKEN_CHAIN, "trail", "trail.json is bad"),
        )
        return Verification(str(folder), "", 0, found)
    problems = check_chain(document, keys) + check_manifest(folder, document)
    derived = trace.derive(folder, keys)
    problems += list(derived.findings) + check_steps(document, derived)
    unique: list[trace.Finding] = []
    for item in problems:
        if item not in unique:
            unique.append(item)
    return Verification(
        str(folder),
        str(document.get("head", "")),
        len(document.get("steps", [])),
        tuple(unique),
    )
