"""Reading the retained records of a trail folder and storing its steps."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scmle.trail import records
from scmle.trail.trace import finding


class Reader:
    """One derivation's state and the record access every step uses.

    Attributes:
        directory: The trail folder.
        base: The folder's ``retained/`` directory.
        keys: Verification keys by role.
        findings: Problems found so far.
        steps: Steps derived so far, by kind.
        cache: Retained JSON records already read, by relative path.
        charter: Digest of the charter, once derived.
        content: SHA-256 of the model manifest file, once derived.
        run: Identifier of the run that produced the model, as named in the
            retained evaluation, once derived.
        inputs: The parsed ``inputs.json``; empty if it could not be read.
        derived: Derived reports by relative path or pseudo-file name.
    """

    def __init__(
        self, directory: Path, keys: Mapping[str, str | bytes | None]
    ) -> None:
        """Binds the folder and the verification keys by role.

        Args:
            directory: The trail folder.
            keys: Verification keys by role.
        """
        self.directory = directory
        self.base = directory / records.RETAINED
        self.keys = dict(keys)
        self.findings: list[finding.Finding] = []
        self.steps: dict[str, dict[str, Any]] = {}
        self.cache: dict[str, Any] = {}
        self.charter = ""
        self.content = ""
        self.run = ""
        self.inputs: dict[str, Any] = {}
        self.derived: dict[str, Any] = {}

    def problem(self, kind: str, step: str, detail: str) -> None:
        """Records a finding.

        Args:
            kind: One of the finding kinds defined in `finding`.
            step: The step kind it belongs to.
            detail: One sentence.
        """
        self.findings.append(finding.Finding(kind, step, detail))

    def need(self, relative: str, step: str) -> Path | None:
        """Finds a retained record, recording a finding when it is absent.

        Args:
            relative: Path relative to ``retained/``.
            step: The step that needs the record.

        Returns:
            The path, or None if the record is not a file.
        """
        path = self.base / relative
        if not path.is_file():
            self.problem(
                finding.MISSING_LINK, step, f"retained/{relative} is absent"
            )
            return None
        return path

    def json(self, relative: str, step: str) -> Any:
        """Reads a retained JSON record, or None with a finding.

        Results are cached, including a failure.

        Args:
            relative: Path relative to ``retained/``.
            step: The step that needs the record.

        Returns:
            The parsed record, or None if it is absent or not valid JSON.
        """
        if relative in self.cache:
            return self.cache[relative]
        path = self.need(relative, step)
        value = None
        if path is not None:
            try:
                value = records.read(path)
            except ValueError:
                # handled: recorded as trace problem
                self.problem(
                    finding.TAMPERED,
                    step,
                    f"retained/{relative} is not valid JSON",
                )
        self.cache[relative] = value
        return value

    def rows(self, relative: str, step: str) -> list[dict[str, Any]] | None:
        """Reads a retained JSON-lines record, or None with a finding.

        Args:
            relative: Path relative to ``retained/``.
            step: The step that needs the record.

        Returns:
            The parsed rows, or None if the record is absent or not valid JSON
            lines.
        """
        path = self.need(relative, step)
        if path is None:
            return None
        try:
            return records.scan(path)
        except ValueError:
            # handled: recorded as trace problem
            self.problem(
                finding.TAMPERED,
                step,
                f"retained/{relative} is not valid JSON lines",
            )
            return None

    def add(
        self,
        kind: str,
        files: list[str],
        facts: dict[str, Any],
        requires: list[str],
    ) -> None:
        """Stores a step with the digest of each record it rests on.

        Args:
            kind: The step kind.
            files: Records the step rests on, relative to ``retained/`` (or
                ``inputs.json``); those that do not exist are left out.
            facts: The facts read from the records.
            requires: Kinds of the steps this one builds on.
        """
        held = {}
        for relative in files:
            path = (
                self.directory / relative
                if relative == records.INPUTS
                else self.base / relative
            )
            if path.is_file():
                held[relative] = records.checksum(path)
        self.steps[kind] = {
            "kind": kind,
            "title": finding.TITLES[kind],
            "records": dict(sorted(held.items())),
            "facts": facts,
            "requires": requires,
        }

    def evaluate(
        self, criteria: list[dict[str, Any]], step: str
    ) -> list[dict[str, Any]]:
        """Checks each criterion against the retained records.

        Args:
            criteria: Criteria with `id`, `path`, `op` and `value`.
            step: The step kind findings are recorded against.

        Returns:
            One result per criterion with its `id`, `path`, `op`, `value`,
            `reason`, the observed value and whether it passed. A criterion
            whose path cannot be resolved, or whose comparison cannot be made
            (unknown operator, incomparable types), is recorded as a finding
            and counts as not passed.
        """
        results = []
        for item in criteria:
            observed: Any = None
            passed = False
            try:
                observed = records.resolve(
                    self.directory, item["path"], self.derived
                )
                passed = records.compare(observed, item["op"], item["value"])
            except (KeyError, FileNotFoundError, ValueError, TypeError):
                # handled: recorded as trace problem
                self.problem(
                    finding.MISSING_LINK,
                    step,
                    f"criterion {item.get('id')} reads {item.get('path')}, "
                    "which cannot be resolved",
                )
            results.append(
                {
                    "id": item.get("id"),
                    "path": item.get("path"),
                    "op": item.get("op"),
                    "value": item.get("value"),
                    "observed": observed,
                    "passed": passed,
                    "reason": item.get("reason", ""),
                }
            )
        return results

    def recompute(
        self, name: str, relative: str, fresh: dict[str, Any], files: list[str]
    ) -> None:
        """Derives a report step: the stored copy must equal the recomputation.

        A finding is recorded if the stored report is absent, or is unreadable
        or differs from `fresh`. The recomputed report is kept for later
        steps.

        Args:
            name: The step kind.
            relative: Path of the stored report, relative to the trail folder.
            fresh: The report recomputed from the retained records.
            files: Records the step rests on.
        """
        stored_path = self.directory / relative
        if not stored_path.is_file():
            self.problem(finding.MISSING_LINK, name, f"{relative} is absent")
        else:
            try:
                stored = records.read(stored_path)
            except ValueError:
                # handled: unparseable stored file reported as mismatch problem
                stored = None
            if stored != fresh:
                self.problem(
                    finding.MISMATCH,
                    name,
                    f"{relative} is not what the retained records give",
                )
        self.derived[relative] = fresh
        self.add(
            name,
            files,
            {"report": relative, "sha256": fresh.get("sha256")},
            ["question"],
        )
