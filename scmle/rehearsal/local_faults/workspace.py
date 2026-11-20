"""The evidence workspace every local fault scenario writes into."""

import dataclasses
import json
import pathlib
import subprocess  # nosec B404 - bounded child process is the fault target.
import sys
from typing import Any

from scmle.contracts import reliability
import scmle.governance.reliability


@dataclasses.dataclass(frozen=True, slots=True)
class Workspace:
    """An isolated evidence workspace and the artifacts written into it."""

    root: pathlib.Path

    def __post_init__(self) -> None:
        """Creates and validates the isolated evidence workspace."""
        workspace = pathlib.Path(self.root).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "root", workspace)

    def artifact(
        self,
        scenario: reliability.Contingency,
        evidence_type: str,
        observation: str,
        details: dict[str, Any],
    ) -> scmle.governance.reliability.Emission:
        """Builds an evidence artifact that records a safe outcome as observed.

        The outcome is always marked as observed; nothing here checks that
        `details` supports it.

        Args:
            scenario: The scenario the evidence belongs to.
            evidence_type: The kind of evidence.
            observation: What was observed.
            details: Facts recorded with the observation; they must be JSON
                serializable.

        Returns:
            The artifact: compact, key-sorted ASCII JSON bytes holding the
            scenario value, evidence type, observation and details, with the
            safe-outcome-observed outcome and the observation text.
        """
        content = json.dumps(
            {
                "scenario": scenario.value,
                "evidence_type": evidence_type,
                "observation": observation,
                "details": details,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return scmle.governance.reliability.Emission(
            kind=evidence_type,
            content=content,
            outcome=reliability.Outcome.SAFE_OUTCOME_OBSERVED,
            observation=observation,
        )

    def write(self, path: pathlib.Path, record: dict[str, Any]) -> None:
        """Writes JSON with the evidence harness's atomic, synced write.

        Args:
            path: The destination file.
            record: The content, written as compact sorted JSON.
        """
        scmle.governance.reliability.Harness.write(
            path,
            json.dumps(
                record,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8"),
        )

    def halt(self) -> tuple[bool, int | None]:
        """Starts a sleeping child process, terminates it and reports how.

        The child is a Python interpreter that sleeps for 60 seconds. It is
        given 0.05 seconds to exit on its own, then terminated and waited for
        up to 2 seconds. A child that outlives that is killed and
        `subprocess.TimeoutExpired` propagates.

        Returns:
            A pair: whether the child had to be terminated, and its exit code
            (negative signal number on POSIX after a termination).
        """
        child = subprocess.Popen(  # nosec B603 - constant local rehearsal command.
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        terminated = False
        try:
            try:
                child.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                # handled: a timed-out child is terminated and reported so
                child.terminate()
                child.wait(timeout=2)
                terminated = True
            return terminated, child.returncode
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=2)
