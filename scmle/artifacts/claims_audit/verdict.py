"""Verdict labels and the `Finding` and `Verification` records."""

import dataclasses
from typing import Any

PASS = "pass"  # noqa: S105 -- a verdict label, not a credential # nosec B105
FAIL = "fail"
SKIPPED = "skipped"
TOLERANCE = 1e-12


@dataclasses.dataclass(frozen=True, slots=True)
class Finding:
    """One verification step.

    Attributes:
        name: What was checked.
        status: `pass`, `fail` or `skipped` (the input does not exist or
            does not apply; a skip never counts as a pass).
        detail: One sentence.
    """

    name: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        """Builds a JSON-ready record.

        Returns:
            A dict with `name`, `status` and `detail`.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class Verification:
    """The result of `verify`.

    Attributes:
        run_dir: The directory verified.
        findings: Every step, in order.
    """

    run_dir: str
    findings: tuple[Finding, ...]

    @property
    def ok(self) -> bool:
        """True when no step failed."""
        return all(f.status != FAIL for f in self.findings)

    @property
    def verified(self) -> int:
        """The number of steps that passed."""
        return sum(1 for f in self.findings if f.status == PASS)

    def to_dict(self) -> dict[str, Any]:
        """Builds a JSON-ready record.

        Returns:
            A dict with the directory, `ok`, the pass count and the findings.
        """
        return {
            "run_dir": self.run_dir,
            "ok": self.ok,
            "passed": self.verified,
            "findings": [f.to_dict() for f in self.findings],
        }
