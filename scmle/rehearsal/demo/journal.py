"""The transcript: what the demo did and whether each step held."""

import dataclasses
import time
from typing import Any


@dataclasses.dataclass
class Step:
    """One recorded step of the demo.

    Attributes:
        name: Description of the step.
        ok: Whether the step held.
        seconds: Wall-clock seconds the step took.
        detail: Measured values and findings of the step.
    """

    name: str
    ok: bool
    seconds: float
    detail: dict[str, Any]


class Transcript:
    """The ordered record of what the demo did and whether it held.

    Attributes:
        steps: The recorded steps, in order.
    """

    def __init__(self) -> None:
        """Starts empty."""
        self.steps: list[Step] = []

    def record(
        self, name: str, started: float, ok: bool, **detail: Any
    ) -> None:
        """Appends a step and prints it.

        Args:
            name: Description of the step.
            started: Value of `time.monotonic()` when the step began.
            ok: Whether the step held.
            **detail: Measured values and findings, kept on the step and
                printed.
        """
        step = Step(name, ok, round(time.monotonic() - started, 3), detail)
        self.steps.append(step)
        print(
            f"[{'ok' if ok else 'FAIL'}] {name} ({step.seconds}s)", flush=True
        )
        for key, value in detail.items():
            print(f"      {key}: {value}", flush=True)

    @property
    def passed(self) -> bool:
        """Whether at least one step was recorded and every step held."""
        return bool(self.steps) and all(step.ok for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Builds a JSON-compatible record.

        Returns:
            A dict with `passed` and `steps` (each step as a dict).
        """
        return {
            "passed": self.passed,
            "steps": [dataclasses.asdict(step) for step in self.steps],
        }
