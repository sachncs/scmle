"""`Tracer`, one derivation over one trail folder, and `derive`."""

from collections.abc import Mapping
import dataclasses
from pathlib import Path
from typing import Any

from scmle.trail import records
from scmle.trail.trace import evidence
from scmle.trail.trace import finding
from scmle.trail.trace import planning
from scmle.trail.trace import release


@dataclasses.dataclass(frozen=True, slots=True)
class Derived:
    """What `derive` found.

    Attributes:
        steps: The ten steps in the order of `KINDS` (without chain fields).
        findings: Every problem, in the order found.
        reports: The exploratory report, the experiment comparison and the
            outcome tracking, by file name under ``reports/``; a report that
            could not be rebuilt is None.
    """

    steps: tuple[dict[str, Any], ...]
    findings: tuple[finding.Finding, ...]
    reports: Mapping[str, Any]


class Tracer(planning.Planning, evidence.Evidence, release.Release):
    """One derivation over one trail folder, with every step of `KINDS`.

    The state and record access are `reader.Reader`'s; the steps are the
    planning, evidence and release steps.
    """


def derive(
    directory: str | Path, keys: Mapping[str, str | bytes | None]
) -> Derived:
    """Derives every step and finding from a trail folder's retained records.

    Args:
        directory: The trail folder.
        keys: Verification keys by role: ``analyst``, ``stakeholder``,
            ``reviewer``. A missing key makes the signature it covers
            unverifiable, which is reported as unsigned.

    Returns:
        The ten steps in order, every finding, and the derived reports (None
        for a report that could not be rebuilt). Problems are recorded as
        findings rather than raised.
    """
    folder = Path(directory)
    tracer = Tracer(folder, keys)
    path = folder / records.INPUTS
    if path.is_file():
        try:
            tracer.inputs = records.read(path)
        except ValueError:
            # handled: recorded as trace problem
            tracer.problem(
                finding.TAMPERED, "stakeholder", "inputs.json is not JSON"
            )
    else:
        tracer.problem(
            finding.MISSING_LINK, "stakeholder", "inputs.json is absent"
        )
    tracer.question()
    tracer.framing()
    tracer.exploration()
    tracer.experiments()
    tracer.selection()
    tracer.observe()
    tracer.stakeholder()
    tracer.quality()
    tracer.approval()
    tracer.deployment()
    ordered = tuple(tracer.steps[k] for k in finding.KINDS)
    return Derived(
        ordered,
        tuple(tracer.findings),
        {
            "exploratory_report.json": tracer.derived.get(records.EXPLORATORY),
            "experiment_comparison.json": tracer.derived.get(
                records.EXPERIMENTS
            ),
            "outcome_tracking.json": tracer.derived.get(
                records.OUTCOME_TRACKING
            ),
        },
    )
