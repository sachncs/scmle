"""Time estimates that keep the final fits inside the run's budget."""

from typing import Any

import pandas

from scmle.contracts import experiments
from scmle.contracts import tasks
import scmle.data.validation.target
from scmle.evaluation import resources
from scmle.pipeline import engine
from scmle.pipeline.stages import profiling

# A fold fit of the selected model in the main process, estimated from its
# trial, whose `fit_seconds` covers fitting and scoring every fold in a
# worker. Measured on the seven overrun tasks at a 60 s budget (2026-11-17,
# two runs at a time): where the refit dominates the final evaluation
# (stacking, 9,184 rows) it took 3.47 s against 3.30 s predicted unmargined,
# and the rest of the final evaluation took at most 0.9 s; the margin covers
# that ratio, the main process sharing the machine, and run-to-run spread.
FIT_MARGIN = 1.5


# Packaging after its last fit (bundle measurement, saving, model card,
# report) took 0.25-0.6 s on the same runs; held at 1 s so every fit of the
# packaging stage leaves it room before the run's end.
TAIL_SECONDS = 1.0


def foldcost(trial: experiments.Trial) -> float:
    """Estimates one fold fit (and its scoring) of a trial's pipeline.

    Args:
        trial: The trial, whose `fit_seconds` covered its folds.

    Returns:
        The trial's seconds per fold times `FIT_MARGIN`.
    """
    folds = max(len(trial.folds), 1)
    return FIT_MARGIN * trial.fit / folds


def refitcost(board: engine.Board, trial: experiments.Trial) -> float:
    """Estimates refitting a trial's pipeline on all development rows.

    A fold trains on (k - 1) / k of the development rows, so the refit is a
    fold fit scaled by k / (k - 1).

    Args:
        board: The run's board, holding the split.
        trial: The trial to refit.

    Returns:
        The estimated seconds.
    """
    split = board.get(profiling.Partition).split
    folds = len(split.manifest.folds)
    scale = folds / (folds - 1) if folds > 1 else 1.0
    return foldcost(trial) * scale


def affordable(
    board: engine.Board, ranked: list[experiments.Trial]
) -> experiments.Trial:
    """Returns the best-ranked trial whose refit ends before the run's end.

    Each better trial passed over is recorded with the reason. When none
    fits, the one with the cheapest refit is taken, also on record.

    Args:
        board: The run's board, holding the governor.
        ranked: Trials in policy order, best first.

    Returns:
        The trial to select.
    """
    governor = board.services.governor
    for trial in ranked:
        reason = governor.overdue(refitcost(board, trial))
        if reason is None:
            return trial
        board.journal.forgo(
            f"selecting {trial.model}/{trial.strategy}",
            f"its refit {reason}",
        )
    cheapest = min(ranked, key=lambda t: refitcost(board, t))
    board.journal.forgo(
        f"selecting {cheapest.model}/{cheapest.strategy}",
        "no refit fits before the run's end; the cheapest refit is taken",
    )
    return cheapest


def late(board: engine.Board, trial: experiments.Trial) -> bool:
    """Whether a candidate's refit and measurement cannot end in time.

    A late candidate is rejected on record, unmeasured.

    Args:
        board: The run's board, holding the governor.
        trial: The candidate.

    Returns:
        True if its refit would not end before the run's end.
    """
    reason = board.services.governor.overdue(refitcost(board, trial))
    if reason is None:
        return False
    label = f"{trial.model}/{trial.strategy}"
    board.journal.enter(label, experiments.Edict.REJECT, f"refit {reason}")
    board.journal.forgo(f"measuring {label}", f"its refit {reason}")
    return True


def halt(
    board: engine.Board,
    selected: experiments.Trial,
    work: str,
) -> str | None:
    """Why the next fit of the selected pipeline must not start, or None.

    Args:
        board: The run's board, holding the governor.
        selected: The selected trial, whose fold fits are timed.
        work: What the fit is for, named in the record.

    Returns:
        The reason (also recorded with `forgo`), or None if a fold fit ends
        `TAIL_SECONDS` before the run's end.
    """
    reason = board.services.governor.overdue(foldcost(selected), TAIL_SECONDS)
    if reason is not None:
        board.journal.forgo(work, reason)
    return reason


def meter(
    board: engine.Board,
    pipeline: Any,
    features: pandas.DataFrame,
) -> resources.Usage:
    """Measures the final pipeline, timing fewer rows if time is short.

    One prediction is timed first; the number of timed rows is reduced (on
    record) so the measurement ends before the packaging reserve.

    Args:
        board: The run's board, holding the governor.
        pipeline: The fitted final pipeline.
        features: Rows to time predictions on.

    Returns:
        The measured usage.
    """
    governor = board.services.governor
    started = board.services.clock.monotonic()
    pipeline.predict(features.head(1))
    single = FIT_MARGIN * (board.services.clock.monotonic() - started)
    left = governor.left("finish") - governor.reserve.parts.get("package", 0.0)
    rows = resources.LATENCY_ROWS
    if single * rows > left:
        rows = max(int(left / single), 1) if single > 0 else rows
        board.journal.forgo(
            "serving measurement",
            f"timed {rows} of {resources.LATENCY_ROWS} rows: one prediction "
            f"takes about {single:.2f} s and {max(left, 0.0):.1f} s remain",
        )
    return resources.measure(pipeline, features, rows)


def dimensions(
    spec: tasks.Config, frame: pandas.DataFrame, task: tasks.Task | None
) -> tuple[int, int, int]:
    """Returns the table shape the time reserve is estimated from.

    Args:
        spec: The project specification, naming the targets.
        frame: The rows: the ingested ones, or the partition's once made.
        task: The resolved task, or None to resolve it from the target.

    Returns:
        Labelled rows, feature columns (every column but the targets) and
        target classes (0 when the task is not classification).
    """
    data = spec.data
    target = frame[data.target].dropna()
    named = {data.target, *data.targets}
    columns = sum(1 for column in frame.columns if column not in named)
    task = task or scmle.data.validation.target.Resolver(spec).resolve(target)
    classes = int(target.nunique()) if task.classification else 0
    return len(target), columns, classes
