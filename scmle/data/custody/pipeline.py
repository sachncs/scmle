"""Applying custody to a split: development rows become `X`, `C` is sealed.

Runs after `SplitManager.split` and before profiling, so the profiler,
feature selection and every trial see only exploration rows. The final test
set keeps its role; `C` is a separate carve-out of the
development rows, so confirmed claims never shape the model that `C` scores.
"""

import dataclasses

import numpy
import pandas

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.charter import contracts
from scmle.data.charter import feasibility
from scmle.data.custody import carve
from scmle.data.custody import manifest as manifest_module
from scmle.data.custody import merkle
from scmle.data.custody import sealed
from scmle.data.splitting import duplicates
from scmle.data.splitting import splitters

TEMPORAL = frozenset({"temporal", "rolling"})


@dataclasses.dataclass(slots=True)
class CustodyOutcome:
    """Everything custody produced for a run.

    Attributes:
        split: The split with development data restricted to `X`.
        view: What exploration code may receive.
        manifest: The v2 manifest.
        feasibility: Whether `C` can support the promised power.
    """

    split: splitters.SplitResult
    view: sealed.ExplorationView
    manifest: manifest_module.CustodyManifest
    feasibility: feasibility.Feasibility


def minority_rows(labels: numpy.ndarray | None, fraction: float) -> int | None:
    """Estimates the minority-class rows in `C`.

    Args:
        labels: Class label per development row, or None for a task without
            classes.
        fraction: Share of development rows placed in `C`.

    Returns:
        The size of the smallest class times `fraction`, truncated to an
        integer; None when `labels` is None.
    """
    if labels is None:
        return None
    counts = numpy.unique(labels, return_counts=True)[1]
    return int(numpy.min(counts) * fraction)


def assess(
    charter: contracts.AnalysisCharter,
    n_rows: int,
    minority: int | None,
) -> feasibility.Feasibility:
    """Assesses whether a confirmation set supports the charter's power.

    Args:
        charter: The governing charter.
        n_rows: Number of rows in the confirmation set.
        minority: Minority-class rows in it, or None when there are no
            classes.

    Returns:
        The feasibility assessment for `n_rows` rows.
    """
    return feasibility.assess(
        n_confirmation=n_rows,
        min_effect=charter.min_effect,
        alpha=float(charter.error_budget.confirmation.alpha_confirm),
        k_max=charter.k_max,
        power=charter.power,
        n_minority=minority,
    )


def classes_complete(
    labels: numpy.ndarray | None, carved: carve.Carve, dev_ids: numpy.ndarray
) -> bool:
    """Checks that `C` holds every class that `X` holds.

    Args:
        labels: Class label per development row, or None for a task without
            classes.
        carved: The carve to check.
        dev_ids: Row ids of the development rows, aligned with `labels`.

    Returns:
        True when `labels` is None or every class in `X` also occurs in
        `C`.
    """
    if labels is None:
        return True
    position = {int(row): index for index, row in enumerate(dev_ids)}
    in_c = {labels[position[int(r)]] for r in carved.confirmation_ids}
    in_x = {labels[position[int(r)]] for r in carved.exploration_ids}
    return in_x <= in_c


def empty_carve(
    dev_ids: numpy.ndarray,
    folds: list[tuple[numpy.ndarray, numpy.ndarray]],
    note: str,
) -> carve.Carve:
    """Builds a carve that leaves every development row in `X`.

    Args:
        dev_ids: Row ids of the development rows.
        folds: Existing folds, kept unchanged.
        note: Why nothing was carved.

    Returns:
        A carve with empty `C` and embargo and `note` as its only note.
    """
    return carve.Carve(
        exploration_ids=dev_ids,
        confirmation_ids=dev_ids[:0],
        embargo_ids=dev_ids[:0],
        folds=folds,
        notes=(note,),
    )


def measured_embargo(
    frame: pandas.DataFrame,
    dev_ids: numpy.ndarray,
    data: tasks.DataSpec,
    times: numpy.ndarray | None,
    task: tasks.TaskType,
) -> int:
    """Measures the embargo in rows from the target's autocorrelation, or 0.

    Only temporal strategies carry an embargo. The target is read in time
    order over the development rows; a target that is not numeric gives 0.

    Args:
        frame: The full dataset, indexed by row id.
        dev_ids: Row ids of the development rows.
        data: Data specification naming the target column.
        times: Ordering value per development row, or None for a
            non-temporal strategy.
        task: Resolved task type; class labels are coded as integers first.

    Returns:
        The embargo in rows; 0 when `times` is None or, for a regression,
        the target has missing or non-numeric values.
    """
    if times is None:
        return 0
    target = frame[data.target].iloc[dev_ids]
    if task.is_classification:
        target = pandas.Series(pandas.factorize(target)[0], index=target.index)
    values = pandas.to_numeric(target, errors="coerce").to_numpy(dtype=float)
    ordered = values[numpy.argsort(times, kind="stable")]
    if numpy.isnan(ordered).any():
        return 0
    return carve.acf_embargo(ordered)


def apply_custody(
    *,
    charter: contracts.AnalysisCharter,
    split: splitters.SplitResult,
    frame: pandas.DataFrame,
    spec: tasks.ProjectSpec,
    task: tasks.TaskType,
    data_hash: str,
    vault: sealed.CustodyVault,
) -> CustodyOutcome:
    """Carves `C` out of the development rows and seals it.

    Args:
        charter: The governing charter.
        split: Result of the ordinary split.
        frame: The full dataset (target included), indexed by row id.
        spec: Project specification.
        task: Resolved task type.
        data_hash: Hash of the dataset.
        vault: Where sealed sets are held.

    Returns:
        The restricted split, the exploration view, the v2 manifest and the
        feasibility verdict. If `C` cannot be confirmed the carve is skipped:
        all development rows stay in `X` and the scope is capped. The
        embargo is the larger of the charter's and the measured one, or the
        charter's when no `C` remains.
    """
    data, old = spec.data, split.manifest
    dev_ids = numpy.asarray(old.train_ids, dtype=int)
    folds = [
        (numpy.asarray(a, int), numpy.asarray(b, int)) for a, b in old.folds
    ]
    labels = (
        frame[data.target].iloc[dev_ids].to_numpy()
        if task.is_classification
        else None
    )
    known_entity = charter.unit_of_prediction == contracts.KNOWN_ENTITY_FUTURE
    # A known entity is not a unit: its rows may sit in X and in C, so the
    # carve (time blocks, with an embargo) does not tie them together.
    groups = (
        frame[data.group_column].iloc[dev_ids].to_numpy()
        if data.group_column and not known_entity
        else None
    )
    times = (
        frame[data.time_column].iloc[dev_ids].to_numpy()
        if old.strategy in TEMPORAL and data.time_column
        else None
    )
    embargo_rows = charter.sets.embargo_rows
    measured = measured_embargo(frame, dev_ids, data, times, task)
    acf_note = None
    if measured > embargo_rows:
        acf_note = (
            f"embargo raised from {embargo_rows} to {measured} rows: "
            "measured autocorrelation of the target"
        )
        embargo_rows = measured
    columns = splitters.SplitManager(spec, task).feature_columns(frame)
    copies = duplicates.duplicate_groups(frame, columns)[dev_ids]
    fraction = charter.sets.confirmation_fraction
    estimate = assess(
        charter,
        round(fraction * len(dev_ids)),
        minority_rows(labels, fraction),
    )
    if estimate.verdict is feasibility.Verdict.INFEASIBLE:
        carved = empty_carve(dev_ids, folds, estimate.reason)
    else:
        try:
            carved = carve.carve_confirmation(
                dev_ids,
                folds,
                fraction=fraction,
                seed=charter.seed,
                charter_hash=charter.charter_hash,
                data_hash=data_hash,
                labels=labels,
                groups=groups,
                times=times,
                duplicates=copies,
                embargo_rows=embargo_rows,
            )
        except errors.ConfigError as refusal:
            # The carve refuses explicitly (for example when it would leave a
            # validation fold empty). The run continues with no confirmation
            # set and an unconfirmed scope, as for an infeasible size.
            carved = empty_carve(
                dev_ids, folds, f"confirmation set not carved: {refusal}"
            )
        if len(carved.confirmation_ids) and not classes_complete(
            labels, carved, dev_ids
        ):
            carved = empty_carve(
                dev_ids, folds, "confirmation set misses a class: not carved"
            )
    actual = assess(
        charter,
        len(carved.confirmation_ids),
        minority_rows(labels, fraction)
        if len(carved.confirmation_ids)
        else None,
    )
    if actual.verdict is feasibility.Verdict.INFEASIBLE and len(
        carved.confirmation_ids
    ):
        carved = empty_carve(dev_ids, folds, actual.reason)
        actual = assess(charter, 0, None)
    if not len(carved.confirmation_ids):
        embargo_rows = charter.sets.embargo_rows
    elif acf_note:
        carved = dataclasses.replace(carved, notes=(*carved.notes, acf_note))
    return finish(
        charter,
        split,
        frame,
        spec,
        carved,
        dev_ids,
        actual,
        data_hash,
        vault,
        embargo_rows,
    )


def finish(
    charter: contracts.AnalysisCharter,
    split: splitters.SplitResult,
    frame: pandas.DataFrame,
    spec: tasks.ProjectSpec,
    carved: carve.Carve,
    dev_ids: numpy.ndarray,
    report: feasibility.Feasibility,
    data_hash: str,
    vault: sealed.CustodyVault,
    embargo_rows: int,
) -> CustodyOutcome:
    """Builds the restricted split, seals `C` and records the manifest.

    Args:
        charter: The governing charter.
        split: Result of the ordinary split.
        frame: The full dataset (target included), indexed by row id.
        spec: Project specification.
        carved: The division of the development rows.
        dev_ids: Row ids of the development rows.
        report: Feasibility assessment of the final confirmation set.
        data_hash: Hash of the dataset.
        vault: Where sealed sets are held; `C` is sealed in it when it is
            not empty.
        embargo_rows: Embargo recorded in the manifest.

    Returns:
        The restricted split, the exploration view, the v2 manifest and the
        feasibility assessment.
    """
    old = split.manifest
    keep = numpy.flatnonzero(numpy.isin(dev_ids, carved.exploration_ids))
    folds = tuple(
        (tuple(int(i) for i in fit), tuple(int(i) for i in held))
        for fit, held in carved.folds
    )
    exploration = tuple(int(i) for i in carved.exploration_ids)
    isolation: list[tuple[str, str]] = [("final_test", vault.level)]
    confirmation = tuple(int(i) for i in carved.confirmation_ids)
    if confirmation:
        target = spec.data.target
        rows = numpy.asarray(confirmation)
        vault.seal(
            "C",
            frame.drop(columns=[target]).iloc[rows].reset_index(drop=True),
            frame[target].iloc[rows].reset_index(drop=True),
            confirmation,
        )
        isolation.insert(0, ("C", vault.level))
    notes = (*old.notes, *carved.notes)
    restricted = splitters.SplitManifest(
        strategy=old.strategy,
        seed=old.seed,
        train_ids=exploration,
        test_ids=old.test_ids,
        folds=folds,
        notes=notes,
    )
    manifest = manifest_module.CustodyManifest(
        strategy=old.strategy,
        seed=charter.seed,
        embargo_rows=embargo_rows,
        charter_hash=charter.charter_hash,
        data_hash=data_hash,
        exploration_ids=exploration,
        confirmation_ids=confirmation,
        embargo_ids=tuple(int(i) for i in carved.embargo_ids),
        final_test_ids=tuple(old.test_ids),
        folds=folds,
        isolation=tuple(isolation),
        scope=report.scope,
        notes=notes,
        row_roots=tuple(
            (name, merkle.frame_root(frame, members))
            for name, members in (
                ("exploration", exploration),
                ("confirmation", confirmation),
                ("final_test", tuple(old.test_ids)),
            )
        ),
    )
    result = splitters.SplitResult(
        restricted,
        split.dev_features.iloc[keep].reset_index(drop=True),
        split.dev_target.iloc[keep].reset_index(drop=True),
        split.locked_test,
    )
    view = sealed.ExplorationView(
        features=result.dev_features,
        target=result.dev_target,
        ids=exploration,
        folds=folds,
        manifest_sha256=manifest.sha256,
    )
    return CustodyOutcome(result, view, manifest, report)
