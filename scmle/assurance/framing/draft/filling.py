"""Slots filled by rules, the data-side checks and the charter they build."""

from collections.abc import Mapping
from collections.abc import Sequence
import re
from typing import Any

import pandas

from scmle.assurance.charter import contracts
from scmle.assurance.framing import costs
from scmle.assurance.framing import refusal
from scmle.assurance.framing.draft import records


def normalise(name: str) -> str:
    """Lower-cases a name and removes separators.

    Args:
        name: The name to normalise.

    Returns:
        The name in lower case with every non-word character and underscore
        removed.
    """
    return re.sub(r"[\W_]+", "", name.lower())


def mentioned(text: str, columns: Sequence[str]) -> list[str]:
    """Finds the columns that the request names.

    Args:
        text: The request.
        columns: Column names to look for.

    Returns:
        The columns whose normalised name is a word of the request, or, for
        names longer than four characters, appears anywhere in the
        normalised request. They keep the order of `columns`.
    """
    words = {normalise(w) for w in re.findall(r"[\w]+", text)}
    flat = normalise(text)
    return [
        c
        for c in columns
        if normalise(c) in words
        or (
            len(normalise(c)) > records.MIN_SUBSTRING_CHARS
            and normalise(c) in flat
        )
    ]


def prefill(
    request: records.Request, frame: pandas.DataFrame
) -> dict[str, Any]:
    """Fills slots by rules from the request and the schema.

    Args:
        request: The request and the source description.
        frame: The data sample, used to find datetime columns.

    Returns:
        The slots found: `target` (an explicit mention, a single column the
        request names, or a single column with a conventional target name),
        `prediction_moment` (the only datetime column other than the
        target-time column), `unit` (the declared primary key) and
        `question` (the request text). A slot that cannot be found is
        absent.
    """
    columns = list(request.source.columns)
    slots: dict[str, Any] = {}
    explicit = re.search(
        r"\b(?:target|label|predict(?:ing)?|outcome)\s*(?:column)?\s*[:=]?\s*`?(\w+)`?",
        request.text,
        re.IGNORECASE,
    )
    named = {normalise(c): c for c in columns}
    if explicit and normalise(explicit.group(1)) in named:
        slots["target"] = named[normalise(explicit.group(1))]
    else:
        hits = mentioned(request.text, columns)
        if len(hits) == 1:
            slots["target"] = hits[0]
        else:
            fallback = [
                c for c in columns if normalise(c) in records.TARGET_WORDS
            ]
            if len(fallback) == 1:
                slots["target"] = fallback[0]
    stamps = [
        c
        for c in columns
        if c in frame.columns
        and pandas.api.types.is_datetime64_any_dtype(frame[c])
        and c != request.source.target_time_column
    ]
    if len(stamps) == 1:
        slots["prediction_moment"] = stamps[0]
    if request.source.primary_key:
        slots["unit"] = list(request.source.primary_key)
    slots["question"] = request.text.strip()
    return slots


def vet(
    frame: pandas.DataFrame,
    slots: Mapping[str, Any],
    source: records.Offer,
) -> tuple[records.Probe, ...]:
    """Runs the data-side checks on the slots, using `X` rows only.

    Args:
        frame: The exploration rows.
        slots: The drafted slots; only the ones present are checked.
        source: The source description, for the target-time column.

    Returns:
        The checks in the order target, prediction moment, unit and
        population, for the slots present.
    """
    found: list[records.Probe] = []
    target = slots.get("target")
    if target is not None:
        found.append(audit(frame, target))
    moment = slots.get("prediction_moment")
    if moment is not None:
        timing = order(frame, moment, source)
        if timing is not None:
            found.append(timing)
    declared = slots.get("unit")
    if declared:
        keys = [declared] if isinstance(declared, str) else list(declared)
        if not all(k in frame.columns for k in keys):
            found.append(
                records.Probe(
                    "unit_exists",
                    ok=False,
                    blocking=True,
                    detail=f"{keys} missing",
                )
            )
        else:
            share = float(frame.duplicated(subset=keys).mean())
            found.append(
                records.Probe(
                    "unit_unique",
                    ok=share <= records.UNIT_DUPLICATE_TOLERANCE,
                    blocking=False,
                    detail=f"duplicate share {share:.4f}",
                )
            )
    population = slots.get("population")
    if population:
        ids = refusal.members(frame, population)
        coverage = None if ids is None else len(ids) / max(len(frame), 1)
        ok = (
            coverage is not None
            and records.MIN_POPULATION_SHARE <= coverage < 1.0
        )
        found.append(
            records.Probe(
                "population_coverage",
                ok=ok,
                blocking=ids is None,
                detail="predicate not understood"
                if coverage is None
                else f"selects {coverage:.3f} of the rows",
            )
        )
    return tuple(found)


def audit(frame: pandas.DataFrame, target: Any) -> records.Probe:
    """Checks that the target column exists and is usable.

    Args:
        frame: The exploration rows.
        target: The drafted target slot.

    Returns:
        A blocking `target_exists` failure when the column is missing,
        otherwise `target_complete`, which fails (without blocking) when too
        many values are missing or the column is constant.
    """
    if target not in frame.columns:
        return records.Probe(
            "target_exists",
            ok=False,
            blocking=True,
            detail=f"{target} missing",
        )
    missing = float(frame[target].isna().mean())
    constant = frame[target].nunique(dropna=True) < records.MIN_DISTINCT_VALUES
    return records.Probe(
        "target_complete",
        ok=missing <= records.MAX_TARGET_MISSING and not constant,
        blocking=False,
        detail=f"missing share {missing:.3f}; constant={constant}",
    )


def order(
    frame: pandas.DataFrame, moment: Any, source: records.Offer
) -> records.Probe | None:
    """Checks that the prediction moment comes before the target time.

    Args:
        frame: The exploration rows.
        moment: The drafted prediction-moment slot.
        source: The source description, for the target-time column.

    Returns:
        A blocking `moment_exists` failure when the column is missing, the
        blocking `moment_before_target` check when the target-time column is
        in the rows, otherwise None.
    """
    if moment not in frame.columns:
        return records.Probe(
            "moment_exists",
            ok=False,
            blocking=True,
            detail=f"{moment} missing",
        )
    if source.target_time_column not in frame.columns:
        return None
    start = pandas.to_datetime(frame[moment], errors="coerce")
    end = pandas.to_datetime(frame[source.target_time_column], errors="coerce")
    bad = int((start >= end).sum())
    return records.Probe(
        "moment_before_target",
        ok=bad <= records.MAX_MOMENT_VIOLATIONS,
        blocking=True,
        detail=f"{bad} rows have the moment at or after the target time",
    )


def assemble(
    slots: Mapping[str, Any],
    request: records.Request,
    cost_model: costs.Tariff | None,
) -> tuple[contracts.AnalysisCharter, list[str]]:
    """Builds the unsigned charter, filling blanks with placeholders.

    Args:
        slots: The drafted slots. `unit`, `target` and `prediction_moment`
            must be present.
        request: The request and the source description.
        cost_model: The costs, or None if none were supplied.

    Returns:
        The charter, and the names of the slots left at a placeholder.
        A missing required slot raises `KeyError`, and a record the charter
        rejects raises `ValueError` or `ConfigError`.
    """
    record = contracts.template()
    open_items: list[str] = []

    def pick(name: str, default: str) -> str:
        value = slots.get(name)
        if value in (None, "", []):
            open_items.append(name)
            return default
        return str(value)

    kind = (
        "causal" if records.CAUSAL_WORDS.search(request.text) else "predictive"
    )
    unit = slots["unit"]
    record["question"] = slots.get("question") or records.UNSPECIFIED
    record["estimand"].update(
        kind=kind,
        population=pick("population", f"all rows of {request.source.uri}"),
        target=str(slots["target"]),
        prediction_moment=str(slots["prediction_moment"]),
        intercurrent_events=[{"event": "none", "strategy": "not applicable"}],
        summary_measure=(
            "net benefit over thresholds "
            f"[{cost_model.t_lo}, {cost_model.t_hi}]"
            if cost_model
            else pick("summary_measure", records.UNSPECIFIED)
        ),
    )
    record["unit"] = unit if isinstance(unit, str) else "-".join(map(str, unit))
    record["deployment_population"] = pick(
        "deployment_population", f"same as {request.source.uri}"
    )
    open_items.append("intercurrent_events")
    if cost_model is not None:
        record["cost_model"] = cost_model.to_dict()
    else:
        open_items.append("cost_model")
    return contracts.AnalysisCharter.from_dict(record), open_items


def linkage(
    charter: contracts.AnalysisCharter, columns: Sequence[str]
) -> dict[str, Any]:
    """Reports whether the charter's words name real columns, descriptively.

    A name that matches no column is reported `unresolved`, never a
    failure: the charter allows free text for these fields.

    Args:
        charter: The charter to check.
        columns: Column names of the data.

    Returns:
        A JSON-ready mapping with, for `target`, `prediction_moment` and
        `unit`, the text, the matching columns and a status of `resolved`
        or `unresolved`, and a `gating` flag that is always False.
    """
    known = {normalise(c): c for c in columns}
    fields = {
        "target": charter.estimand.target,
        "prediction_moment": charter.estimand.prediction_moment,
        "unit": charter.unit,
    }
    out = {}
    for name, value in fields.items():
        parts = [p for p in re.split(r"[\s,;+&-]+", value) if p]
        hits = [known[normalise(p)] for p in parts if normalise(p) in known]
        out[name] = {
            "text": value,
            "columns": hits,
            "status": "resolved" if hits else "unresolved",
        }
    return {"schema_version": 1, "fields": out, "gating": False}
