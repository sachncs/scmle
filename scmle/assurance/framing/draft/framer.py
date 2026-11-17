"""The framing of a request into a draft charter, and its artifacts."""

from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy
import pandas

from scmle.assurance.framing import baseline
from scmle.assurance.framing import costs
from scmle.assurance.framing import provenance
from scmle.assurance.framing import refusal
from scmle.assurance.framing.draft import filling
from scmle.assurance.framing.draft import records
import scmle.assurance.framing.questions
from scmle.contracts import errors


def refused(
    reason: str,
    detail: Mapping[str, Any],
    checks: Sequence[records.Probe] = (),
    llm: dict[str, Any] | None = None,
    discarded: Sequence[str] = (),
) -> records.Draft:
    """Builds a refused draft, with the reason on file.

    Args:
        reason: Why the draft was refused.
        detail: Further facts, stored with the reason.
        checks: The data-side checks run so far.
        llm: The LLM record, if the LLM path was used.
        discarded: What was dropped and why.

    Returns:
        A draft with status `refused`, no charter, and the reason and detail
        as its refusal.
    """
    return records.Draft(
        records.REFUSED,
        None,
        {},
        tuple(checks),
        (),
        (),
        (),
        None,
        {"reason": reason, **detail},
        llm,
        tuple(discarded),
    )


def mismatch(
    request: records.Request, data_sample: pandas.DataFrame
) -> records.Draft | None:
    """Refuses a request whose columns do not match the source or the sample.

    Args:
        request: The request and the source description.
        data_sample: Exploration rows only.

    Returns:
        A refused draft when the request names columns the source does not
        have or the sample lacks declared columns; otherwise None.
    """
    source = request.source
    named = [
        c for c in records.BACKTICKED.findall(request.text) if " " not in c
    ]
    missing = [c for c in named if c not in source.columns]
    if missing:
        return refused(
            "request names columns the source does not have",
            {"missing_columns": missing},
        )
    absent = [c for c in source.columns if c not in data_sample.columns]
    if absent:
        return refused(
            "the data sample lacks declared columns",
            {"missing_columns": absent},
        )
    return None


def consult(
    request: records.Request,
    data_sample: pandas.DataFrame,
    slots: dict[str, Any],
    sources: dict[str, provenance.Filling],
    sampler: refusal.Sampler | None,
    tau: float | None,
    k: int,
) -> tuple[dict[str, Any] | None, list[str], records.Draft | None]:
    """Asks the LLM slot-filler for slots and keeps the ones that pass.

    Slots the template left empty are filled from the consensus of the
    drafts, unless the consensus fails a blocking data-side check.

    Args:
        request: The request and the source description.
        data_sample: Exploration rows only.
        slots: The slots drafted so far; filled in place.
        sources: Where each slot came from; updated in place.
        sampler: The LLM slot-filler, or None for no LLM path.
        tau: Calibrated ambiguity threshold.
        k: Drafts to draw.

    Returns:
        The LLM record (None without a sampler), what was discarded, and a
        refused draft when the ambiguity is above `tau` or no `tau` is on
        file (otherwise None).
    """
    if sampler is None:
        return None, [], None
    source = request.source
    prompt = refusal.instruct(request.text, source.columns)
    samples = refusal.draw(sampler, prompt, k)
    agreed = refusal.consensus(data_sample, samples)
    llm = {
        "model_id": sampler.model_id,
        "prompt": prompt,
        "samples": samples,
        "samples_hash": refusal.samples_hash(sampler.model_id, prompt, samples),
        "ambiguity": agreed.ambiguity,
        "agreement": agreed.agreement,
        "tau": tau,
        "tier": "unverified",
    }
    if refusal.refuse(agreed.ambiguity, tau):
        stopped = refused(
            "ambiguity above tau, or no calibrated tau on file",
            {
                "ambiguity": agreed.ambiguity,
                "tau": tau,
                "conflicts": list(agreed.conflicts),
            },
            llm=llm,
        )
        return llm, [], stopped
    discarded: list[str] = []
    proposal = {
        k_: v
        for k_, v in agreed.values.items()  # noqa: PD011 -- `Consensus.values` is a dataclass field, not a pandas attribute
        if v not in (None, "", [])
    }
    trial_checks = filling.vet(data_sample, proposal, source)
    if any(not c.ok and c.blocking for c in trial_checks):
        discarded.append("llm draft failed a blocking data-side check")
    else:
        for name, value in proposal.items():
            if name not in slots:
                slots[name] = value
                sources[name] = provenance.Filling.LLM
    return llm, discarded, None


def interview(
    data_sample: pandas.DataFrame,
    target: str,
    baseline_column: str | None,
    cost_model: costs.Tariff | None,
    pilot_actions: Sequence[int] | None,
    open_items: list[str],
) -> tuple[
    dict[str, Any] | None,
    list[scmle.assurance.framing.questions.Question],
    list[dict[str, Any]],
]:
    """Profiles the baseline on `X` and asks for the unknown cost threshold.

    Args:
        data_sample: Exploration rows only.
        target: The target column.
        baseline_column: Column of the current process's 0/1 decisions.
        cost_model: The costs, if a person supplied them.
        pilot_actions: Actions of a rule fitted on `X`.
        open_items: Items still open; changed in place, by adding
            `baseline_decision_column` when there is no baseline column and
            by removing `cost_model` when the threshold question is asked
            and answered by the data.

    Returns:
        The baseline profile (None when it cannot be measured), the
        questions that remain for a person, and the robust verdicts.
    """
    profile = None
    asked: list[scmle.assurance.framing.questions.Question] = []
    robust: list[dict[str, Any]] = []
    if (
        baseline_column
        and baseline_column in data_sample.columns
        and cost_model
    ):
        profile = baseline.gauge(
            pandas.to_numeric(data_sample[target], errors="coerce").to_numpy(),
            data_sample[baseline_column].to_numpy(),
            cost_model,
        )
    elif not baseline_column:
        open_items.append("baseline_decision_column")
    if (
        cost_model is None
        and baseline_column in data_sample.columns
        and pilot_actions is not None
    ):
        outcome = pandas.to_numeric(data_sample[target], errors="coerce")
        seen = outcome.notna().to_numpy()
        asked, robust = scmle.assurance.framing.questions.ask(
            [
                scmle.assurance.framing.questions.Unknown(
                    "cost_threshold",
                    records.COST_RANGE[0],
                    records.COST_RANGE[1],
                    "a threshold t = c_FP / (c_FP + c_FN) in (0, 1)",
                    scmle.assurance.framing.questions.judge(
                        outcome.to_numpy()[seen],
                        numpy.asarray(pilot_actions)[seen],
                        data_sample[baseline_column].to_numpy()[seen],
                    ),
                )
            ]
        )
        if not asked:
            open_items.remove("cost_model")
    return profile, asked, robust


def frame(
    request: records.Request,
    data_sample: pandas.DataFrame,
    *,
    cost_model: costs.Tariff | None = None,
    baseline_column: str | None = None,
    pilot_actions: Sequence[int] | None = None,
    human: Mapping[str, Any] | None = None,
    sampler: refusal.Sampler | None = None,
    tau: float | None = None,
    k: int = refusal.DEFAULT_K,
) -> records.Draft:
    """Drafts a charter from a request, reading only `X` rows.

    Args:
        request: The request and the source description.
        data_sample: Exploration rows only.
        cost_model: The costs, if a person supplied them.
        baseline_column: Column of the current process's 0/1 decisions.
        pilot_actions: Actions of a rule fitted on `X`, used only to test
            whether the unknown cost threshold changes the verdict.
        human: Slots a person fixed, which override every other source.
        sampler: Optional LLM slot-filler (or a replay of recorded drafts).
        tau: Calibrated ambiguity threshold; without it an LLM draft is
            refused, because no refusal rule is on file.
        k: Drafts to draw.

    Returns:
        The draft. It is `refused` when named columns are missing, a
        blocking check fails, a critical slot cannot be filled, or the LLM
        drafts disagree by more than `tau`.
    """
    stopped = mismatch(request, data_sample)
    if stopped is not None:
        return stopped
    slots = filling.prefill(request, data_sample)
    sources = dict.fromkeys(slots, provenance.Filling.TEMPLATE)
    llm, discarded, stopped = consult(
        request, data_sample, slots, sources, sampler, tau, k
    )
    if stopped is not None:
        return stopped
    slots.update(human or {})
    sources.update(dict.fromkeys(human or {}, provenance.Filling.HUMAN))
    checks = filling.vet(data_sample, slots, request.source)
    held = screen(slots, checks)
    if held is not None:
        return refused(*held, checks, llm, discarded)
    try:
        charter, open_items = filling.assemble(slots, request, cost_model)
    except (ValueError, KeyError, TypeError, errors.ConfigError) as error:
        # handled: returned as refused draft with error
        return refused(
            "the draft does not form a valid charter",
            {"error": str(error)},
            checks,
            llm,
            discarded,
        )
    profile, asked, robust = interview(
        data_sample,
        slots["target"],
        baseline_column,
        cost_model,
        pilot_actions,
        open_items,
    )
    return records.Draft(
        records.ISSUED if not open_items else records.WITH_QUESTIONS,
        charter,
        attribute(sources, open_items),
        checks,
        tuple(asked),
        tuple(robust),
        tuple(open_items),
        profile,
        None,
        llm,
        tuple(discarded),
    )


def screen(
    slots: Mapping[str, Any], checks: Sequence[records.Probe]
) -> tuple[str, Mapping[str, Any]] | None:
    """Finds why a draft must be refused before its charter is built.

    Args:
        slots: The drafted slots.
        checks: The data-side checks on the slots.

    Returns:
        The reason and its detail when a blocking check failed, or else when
        a critical slot is unfilled; None when neither applies.
    """
    blocked = [c for c in checks if not c.ok and c.blocking]
    if blocked:
        return (
            "a blocking data-side check failed",
            {"failed": [c.name for c in blocked]},
        )
    unfilled = [f for f in records.CRITICAL_FIELDS if not slots.get(f)]
    if unfilled:
        return (
            "critical slots cannot be filled from the request and the schema",
            {
                "unfilled": unfilled,
                "ask": dict.fromkeys(unfilled, "name the column"),
            },
        )
    return None


def attribute(
    sources: Mapping[str, provenance.Filling], open_items: Sequence[str]
) -> dict[str, provenance.Filling]:
    """Records where every drafted or open field came from.

    Args:
        sources: Where each drafted slot came from.
        open_items: Slots left at a placeholder.

    Returns:
        `sources`, plus `template` for each open item without a source.
    """
    return {
        **sources,
        **{
            n: provenance.Filling.TEMPLATE
            for n in open_items
            if n not in sources
        },
    }


def emit(draft: records.Draft, directory: str | Path) -> list[str]:
    """Writes the framing artifacts to a directory.

    The directory is created if needed. The files are the draft, the
    questions and, where they exist, the cost model, the baseline report and
    the refusal.

    Args:
        draft: The draft to write.
        directory: Where to write.

    Returns:
        The sorted names of the files written.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    documents: dict[str, Any] = {"framing_draft.json": draft.to_dict()}
    documents["framing_questions.json"] = {
        "questions": [q.to_dict() for q in draft.questions],
        "robust": list(draft.robust),
        "open_items": list(draft.open_items),
    }
    if draft.charter is not None and draft.charter.costs is not None:
        documents["decision_cost_model.json"] = draft.charter.costs.to_dict()
    if draft.baseline is not None:
        documents["baseline_report.json"] = draft.baseline
    if draft.refusal is not None:
        documents["framing_refusal.json"] = draft.refusal
    for name, document in documents.items():
        (target / name).write_text(
            refusal.dump(document) + "\n", encoding="utf-8", newline="\n"
        )
    return sorted(documents)
