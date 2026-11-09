"""Draft charter, data-side checks and the framing result.

`frame` turns a plain-language request and a source description into a draft
`AnalysisCharter` (unsigned), reads only `X` rows, and records where every
field came from. The deterministic drafter fills slots by rules; an optional
LLM slot-filler may fill the rest, behind the refusal rule in `refusal`. Every
LLM output must round-trip `AnalysisCharter.from_dict` and pass the data-side
checks, or it is discarded and the deterministic result is used. Nothing here
claims a drafted charter is correct.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import hashlib
from pathlib import Path
import re
from typing import Any

import numpy
import pandas

from scmle.contracts import errors
from scmle.contracts import signing
from scmle.data.charter import contracts
from scmle.data.framing import baseline
from scmle.data.framing import costs
from scmle.data.framing import provenance
from scmle.data.framing import refusal
import scmle.data.framing.questions

ISSUED = "issued"
WITH_QUESTIONS = "issued_with_questions"
REFUSED = "refused"
MAX_TARGET_MISSING = 0.2
MIN_POPULATION_SHARE = 0.01
UNIT_DUPLICATE_TOLERANCE = 0.01
TARGET_WORDS = ("target", "label", "outcome", "y")
CAUSAL_WORDS = re.compile(
    r"\b(cause|causal|effect of|impact of|because of|treatment effect)\b", re.I
)
BACKTICKED = re.compile(r"`([^`]+)`")
UNSPECIFIED = "unspecified (open question)"
CRITICAL_FIELDS = ("target", "prediction_moment", "unit")
MAX_MOMENT_VIOLATIONS = 0
COST_RANGE = (0.05, 0.5)


@dataclasses.dataclass(frozen=True, slots=True)
class SourceDescription:
    """What the data source offers.

    Attributes:
        uri: Where the data lives.
        columns: Column names the source has.
        primary_key: Declared key columns, if any.
        target_time_column: Column holding when the outcome is known, if any.
    """

    uri: str
    columns: tuple[str, ...]
    primary_key: tuple[str, ...] = ()
    target_time_column: str | None = None

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "SourceDescription":
        """Builds the description from a JSON object.

        Args:
            record: A mapping with `uri` and `columns`, and optionally
                `primary_key` and `target_time_column`.

        Returns:
            The description. A missing `uri` or `columns` raises `KeyError`.
        """
        return cls(
            uri=str(record["uri"]),
            columns=tuple(record["columns"]),
            primary_key=tuple(record.get("primary_key", ())),
            target_time_column=record.get("target_time_column"),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class FramingRequest:
    """A plain-language request and the source it is about.

    Attributes:
        text: The request in plain language.
        source: What the data source offers.
    """

    text: str
    source: SourceDescription


@dataclasses.dataclass(frozen=True, slots=True)
class DataCheck:
    """One LLM-free check of a draft against `X`.

    Attributes:
        name: The check.
        ok: Whether it passed.
        blocking: Whether a failure stops the draft.
        detail: Counts behind the result.
    """

    name: str
    ok: bool
    blocking: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Returns the check as a JSON-ready record.

        Returns:
            A mapping of every field to its value.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class FramingDraft:
    """The result of `frame`.

    Attributes:
        status: `issued`, `issued_with_questions` or `refused`.
        charter: The unsigned draft charter, or None when refused.
        fields: Source of each drafted field.
        checks: Data-side checks.
        questions: Tipping-point questions for a person.
        robust: Unknowns whose plausible range leaves the verdict unchanged.
        open_items: Slots left at a placeholder, to be answered.
        baseline: The baseline measured on `X`, if a decision column exists.
        refusal: Why the draft was refused, if it was.
        llm: Model id, prompt, raw samples, hash and ambiguity, if used.
        discarded: What was dropped and why (the LLM path, for example).
    """

    status: str
    charter: contracts.AnalysisCharter | None
    fields: dict[str, provenance.FieldSource]
    checks: tuple[DataCheck, ...]
    questions: tuple[scmle.data.framing.questions.HumanQuestion, ...]
    robust: tuple[dict[str, Any], ...]
    open_items: tuple[str, ...]
    baseline: dict[str, Any] | None
    refusal: dict[str, Any] | None
    llm: dict[str, Any] | None
    discarded: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Returns the `framing_draft.json` document.

        Returns:
            A JSON-ready mapping of the status, the charter and its hash
            (both None when refused), the source of each field, the checks,
            the open items, the LLM record, what was discarded, and a label
            that the draft is unverified.
        """
        return {
            "schema_version": 1,
            "status": self.status,
            "charter": None if self.charter is None else self.charter.to_dict(),
            "charter_hash": (
                None if self.charter is None else self.charter.charter_hash
            ),
            "fields": {k: v.value for k, v in sorted(self.fields.items())},
            "checks": [c.to_dict() for c in self.checks],
            "open_items": list(self.open_items),
            "llm": self.llm,
            "discarded": list(self.discarded),
            "label": (
                "a drafted charter is a draft: nothing here shows it is "
                "correct; llm fields are unverified until a person signs"
            ),
        }

    @property
    def draft_hash(self) -> str:
        """The SHA-256 hex digest of the stored draft document."""
        return hashlib.sha256(
            signing.canonical_payload(self.to_dict())
        ).hexdigest()


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
        or (len(normalise(c)) > 4 and normalise(c) in flat)
    ]


def deterministic_slots(
    request: FramingRequest, frame: pandas.DataFrame
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
        re.I,
    )
    named = {normalise(c): c for c in columns}
    if explicit and normalise(explicit.group(1)) in named:
        slots["target"] = named[normalise(explicit.group(1))]
    else:
        hits = mentioned(request.text, columns)
        if len(hits) == 1:
            slots["target"] = hits[0]
        else:
            fallback = [c for c in columns if normalise(c) in TARGET_WORDS]
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


def check_draft(
    frame: pandas.DataFrame,
    slots: Mapping[str, Any],
    source: SourceDescription,
) -> tuple[DataCheck, ...]:
    """Runs the data-side checks on the slots, using `X` rows only.

    Args:
        frame: The exploration rows.
        slots: The drafted slots; only the ones present are checked.
        source: The source description, for the target-time column.

    Returns:
        The checks in the order target, prediction moment, unit and
        population, for the slots present.
    """
    found: list[DataCheck] = []
    target = slots.get("target")
    if target is not None:
        if target not in frame.columns:
            found.append(
                DataCheck("target_exists", False, True, f"{target} missing")
            )
        else:
            missing = float(frame[target].isna().mean())
            constant = frame[target].nunique(dropna=True) < 2
            found.append(
                DataCheck(
                    "target_complete",
                    missing <= MAX_TARGET_MISSING and not constant,
                    False,
                    f"missing share {missing:.3f}; constant={constant}",
                )
            )
    moment = slots.get("prediction_moment")
    if moment is not None:
        if moment not in frame.columns:
            found.append(
                DataCheck("moment_exists", False, True, f"{moment} missing")
            )
        elif source.target_time_column in frame.columns:
            start = pandas.to_datetime(frame[moment], errors="coerce")
            end = pandas.to_datetime(
                frame[source.target_time_column], errors="coerce"
            )
            bad = int((start >= end).sum())
            found.append(
                DataCheck(
                    "moment_before_target",
                    bad <= MAX_MOMENT_VIOLATIONS,
                    True,
                    f"{bad} rows have the moment at or after the target time",
                )
            )
    declared = slots.get("unit")
    if declared:
        keys = [declared] if isinstance(declared, str) else list(declared)
        if not all(k in frame.columns for k in keys):
            found.append(
                DataCheck("unit_exists", False, True, f"{keys} missing")
            )
        else:
            share = float(frame.duplicated(subset=keys).mean())
            found.append(
                DataCheck(
                    "unit_unique",
                    share <= UNIT_DUPLICATE_TOLERANCE,
                    False,
                    f"duplicate share {share:.4f}",
                )
            )
    population = slots.get("population")
    if population:
        ids = refusal.population_ids(frame, population)
        coverage = None if ids is None else len(ids) / max(len(frame), 1)
        ok = coverage is not None and MIN_POPULATION_SHARE <= coverage < 1.0
        found.append(
            DataCheck(
                "population_coverage",
                ok,
                ids is None,
                "predicate not understood"
                if coverage is None
                else f"selects {coverage:.3f} of the rows",
            )
        )
    return tuple(found)


def assemble(
    slots: Mapping[str, Any],
    request: FramingRequest,
    cost_model: costs.DecisionCostModel | None,
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

    kind = "causal" if CAUSAL_WORDS.search(request.text) else "predictive"
    unit = slots["unit"]
    record["question"] = slots.get("question") or UNSPECIFIED
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
            else pick("summary_measure", UNSPECIFIED)
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


def refused(
    reason: str,
    detail: Mapping[str, Any],
    checks: Sequence[DataCheck] = (),
    llm: dict[str, Any] | None = None,
    discarded: Sequence[str] = (),
) -> FramingDraft:
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
    return FramingDraft(
        REFUSED,
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


def frame(
    request: FramingRequest,
    data_sample: pandas.DataFrame,
    *,
    cost_model: costs.DecisionCostModel | None = None,
    baseline_column: str | None = None,
    pilot_actions: Sequence[int] | None = None,
    human: Mapping[str, Any] | None = None,
    sampler: refusal.SlotSampler | None = None,
    tau: float | None = None,
    k: int = refusal.DEFAULT_K,
) -> FramingDraft:
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
    source = request.source
    named = [c for c in BACKTICKED.findall(request.text) if " " not in c]
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
    slots = deterministic_slots(request, data_sample)
    sources = dict.fromkeys(slots, provenance.FieldSource.TEMPLATE)
    llm: dict[str, Any] | None = None
    discarded: list[str] = []
    if sampler is not None:
        prompt = refusal.prompt_for(request.text, source.columns)
        samples = refusal.draw(sampler, prompt, k)
        agreed = refusal.consensus(data_sample, samples)
        llm = {
            "model_id": sampler.model_id,
            "prompt": prompt,
            "samples": samples,
            "samples_hash": refusal.samples_hash(
                sampler.model_id, prompt, samples
            ),
            "ambiguity": agreed.ambiguity,
            "agreement": agreed.agreement,
            "tau": tau,
            "tier": "unverified",
        }
        if refusal.refuse(agreed.ambiguity, tau):
            return refused(
                "ambiguity above tau, or no calibrated tau on file",
                {
                    "ambiguity": agreed.ambiguity,
                    "tau": tau,
                    "conflicts": list(agreed.conflicts),
                },
                llm=llm,
            )
        proposal = {
            k_: v for k_, v in agreed.values.items() if v not in (None, "", [])
        }
        trial_checks = check_draft(data_sample, proposal, source)
        if any(not c.ok and c.blocking for c in trial_checks):
            discarded.append("llm draft failed a blocking data-side check")
        else:
            for name, value in proposal.items():
                if name not in slots:
                    slots[name] = value
                    sources[name] = provenance.FieldSource.LLM
    for name, value in (human or {}).items():
        slots[name] = value
        sources[name] = provenance.FieldSource.HUMAN
    checks = check_draft(data_sample, slots, source)
    blocked = [c for c in checks if not c.ok and c.blocking]
    if blocked:
        return refused(
            "a blocking data-side check failed",
            {"failed": [c.name for c in blocked]},
            checks,
            llm,
            discarded,
        )
    unfilled = [f for f in CRITICAL_FIELDS if not slots.get(f)]
    if unfilled:
        return refused(
            "critical slots cannot be filled from the request and the schema",
            {
                "unfilled": unfilled,
                "ask": dict.fromkeys(unfilled, "name the column"),
            },
            checks,
            llm,
            discarded,
        )
    try:
        charter, open_items = assemble(slots, request, cost_model)
    except (ValueError, KeyError, TypeError, errors.ConfigError) as error:
        return refused(
            "the draft does not form a valid charter",
            {"error": str(error)},
            checks,
            llm,
            discarded,
        )
    profile = None
    asked: list[scmle.data.framing.questions.HumanQuestion] = []
    robust: list[dict[str, Any]] = []
    target = slots["target"]
    if (
        baseline_column
        and baseline_column in data_sample.columns
        and cost_model
    ):
        profile = baseline.profile_on_exploration(
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
        asked, robust = scmle.data.framing.questions.ask(
            [
                scmle.data.framing.questions.Unknown(
                    "cost_threshold",
                    COST_RANGE[0],
                    COST_RANGE[1],
                    "a threshold t = c_FP / (c_FP + c_FN) in (0, 1)",
                    scmle.data.framing.questions.gain_sign_verdict(
                        outcome.to_numpy()[seen],
                        numpy.asarray(pilot_actions)[seen],
                        data_sample[baseline_column].to_numpy()[seen],
                    ),
                )
            ]
        )
        if not asked:
            open_items.remove("cost_model")
    return FramingDraft(
        ISSUED if not open_items else WITH_QUESTIONS,
        charter,
        {
            **sources,
            **{
                n: provenance.FieldSource.TEMPLATE
                for n in open_items
                if n not in sources
            },
        },
        checks,
        tuple(asked),
        tuple(robust),
        tuple(open_items),
        profile,
        None,
        llm,
        tuple(discarded),
    )


def write_artifacts(draft: FramingDraft, directory: str | Path) -> list[str]:
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
    if draft.charter is not None and draft.charter.cost_model is not None:
        documents["decision_cost_model.json"] = (
            draft.charter.cost_model.to_dict()
        )
    if draft.baseline is not None:
        documents["baseline_report.json"] = draft.baseline
    if draft.refusal is not None:
        documents["framing_refusal.json"] = draft.refusal
    for name, document in documents.items():
        (target / name).write_text(
            refusal.dump(document) + "\n", encoding="utf-8", newline="\n"
        )
    return sorted(documents)


def link_report(
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
