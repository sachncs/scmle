"""The baseline claim of F8 as a registered claim of a charter run.

The comparison is "the model rule has a higher net benefit than the current
process over the charter's cost range" (`baseline`). In a run it works like
this:

1. A rule is frozen on the exploration rows `X` before `C` is opened: a
   standardised logistic regression on the numeric features of the plan,
   fitted with the run seed, acting where its probability reaches the cost
   model's implied threshold. Its fitted parameters are hashed.
2. The registration (cost-model hash, rule hash, level, range) is written to
   the ledger as a `claim` row by hash, and `BaselineGuard` evaluates it once.
3. The runner receives the frame `C` releases for the confirmation, in the
   same opening, and tests both ends of the cost range.

The claim is about this frozen rule, not about the model the run goes on to
select and train later; the later model is judged on the locked test set. The
selection of outcomes on `X` and on the rows set aside for a missing target is
checked on those rows (`C` has no unlabeled rows, they are removed before the
split), and a flag there makes the claim `not_issued` without a test.

The tier a baseline claim may reach is capped by `provenance.claim_tier`: an
unsigned charter with an LLM-filled field, or a cost that came from data or a
template, leaves it `unverified`. The cap is read from `framing_draft.json`
when the data spec names it.

Limits. The level of this claim is the charter's `alpha_confirm`, spent on
this one claim; the guarantee of the bivariate family is separate and no
joint correction across the two is claimed. The rule is a plain linear model:
a claim that it beats the current process says nothing about a better model.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn import linear_model

from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.data.framing import baseline
from scmle.data.framing import costs
from scmle.data.framing import provenance
from scmle.data.ledger import confirm
from scmle.data.ledger import explore

DOCUMENT = "baseline_report.json"
PROVENANCE = "framing_provenance.json"
KIND = "baseline"
RIDGE = 1.0


@dataclasses.dataclass(frozen=True, slots=True)
class FrozenRule:
    """A linear rule fixed on `X`.

    Attributes:
        columns: Numeric feature names, in order.
        medians: Value that replaces a missing entry.
        means: Column means used to standardise.
        scales: Column scales used to standardise (never zero).
        coefficients: Logistic coefficients on the standardised columns.
        intercept: Logistic intercept.
        cut: Probability at which the rule acts (the cost model's threshold).
    """

    columns: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    cut: float

    def describe(self) -> dict[str, Any]:
        """JSON-ready description, the thing that is hashed."""
        return {
            "kind": "standardised_logistic",
            "ridge": RIDGE,
            "columns": list(self.columns),
            "medians": [round(v, 12) for v in self.medians],
            "means": [round(v, 12) for v in self.means],
            "scales": [round(v, 12) for v in self.scales],
            "coefficients": [round(v, 12) for v in self.coefficients],
            "intercept": round(self.intercept, 12),
            "cut": self.cut,
        }

    def matrix(self, features: pd.DataFrame) -> np.ndarray:
        """The standardised design for any rows."""
        raw = (
            features[list(self.columns)]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        raw = np.where(np.isnan(raw), np.asarray(self.medians), raw)
        return (raw - np.asarray(self.means)) / np.asarray(self.scales)

    def actions(self, features: pd.DataFrame) -> np.ndarray:
        """0/1 actions on any rows."""
        score = self.matrix(features) @ np.asarray(self.coefficients)
        probability = 1.0 / (1.0 + np.exp(-(score + self.intercept)))
        return (probability >= self.cut).astype(float)


def freeze_rule(
    features: pd.DataFrame,
    outcome: Sequence[float],
    columns: Sequence[str],
    cut: float,
    seed: int,
) -> FrozenRule:
    """Fits the rule on exploration rows with an observed outcome.

    Raises:
        ConfigError: With no numeric feature, or one outcome class only.
    """
    names = tuple(c for c in columns if c in features.columns)
    if not names:
        raise errors.ConfigError(
            "the baseline claim needs at least one numeric feature to freeze "
            "a rule on the exploration rows"
        )
    y = np.asarray(outcome, dtype=float)
    seen = ~np.isnan(y)
    if len(np.unique(y[seen])) < 2:
        raise errors.ConfigError(
            "the baseline claim needs both outcome classes in the exploration "
            "rows"
        )
    raw = (
        features.loc[:, list(names)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    medians = np.nanmedian(raw, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    filled = np.where(np.isnan(raw), medians, raw)
    means = filled.mean(axis=0)
    scales = filled.std(axis=0)
    scales = np.where(scales == 0.0, 1.0, scales)
    model = linear_model.LogisticRegression(
        C=RIDGE, max_iter=1000, random_state=seed
    )
    model.fit(((filled - means) / scales)[seen], y[seen])
    return FrozenRule(
        names,
        tuple(float(v) for v in medians),
        tuple(float(v) for v in means),
        tuple(float(v) for v in scales),
        tuple(float(v) for v in model.coef_[0]),
        float(model.intercept_[0]),
        float(cut),
    )


def source_ceiling(
    draft_path: str | None,
    charter_hash: str,
    model: costs.DecisionCostModel,
    *,
    signed: bool,
) -> tuple[qualification.Tier, dict[str, Any]]:
    """Highest tier the inputs of the baseline claim allow, and why.

    Reads `framing_draft.json` when given. A draft that does not describe this
    charter (its hash differs, because a person edited the charter) keeps its
    field sources, since the edit is not known to have replaced an LLM field;
    the charter's signature is what lifts the cap.
    """
    fields: dict[str, str] = {}
    note = "no framing draft was named: no field is known to be LLM-filled"
    if draft_path:
        try:
            draft = json.loads(Path(draft_path).read_text(encoding="utf-8"))
            fields = {str(k): str(v) for k, v in draft["fields"].items()}
            for value in fields.values():
                provenance.FieldSource(value)
        except (OSError, ValueError, KeyError, AttributeError) as error:
            raise errors.ConfigError(
                f"cannot read framing draft {draft_path}: {error}"
            ) from error
        note = (
            "the draft describes this charter"
            if draft.get("charter_hash") == charter_hash
            else "the charter differs from the draft (edited after drafting)"
        )
    sources = [provenance.FieldSource(v) for v in fields.values()]
    from_data = bool(model.unverified_entries)
    tier = provenance.claim_tier(
        qualification.Tier.CONFIRMED,
        sources,
        signed=signed,
        cost_from_data=from_data,
    )
    return tier, {
        "tier_ceiling": tier.value,
        "charter_signed": signed,
        "llm_fields": sorted(
            k
            for k, v in fields.items()
            if v == provenance.FieldSource.LLM.value
        ),
        "cost_entries_not_from_a_person": list(model.unverified_entries),
        "draft": note,
    }


def jsonable(document: Mapping[str, Any]) -> dict[str, Any]:
    """A JSON round trip, so the hashed document is the written one."""
    return dict(json.loads(json.dumps(document, default=str, sort_keys=True)))


def build_runner(
    *,
    rule: FrozenRule,
    model: costs.DecisionCostModel,
    alpha: float,
    baseline_column: str,
    group_column: str | None,
    classes: Sequence[Any],
    selection: Mapping[str, Any],
    ceiling: qualification.Tier,
    seed: int,
) -> confirm.ClaimRunner:
    """The registered baseline claim and the function that tests it on `C`.

    Args:
        rule: The rule frozen on `X`.
        model: The charter's cost model.
        alpha: The confirmation level.
        baseline_column: Column of the current process's 0/1 decisions.
        group_column: Entity column for the cluster-robust error, or None
            (each row is then its own cluster).
        classes: The target's classes in coding order; the last is positive.
        selection: Selective-label finding on `X` and the unlabeled rows
            (`baseline.Selectivity.to_dict()`), with the `flagged` key.
        ceiling: Highest tier the inputs allow.
        seed: The run seed.
    """
    description = rule.describe()
    registration = baseline.register(model, description, alpha)
    guard = baseline.BaselineGuard(registration)
    claim = confirm.RegisteredClaim(
        baseline.CLAIM_ID,
        KIND,
        "net benefit of the frozen rule exceeds net benefit of the current "
        f"process over thresholds [{model.t_lo:.4f}, {model.t_hi:.4f}]",
        registration.registration_hash,
    )

    def run(features: pd.DataFrame, target: pd.Series) -> confirm.ClaimRun:
        if selection["flagged"]:
            document = refused(registration, selection, ceiling)
        else:
            y = explore.encode_target(target, True, classes).to_numpy(float)
            clusters = (
                features[group_column].astype(str).to_numpy()
                if group_column and group_column in features.columns
                else np.arange(len(features))
            )
            report = guard.evaluate(
                y,
                rule.actions(features),
                features[baseline_column].to_numpy(dtype=float),
                clusters,
                rule=description,
                model=model,
                input_tier=ceiling,
                seed=seed,
            )
            document = report.to_dict()
        document["rule"] = description
        document["selection_on_x"] = dict(selection)
        document["tier_ceiling"] = ceiling.value
        document = jsonable(document)
        return confirm.ClaimRun(
            {baseline.CLAIM_ID: str(document["status"])}, {DOCUMENT: document}
        )

    return confirm.ClaimRunner((claim,), run)


def refused(
    registration: baseline.BaselineRegistration,
    selection: Mapping[str, Any],
    ceiling: qualification.Tier,
) -> dict[str, Any]:
    """The report when selective labels on `X` stop the comparison."""
    reason = (
        "outcomes are observed depending on the baseline decision on the "
        "exploration and unlabeled rows (selective labels); the naive "
        "comparison is biased and was not run"
    )
    return {
        "schema_version": 1,
        "status": "not_issued",
        "issued": False,
        "reasons": [reason],
        "p_value_intersection_union": None,
        "endpoints": [],
        "selective_labels": dict(selection),
        "registration_hash": registration.registration_hash,
        "checks": [],
        "claim": {
            "claim_id": baseline.CLAIM_ID,
            "tier": qualification.Tier.UNVERIFIED.value,
            "statement": "not evaluated: " + reason,
        },
    }
