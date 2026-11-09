"""The baseline claim as a registered claim of a charter run.

The comparison is "the model rule has a higher net benefit than the current
process over the charter's cost range" (`baseline`). In a run it works like
this:

1. A rule is frozen on the exploration rows `X` before `C` is opened: a
   standardised logistic regression on the numeric features of the run,
   fitted with the run seed, acting where its probability reaches the cost
   model's implied threshold. Its fitted parameters are hashed.
2. The registration (cost-model hash, rule hash, level, range) is written to
   the ledger as a `claim` row by hash, and `BaselineGuard` evaluates it once.
3. The runner receives the frame `C` releases for the confirmation, in the
   same opening, and tests both ends of the cost range.

By default the claim is about this frozen rule, not about the model the run
goes on to select and train later; the later model is judged on the locked
test set. With `data.baseline_subject = "selected_model"` the claim is about
the selected model instead: the opening of `C` is deferred until the
selection is frozen (`selection.frozen`), the registration hashes the
selection (model, strategy, parameters, trial and the split it was made on),
and the logistic rule above is reported next to it as the default reference. The
selection of outcomes on `X` and on the rows set aside for a missing target is
checked on those rows (`C` has no unlabeled rows, they are removed before the
split), and a flag there makes the claim `not_issued` without a test.

The tier a baseline claim may reach is capped by `provenance.claim_tier`: an
unsigned charter with an LLM-filled field, or a cost that came from data or a
template, leaves it `unverified`. The cap is read from `framing_draft.json`
when the data spec names it.

Limits. The level of this claim is the charter's `alpha_confirm`, spent on
this one claim; the guarantee of the bivariate family is separate and no
joint correction across the two is claimed. The default rule is a plain linear
model: a claim that it beats the current process says nothing about a better
model. The selected-model mode is valid because the selection used `X`
only; a selection that read `C` would void it.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import json
from pathlib import Path
from typing import Any
from typing import Protocol

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
from scmle.data.ledger import ledger as ledger_module

DOCUMENT = "baseline_report.json"
SELECTED = "selected_model"
FROZEN = "frozen_rule"
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
        """Describes the rule in the form that is hashed.

        Returns:
            A JSON-ready mapping of the rule kind, the ridge strength, the
            columns and the fitted parameters (rounded to 12 places), and
            the cut.
        """
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
        """Builds the standardised design matrix for any rows.

        Args:
            features: Rows holding every column of the rule. Entries that
                are missing or not numeric are replaced by the medians.

        Returns:
            An array with one row per input row and one column per rule
            column, centred by `means` and divided by `scales`.
        """
        raw = (
            features[list(self.columns)]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        raw = np.where(np.isnan(raw), np.asarray(self.medians), raw)
        return (raw - np.asarray(self.means)) / np.asarray(self.scales)

    def actions(self, features: pd.DataFrame) -> np.ndarray:
        """Computes the actions of the rule on any rows.

        Args:
            features: Rows holding every column of the rule.

        Returns:
            1.0 for each row whose predicted probability reaches `cut`,
            otherwise 0.0.
        """
        score = self.matrix(features) @ np.asarray(self.coefficients)
        probability = 1.0 / (1.0 + np.exp(-(score + self.intercept)))
        return (probability >= self.cut).astype(float)


class Rule(Protocol):
    """What the claim needs of the rule it is about."""

    def describe(self) -> dict[str, Any]:
        """Describes the rule in the form that is hashed.

        Returns:
            A JSON-ready mapping that identifies the rule.
        """

    def actions(self, features: pd.DataFrame) -> np.ndarray:
        """Computes the actions of the rule on any rows.

        Args:
            features: Rows to act on.

        Returns:
            1.0 for each row the rule acts on, otherwise 0.0.
        """


@dataclasses.dataclass(frozen=True, slots=True)
class SelectedModelRule:
    """The model a run selected, as the rule of the claim.

    The model was chosen and fitted on the development rows (never on `C`)
    and the selection was frozen before the claim was registered. `identity`
    names the choice and the data it was made on; its hash is part of the
    registration.

    Attributes:
        pipeline: The fitted candidate with `predict_proba` and `classes_`.
        positive: The positive class label.
        cut: Probability at which the model acts (the cost model's
            implied threshold).
        identity: Model name, strategy, parameters, trial id and the split
            fingerprint the selection was made on.
    """

    pipeline: Any
    positive: Any
    cut: float
    identity: Mapping[str, Any]

    def describe(self) -> dict[str, Any]:
        """Describes the rule in the form that is hashed.

        Returns:
            A JSON-ready mapping of the kind, the selection identity, the
            cut, and `selection_sha256`, the digest of the identity.
        """
        document = jsonable(
            {"kind": SELECTED, **self.identity, "cut": self.cut}
        )
        document["selection_sha256"] = ledger_module.digest(
            jsonable(self.identity)
        )
        return document

    def actions(self, features: pd.DataFrame) -> np.ndarray:
        """Computes the actions of the selected model on any rows.

        Args:
            features: Rows to score with the fitted pipeline.

        Returns:
            1.0 for each row whose predicted probability of the positive
            class reaches `cut`, otherwise 0.0.
        """
        classes = list(self.pipeline.classes_)
        column = classes.index(self.positive)
        probability = np.asarray(self.pipeline.predict_proba(features))[
            :, column
        ]
        return (probability >= self.cut).astype(float)


def freeze_rule(
    features: pd.DataFrame,
    outcome: Sequence[float],
    columns: Sequence[str],
    cut: float,
    seed: int,
) -> FrozenRule:
    """Fits the rule on exploration rows with an observed outcome.

    Args:
        features: Exploration rows.
        outcome: Outcome of each row, 0 or 1, NaN where unobserved.
        columns: Candidate numeric feature names. Names absent from
            `features` are dropped.
        cut: Probability at which the rule acts.
        seed: Seed of the logistic regression.

    Returns:
        The frozen rule, with medians, means and scales taken over all
        exploration rows and the regression fitted on the observed ones.

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
    """Finds the highest tier the inputs of the baseline claim allow, and why.

    Reads `framing_draft.json` when given. A draft that does not describe this
    charter (its hash differs, because a person edited the charter) keeps its
    field sources, since the edit is not known to have replaced an LLM field;
    the charter's signature is what lifts the cap.

    Args:
        draft_path: Path of the framing draft, or None when none was named.
        charter_hash: Hash of the charter being run.
        model: The charter's cost model.
        signed: Whether the charter carries a verified signature.

    Returns:
        The tier ceiling, and a JSON-ready explanation holding the ceiling,
        whether the charter is signed, the LLM-filled fields, the costs not
        given by a person, and how the draft relates to the charter.

    Raises:
        ConfigError: If the draft cannot be read, is not valid JSON, lacks a
            `fields` mapping, or names an unknown field source.
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
    """Round-trips a document through JSON.

    The hashed document is then the one that is written.

    Args:
        document: The document to convert. Values JSON cannot encode are
            converted with `str`.

    Returns:
        A plain dict with sorted keys, built from JSON types only.
    """
    return dict(json.loads(json.dumps(document, default=str, sort_keys=True)))


def build_runner(
    *,
    rule: Rule,
    reference: FrozenRule | None = None,
    model: costs.DecisionCostModel,
    alpha: float,
    baseline_column: str,
    group_column: str | None,
    classes: Sequence[Any],
    selection: Mapping[str, Any],
    ceiling: qualification.Tier,
    seed: int,
) -> confirm.ClaimRunner:
    """Registers the baseline claim and builds the function that tests it.

    Args:
        rule: The rule the claim is about: the one frozen on `X`, or the
            model the run selected.
        reference: The frozen logistic rule, reported next to a selected
            model as the default declared reference (descriptive).
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

    Returns:
        A runner holding the one registered claim and the function that
        evaluates it on the frame released for confirmation. That function
        writes the report document `baseline_report.json` and reports the
        claim's status; with a flagged `selection` it runs no test.
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
        """Tests the registered claim on the rows released for confirmation.

        Args:
            features: Confirmation features.
            target: Confirmation target.

        Returns:
            The status of the claim and the report document.
        """
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
        if reference is not None:
            document["reference_rule"] = reference.describe()
            if not selection["flagged"]:
                document["reference_agreement"] = float(
                    np.mean(
                        reference.actions(features) == rule.actions(features)
                    )
                )
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
    """Builds the report for a claim stopped by selective labels on `X`.

    Args:
        registration: The registration of the claim.
        selection: The selective-label finding on `X` and the unlabeled rows.
        ceiling: Highest tier the inputs allow (not used in the report, as
            the claim is not evaluated).

    Returns:
        A report document with status `not_issued`, the reason, no
        endpoints or checks, and a claim at tier `unverified`.
    """
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
