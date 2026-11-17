"""The analysis charter itself and the readers of a charter file."""

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any
from typing import Self

from scmle.assurance.charter.contracts import parts
import scmle.assurance.charter.decision
import scmle.assurance.charter.ingest
import scmle.assurance.charter.rules
import scmle.assurance.charter.uncertainty
from scmle.assurance.framing import costs
from scmle.assurance.ledger import budget
from scmle.contracts import errors
from scmle.contracts import evidence
from scmle.contracts import qualification
from scmle.contracts import signing


@dataclasses.dataclass(frozen=True, slots=True)
class AnalysisCharter:
    """The signed specification of an analysis, fixed before any exploration.

    Attributes:
        question: The question being answered.
        estimand: The estimand.
        unit: Unit of analysis (for example "customer-month").
        deployment_population: Reference sample URI or stated marginals.
        gating_checks: Checks allowed to gate a run; each needs a decision
            row and a certificate id.
        decision_map: Check to action bindings.
        sets: How development rows are divided.
        min_effect: Smallest standardised effect worth acting on.
        k_max: Most claims the run may carry into confirmation.
        power: Target power at `min_effect`, in (0, 1).
        error_budget: Alpha, exploration wealth, permutations, tolerances.
        multiverse: Defensible choices declared in advance.
        seed: Seed of every random draw the charter governs.
        cost_model: Costs of acting and not acting, fixed before `C`
            is opened; None for a charter that does not claim a decision
            benefit. Format 1 charters have none and load unchanged.
        rules: Pre-registered ledger rule, confirmation rule, failure
            policy and stability thresholds (format 3); None for a charter
            that does not fix them, which serialises, hashes and loads
            exactly as a format 1 or 2 charter did.
        decision: The decision layer (format 5): how the cost model is used,
            fairness constraints and the rollout plan; None for a charter
            that makes no decision claim. It needs a `cost_model`.
        unit_of_prediction: What a deployed prediction is for (format 6):
            `new_entity` (a row of an entity not seen in training) or
            `known_entity_future` (a later row of a known entity); None when
            the charter does not say, which serialises, hashes and loads as
            a format 1-5 charter did. `new_entity` needs a group column in
            the data and makes interval calibration use one score per entity.
        ingest: The ingestion block (format 7): source budget, `B_feat`,
            windows, lag, reconciliation delay and key tolerances; None for
            a charter that reads a file, which serialises, hashes and loads
            as a format 1-6 charter did.
        uncertainty: The uncertainty block (format 10, or 11 with the
            coverage claim on the confirmation rows): the cap on the fold
            models a bundle keeps for CV+; None for a charter that does not
            fix it, which serialises, hashes and loads as a format 1-9
            charter did.
    """

    question: str
    estimand: parts.AnalysisEstimand
    unit: str
    deployment_population: str
    gating_checks: tuple[str, ...]
    decision_map: tuple[parts.Trigger, ...]
    sets: parts.SetSizes
    min_effect: float
    k_max: int
    power: float
    error_budget: budget.ErrorBudget
    multiverse: tuple[parts.Choice, ...]
    seed: int
    cost_model: costs.Tariff | None = None
    rules: scmle.assurance.charter.rules.Procedure | None = None
    decision: scmle.assurance.charter.decision.Layer | None = None
    unit_of_prediction: str | None = None
    ingest: scmle.assurance.charter.ingest.Ingestion | None = None
    uncertainty: scmle.assurance.charter.uncertainty.Uncertainty | None = None

    def __post_init__(self) -> None:
        """Validates the charter, including the decision-map lint.

        Raises:
            ValueError: If `question`, `unit` or `deployment_population` is
                blank; `gating_checks`, `decision_map` or `multiverse` is
                not a tuple; `min_effect` is not finite and positive;
                `k_max` is not an integer of at least 1; `power` is not
                strictly in (0, 1); `seed` is not a non-negative integer; a
                multiverse name repeats; the decision map fails
                `cover`; a decision layer is given without a
                `cost_model`; or `unit_of_prediction` is set and not in
                `UNITS_OF_PREDICTION`.
        """
        evidence.require_text(self.question, "question")
        evidence.require_text(self.unit, "unit")
        evidence.require_text(
            self.deployment_population, "deployment_population"
        )
        for name in ("gating_checks", "decision_map", "multiverse"):
            qualification.tupled(getattr(self, name), name)
        parts.positive(self.min_effect, "min_effect")
        qualification.count(self.k_max, "k_max", minimum=1)
        power = qualification.finite(self.power, "power")
        if not 0 < power < 1:
            raise errors.InputError("power must be strictly between 0 and 1")
        qualification.count(self.seed, "seed")
        names = [c.name for c in self.multiverse]
        if len(set(names)) != len(names):
            raise errors.InputError("duplicate name in multiverse")
        self.cover()
        if self.decision is not None and self.cost_model is None:
            raise errors.InputError("a decision layer needs a cost_model")
        if (
            self.unit_of_prediction is not None
            and self.unit_of_prediction not in parts.UNITS_OF_PREDICTION
        ):
            raise errors.InputError(
                f"unit_of_prediction must be one of {parts.UNITS_OF_PREDICTION}"
            )

    def cover(self) -> None:
        """Checks that every gating check has one row and a certificate id.

        Raises:
            ValueError: If two rows name the same check, a gating check has
                no row, or its row has a blank `certificate_id`.
        """
        rows: dict[str, parts.Trigger] = {}
        for row in self.decision_map:
            if row.check_id in rows:
                raise errors.InputError(
                    f"duplicate decision row for {row.check_id}"
                )
            rows[row.check_id] = row
        for check_id in self.gating_checks:
            bound = rows.get(check_id)
            if bound is None:
                raise errors.InputError(
                    f"gating check {check_id} has no decision row"
                )
            if not bound.certificate_id.strip():
                raise errors.InputError(
                    f"gating check {check_id} needs a certificate_id"
                )

    def lint(self, known_checks: Iterable[str]) -> None:
        """Rejects rows or gates that name a check that does not exist.

        Args:
            known_checks: Identifiers of the registered checks.

        Raises:
            ValueError: If a decision row or a gating check names an
                identifier that is not in `known_checks`.
        """
        known = set(known_checks)
        for check_id in (
            *(r.check_id for r in self.decision_map),
            *self.gating_checks,
        ):
            if check_id not in known:
                raise errors.InputError(
                    f"unknown check {check_id} in the charter"
                )

    @property
    def charter_hash(self) -> str:
        """SHA-256 of the canonical JSON of the charter (64 hex digits)."""
        return hashlib.sha256(signing.canonicalise(self.to_dict())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serializes the charter.

        The optional blocks (`cost_model`, `rules`, `decision`,
        `unit_of_prediction`, `ingest`, `uncertainty`) are included only
        when set, so a charter that leaves them out keeps the same
        serialization and hash.

        Returns:
            A JSON-ready dict of the charter; fractions in the error budget
            are written as ``"p/q"`` strings.
        """
        record = {
            "question": self.question,
            "estimand": self.estimand.to_dict(),
            "unit": self.unit,
            "deployment_population": self.deployment_population,
            "gating_checks": list(self.gating_checks),
            "decision_map": [r.to_dict() for r in self.decision_map],
            "sets": self.sets.to_dict(),
            "min_effect": self.min_effect,
            "k_max": self.k_max,
            "power": self.power,
            "error_budget": self.error_budget.to_dict(),
            "multiverse": [c.to_dict() for c in self.multiverse],
            "seed": self.seed,
        }
        if self.cost_model is not None:
            record["cost_model"] = self.cost_model.to_dict()
        if self.rules is not None:
            record["rules"] = self.rules.to_dict()
        if self.decision is not None:
            record["decision"] = self.decision.to_dict()
        if self.unit_of_prediction is not None:
            record["unit_of_prediction"] = self.unit_of_prediction
        if self.ingest is not None:
            record["ingest"] = self.ingest.to_dict()
        if self.uncertainty is not None:
            record["uncertainty"] = self.uncertainty.to_dict()
        return record

    @property
    def format_version(self) -> int:
        """The oldest charter format that can carry this charter.

        It is 2 for a charter without optional blocks; 3 with rules; 4 with
        a causal block; 5 with a decision layer; 6 with a unit of
        prediction; 7 with an ingest block; 8 when the ingest block carries
        an entity or reconcile sub-block; 9 when the causal block carries
        subgroups, attestors or an attestations file; 10 with an
        uncertainty block; 11 when that block asks for the coverage claim on
        the confirmation rows. The largest applicable value is used.
        """
        found = 2 if self.rules is None else 3
        causal_block = self.estimand.causal
        if causal_block is not None:
            extended = (
                causal_block.subgroups
                or causal_block.attestors
                or causal_block.attestations
            )
            found = max(found, 9 if extended else 4)
        if self.decision is not None:
            found = max(found, 5)
        if self.unit_of_prediction is not None:
            found = max(found, 6)
        if self.ingest is not None:
            sub = self.ingest.entity or self.ingest.reconcile
            found = max(found, 7 if sub is None else 8)
        if self.uncertainty is not None:
            found = max(
                found, 11 if self.uncertainty.coverage_on_confirmation else 10
            )
        return found

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the charter from a dict.

        The optional fields `cost_model`, `rules`, `decision`,
        `unit_of_prediction`, `ingest` and `uncertainty` may be absent and
        then default to None. Other missing fields, and keys that are not
        fields, are rejected with the `ValueError` of
        `qualification.require`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt charter.
        """
        record = {
            "cost_model": None,
            "rules": None,
            "decision": None,
            "unit_of_prediction": None,
            "ingest": None,
            "uncertainty": None,
            **record,
        }
        qualification.require(cls, record)
        return cls(
            question=record["question"],
            estimand=parts.AnalysisEstimand.from_dict(record["estimand"]),
            unit=record["unit"],
            deployment_population=record["deployment_population"],
            gating_checks=tuple(record["gating_checks"]),
            decision_map=tuple(
                parts.Trigger.from_dict(r) for r in record["decision_map"]
            ),
            sets=parts.SetSizes.from_dict(record["sets"]),
            min_effect=record["min_effect"],
            k_max=record["k_max"],
            power=record["power"],
            error_budget=budget.ErrorBudget.from_dict(record["error_budget"]),
            multiverse=tuple(
                parts.Choice.from_dict(c) for c in record["multiverse"]
            ),
            seed=record["seed"],
            cost_model=revive(record["cost_model"], costs.Tariff.from_dict),
            rules=revive(
                record["rules"],
                scmle.assurance.charter.rules.Procedure.from_dict,
            ),
            decision=revive(
                record["decision"],
                scmle.assurance.charter.decision.Layer.from_dict,
            ),
            unit_of_prediction=record["unit_of_prediction"],
            ingest=revive(
                record["ingest"],
                scmle.assurance.charter.ingest.Ingestion.from_dict,
            ),
            uncertainty=revive(
                record["uncertainty"],
                scmle.assurance.charter.uncertainty.Uncertainty.from_dict,
            ),
        )


def revive(value: Any, reader: Callable[[Any], Any]) -> Any:
    """Rebuilds an optional charter block.

    Args:
        value: The stored block, or None when the charter has none.
        reader: The `from_dict` of the block's class.

    Returns:
        None when `value` is None, otherwise what `reader` builds from it.
    """
    return None if value is None else reader(value)


def template() -> dict[str, Any]:
    """Builds a skeleton charter record for the analyst to fill in.

    The required text fields are blank and the lists are empty, so the
    record fails validation until they are filled in. The numeric fields
    carry starting values: a confirmation fraction of 0.25, no embargo, a
    minimum effect of 0.2, `k_max` 10, power 0.8, an error budget created
    with levels 0.05 and 0.1, and seed 0.

    Returns:
        A dict in the shape of `AnalysisCharter.to_dict`.
    """
    return {
        "question": "",
        "estimand": {
            "kind": "predictive",
            "population": "",
            "target": "",
            "prediction_moment": "",
            "intercurrent_events": [],
            "summary_measure": "",
        },
        "unit": "",
        "deployment_population": "",
        "gating_checks": [],
        "decision_map": [],
        "sets": {"confirmation_fraction": 0.25, "embargo_rows": 0},
        "min_effect": 0.2,
        "k_max": 10,
        "power": 0.8,
        "error_budget": budget.ErrorBudget.create(0.05, 0.1).to_dict(),
        "multiverse": [],
        "seed": 0,
    }


def parse(record: Mapping[str, Any]) -> AnalysisCharter:
    """Builds a charter from a record, reporting problems as `ConfigError`.

    Args:
        record: A charter record as written by `AnalysisCharter.to_dict`.

    Returns:
        The validated charter.

    Raises:
        ConfigError: If `AnalysisCharter.from_dict` raises `ValueError`,
            `KeyError` or `TypeError` for the record.
    """
    try:
        return AnalysisCharter.from_dict(record)
    except (ValueError, KeyError, TypeError) as error:
        raise errors.ConfigError(
            f"invalid analysis charter: {error}"
        ) from error


def deserialize(path: str | Path) -> dict[str, Any]:
    """Reads a JSON object from a file.

    Args:
        path: The file to read, as UTF-8 text.

    Returns:
        The decoded top-level object.

    Raises:
        ConfigError: If the file cannot be read or decoded as JSON, or its
            top-level value is not an object.
    """
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise errors.ConfigError(
            f"cannot read charter {path}: {error}"
        ) from error
    if not isinstance(record, dict):
        raise errors.ConfigError(f"charter {path} is not a JSON object")
    return record


def peek(path: str | Path) -> AnalysisCharter:
    """Reads a charter file without verifying its hash or signature.

    A file with a `charter` key is read as the signed wrapper and that
    entry is used; any other object is read as a bare charter. Problems
    raise `ConfigError` (from `deserialize` and `parse`).

    Args:
        path: The charter file.

    Returns:
        The validated charter.
    """
    record = deserialize(path)
    inner = record.get("charter", record)
    return parse(inner)
