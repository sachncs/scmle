"""The analysis charter itself and the readers of a charter file."""

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.assurance.charter.contracts import parts
import scmle.assurance.charter.decision
import scmle.assurance.charter.ingest
import scmle.assurance.charter.rules
import scmle.assurance.charter.uncertainty
import scmle.assurance.framing.costs
import scmle.assurance.ledger.budget
from scmle.contracts import errors
from scmle.contracts import evidence
from scmle.contracts import qualification
from scmle.contracts import records
from scmle.contracts import signing

# The charter's version steps: version 1 spelled the keys before the
# record-key vocabulary; its parts are moved by their own steps.
STEPS = (
    records.nest(
        renames={
            "deployment_population": "population",
            "gating_checks": "gates",
            "decision_map": "triggers",
            "min_effect": "effect",
            "k_max": "quota",
            "error_budget": "budget",
            "cost_model": "costs",
            "unit_of_prediction": "purpose",
        },
        parts={
            "estimand": parts.AnalysisEstimand.migrations[0],
            "triggers": parts.Trigger.migrations[0],
            "sets": parts.SetSizes.migrations[0],
            "budget": (scmle.assurance.ledger.budget.ErrorBudget.migrations[0]),
            "costs": scmle.assurance.framing.costs.Tariff.migrations[0],
            "rules": scmle.assurance.charter.rules.Procedure.migrations[0],
            "decision": scmle.assurance.charter.decision.Layer.migrations[0],
            "ingest": (scmle.assurance.charter.ingest.Ingestion.migrations[0]),
            "uncertainty": (
                scmle.assurance.charter.uncertainty.Uncertainty.migrations[0]
            ),
        },
    ),
)

# The optional blocks a stored charter may leave out, with their readers.
BLOCKS: dict[str, Callable[[Any], Any]] = {
    "costs": scmle.assurance.framing.costs.Tariff.from_dict,
    "rules": scmle.assurance.charter.rules.Procedure.from_dict,
    "decision": scmle.assurance.charter.decision.Layer.from_dict,
    "ingest": scmle.assurance.charter.ingest.Ingestion.from_dict,
    "uncertainty": scmle.assurance.charter.uncertainty.Uncertainty.from_dict,
}


@dataclasses.dataclass(frozen=True, slots=True)
class AnalysisCharter:
    """The signed specification of an analysis, fixed before any exploration.

    The charter is stored at schema version 2; a version 1 record (the keys
    before the record-key vocabulary) loads through `migrations`. The
    charter hash and the signature are computed on the version 1 layout
    (`anchor`), so the rename leaves the identity of every charter as it was.

    Attributes:
        question: The question being answered.
        estimand: The estimand.
        unit: Unit of analysis (for example "customer-month").
        population: Deployment population: reference sample URI or stated
            marginals.
        gates: Checks allowed to gate a run; each needs a trigger row and a
            certificate id.
        triggers: Check to action bindings (the decision map).
        sets: How development rows are divided.
        effect: Smallest standardised effect worth acting on.
        quota: Most claims the run may carry into confirmation (`K_max`).
        power: Target power at `effect`, in (0, 1).
        budget: Error budget: alpha, exploration wealth, permutations,
            tolerances.
        multiverse: Defensible choices declared in advance.
        seed: Seed of every random draw the charter governs.
        costs: Cost model: costs of acting and not acting, fixed before `C`
            is opened; None for a charter that does not claim a decision
            benefit. Format 1 charters have none and load unchanged.
        rules: Pre-registered ledger rule, confirmation rule, failure
            policy and stability thresholds (format 3); None for a charter
            that does not fix them, which serialises, hashes and loads
            exactly as a format 1 or 2 charter did.
        decision: The decision layer (format 5): how the cost model is used,
            fairness constraints and the rollout plan; None for a charter
            that makes no decision claim. It needs `costs`.
        purpose: Unit of prediction, what a deployed prediction is for
            (format 6): `new_entity` (a row of an entity not seen in
            training) or `known_entity_future` (a later row of a known
            entity); None when the charter does not say, which serialises,
            hashes and loads as a format 1-5 charter did. `new_entity` needs
            a group column in the data and makes interval calibration use one
            score per entity.
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

    migrations: ClassVar[tuple[records.Migration, ...]] = STEPS

    question: str
    estimand: parts.AnalysisEstimand
    unit: str
    population: str
    gates: tuple[str, ...]
    triggers: tuple[parts.Trigger, ...]
    sets: parts.SetSizes
    effect: float
    quota: int
    power: float
    budget: scmle.assurance.ledger.budget.ErrorBudget
    multiverse: tuple[parts.Choice, ...]
    seed: int
    costs: scmle.assurance.framing.costs.Tariff | None = None
    rules: scmle.assurance.charter.rules.Procedure | None = None
    decision: scmle.assurance.charter.decision.Layer | None = None
    purpose: str | None = None
    ingest: scmle.assurance.charter.ingest.Ingestion | None = None
    uncertainty: scmle.assurance.charter.uncertainty.Uncertainty | None = None

    def __post_init__(self) -> None:
        """Validates the charter, including the decision-map lint.

        Raises:
            ValueError: If `question`, `unit` or `population` is blank;
                `gates`, `triggers` or `multiverse` is not a tuple; `effect`
                is not finite and positive; `quota` is not an integer of at
                least 1; `power` is not strictly in (0, 1); `seed` is not a
                non-negative integer; a multiverse name repeats; the trigger
                rows fail `cover`; a decision layer is given without
                `costs`; or `purpose` is set and not in
                `UNITS_OF_PREDICTION`.
        """
        evidence.require_text(self.question, "question")
        evidence.require_text(self.unit, "unit")
        evidence.require_text(self.population, "population")
        for name in ("gates", "triggers", "multiverse"):
            qualification.tupled(getattr(self, name), name)
        parts.positive(self.effect, "effect")
        qualification.count(self.quota, "quota", minimum=1)
        power = qualification.finite(self.power, "power")
        if not 0 < power < 1:
            raise errors.InputError("power must be strictly between 0 and 1")
        qualification.count(self.seed, "seed")
        names = [c.name for c in self.multiverse]
        if len(set(names)) != len(names):
            raise errors.InputError("duplicate name in multiverse")
        self.cover()
        if self.decision is not None and self.costs is None:
            raise errors.InputError("a decision layer needs costs")
        if (
            self.purpose is not None
            and self.purpose not in parts.UNITS_OF_PREDICTION
        ):
            raise errors.InputError(
                f"purpose must be one of {parts.UNITS_OF_PREDICTION}"
            )

    def cover(self) -> None:
        """Checks that every gating check has one row and a certificate id.

        Raises:
            ValueError: If two rows name the same check, a gating check has
                no row, or its row has a blank `certificate`.
        """
        rows: dict[str, parts.Trigger] = {}
        for row in self.triggers:
            if row.check in rows:
                raise errors.InputError(
                    f"duplicate decision row for {row.check}"
                )
            rows[row.check] = row
        for check in self.gates:
            bound = rows.get(check)
            if bound is None:
                raise errors.InputError(
                    f"gating check {check} has no decision row"
                )
            if not bound.certificate.strip():
                raise errors.InputError(
                    f"gating check {check} needs a certificate"
                )

    def lint(self, known_checks: Iterable[str]) -> None:
        """Rejects rows or gates that name a check that does not exist.

        Args:
            known_checks: Identifiers of the registered checks.

        Raises:
            ValueError: If a trigger row or a gating check names an
                identifier that is not in `known_checks`.
        """
        known = set(known_checks)
        for check in (*(r.check for r in self.triggers), *self.gates):
            if check not in known:
                raise errors.InputError(f"unknown check {check} in the charter")

    @property
    def charter_hash(self) -> str:
        """SHA-256 of the canonical JSON of the version 1 charter (64 hex)."""
        return hashlib.sha256(signing.canonicalise(self.anchor())).hexdigest()

    def anchor(self) -> records.Layout:
        """Serializes the charter in its version 1 layout.

        The charter hash and the signature cover this layout, so they stay
        what they were before the keys were renamed.

        Returns:
            The JSON-ready version 1 record, without `schema_version`.
        """
        current = self.to_dict()
        del current[records.KEY]
        return records.downgrade(self.migrations, current)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the charter at its current schema version.

        The optional blocks (`costs`, `rules`, `decision`, `purpose`,
        `ingest`, `uncertainty`) are included only when set, so a charter
        that leaves them out keeps the same serialization and hash.

        Returns:
            A JSON-ready dict of the charter with its `schema_version`;
            fractions in the error budget are written as ``"p/q"`` strings.
        """
        record = {
            "question": self.question,
            "estimand": self.estimand.to_dict(),
            "unit": self.unit,
            "population": self.population,
            "gates": list(self.gates),
            "triggers": [r.to_dict() for r in self.triggers],
            "sets": self.sets.to_dict(),
            "effect": self.effect,
            "quota": self.quota,
            "power": self.power,
            "budget": self.budget.to_dict(),
            "multiverse": [c.to_dict() for c in self.multiverse],
            "seed": self.seed,
        }
        if self.costs is not None:
            record["costs"] = self.costs.to_dict()
        if self.rules is not None:
            record["rules"] = self.rules.to_dict()
        if self.decision is not None:
            record["decision"] = self.decision.to_dict()
        if self.purpose is not None:
            record["purpose"] = self.purpose
        if self.ingest is not None:
            record["ingest"] = self.ingest.to_dict()
        if self.uncertainty is not None:
            record["uncertainty"] = self.uncertainty.to_dict()
        return {records.KEY: records.ORIGIN + len(self.migrations), **record}

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
        if self.purpose is not None:
            found = max(found, 6)
        if self.ingest is not None:
            sub = self.ingest.entity or self.ingest.reconcile
            found = max(found, 7 if sub is None else 8)
        if self.uncertainty is not None:
            found = max(found, 11 if self.uncertainty.confirmatory else 10)
        return found

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds the charter from a dict of any schema version.

        The record is first moved to the current version (`records.load`).
        The optional fields `costs`, `rules`, `decision`, `purpose`, `ingest`
        and `uncertainty` may be absent and then default to None. Other
        missing fields, and keys that are not fields, are rejected with the
        `ValueError` of `qualification.require`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The rebuilt charter.
        """
        record = qualification.require(
            cls,
            {
                **dict.fromkeys((*BLOCKS, "purpose")),
                **records.load(cls.__name__, cls.migrations, record),
            },
        )
        blocks = {
            name: revive(record[name], reader)
            for name, reader in BLOCKS.items()
        }
        return cls(
            question=record["question"],
            estimand=parts.AnalysisEstimand.from_dict(record["estimand"]),
            unit=record["unit"],
            population=record["population"],
            gates=tuple(record["gates"]),
            triggers=tuple(
                parts.Trigger.from_dict(r) for r in record["triggers"]
            ),
            sets=parts.SetSizes.from_dict(record["sets"]),
            effect=record["effect"],
            quota=record["quota"],
            power=record["power"],
            budget=scmle.assurance.ledger.budget.ErrorBudget.from_dict(
                record["budget"]
            ),
            multiverse=tuple(
                parts.Choice.from_dict(c) for c in record["multiverse"]
            ),
            seed=record["seed"],
            purpose=record["purpose"],
            **blocks,
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
    minimum effect of 0.2, a quota (`K_max`) of 10, power 0.8, an error
    budget created with levels 0.05 and 0.1, and seed 0.

    Returns:
        A dict in the shape of `AnalysisCharter.to_dict`.
    """
    return {
        records.KEY: records.ORIGIN + len(AnalysisCharter.migrations),
        "question": "",
        "estimand": {
            "kind": "predictive",
            "population": "",
            "target": "",
            "moment": "",
            "intercurrent": [],
            "summary": "",
        },
        "unit": "",
        "population": "",
        "gates": [],
        "triggers": [],
        "sets": {"confirmation": 0.25, "embargo": 0},
        "effect": 0.2,
        "quota": 10,
        "power": 0.8,
        "budget": scmle.assurance.ledger.budget.ErrorBudget.create(
            0.05, 0.1
        ).to_dict(),
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
