"""Identifiers, limits and settings of the leakage tests.

`CERTIFIED` holds the settings the skill test was certified with.
"""

import dataclasses

import numpy

from scmle.assurance.ledger import budget
from scmle.contracts import errors
from scmle.contracts import qualification

DETECTOR_ID = "f4-leakage-skill"


LEAK_DEFECTS = (
    "leak_copy",
    "leak_noisy_copy",
    "leak_threshold",
    "leak_missing_for_class",
)


TASKS = ("classification", "regression")


MAX_LEVELS = 50


EXPECTED_FISHER = 5.0


TARGET_BINS = 4


MIN_STRATUM_ROWS = 2


SKILL = "leakage.skill"


CONDITIONAL = "leakage.conditional"


MISSING = "leakage.missingness"


MIN_CLASSES = 2


BINARY_CLASSES = 2


Splits = list[tuple[numpy.ndarray, numpy.ndarray]]


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    """Pre-registered settings of the leakage tests.

    Attributes:
        skill_ceiling: Highest skill (AUC or R^2) plausible for a legitimate
            single feature in this domain. Set by the charter; no default.
        association_ceiling: Cramer's V of missingness against the target
            above which missingness is suspect. Set by the charter.
        alpha: Level of the permutation tests.
        permutations: `B`; at least `budget.MIN_PERMUTATIONS` and at least
            `1 / alpha - 1`, so the level is reachable.
        folds: Cross-fitting folds.
        confidence: Level of the one-sided lower bound.
        bootstrap: Bootstrap draws for the lower bound.
        min_rows: Fewer rows than this is reported as low power.
        max_leaf_nodes: Size of the small tree.
        min_samples_leaf: Smallest leaf of the small tree.
        n_strata: Leaves of the tree that defines strata.
        run_seed: Run seed for every random stream.
    """

    skill_ceiling: float
    association_ceiling: float
    alpha: float = 0.05
    permutations: int = 199
    folds: int = 5
    confidence: float = 0.95
    bootstrap: int = 200
    min_rows: int = 100
    max_leaf_nodes: int = 8
    min_samples_leaf: int = 5
    n_strata: int = 8
    run_seed: int = 0

    def __post_init__(self) -> None:
        """Validates the settings.

        Raises:
            ValueError: If a ceiling is outside (0, 1], `alpha` or
                `confidence` is outside (0, 1), `permutations` is too few
                to reach `alpha` (`permutations + 1 < 1 / alpha`), or a
                count is not an integer at or above its minimum.
        """
        for name in ("skill_ceiling", "association_ceiling"):
            value = qualification.finite(getattr(self, name), name)
            if not 0.0 < value <= 1.0:
                raise errors.InputError(f"{name} must be within (0, 1]")
        for name in ("alpha", "confidence"):
            value = qualification.finite(getattr(self, name), name)
            if not 0.0 < value < 1.0:
                raise errors.InputError(f"{name} must be within (0, 1)")
        qualification.count(
            self.permutations, "permutations", budget.MIN_PERMUTATIONS
        )
        if self.permutations + 1 < 1.0 / self.alpha:
            raise errors.InputError(
                "permutations too few to reach alpha: need B + 1 >= 1 / alpha"
            )
        qualification.count(self.folds, "folds", minimum=2)
        qualification.count(self.bootstrap, "bootstrap", minimum=2)
        qualification.count(self.min_rows, "min_rows", minimum=1)
        qualification.count(self.max_leaf_nodes, "max_leaf_nodes", minimum=2)
        qualification.count(self.min_samples_leaf, "min_samples_leaf", 1)
        qualification.count(self.n_strata, "n_strata", minimum=2)
        qualification.count(self.run_seed, "run_seed")


CERTIFIED = Config(skill_ceiling=0.95, association_ceiling=0.9)
