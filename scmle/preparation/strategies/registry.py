"""The strategy registry."""

from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.preparation.strategies.features import Construction
from scmle.preparation.strategies.features import Encoding
from scmle.preparation.strategies.features import Indicators
from scmle.preparation.strategies.features import Interactions
from scmle.preparation.strategies.features import IterativeImputation
from scmle.preparation.strategies.features import Logarithm
from scmle.preparation.strategies.features import Rarity
from scmle.preparation.strategies.features import Selection
from scmle.preparation.strategies.features import Transformation
from scmle.preparation.strategies.interface import Default
from scmle.preparation.strategies.interface import Strategy
from scmle.preparation.strategies.rebalancing import Adasyn
from scmle.preparation.strategies.rebalancing import Borderline
from scmle.preparation.strategies.rebalancing import Copula
from scmle.preparation.strategies.rebalancing import Jitter
from scmle.preparation.strategies.rebalancing import Oversampling
from scmle.preparation.strategies.rebalancing import PlainSmote
from scmle.preparation.strategies.rebalancing import Recalibration
from scmle.preparation.strategies.rebalancing import Undersampling
from scmle.preparation.strategies.rebalancing import Weighting

STRATEGIES: dict[str, Strategy] = {
    strategy.name: strategy
    for strategy in (
        Default(),
        Weighting(),
        Oversampling(),
        Undersampling(),
        PlainSmote(),
        Borderline(),
        Adasyn(),
        Jitter(),
        Copula(),
        Recalibration(),
        Logarithm(),
        Interactions(),
        Construction(),
        Selection(),
        Indicators(),
        IterativeImputation(),
        Transformation(),
        Encoding(),
        Rarity(),
    )
}


EXPERIMENTAL = tuple(
    name for name in STRATEGIES if name != experiments.DEFAULT_STRATEGY
)


OPT_IN = frozenset({"jitter_augment", "copula_augment"})


DEFAULT_NAMES = tuple(n for n in EXPERIMENTAL if n not in OPT_IN)


def lookup(name: str) -> Strategy:
    """Looks up a registered data strategy.

    Args:
        name: The strategy name, as recorded on trials.

    Returns:
        The registered strategy.

    Raises:
        ConfigError: If the name is unknown.
    """
    if name not in STRATEGIES:
        raise errors.ConfigError(
            f"unknown strategy '{name}'; available: {sorted(STRATEGIES)}"
        )
    return STRATEGIES[name]
