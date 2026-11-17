"""Data strategies: end-to-end variants of how a candidate is prepared.

A strategy decides *whether it applies* (with a reason when it does not) and
*how to build* the complete candidate pipeline. The search evaluates each
applicable strategy as its own independent, comparable candidate on the
original validation distribution, so a technique is adopted only on evidence.
"""

from scmle.preparation.strategies.features import Construction
from scmle.preparation.strategies.features import Encoding
from scmle.preparation.strategies.features import Features
from scmle.preparation.strategies.features import Imputation
from scmle.preparation.strategies.features import Indicators
from scmle.preparation.strategies.features import Interactions
from scmle.preparation.strategies.features import IterativeImputation
from scmle.preparation.strategies.features import Logarithm
from scmle.preparation.strategies.features import Rarity
from scmle.preparation.strategies.features import Selection
from scmle.preparation.strategies.features import Transformation
from scmle.preparation.strategies.interface import CALIBRATION_FOLDS
from scmle.preparation.strategies.interface import Default
from scmle.preparation.strategies.interface import MAX_INTERACTION_FEATURES
from scmle.preparation.strategies.interface import MAX_ITERATIVE_FEATURES
from scmle.preparation.strategies.interface import MIN_IMBALANCE_RATIO
from scmle.preparation.strategies.interface import MIN_INTERACTION_FEATURES
from scmle.preparation.strategies.interface import MIN_MISSING_RATE
from scmle.preparation.strategies.interface import MIN_SELECTION_FEATURES
from scmle.preparation.strategies.interface import SKEW_THRESHOLD
from scmle.preparation.strategies.interface import Strategy
from scmle.preparation.strategies.interface import TARGET_SKEW_THRESHOLD
from scmle.preparation.strategies.rebalancing import Adasyn
from scmle.preparation.strategies.rebalancing import Borderline
from scmle.preparation.strategies.rebalancing import Copula
from scmle.preparation.strategies.rebalancing import Jitter
from scmle.preparation.strategies.rebalancing import Oversampling
from scmle.preparation.strategies.rebalancing import PlainSmote
from scmle.preparation.strategies.rebalancing import Rebalancing
from scmle.preparation.strategies.rebalancing import Recalibration
from scmle.preparation.strategies.rebalancing import Resampling
from scmle.preparation.strategies.rebalancing import Smote
from scmle.preparation.strategies.rebalancing import Undersampling
from scmle.preparation.strategies.rebalancing import Weighting
from scmle.preparation.strategies.registry import DEFAULT_NAMES
from scmle.preparation.strategies.registry import EXPERIMENTAL
from scmle.preparation.strategies.registry import lookup
from scmle.preparation.strategies.registry import OPT_IN
from scmle.preparation.strategies.registry import STRATEGIES

__all__ = [
    "CALIBRATION_FOLDS",
    "DEFAULT_NAMES",
    "EXPERIMENTAL",
    "MAX_INTERACTION_FEATURES",
    "MAX_ITERATIVE_FEATURES",
    "MIN_IMBALANCE_RATIO",
    "MIN_INTERACTION_FEATURES",
    "MIN_MISSING_RATE",
    "MIN_SELECTION_FEATURES",
    "OPT_IN",
    "SKEW_THRESHOLD",
    "STRATEGIES",
    "TARGET_SKEW_THRESHOLD",
    "Adasyn",
    "Borderline",
    "Construction",
    "Copula",
    "Default",
    "Encoding",
    "Features",
    "Imputation",
    "Indicators",
    "Interactions",
    "IterativeImputation",
    "Jitter",
    "Logarithm",
    "Oversampling",
    "PlainSmote",
    "Rarity",
    "Rebalancing",
    "Recalibration",
    "Resampling",
    "Selection",
    "Smote",
    "Strategy",
    "Transformation",
    "Undersampling",
    "Weighting",
    "lookup",
]
