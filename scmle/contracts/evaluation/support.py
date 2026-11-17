"""What each task supports: metrics, their directions and policy choices.

The tables are the explicit boundary of the evaluation policy contract: a
policy is not inferred from the presence of an estimator or a diagnostic
class, so every supported combination is declared here.
"""

from collections.abc import Mapping
import enum

from scmle.contracts import errors
from scmle.contracts import tasks


class Orientation(enum.StrEnum):
    """Direction in which a metric improves."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class UncertaintyMethod(enum.StrEnum):
    """Uncertainty contract promised by an evaluation policy."""

    NONE = "none"
    CALIBRATED_PROBABILITY = "calibrated_probability"
    PREDICTION_INTERVAL = "prediction_interval"
    CONFORMAL = "conformal"
    SURVIVAL_CURVE = "survival_curve"
    ABSTENTION = "abstention"


class Procedure(enum.StrEnum):
    """Calibration procedure or diagnostic promised by a policy."""

    NONE = "none"
    RELIABILITY_CURVE = "reliability_curve"
    PLATT = "platt"
    ISOTONIC = "isotonic"
    DIRICHLET = "dirichlet"


class Technique(enum.StrEnum):
    """Fairness analysis or mitigation promised by a policy."""

    NONE = "none"
    GROUP_REPORT = "group_report"
    INTERSECTIONAL_REPORT = "intersectional_report"
    MITIGATION = "mitigation"


class Explicability(enum.StrEnum):
    """Level of model explanation the evaluation must make available."""

    NONE = "none"
    GLOBAL = "global"
    LOCAL = "local"
    COUNTERFACTUAL = "counterfactual"
    FULL = "full"


class Remedy(enum.StrEnum):
    """Action taken when a required diagnostic does not pass."""

    WARN = "warn"
    BLOCK = "block"


class UnsupportedError(errors.ConfigError):
    """Raised when a task and evaluation policy combination is unsupported."""

    category = "evaluation_policy"


class UndeclaredError(UnsupportedError):
    """Raised when no policy was declared for a requested task."""


SUPPORTED_METRICS: Mapping[tasks.Task, frozenset[str]] = {
    tasks.Task.BINARY: frozenset(
        {
            "roc_auc",
            "pr_auc",
            "log_loss",
            "f1",
            "f1_macro",
            "balanced_accuracy",
            "accuracy",
        }
    ),
    tasks.Task.MULTICLASS: frozenset(
        {"roc_auc_ovr", "log_loss", "f1_macro", "balanced_accuracy", "accuracy"}
    ),
    tasks.Task.REGRESSION: frozenset({"mae", "rmse", "r2"}),
    tasks.Task.SURVIVAL: frozenset({"c_index", "ibs", "td_auc"}),
    tasks.Task.ORDINAL: frozenset({"qwk", "mace", "accuracy_levels"}),
    tasks.Task.RANKING: frozenset(
        {"ndcg", "map", "mrr", "precision_at_10", "recall_at_10"}
    ),
    tasks.Task.MULTILABEL: frozenset(
        {"f1_macro_labels", "f1_micro_labels", "hamming", "subset_accuracy"}
    ),
    tasks.Task.MULTIOUTPUT: frozenset({"r2_mean", "mae_scaled"}),
    tasks.Task.FORECAST: frozenset({"mase", "forecast_mae", "forecast_smape"}),
    tasks.Task.UPLIFT: frozenset({"qini", "uplift_at_30"}),
}
"""Built-in metric names that the current evaluation catalog can score."""


METRIC_DIRECTIONS: Mapping[str, Orientation] = {
    "roc_auc": Orientation.MAXIMIZE,
    "roc_auc_ovr": Orientation.MAXIMIZE,
    "pr_auc": Orientation.MAXIMIZE,
    "log_loss": Orientation.MINIMIZE,
    "f1": Orientation.MAXIMIZE,
    "f1_macro": Orientation.MAXIMIZE,
    "balanced_accuracy": Orientation.MAXIMIZE,
    "accuracy": Orientation.MAXIMIZE,
    "mae": Orientation.MINIMIZE,
    "rmse": Orientation.MINIMIZE,
    "r2": Orientation.MAXIMIZE,
    "qwk": Orientation.MAXIMIZE,
    "mace": Orientation.MINIMIZE,
    "accuracy_levels": Orientation.MAXIMIZE,
    "f1_macro_labels": Orientation.MAXIMIZE,
    "f1_micro_labels": Orientation.MAXIMIZE,
    "hamming": Orientation.MINIMIZE,
    "subset_accuracy": Orientation.MAXIMIZE,
    "r2_mean": Orientation.MAXIMIZE,
    "mae_scaled": Orientation.MINIMIZE,
    "mase": Orientation.MINIMIZE,
    "forecast_mae": Orientation.MINIMIZE,
    "forecast_smape": Orientation.MINIMIZE,
    "qini": Orientation.MAXIMIZE,
    "uplift_at_30": Orientation.MAXIMIZE,
    "ndcg": Orientation.MAXIMIZE,
    "map": Orientation.MAXIMIZE,
    "mrr": Orientation.MAXIMIZE,
    "precision_at_10": Orientation.MAXIMIZE,
    "recall_at_10": Orientation.MAXIMIZE,
    "c_index": Orientation.MAXIMIZE,
    "ibs": Orientation.MINIMIZE,
    "td_auc": Orientation.MAXIMIZE,
}
"""Direction for every built-in metric declared in ``SUPPORTED_METRICS``."""


DEFAULT_METRICS: Mapping[tasks.Task, str] = {
    tasks.Task.BINARY: "roc_auc",
    tasks.Task.MULTICLASS: "log_loss",
    tasks.Task.REGRESSION: "rmse",
    tasks.Task.SURVIVAL: "c_index",
    tasks.Task.ORDINAL: "qwk",
    tasks.Task.RANKING: "ndcg",
    tasks.Task.MULTILABEL: "f1_macro_labels",
    tasks.Task.MULTIOUTPUT: "r2_mean",
    tasks.Task.FORECAST: "mase",
    tasks.Task.UPLIFT: "qini",
}


SUPPORTED_UNCERTAINTY: Mapping[tasks.Task, frozenset[UncertaintyMethod]] = {
    **{task: frozenset({UncertaintyMethod.NONE}) for task in tasks.Task},
    tasks.Task.BINARY: frozenset(
        {
            UncertaintyMethod.NONE,
            UncertaintyMethod.CALIBRATED_PROBABILITY,
            UncertaintyMethod.CONFORMAL,
            UncertaintyMethod.ABSTENTION,
        }
    ),
    tasks.Task.MULTICLASS: frozenset(
        {
            UncertaintyMethod.NONE,
            UncertaintyMethod.CALIBRATED_PROBABILITY,
            UncertaintyMethod.CONFORMAL,
            UncertaintyMethod.ABSTENTION,
        }
    ),
    tasks.Task.REGRESSION: frozenset(
        {
            UncertaintyMethod.NONE,
            UncertaintyMethod.PREDICTION_INTERVAL,
            UncertaintyMethod.CONFORMAL,
            UncertaintyMethod.ABSTENTION,
        }
    ),
    tasks.Task.SURVIVAL: frozenset(
        {UncertaintyMethod.NONE, UncertaintyMethod.SURVIVAL_CURVE}
    ),
}


SUPPORTED_CALIBRATION: Mapping[tasks.Task, frozenset[Procedure]] = {
    **{task: frozenset({Procedure.NONE}) for task in tasks.Task},
    tasks.Task.BINARY: frozenset(
        {
            Procedure.NONE,
            Procedure.RELIABILITY_CURVE,
            Procedure.PLATT,
            Procedure.ISOTONIC,
        }
    ),
    tasks.Task.MULTICLASS: frozenset(
        {
            Procedure.NONE,
            Procedure.RELIABILITY_CURVE,
            Procedure.DIRICHLET,
        }
    ),
}


SUPPORTED_FAIRNESS: Mapping[tasks.Task, frozenset[Technique]] = {
    **{task: frozenset({Technique.NONE}) for task in tasks.Task},
    tasks.Task.BINARY: frozenset(
        {
            Technique.NONE,
            Technique.GROUP_REPORT,
            Technique.MITIGATION,
        }
    ),
    tasks.Task.REGRESSION: frozenset(
        {
            Technique.NONE,
            Technique.GROUP_REPORT,
            Technique.MITIGATION,
        }
    ),
}


SUPPORTED_EXPLANATIONS: Mapping[tasks.Task, frozenset[Explicability]] = {
    **{
        task: frozenset({Explicability.NONE, Explicability.GLOBAL})
        for task in tasks.Task
    },
    tasks.Task.BINARY: frozenset(
        {
            Explicability.NONE,
            Explicability.GLOBAL,
            Explicability.LOCAL,
            Explicability.COUNTERFACTUAL,
            Explicability.FULL,
        }
    ),
    tasks.Task.REGRESSION: frozenset(
        {
            Explicability.NONE,
            Explicability.GLOBAL,
            Explicability.LOCAL,
            Explicability.COUNTERFACTUAL,
            Explicability.FULL,
        }
    ),
}


COMMON_DIAGNOSTICS = frozenset(
    {
        "metric_uncertainty",
        "slice_performance",
        "robustness",
        "adversarial_validation",
    }
)
SUPPORTED_DIAGNOSTICS: Mapping[tasks.Task, frozenset[str]] = {
    **dict.fromkeys(tasks.Task, COMMON_DIAGNOSTICS),
    tasks.Task.BINARY: COMMON_DIAGNOSTICS
    | frozenset(
        {
            "classification_report",
            "calibration",
            "threshold_analysis",
            "fairness",
        }
    ),
    tasks.Task.MULTICLASS: COMMON_DIAGNOSTICS
    | frozenset({"classification_report", "calibration"}),
    tasks.Task.REGRESSION: COMMON_DIAGNOSTICS
    | frozenset({"residuals", "fairness"}),
}


FAIRNESS_METRICS: Mapping[tasks.Task, frozenset[str]] = {
    tasks.Task.BINARY: frozenset(
        {
            "demographic_parity_difference",
            "equal_opportunity_difference",
            "equalized_odds_difference",
            "disparate_impact_ratio",
        }
    ),
    tasks.Task.REGRESSION: frozenset({"bias_gap"}),
}
