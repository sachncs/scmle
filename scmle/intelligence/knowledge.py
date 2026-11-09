"""Meta-learning: dataset fingerprints and a nearest-neighbour model prior.

Historical runs influence only the *ordering* of the search; every candidate is
still evaluated on the new data, and runs on the same dataset are excluded so
past results never leak into their own evaluation.
"""

import math

import numpy

from scmle.contracts import data
from scmle.contracts import experiments
from scmle.contracts import tasks

FIELDS = (
    "log_rows",
    "log_features",
    "categorical_share",
    "missing_rate",
    "log_imbalance",
    "mean_abs_skew",
)
NEIGHBORS = 5


def fingerprint(
    profile: data.DatasetProfile, task: tasks.Task
) -> dict[str, float]:
    """Summarizes a dataset as a small numeric vector.

    Args:
        profile: Dataset profile.
        task: Resolved task type.

    Returns:
        A mapping with one float per entry of `FIELDS`.
    """
    plan = profile.feature_plan
    stats = profile.column_statistics
    features = tuple(c for c in plan.features if c in stats)
    numeric_skews = [
        abs(stats[c].get("skew", 0.0)) for c in plan.numeric if c in stats
    ]
    imbalance = profile.target_summary.get("imbalance_ratio", 1.0)
    missing_rate = (
        float(numpy.mean([stats[c]["missing_rate"] for c in features]))
        if features
        else 0.0
    )
    return {
        "log_rows": math.log10(max(profile.manifest.row_count, 1)),
        "log_features": math.log10(max(len(features), 1)),
        "categorical_share": len(plan.categorical) / max(len(features), 1),
        "missing_rate": missing_rate,
        "log_imbalance": math.log10(max(imbalance, 1.0)),
        "mean_abs_skew": float(numpy.mean(numeric_skews))
        if numeric_skews
        else 0.0,
    }


class MetaLearner:
    """Recommends models by similarity to historical datasets.

    Attributes:
        history: Past runs.
        neighbors: How many similar datasets vote.
    """

    def __init__(
        self,
        history: list[experiments.HistoricalRun],
        neighbors: int = NEIGHBORS,
    ) -> None:
        """Initializes the learner.

        Args:
            history: Past runs.
            neighbors: How many similar datasets vote.
        """
        self.history = history
        self.neighbors = neighbors

    def eligible(
        self, task: tasks.Task, dataset_id: str, data_hash: str
    ) -> list[experiments.HistoricalRun]:
        """Selects past runs on a different dataset with the same task.

        Args:
            task: Resolved task type.
            dataset_id: Identity of the new dataset, which is excluded.
            data_hash: Content hash of the new dataset, which is excluded.

        Returns:
            The past runs with the same task, a different dataset identity and
            content hash, and a score that is not NaN.
        """
        return [
            run
            for run in self.history
            if run.task == task.value
            and run.dataset_id != dataset_id
            and run.data_hash != data_hash
            and not math.isnan(run.cv_score)
        ]

    def recommend(
        self,
        current: dict[str, float],
        task: tasks.Task,
        dataset_id: str,
        data_hash: str,
    ) -> list[tuple[str, float]]:
        """Scores models that won on the most similar past datasets.

        Args:
            current: Fingerprint of the new dataset.
            task: Resolved task type.
            dataset_id: Identity of the new dataset (excluded from history).
            data_hash: Content hash of the new dataset (excluded as well).

        Returns:
            `(model, weight)` pairs, best first; empty without history.
        """
        pool = self.eligible(task, dataset_id, data_hash)
        if not pool:
            return []
        matrix = numpy.array([[r.fingerprint[f] for f in FIELDS] for r in pool])
        target = numpy.array([current[f] for f in FIELDS])
        scale = numpy.vstack([matrix, target]).std(axis=0)
        scale[scale == 0] = 1.0
        distances = numpy.linalg.norm((matrix - target) / scale, axis=1)
        order = numpy.argsort(distances)[: self.neighbors]
        weights: dict[str, float] = {}
        for index in order:
            model = pool[index].selected_model
            weights[model] = weights.get(model, 0.0) + 1.0 / (
                1.0 + float(distances[index])
            )
        return sorted(weights.items(), key=lambda pair: -pair[1])
