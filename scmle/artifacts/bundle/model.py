"""`Bundle`, a fitted pipeline with the schema and metadata to serve it."""

import dataclasses
import pathlib
import platform
from typing import Any
from typing import Self

import numpy
import pandas
from sklearn import calibration
from sklearn import frozen

from scmle.artifacts.bundle import archive
from scmle.artifacts.bundle import environment
from scmle.artifacts.bundle import explanation
from scmle.artifacts.bundle import intervals
from scmle.artifacts.bundle import outputs
from scmle.artifacts.bundle import provenance
from scmle.contracts import artifacts
from scmle.contracts import errors
from scmle.contracts import tasks
import scmle.contracts.data
import scmle.data.reference
from scmle.evaluation import dirichlet
from scmle.evaluation import reliability
from scmle.models import thresholds

PARITY_TOLERANCE = 1e-9
DIRICHLET_MIN_CLASSES = 3
MAX_KNOWN_CATEGORIES = 200


def brier(
    probabilities: numpy.ndarray,
    labels: numpy.ndarray,
    classes: tuple[Any, ...],
) -> float:
    """Computes the multiclass Brier score of predicted probabilities.

    Args:
        probabilities: Predicted class probabilities, one row per sample.
        labels: True labels.
        classes: Class labels in probability-column order.

    Returns:
        The mean over samples of the squared differences to the one-hot truth,
        summed over classes (0 is perfect).
    """
    truth = (labels[:, None] == numpy.asarray(classes)[None, :]).astype(float)
    return float(numpy.mean(numpy.sum((probabilities - truth) ** 2, axis=1)))


def describe(
    pipeline: Any,
    plan: scmle.contracts.data.Blueprint,
    dev_features: pandas.DataFrame,
    target: str,
    task: str,
    class_labels: tuple[Any, ...] | None,
    graph_fingerprint: str,
    group_column: str | None,
    time_column: str | None,
) -> artifacts.Interface:
    """Builds the input contract of a fitted pipeline from its training data.

    Args:
        pipeline: Fitted pipeline.
        plan: Feature columns used.
        dev_features: Training features (for dtypes and categories).
        target: Target column name.
        task: Task type value.
        class_labels: Labels of the classes in probability-column order
            when the pipeline works on codes (ordinal tasks), else None.
        graph_fingerprint: Hash of the feature graph, if one was used.
        group_column: Entity column for series features.
        time_column: Ordering column for series features.

    Returns:
        The schema. A categorical column keeps at most
        `MAX_KNOWN_CATEGORIES` of its sorted training categories.
    """
    computed = plan.engineered
    series_names = {n.name for n in plan.series}
    numeric = tuple(c for c in plan.numeric if c not in computed)
    categorical = tuple(c for c in plan.categorical if c not in computed)
    extra = tuple(
        c
        for c in plan.extra_inputs
        if c not in computed and c not in numeric + categorical
    )
    series_columns = tuple(
        c for c in (*plan.features, *plan.extra_inputs) if c in series_names
    )
    known = {
        column: sorted(map(str, dev_features[column].dropna().unique()))[
            :MAX_KNOWN_CATEGORIES
        ]
        for column in categorical
    }
    classes = getattr(pipeline, "classes_", None)
    return artifacts.Interface(
        target=target,
        task=task,
        numeric=numeric,
        categorical=categorical,
        dtypes={
            c: str(dev_features[c].dtype) for c in (*numeric, *categorical)
        },
        categories=known,
        classes=class_labels
        if class_labels is not None
        else None
        if classes is None
        else tuple(numpy.asarray(classes).tolist()),
        extras=extra,
        series=tuple(n.model_dump(mode="json") for n in plan.series),
        derived=series_columns,
        digests={"graph": graph_fingerprint} if graph_fingerprint else {},
        group=group_column,
        time=time_column,
        targets=tuple(getattr(pipeline, "target_columns", ())),
        optional=(
            (pipeline.column,)
            if isinstance(pipeline, thresholds.Segmenter)
            else ()
        ),
    )


def stamp(
    model_name: str,
    model_version: str,
    metric: str,
    model_params: dict[str, Any] | None,
    strategy: str,
    decision_threshold: float | None,
    seed: int | None,
    configuration: str,
    dataset: str,
    split: str,
) -> provenance.Manifest:
    """Builds the manifest of a new bundle, with its build environment.

    Args:
        model_name: Selected model family.
        model_version: Bundle version label.
        metric: Optimized metric name.
        model_params: Hyperparameter overrides of the selected model.
        strategy: Data strategy of the selected candidate.
        decision_threshold: Optimized binary decision threshold, if any.
        seed: Random seed of the run.
        configuration: Hash of the run configuration.
        dataset: Hash of the training data.
        split: Fingerprint of the train and test split.

    Returns:
        The manifest. It also records every installed distribution, the
        platform, the scmle commit and the math-library threads and BLAS in
        use (ML-10), read before the tracked libraries are imported.
    """
    threads, blas = environment.pools()
    return provenance.Manifest(
        model_name,
        model_version,
        metric,
        model_params=model_params or {},
        strategy=strategy,
        decision_threshold=decision_threshold,
        environment=environment.inventory(),
        platform=platform.platform(),
        seed=seed,
        configuration=configuration,
        dataset=dataset,
        split=split,
        commit=environment.revision(),
        threads=threads,
        blas=blas,
    )


class Bundle(
    archive.Archive,
    intervals.Intervals,
    explanation.Explanation,
    outputs.Outputs,
):
    """A fitted pipeline plus the schema and metadata needed to serve it.

    Attributes:
        pipeline: Fitted preprocessing and estimator pipeline.
        schema: Input contract.
        info: Provenance metadata.
        reference: Training-time distributions for monitoring, if any.
        history: Recent rows per entity for series features, if any.
        calibrator: Probability calibrator, if one was fitted.
        conformal: Conformal calibrator, if any.
        novelty: Novelty detector, if any.
        quantile: Quantile interval models, if any.
        cv_plus: CV+ fold models, if any.
        shift_weights: Covariate-shift weights, if attached.
        coverage_report: Measured coverage of the stored intervals or sets on
            the final test rows, if any.
        history_lock: Serializes updates of the history.
    """

    @classmethod
    def create(
        cls,
        pipeline: Any,
        plan: scmle.contracts.data.Blueprint,
        dev_features: pandas.DataFrame,
        target: str,
        task: str,
        metric: str,
        model_name: str,
        model_version: str = "v1",
        model_params: dict[str, Any] | None = None,
        strategy: str = "default",
        decision_threshold: float | None = None,
        reference: scmle.data.reference.Baseline | None = None,
        graph_fingerprint: str = "",
        group_column: str | None = None,
        time_column: str | None = None,
        history_source: pandas.DataFrame | None = None,
        class_labels: tuple[Any, ...] | None = None,
        seed: int | None = None,
        configuration: str = "",
        dataset: str = "",
        split: str = "",
    ) -> Self:
        """Builds a bundle from a fitted pipeline and its training data.

        Args:
            pipeline: Fitted pipeline.
            plan: Feature columns used.
            dev_features: Training features (for dtypes and categories).
            target: Target column name.
            task: Task type value.
            metric: Optimized metric name.
            model_name: Selected model family.
            model_version: Bundle version label.
            model_params: Hyperparameter overrides of the selected model.
            strategy: Data strategy of the selected candidate.
            decision_threshold: Optimized binary decision threshold, if any.
            reference: Training-time distributions for monitoring.
            graph_fingerprint: Hash of the feature graph, if one was used.
            group_column: Entity column for series features.
            time_column: Ordering column for series features.
            history_source: All rows known at training time; the recent ones
                per entity are stored as the series history, if any.
            class_labels: Labels per probability column for an ordinal model.
            seed: Random seed of the run.
            configuration: Hash of the run configuration.
            dataset: Hash of the training data.
            split: Fingerprint of the train and test split.

        Returns:
            The bundle, with the manifest `stamp` writes.
        """
        schema = describe(
            pipeline,
            plan,
            dev_features,
            target,
            task,
            class_labels,
            graph_fingerprint,
            group_column,
            time_column,
        )
        info = stamp(
            model_name,
            model_version,
            metric,
            model_params,
            strategy,
            decision_threshold,
            seed,
            configuration,
            dataset,
            split,
        )
        bundle = cls(pipeline, schema, info, reference)
        if schema.series and history_source is not None:
            bundle.history = bundle.assemble().retain(history_source)
        return bundle

    def calibrate(
        self,
        data: str | pathlib.Path | pandas.DataFrame,
        target: Any,
        method: str = "sigmoid",
    ) -> dict[str, float]:
        """Fits an optional probability calibrator on labeled rows.

        The trained pipeline is frozen; only a mapping from its probabilities
        to calibrated ones is learned. Use rows the model was not trained on.
        `predict_proba` then returns calibrated probabilities; `predict` is
        unchanged.

        Args:
            data: Labeled rows.
            target: Their true labels.
            method: `sigmoid` (Platt), `isotonic`, or `dirichlet` (a full
                map over all class probabilities; multiclass only).

        Returns:
            A dict with `brier_before`, `brier_after`, `ece_before` and
            `ece_after` (top-label expected calibration error), computed on
            the calibration rows (a diagnostic; the calibrator is fitted on
            the same rows).

        Raises:
            ArtifactError: For regression and ordinal models.
            ConfigError: For an unknown method, or `dirichlet` with fewer than
                three classes.
        """
        if (
            self.schema.classes is None
            or self.schema.task == tasks.Task.ORDINAL.value
        ):
            raise errors.ArtifactError("only classifiers can be calibrated")
        if method not in ("sigmoid", "isotonic", "dirichlet"):
            raise errors.ConfigError(
                f"unknown calibration method '{method}'; use sigmoid, "
                "isotonic or dirichlet"
            )
        if (
            method == "dirichlet"
            and len(self.schema.classes) < DIRICHLET_MIN_CLASSES
        ):
            raise errors.ConfigError(
                "dirichlet calibration needs three or more classes"
            )
        frame = self.prepare(data)
        labels = numpy.asarray(target)
        before = self.pipeline.predict_proba(frame)
        if method == "dirichlet":
            calibrator: Any = dirichlet.Dirichlet(self.pipeline).fit(
                frame, labels
            )
        else:
            calibrator = calibration.CalibratedClassifierCV(
                frozen.FrozenEstimator(self.pipeline), method=method
            ).fit(frame, labels)
        after = calibrator.predict_proba(frame)
        self.calibrator = calibrator
        self.info = dataclasses.replace(
            self.info, calibration=f"{method} on {len(frame)} rows"
        )
        classes = self.schema.classes
        return {
            "brier_before": brier(before, labels, classes),
            "brier_after": brier(after, labels, classes),
            "ece_before": reliability.topclass(before, labels, classes)[0],
            "ece_after": reliability.topclass(after, labels, classes)[0],
        }

    def roundtrip(
        self,
        directory: str | pathlib.Path,
        sample: pandas.DataFrame,
        tolerance: float = PARITY_TOLERANCE,
    ) -> dict[str, Any]:
        """Reloads the saved bundle and checks it predicts like this one.

        Args:
            directory: Where the bundle was saved.
            sample: Raw rows to score with both copies.
            tolerance: Largest allowed absolute difference of probabilities
                and of numeric predictions. Non-numeric labels are compared
                by the count of differing rows, so any difference fails
                unless the tolerance is at least 1.

        Returns:
            A dict with `rows`, `tolerance` and `largest_difference`.

        Raises:
            ArtifactError: If the reloaded bundle disagrees with this one. The
                reload uses `trusted=True`, so it is refused when
                `$SCMLE_ENV` is `production`.
        """
        reloaded = Bundle.load(directory, trusted=True)
        before = numpy.asarray(self.predict(sample))
        after = numpy.asarray(reloaded.predict(sample))
        if before.dtype.kind in "fiu":
            worst = float(numpy.max(numpy.abs(before - after), initial=0.0))
        else:
            worst = float(numpy.sum(before != after))
        if self.schema.classes is not None:
            worst = max(
                worst,
                float(
                    numpy.max(
                        numpy.abs(
                            self.predict_proba(sample)
                            - reloaded.predict_proba(sample)
                        ),
                        initial=0.0,
                    )
                ),
            )
        if worst > tolerance:
            raise errors.ArtifactError(
                "the saved bundle does not reproduce this model's "
                f"predictions (largest difference {worst:.3g} > {tolerance:g})"
            )
        return {
            "rows": len(sample),
            "tolerance": tolerance,
            "largest_difference": worst,
        }
