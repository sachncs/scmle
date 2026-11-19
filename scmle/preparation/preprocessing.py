"""Candidate pipelines: preprocessing and estimator as one unit.

Everything is a single scikit-learn `Pipeline`, so cross-validation clones
and refits the whole thing per fold (fold-local fitting) and inference reuses
the identical fitted graph.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import cast

import numpy
import pandas
from sklearn import base
from sklearn import compose
from sklearn import pipeline
from sklearn import preprocessing

from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import models
from scmle.contracts import tasks
from scmle.data import feature_graph
from scmle.data.validation import columns
from scmle.preparation import forecasters
from scmle.preparation import selection
from scmle.preparation import weighting
from scmle.preparation import wrappers
import scmle.preparation.multitarget
import scmle.preparation.ranking
import scmle.preparation.survival
import scmle.preparation.uplift

SPARSE_THRESHOLD = 0.3
# HistGradientBoosting bins a native categorical into at most `max_bins`
# (default 255) bins, one per level, and refuses a column with more levels.
# Integer codes are therefore capped at 255 levels: the 254 most frequent
# keep their own code and the rest share one infrequent code, which the
# fitted encoder records in `infrequent_categories_`.
NATIVE_LEVELS = 255


class Overflow(base.TransformerMixin, base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information
    """Target-encodes the columns with more levels than native codes hold.

    Integer codes pool every level past `NATIVE_LEVELS` into one code, so a
    column with more levels would silently lose the target signal of its
    tail. Each such column also gets scikit-learn's `TargetEncoder`, chosen
    over the alternatives because:

    * ordinal codes beyond `NATIVE_LEVELS` cannot be native categories, and
      as plain numbers they impose an order the levels do not have;
    * frequency encoding merges every pair of levels of equal count and
      carries no information about the target;
    * target encoding keeps one number per level, its empirical-Bayes
      smoothing (`smooth="auto"`) shrinks a rare level toward the prior in
      proportion to its own variance, and `fit_transform` cross-fits
      (`cv` folds), so no training row's encoding has seen its own label.

    It is leakage-safe inside cross-validation: the whole pipeline is
    refitted per fold, so validation rows are encoded only with statistics
    of the training fold, and the training rows with cross-fitted ones. The
    encoded columns are placed after every other block, so the positions of
    the native codes and of the numeric columns do not move. A column
    within `NATIVE_LEVELS` adds nothing.

    Attributes:
        levels: The native code limit; a column with more levels is encoded.
        seed: Seed of the cross-fitting folds.
        folds: Cross-fitting folds.
    """

    def __init__(
        self, levels: int = NATIVE_LEVELS, seed: int = 0, folds: int = 5
    ) -> None:
        """Initializes the encoder.

        Args:
            levels: The native code limit.
            seed: Seed of the cross-fitting folds.
            folds: Cross-fitting folds.
        """
        self.levels = levels
        self.seed = seed
        self.folds = folds

    def fit(self, features: Any, target: Any) -> "Overflow":
        """Fits the encoder (through the cross-fitted `fit_transform`).

        Args:
            features: Categorical block, missing values already filled.
            target: Labels or values.

        Returns:
            The fitted encoder.
        """
        self.fit_transform(features, target)
        return self

    def fit_transform(
        self,
        features: Any,
        target: Any = None,
        **params: Any,  # noqa: ARG002 -- scikit-learn's fit_transform signature
    ) -> numpy.ndarray:
        """Finds the wide columns and cross-fits their encoding.

        Args:
            features: Categorical block, missing values already filled.
            target: Labels or values.
            **params: Unused fit parameters.

        Returns:
            The cross-fitted encoding of the wide columns, one column per
            wide column (one per class and column for more than two
            classes); no column when none is wide.
        """
        block = numpy.asarray(features, dtype=object)
        self.n_features_in_ = block.shape[1]
        self.wide_ = [
            index
            for index in range(block.shape[1])
            if len(pandas.unique(block[:, index])) > self.levels
        ]
        if not self.wide_:
            self.encoder_ = None
            return numpy.zeros((len(block), 0))
        self.encoder_ = preprocessing.TargetEncoder(
            smooth="auto",
            cv=self.folds,
            shuffle=True,
            random_state=self.seed,  # pyright: ignore[reportArgumentType]  # scikit-learn infers this parameter as str from its default
        )
        encoded: numpy.ndarray = self.encoder_.fit_transform(
            block[:, self.wide_], target
        )
        return encoded

    def transform(self, features: Any) -> numpy.ndarray:
        """Encodes the wide columns with the statistics of the fit.

        Args:
            features: Categorical block, missing values already filled.

        Returns:
            The encoding, or an empty block when no column is wide.
        """
        block = numpy.asarray(features, dtype=object)
        if self.encoder_ is None:
            return numpy.zeros((len(block), 0))
        encoded: numpy.ndarray = self.encoder_.transform(block[:, self.wide_])
        return encoded

    def get_feature_names_out(
        self, input_features: Any = None
    ) -> numpy.ndarray:
        """Names the encoded columns `<column>__target`.

        Args:
            input_features: Names of the input columns.

        Returns:
            The output names; `_<class>` precedes the suffix for more than
            two classes.
        """
        names = (
            [f"x{i}" for i in range(self.n_features_in_)]
            if input_features is None
            else [str(n) for n in input_features]
        )
        if self.encoder_ is None:
            return numpy.asarray([], dtype=object)
        inner = self.encoder_.get_feature_names_out(
            [names[i] for i in self.wide_]
        )
        return numpy.asarray([f"{n}__target" for n in inner], dtype=object)


def build_preprocessor(
    plan: data.Blueprint,
    options: data.Preprocessing,
) -> compose.ColumnTransformer:
    """Builds the column transformer for a feature plan.

    Args:
        plan: Feature columns grouped by kind.
        options: Preprocessing switches.

    Returns:
        An unfitted transformer that drops every non-feature column.
    """
    parts = []
    special = isolate(plan, options)
    if special:
        encoder = encode(options.high_cardinality)
        parts.append(("high_cardinality", encoder, list(special)))
    for kind in columns.KINDS:
        selected = [c for c in plan.pick(kind.name) if c not in special]
        if not selected:
            continue
        kind_options = options
        if kind.name == "numeric" and options.missing == "indicator":
            flagged, selected = partition(plan, selected)
            if flagged:
                parts.append(
                    ("numeric_flagged", kind.transformer(options), flagged)
                )
            kind_options = dataclasses.replace(options, missing="median")
        transformer = kind.transformer(kind_options)
        if selected and transformer is not None:
            parts.append((kind.name, transformer, selected))
    if options.construct:
        parts.append(
            (
                "constructed",
                columns.Quantitative().transformer(options),
                compose.make_column_selector(pattern=f"^{selection.PREFIX}"),
            )
        )
    if special and options.high_cardinality == "ordinal":
        parts.append(
            (
                "overflow",
                columns.Categorical().encode(Overflow()),
                list(special),
            )
        )
    return compose.ColumnTransformer(
        parts,
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=SPARSE_THRESHOLD if options.sparse else 0.0,
    )


def encode(encoding: str) -> pipeline.Pipeline | None:
    """Encoder for the high-cardinality columns.

    Integer codes for native categoricals are capped at `NATIVE_LEVELS`, so
    a column with more levels than the estimator accepts pools its rarest
    levels instead of failing the fit.

    Args:
        encoding: `ordinal` or `target`; anything else means one-hot.

    Returns:
        An unfitted encoder pipeline, or None for one-hot.
    """
    if encoding != "ordinal":
        return columns.Categorical().high_cardinality_transformer(encoding)
    return columns.Categorical().encode(
        preprocessing.OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            max_categories=NATIVE_LEVELS,
        )
    )


def partition(
    plan: data.Blueprint, selected: list[str]
) -> tuple[list[str], list[str]]:
    """Numeric columns whose gaps are informative, and the rest.

    Without a missingness report every column gets indicators.

    Args:
        plan: Feature plan; its `informative_missing` lists the flagged
            columns, or is None when no missingness report exists.
        selected: The numeric columns to split.

    Returns:
        A pair `(flagged, rest)`: the columns whose gaps are informative
        (all of them if there is no report) and the remaining columns, each
        in the order of `selected`.
    """
    if plan.informative_missing is None:
        return list(selected), []
    flagged = [c for c in selected if c in plan.informative_missing]
    return flagged, [c for c in selected if c not in flagged]


def monotonicity(plan: data.Blueprint, offset: int) -> tuple[int, ...]:
    """Monotonic signs for the leading columns of the encoded matrix.

    The encoded matrix starts with the specially encoded high-cardinality
    columns (`offset` of them, one column each), then the numeric features
    in plan order; later columns are unconstrained. The function only builds
    the sign vector from that assumed layout and does not inspect the fitted
    matrix, so it does not follow a reordering of the numeric columns or the
    removal of columns after encoding.

    Args:
        plan: Feature plan with the numeric columns and their monotonic
            signs.
        offset: Number of specially encoded high-cardinality columns.

    Returns:
        `offset` zeros followed by the sign of each numeric column (+1, -1
        or 0 where none is declared).
    """
    return (0,) * offset + tuple(
        plan.monotonic.get(column, 0) for column in plan.numeric
    )


def misalignment(
    plan: data.Blueprint, select: bool, missing: str
) -> str | None:
    """Why the monotonic signs cannot follow the encoded numeric columns.

    Args:
        plan: Feature plan with the numeric columns and their monotonic
            signs.
        select: Whether the sparse column selector follows the encoder.
        missing: The numeric missing-value strategy.

    Returns:
        A reason when the encoded matrix would not hold the numeric columns
        in plan order, or None when the sign vector lines up.
    """
    if not plan.monotonic:
        return None
    if select:
        return "feature selection drops columns the signs are indexed by"
    if missing == "indicator":
        flagged, rest = partition(plan, list(plan.numeric))
        if flagged and rest:
            return (
                "missing indicators move the flagged numeric columns ahead "
                "of the others"
            )
    return None


def isolate(
    plan: data.Blueprint,
    options: data.Preprocessing,
) -> tuple[str, ...]:
    """Columns that get a special encoder instead of one-hot encoding.

    They are placed first in the transformed matrix, so their positions are
    `0..len-1`, which native-categorical estimators can be told about.

    Args:
        plan: Feature plan with the categorical and high-cardinality columns.
        options: Preprocessing switches; with `onehot` encoding no column
            is special.

    Returns:
        The high-cardinality categorical columns, in plan order; empty with
        `onehot` encoding.
    """
    if options.high_cardinality == "onehot":
        return ()
    return tuple(c for c in plan.categorical if c in plan.high_cardinality)


def frontend(
    plan: data.Blueprint,
    options: data.Preprocessing,
) -> Any:
    """Builds the preprocessing front end, with the feature graph if any.

    Args:
        plan: Feature columns by kind (and row-wise graph nodes).
        options: Preprocessing switches.

    Returns:
        The column transformer, or a pipeline of graph transformer followed
        by the column transformer when features are derived.
    """
    preprocessor = build_preprocessor(plan, options)
    if not plan.derived:
        return preprocessor
    return pipeline.Pipeline(
        [
            ("graph", feature_graph.Transformer(plan.derived)),
            ("prep", preprocessor),
        ]
    )


def default_options(
    plan: data.Blueprint, adapter: models.Model
) -> data.Preprocessing:
    """Preprocessing switches implied by an adapter's capabilities.

    Scaling for models that benefit from it, no imputation for models that
    handle missing values, sparse one-hot output for sparse-capable models on
    high-cardinality data, and integer-coded native categoricals where the
    adapter supports them.

    Args:
        plan: Feature plan, used for its high-cardinality columns and text
            settings.
        adapter: Model family whose capabilities decide the switches.

    Returns:
        The preprocessing options for the adapter.
    """
    caps = adapter.capabilities
    native = bool(
        plan.high_cardinality
        and caps.native_categorical
        and adapter.nominals([0])
    )
    return data.Preprocessing(
        scale=caps.needs_scaling,
        impute=not caps.handles_missing,
        high_cardinality="ordinal" if native else "onehot",
        sparse=caps.sparse_input and bool(plan.high_cardinality),
        text_encoding=plan.text_encoding,
        embedding_model=plan.embedding_model,
    )


def weigh(model: Any, plan: data.Blueprint, seed: int) -> Any:
    """Wraps a model so that it reads the plan's weight column.

    Args:
        model: The pipeline or estimator.
        plan: Feature columns grouped by kind.
        seed: Random seed forwarded to the wrapper.

    Returns:
        The weighted model, or `model` itself when the plan has no weight
        column.
    """
    if plan.weight_column:
        return weighting.weighted(model, plan.weight_column, seed)
    return model


def preprocess(
    plan: data.Blueprint,
    options: data.Preprocessing,
    task: tasks.Task,
    seed: int,
) -> list[tuple[str, Any]]:
    """Lists the steps that come before the estimator.

    Args:
        plan: Feature columns grouped by kind.
        options: Preprocessing switches.
        task: Resolved task type.
        seed: Random seed forwarded to the selector.

    Returns:
        The graph, construction, preprocessing and selection steps that the
        plan and the options call for, in that order.
    """
    steps: list[tuple[str, Any]] = []
    if plan.derived:
        steps.append(("graph", feature_graph.Transformer(plan.derived)))
    if options.construct:
        steps.append(("construct", selection.Constructor(tuple(plan.numeric))))
    steps.append(("prep", build_preprocessor(plan, options)))
    if options.select:
        steps.append(
            (
                "select",
                selection.SparseSelector(
                    classification=task.is_classification, random_state=seed
                ),
            )
        )
    return steps


def wrap(built: Any, plan: data.Blueprint, task: tasks.Task) -> Any:
    """Wraps a pipeline in the estimators the plan's columns call for.

    Args:
        built: The pipeline.
        plan: Feature columns grouped by kind.
        task: Resolved task type.

    Returns:
        The pipeline wrapped for survival events, forecasting or several
        targets, then for treatment and query columns, when the plan has
        them.
    """
    if plan.event_column:
        built = scmle.preparation.survival.EventAwareEstimator(
            built, plan.event_column
        )
    if task is tasks.Task.FORECAST:
        built = forecasters.Forecaster(
            built, plan.target_columns, plan.value_column or ""
        )
    elif plan.target_columns:
        kind = (
            scmle.preparation.multitarget.MultiLabelEstimator
            if task is tasks.Task.MULTILABEL
            else scmle.preparation.multitarget.Multioutput
        )
        built = kind(built, plan.target_columns)
    if plan.treatment_column:
        built = scmle.preparation.uplift.TreatmentAwareEstimator(
            built, plan.treatment_column
        )
    if plan.query_column:
        built = scmle.preparation.ranking.Ranker(built, plan.query_column)
    return built


def instantiate(
    plan: data.Blueprint,
    adapter: models.Model,
    task: tasks.Task,
    seed: int,
    params: Mapping[str, Any] | None = None,
    options: data.Preprocessing | None = None,
) -> Any:
    """Builds a complete, independently evaluable candidate pipeline.

    Args:
        plan: Feature columns grouped by kind.
        adapter: Model family to place after preprocessing.
        task: Resolved task type.
        seed: Random seed forwarded to the estimator.
        params: Hyperparameter overrides for the estimator.
        options: Preprocessing switches; by default `default_options`.

    Returns:
        An unfitted pipeline of preprocessing followed by the estimator.

    Raises:
        ConfigError: If the plan's monotonic constraints cannot apply to the
            layout the preprocessing produces.
    """
    if plan.ordinal_levels:
        params = {**(params or {}), "levels": plan.ordinal_levels}
    if plan.treatment_column:
        params = {**(params or {}), "classify": plan.binary_outcome}
    # The family may assemble its own pipeline; the dummy family on a
    # forecast task is a last-value forecaster. None leaves it to this module.
    assembled: Any
    if task is tasks.Task.FORECAST and adapter.name == "dummy":
        assembled = forecasters.Naive(
            None, plan.target_columns, plan.value_column or ""
        )
    else:
        assembled = adapter.assemble(plan, task, seed, params)
    if assembled is not None:
        return weigh(assembled, plan, seed)
    params = dict(params or {})
    if options is None:
        options = default_options(plan, adapter)
    special = isolate(plan, options)
    if options.high_cardinality == "ordinal" and special:
        params |= adapter.nominals(list(range(len(special))))
    steps = preprocess(plan, options, task, seed)
    estimator = adapter.build(task, seed, params)
    problem = misalignment(plan, options.select, options.missing)
    if problem:
        raise errors.ConfigError(
            f"monotonic constraints cannot apply: {problem}"
        )
    if plan.monotonic:
        kind = (
            wrappers.MonotoneClassifier
            if base.is_classifier(estimator)
            else wrappers.MonotoneRegressor
        )
        estimator = kind(
            cast("base.BaseEstimator", estimator),
            adapter.monotonic_param or "",
            monotonicity(plan, len(special)),
        )
    steps.append(("model", estimator))
    built = wrap(pipeline.Pipeline(steps), plan, task)
    return weigh(built, plan, seed)
