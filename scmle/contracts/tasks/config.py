"""The project configuration and the rules that span its sections.

A configuration is stored with single-word keys (see `sections`) and read in
either spelling. Its hash is taken on the old, version-1 layout, so renaming
a key never changes the hash a run is known by.
"""

import hashlib
import json
from typing import Any
from typing import ClassVar
from typing import Literal
from typing import Self

import pydantic

from scmle.contracts import errors
from scmle.contracts import records
from scmle.contracts.tasks import graph
from scmle.contracts.tasks import kinds
from scmle.contracts.tasks import sections

CONFIG_VERSION = 1
FAST_BUDGET_SECONDS = 300
FAST_SEARCH_MODELS = 1
FAST_SEARCH_CONFIGS = 3
MIN_ORDINAL_LEVELS = 3
MIN_TARGETS = 2

FAIRNESS_WEIGHT = "fairness_weight"
"""Weight column the reweighing mitigation adds to development rows."""


class Config(sections.Section):
    """The explicit, versioned specification of an ML project."""

    vocabulary: ClassVar[dict[str, str]] = {
        "feature_graph": "derivation",
        "leakage_policy": "leakage",
        "charter_path": "charter",
        "ledger_rule": "ledger",
        "confirmation_rule": "confirmation",
        "confirmation_failure_policy": "failure",
        "leakage_attestations": "attestations",
    }
    spelling: ClassVar[records.Migration] = sections.spell(vocabulary)

    version: int = CONFIG_VERSION
    data: sections.Dataset
    task: kinds.TaskRequest = "auto"
    metric: str | None = None
    validation: sections.Validation = pydantic.Field(
        default_factory=sections.Validation
    )
    resources: sections.Resources = pydantic.Field(
        default_factory=sections.Resources
    )
    search: sections.Search = pydantic.Field(default_factory=sections.Search)
    strategies: sections.Strategies = pydantic.Field(
        default_factory=sections.Strategies
    )
    models: sections.Candidates = pydantic.Field(
        default_factory=sections.Candidates
    )
    feature_graph: graph.Derivation = pydantic.Field(
        default_factory=graph.Derivation
    )
    ensemble: sections.Stacking = pydantic.Field(
        default_factory=sections.Stacking
    )
    selection: sections.Selection = pydantic.Field(
        default_factory=sections.Selection
    )
    decision: sections.Threshold = pydantic.Field(
        default_factory=sections.Threshold
    )
    fairness: sections.Parity = pydantic.Field(default_factory=sections.Parity)
    problem: sections.Problem = pydantic.Field(default_factory=sections.Problem)
    leakage_policy: Literal["warn", "block"] = "warn"
    # Path of an analysis charter. When set, the run splits the development
    # rows into exploration and sealed confirmation sets.
    charter_path: str | None = None
    # Choices for the claim ledger of a charter run. `ledger_rule`
    # is the online FDR rule for exploration: e-LOND is valid under arbitrary
    # dependence; LORD++ and SAFFRON assume independent p-values.
    # `confirmation_rule` adjusts the survivors tested once on `C`.
    ledger_rule: Literal["e-lond", "lord++", "saffron"] = "e-lond"
    confirmation_rule: Literal["bh", "by", "holm", "bonferroni"] = "by"
    confirmation_failure_policy: Literal["unconfirmed"] = "unconfirmed"
    # JSON file of per-column attestations that a strong predictor flagged by
    # the leakage screen is legitimate (charter runs only; see
    # `scmle.assurance.admissibility.attestations`).
    leakage_attestations: str | None = None

    @pydantic.model_validator(mode="before")
    @classmethod
    def lag(cls, values: Any) -> Any:
        """Turns `data.feature_delays` into lagged nodes and exclusions.

        The expansion is idempotent, so a saved configuration (which already
        contains the nodes) loads unchanged.

        Args:
            values: The raw specification.

        Returns:
            The specification with a lagged node for each delayed column and
            those columns excluded; unchanged if it is not a dict or has no
            delays.
        """
        if not isinstance(values, dict):
            return values
        values = named(cls, values)
        data_record = named(sections.Dataset, values.get("data"))
        delays = data_record.get("feature_delays") or {}
        if not delays:
            return values
        declared = values.get("feature_graph")
        graph_record = (
            declared.model_dump()
            if isinstance(declared, graph.Derivation)
            else dict(declared or {})
        )
        nodes = list(graph_record.get("nodes", []))
        present = {node["name"] for node in nodes}
        for column, delay in delays.items():
            name = f"{column}{graph.DELAYED_SUFFIX}"
            if name not in present:
                nodes.append(
                    {
                        "name": name,
                        "op": "lag",
                        "inputs": [column],
                        "params": {"k": delay},
                    }
                )
        excluded = list(data_record.get("excluded_columns", []))
        excluded += [c for c in delays if c not in excluded]
        return {
            **values,
            "data": {**data_record, "excluded_columns": excluded},
            "feature_graph": {**graph_record, "nodes": nodes},
        }

    @pydantic.model_validator(mode="before")
    @classmethod
    def expand(cls, values: Any) -> Any:
        """Turns a forecasting request into targets and series features.

        `data.target` is the series value; the expansion points the target at
        the first future-value column, lists them all as targets and adds the
        lag / rolling / calendar feature nodes and records the value column.
        It is idempotent, so a saved configuration loads unchanged.

        Args:
            values: The raw specification.

        Returns:
            The expanded specification; unchanged if the task is not
            `forecast`, no horizon is set or `value_column` is already set
            (already expanded).
        """
        if not isinstance(values, dict) or values.get("task") != "forecast":
            return values
        values = named(cls, values)
        record = named(sections.Dataset, values.get("data"))
        if record.get("value_column") or not record.get("horizon"):
            return values
        value, horizon = record["target"], int(record["horizon"])
        names = graph.horizons(value, horizon)
        declared = values.get("feature_graph")
        graph_record = (
            declared.model_dump()
            if isinstance(declared, graph.Derivation)
            else dict(declared or {})
        )
        known = list(graph_record.get("nodes", []))
        present = {node["name"] for node in known}
        added = [
            node
            for node in graph.featurize(record, value)
            if node["name"] not in present
        ]
        return {
            **values,
            "data": {
                **record,
                "value_column": value,
                "target": names[0],
                "targets": names,
            },
            "feature_graph": {**graph_record, "nodes": known + added},
        }

    @pydantic.model_validator(mode="after")
    def check_cross_field_rules(self) -> Self:
        """Validates constraints that span several fields.

        Returns:
            The specification, unchanged.

        Raises:
            ValueError: If the combination of fields is inconsistent, for
                example a validation strategy without the column it needs, or
                a task without (or, for other tasks, with) its task-specific
                data columns.
        """
        data, validation = self.data, self.validation
        validate(self)
        check_survival_and_ordinal_rules(self)
        check_streaming_and_fairness_rules(self)
        check_multi_target_rules(self)
        if (self.task == "uplift") != (data.treatment_column is not None):
            raise errors.InputError(
                "uplift tasks need data.treatment_column, and only uplift "
                "tasks may set it"
            )
        if (self.task == "ranking") != (data.query_column is not None):
            raise errors.InputError(
                "ranking tasks need data.query_column, and only ranking "
                "tasks may set it"
            )
        exclusive(self)
        wanted = self.selection.maximize
        if wanted and self.metric and wanted != self.metric:
            raise errors.InputError(
                f"selection.maximize '{wanted}' conflicts with metric "
                f"'{self.metric}'"
            )
        if validation.embargo_rows and not data.time_column:
            raise errors.InputError(
                "validation.embargo_rows requires data.time_column"
            )
        return self

    @property
    def primary_metric(self) -> str | None:
        """The metric to optimize: `metric`, else `selection.maximize`."""
        return self.metric or self.selection.maximize

    @property
    def brief(self) -> bool:
        """Whether the time budget is short enough to shrink the search."""
        return self.resources.time_budget_seconds < FAST_BUDGET_SECONDS

    def budgeted(self) -> sections.Search:
        """Returns the search settings in force, reduced under a short budget.

        Returns:
            The configured search settings; when the time budget is under 300
            seconds, `top_models` is capped at 1 and `configs_per_model` at 3.
        """
        if not self.brief:
            return self.search
        return self.search.model_copy(
            update={
                "top_models": min(self.search.top_models, FAST_SEARCH_MODELS),
                "configs_per_model": min(
                    self.search.configs_per_model, FAST_SEARCH_CONFIGS
                ),
            }
        )

    def effective(self) -> sections.Strategies:
        """Returns the strategy settings in force, reduced under a short budget.

        Returns:
            The configured strategy settings; when the time budget is under
            300 seconds, `top_models` is capped at 1.
        """
        if not self.brief:
            return self.strategies
        return self.strategies.model_copy(
            update={
                "top_models": min(
                    self.strategies.top_models, FAST_SEARCH_MODELS
                )
            }
        )

    def canonize(self) -> str:
        """Returns a key-sorted, whitespace-free JSON rendering."""
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )

    def anchor(self) -> records.Layout:
        """Returns the configuration in its version-1 layout (field names).

        Returns:
            The JSON dump with every section moved back to its field names,
            the layout `config_hash` is taken on.
        """
        layout = self.spelling.downgrade(self.model_dump(mode="json"))
        for name, kind in parts().items():
            layout[name] = kind.spelling.downgrade(layout[name])
        return layout

    def config_hash(self) -> str:
        """Returns a short deterministic hash of the version-1 layout."""
        text = json.dumps(self.anchor(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()[:16]


def named(kind: type[sections.Section], value: Any) -> records.Layout:
    """Returns a section, given in either spelling, keyed by field name.

    Args:
        kind: The section's class.
        value: The section, a mapping in either spelling, or None.

    Returns:
        The section's fields as a dict keyed by field name.
    """
    layout = (
        value.model_dump()
        if isinstance(value, pydantic.BaseModel)
        else dict(value or {})
    )
    if set(layout) - set(kind.model_fields):
        layout = kind.spelling.downgrade(layout)
    return layout


def parts() -> dict[str, type[sections.Section]]:
    """Returns each configuration field that holds a spelled section."""
    found = {}
    for name, field in Config.model_fields.items():
        kind = field.annotation
        if isinstance(kind, type) and issubclass(kind, sections.Section):
            found[name] = kind
    return found


def renamed() -> dict[str, str]:
    """Returns every renamed configuration key as old to new dotted paths.

    Returns:
        For example `data.group_column` to `data.group` and
        `validation.horizon_rows` to `validation.rolling.horizon`.
    """
    found = dict(Config.vocabulary)
    for name, kind in parts().items():
        found |= {
            f"{name}.{old}": f"{name}.{new}"
            for old, new in kind.vocabulary.items()
        }
    return found


def validate(config: Config) -> None:
    """Checks the validation strategy against the columns it needs.

    Args:
        config: The configuration being validated.

    Raises:
        ValueError: If a strategy lacks its column, the row options do not fit
            the strategy, feature delays are inconsistent, or stratified
            validation is used with a task that has no classes.
    """
    data, validation = config.data, config.validation
    if validation.strategy == "group" and not (
        data.group_column or data.query_column
    ):
        raise errors.InputError(
            "strategy 'group' requires data.group_column (or "
            "data.query_column for ranking)"
        )
    if validation.strategy in ("temporal", "rolling") and (
        not data.time_column
    ):
        raise errors.InputError(
            f"strategy '{validation.strategy}' requires data.time_column"
        )
    if (
        validation.horizon_rows or validation.window_rows
    ) and validation.strategy not in ("rolling", "auto"):
        raise errors.InputError(
            "horizon_rows and window_rows apply to the rolling strategy"
        )
    if data.feature_delays and not data.time_column:
        raise errors.InputError("data.feature_delays requires data.time_column")
    if any(delay < 1 for delay in data.feature_delays.values()):
        raise errors.InputError("feature delays must be at least 1 row")
    stale = [
        node.name
        for node in config.feature_graph.nodes
        if set(node.inputs) & set(data.feature_delays)
        and not node.name.endswith(graph.DELAYED_SUFFIX)
    ]
    if stale:
        raise errors.InputError(
            f"feature nodes {stale} read a delayed column directly; "
            f"read '<column>{graph.DELAYED_SUFFIX}' instead"
        )
    if validation.strategy == "stratified" and config.task in (
        "regression",
        "survival",
        "ordinal",
        "ranking",
        "multilabel",
        "multioutput",
        "forecast",
        "uplift",
    ):
        raise errors.InputError(
            f"stratified validation is undefined for {config.task}"
        )


def check_survival_and_ordinal_rules(config: Config) -> None:
    """Checks that survival and ordinal tasks have their columns.

    Args:
        config: The configuration being validated.

    Raises:
        ValueError: If the event column or the ordinal order does not match the
            task, the order repeats a label, or an ordinal task has fewer than
            three levels.
    """
    data = config.data
    if (config.task == "survival") != (data.event_column is not None):
        raise errors.InputError(
            "survival tasks need data.event_column, and only survival "
            "tasks may set it"
        )
    if (config.task == "ordinal") != bool(data.ordinal_order):
        raise errors.InputError(
            "ordinal tasks need data.ordinal_order, and only ordinal "
            "tasks may set it"
        )
    if len(set(data.ordinal_order)) != len(data.ordinal_order):
        raise errors.InputError("data.ordinal_order must not repeat a label")
    if (
        config.task == "ordinal"
        and len(data.ordinal_order) < MIN_ORDINAL_LEVELS
    ):
        raise errors.InputError("ordinal tasks need at least three levels")


def check_streaming_and_fairness_rules(config: Config) -> None:
    """Checks streaming training and fairness mitigation settings.

    Args:
        config: The configuration being validated.

    Raises:
        ValueError: If streaming is combined with an unsupported option, or the
            training mode lacks its memory budget, or fairness reweighing
            conflicts with a weight column, or a mitigation has no sensitive
            column to act on.
    """
    data = config.data
    if data.memory_budget_gb is not None and (
        data.time_column
        or data.group_column
        or data.query_column
        or data.horizon
        or data.feature_delays
        or config.feature_graph.nodes
        or data.unlabeled != "block"
    ):
        raise errors.InputError(
            "streaming a file larger than memory supports random or "
            "stratified validation only (no time, group, query, "
            "feature graph, forecast or unlabeled-row options)"
        )
    if data.training == "stream" and data.memory_budget_gb is None:
        raise errors.InputError("training='stream' needs data.memory_budget_gb")
    if (
        config.fairness.mitigation == "reweighing"
        and data.weight_column not in (None, FAIRNESS_WEIGHT)
    ):
        raise errors.InputError(
            "fairness reweighing cannot be combined with data.weight_column"
        )
    if config.fairness.mitigation != "none":
        chosen = config.fairness.column or next(
            iter(data.sensitive_columns), None
        )
        if chosen is None or chosen not in data.sensitive_columns:
            raise errors.InputError(
                "fairness mitigation needs a column listed in "
                "data.sensitive_columns"
            )


def check_multi_target_rules(config: Config) -> None:
    """Checks that forecast and multi-target tasks have their columns.

    Args:
        config: The configuration being validated.

    Raises:
        ValueError: If a forecast lacks its horizon or time column, another
            task sets a horizon, or the targets do not match the task, repeat,
            number fewer than two, or do not start with the target.
    """
    data = config.data
    if config.task == "forecast":
        if not (data.horizon and data.time_column):
            raise errors.InputError(
                "forecast tasks need data.horizon and data.time_column"
            )
    elif data.horizon is not None:
        raise errors.InputError("data.horizon applies to forecast tasks only")
    multi = config.task in ("multilabel", "multioutput", "forecast")
    if multi != bool(data.targets):
        raise errors.InputError(
            "multilabel and multioutput tasks need data.targets, and "
            "only they may set it"
        )
    if multi:
        if len(set(data.targets)) != len(data.targets) or (
            len(data.targets) < MIN_TARGETS
        ):
            raise errors.InputError(
                "data.targets needs at least two distinct columns"
            )
        if data.targets[0] != data.target:
            raise errors.InputError("data.targets must start with data.target")


def exclusive(config: Config) -> None:
    """Checks that the target column is not reused for another role.

    Args:
        config: The configuration being validated.

    Raises:
        ValueError: If the event, group or time column equals the target, the
            target is an id, excluded, constrained, sensitive or unavailable
            column, or a monotonic constraint is not +1 or -1.
    """
    data = config.data
    if data.event_column == data.target:
        raise errors.InputError("data.event_column must differ from the target")
    for name in ("group_column", "time_column"):
        if getattr(data, name) == data.target:
            raise errors.InputError(f"data.{name} must differ from the target")
    if data.target in data.id_columns:
        raise errors.InputError("the target cannot be an id column")
    if data.target in data.excluded_columns:
        raise errors.InputError("the target cannot be excluded")
    if any(sign not in (-1, 1) for sign in data.monotonic_constraints.values()):
        raise errors.InputError("monotonic constraints must be +1 or -1")
    if data.target in data.monotonic_constraints:
        raise errors.InputError("the target cannot have a monotonic constraint")
    if data.target in data.sensitive_columns:
        raise errors.InputError("the target cannot be a sensitive column")
    if data.target in data.unavailable_at_prediction:
        raise errors.InputError(
            "the target cannot be unavailable at prediction"
        )
