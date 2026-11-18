"""Behaviour of the task configuration, its sections and the feature graph."""

import hashlib
import json

import pydantic
import pytest

from scmle.contracts.tasks import config
from scmle.contracts.tasks import graph
from scmle.contracts.tasks import sections


def make(task="auto", **changes):
    data = {"uri": "file://x.csv", "target": "y"}
    data.update(changes.pop("data", {}))
    return config.Config(data=data, task=task, **changes)


def refused(message, task="auto", **changes):
    with pytest.raises(pydantic.ValidationError, match=message):
        make(task, **changes)


# ---- graph -----------------------------------------------------------------


def test_graph_constants_are_the_documented_values():
    assert graph.GRAPH_VERSION == 1
    assert graph.DELAYED_SUFFIX == "__available"
    assert graph.HORIZON_SEPARATOR == "__h"
    assert graph.DEFAULT_LAGS == (1, 2, 3, 7)
    assert graph.DEFAULT_WINDOWS == (3, 7)


def test_horizons_name_each_future_step():
    assert graph.horizons("sales", 3) == ["sales__h1", "sales__h2", "sales__h3"]
    assert graph.horizons("sales", 0) == []


def test_derive_builds_a_one_input_node():
    assert graph.derive("n", "lag", "c", k=2) == {
        "name": "n",
        "op": "lag",
        "inputs": ["c"],
        "params": {"k": 2},
    }


def test_featurize_uses_the_defaults_without_a_clock():
    nodes = graph.featurize({}, "v")
    assert [n["name"] for n in nodes] == [
        "v__lag1",
        "v__lag2",
        "v__lag3",
        "v__lag7",
        "v__mean3",
        "v__std3",
        "v__mean7",
        "v__std7",
    ]
    assert nodes[4]["params"] == {"window": 3}
    assert nodes[5]["op"] == "rolling_std"


def test_featurize_adds_the_season_and_calendar_parts():
    record = {
        "forecast_lags": [1],
        "forecast_windows": [5],
        "seasonal_period": 12,
        "time_column": "t",
        "forecast_calendar": ["month", "dow"],
    }
    names = [n["name"] for n in graph.featurize(record, "v")]
    assert names == [
        "v__lag1",
        "v__lag12",
        "v__mean5",
        "v__std5",
        "t__month",
        "t__dow",
    ]
    again = graph.featurize({**record, "forecast_lags": [12]}, "v")
    assert [n["name"] for n in again][:1] == ["v__lag12"]
    assert len(again) == 5


def test_a_derivation_needs_unique_ordered_nodes():
    good = graph.Derivation(
        nodes=[
            {"name": "a", "op": "log1p", "inputs": ["x"]},
            {"name": "b", "op": "ratio", "inputs": ["a", "x"]},
        ]
    )
    assert good.version == 1
    assert good.nodes[0].params == {}
    with pytest.raises(pydantic.ValidationError, match="must be unique"):
        graph.Derivation(
            nodes=[
                {"name": "a", "op": "x", "inputs": ["x"]},
                {"name": "a", "op": "x", "inputs": ["x"]},
            ]
        )
    with pytest.raises(
        pydantic.ValidationError, match="node 'a' has no inputs"
    ):
        graph.Derivation(nodes=[{"name": "a", "op": "x", "inputs": []}])
    with pytest.raises(
        pydantic.ValidationError, match=r"later or on itself: \['b'\]"
    ):
        graph.Derivation(
            nodes=[
                {"name": "a", "op": "x", "inputs": ["b"]},
                {"name": "b", "op": "x", "inputs": ["x"]},
            ]
        )
    with pytest.raises(pydantic.ValidationError, match=r"itself: \['a'\]"):
        graph.Derivation(nodes=[{"name": "a", "op": "x", "inputs": ["a"]}])


def test_a_fingerprint_hashes_the_version_and_the_nodes():
    made = graph.Derivation(nodes=[{"name": "a", "op": "x", "inputs": ["c"]}])
    payload = json.dumps(
        [{"name": "a", "op": "x", "inputs": ["c"], "params": {}}],
        sort_keys=True,
    )
    expected = hashlib.sha256(f"v1:{payload}".encode()).hexdigest()[:16]
    assert made.fingerprint() == expected
    assert graph.Derivation().fingerprint() != expected
    assert graph.Derivation(version=2).fingerprint() != (
        graph.Derivation().fingerprint()
    )


# ---- sections --------------------------------------------------------------


def test_spell_splits_flat_and_dotted_keys():
    step = sections.spell({"a_b": "a", "c_d": "group.c", "e": "group.e"})
    stored = step.upgrade({"a_b": 1, "c_d": 2, "e": 3})
    assert stored == {"a": 1, "group": {"c": 2, "e": 3}}
    assert step.downgrade(stored) == {"a_b": 1, "c_d": 2, "e": 3}


def test_a_section_reads_either_spelling():
    stored = sections.Validation.model_validate(
        {"test": 0.3, "rolling": {"horizon": 4}}
    )
    named = sections.Validation(test_fraction=0.3, horizon_rows=4)
    assert stored == named
    assert stored.window_rows is None


def test_a_field_name_wins_over_its_stored_key():
    made = sections.Resources.model_validate(
        {"budget": 10, "time_budget_seconds": 99}
    )
    assert made.time_budget_seconds == 99
    nested = sections.Resources.model_validate(
        {"limits": {"memory": 2.0, "cpu": 3}, "memory_limit_gb": 4.0}
    )
    assert nested.memory_limit_gb == 4.0
    assert nested.cpu_limit == 3
    only = sections.Resources.model_validate(
        {"limits": {"memory": 2.0}, "memory_limit_gb": 4.0}
    )
    assert only.memory_limit_gb == 4.0


def test_a_section_passes_a_non_mapping_through():
    assert sections.Section.migrate(5) == 5


def test_section_defaults_are_the_documented_values():
    validation = sections.Validation()
    assert (
        validation.strategy,
        validation.test_fraction,
        validation.folds,
    ) == (
        "auto",
        0.2,
        5,
    )
    assert validation.seed == 42
    resources = sections.Resources()
    assert resources.time_budget_seconds == 1800
    assert resources.memory_limit_gb == 8.0
    search = sections.Search()
    assert (search.top_models, search.configs_per_model, search.eta) == (
        2,
        6,
        3,
    )
    assert sections.Stacking().size == 3
    assert sections.Threshold().false_negative_cost == 1.0
    parity = sections.Parity()
    assert (parity.trigger_gap, parity.min_reduction) == (0.05, 0.5)
    assert sections.Problem().success_criteria == {}
    assert "smote" in sections.Strategies().names
    assert sections.Dataset(uri="u", target="y").modality == "tabular"


@pytest.mark.parametrize(
    ("kind", "values"),
    [
        (sections.Validation, {"folds": 1}),
        (sections.Validation, {"test_fraction": 1.0}),
        (sections.Search, {"eta": 1}),
        (sections.Search, {"prune_fraction": 1.0}),
        (sections.Selection, {"latency_safety_factor": 0.99}),
        (sections.Dataset, {"uri": "u", "target": "y", "horizon": 0}),
        (sections.Dataset, {"uri": "u", "target": "y", "seasonal_period": 1}),
        (sections.Dataset, {"uri": "u", "target": "y", "memory_budget_gb": 0}),
        (sections.Parity, {"trigger_gap": 1.0}),
        (sections.Parity, {"min_reduction": 0.0}),
    ],
)
def test_sections_refuse_values_outside_their_bounds(kind, values):
    with pytest.raises(pydantic.ValidationError):
        kind(**values)


def test_sections_accept_values_on_their_bounds():
    assert sections.Validation(folds=2).folds == 2
    assert sections.Search(eta=2, prune_fraction=0.0).eta == 2
    assert sections.Selection(latency_safety_factor=1.0)
    assert sections.Parity(min_reduction=1.0).min_reduction == 1.0
    assert sections.Dataset(uri="u", target="y", seasonal_period=2)


def test_a_dataset_dumps_single_words():
    made = sections.Dataset(
        uri="u", target="y", group_column="g", forecast_lags=[1]
    )
    stored = made.model_dump(mode="json")
    assert stored["group"] == "g"
    assert stored["forecast"]["lags"] == [1]
    assert "group_column" not in stored


# ---- config ----------------------------------------------------------------


def test_config_constants_are_the_documented_values():
    assert config.CONFIG_VERSION == 1
    assert config.FAST_BUDGET_SECONDS == 300
    assert config.FAST_SEARCH_MODELS == 1
    assert config.FAST_SEARCH_CONFIGS == 3
    assert config.MIN_ORDINAL_LEVELS == 3
    assert config.MIN_TARGETS == 2
    assert config.FAIRNESS_WEIGHT == "fairness_weight"


def test_a_minimal_config_has_its_defaults():
    made = make()
    assert made.version == 1
    assert made.task == "auto"
    assert made.leakage_policy == "warn"
    assert made.ledger_rule == "e-lond"
    assert made.confirmation_rule == "by"
    assert made.primary_metric is None


def test_the_primary_metric_prefers_metric_over_selection():
    assert make(metric="auc").primary_metric == "auc"
    assert make(selection={"maximize": "f1"}).primary_metric == "f1"
    refused("conflicts with metric", metric="auc", selection={"maximize": "f1"})
    assert (
        make(metric="f1", selection={"maximize": "f1"}).primary_metric == "f1"
    )


def test_a_short_budget_shrinks_the_search_and_strategies():
    long = make(resources={"time_budget_seconds": 300})
    assert long.brief is False
    assert long.budgeted() is long.search
    assert long.effective() is long.strategies
    short = make(
        resources={"time_budget_seconds": 299},
        search={"top_models": 4, "configs_per_model": 2},
    )
    assert short.brief is True
    assert short.budgeted().top_models == 1
    assert short.budgeted().configs_per_model == 2
    assert (
        make(resources={"time_budget_seconds": 10}).budgeted().configs_per_model
        == 3
    )
    assert short.effective().top_models == 1
    zero = make(resources={"time_budget_seconds": 10}, search={"top_models": 0})
    assert zero.budgeted().top_models == 0


def test_canonize_is_sorted_compact_json():
    made = make()
    text = made.canonize()
    assert " " not in text.replace("file://x.csv", "")
    assert json.loads(text) == made.model_dump(mode="json")


def test_the_config_hash_is_taken_on_the_field_name_layout():
    made = make(data={"group_column": "g"}, validation={"strategy": "group"})
    layout = made.anchor()
    assert layout["data"]["group_column"] == "g"
    assert layout["feature_graph"] == {"version": 1, "nodes": []}
    text = json.dumps(layout, sort_keys=True, separators=(",", ":"))
    assert made.config_hash() == hashlib.sha256(text.encode()).hexdigest()[:16]
    stored = config.Config.model_validate(made.model_dump(mode="json"))
    assert stored.config_hash() == made.config_hash()
    assert make().config_hash() != made.config_hash()


def test_named_reads_either_spelling_by_field_name():
    assert config.named(sections.Validation, {"test": 0.1}) == {
        "test_fraction": 0.1
    }
    assert config.named(sections.Validation, {"folds": 3}) == {"folds": 3}
    assert config.named(sections.Validation, None) == {}
    made = sections.Validation(folds=3)
    assert config.named(sections.Validation, made)["folds"] == 3


def test_parts_and_renamed_list_the_spelled_sections():
    found = config.parts()
    assert found["data"] is sections.Dataset
    assert found["validation"] is sections.Validation
    assert "feature_graph" not in found
    assert "ensemble" not in found
    names = config.renamed()
    assert names["feature_graph"] == "derivation"
    assert names["data.group_column"] == "data.group"
    assert names["validation.horizon_rows"] == "validation.rolling.horizon"


def test_feature_delays_become_lagged_nodes_and_exclusions():
    made = make(data={"time_column": "t", "feature_delays": {"x": 2}})
    node = made.feature_graph.nodes[0]
    assert (node.name, node.op, node.inputs, node.params) == (
        "x__available",
        "lag",
        ["x"],
        {"k": 2},
    )
    assert made.data.excluded_columns == ["x"]
    again = config.Config.model_validate(made.model_dump(mode="json"))
    assert len(again.feature_graph.nodes) == 1
    assert again.data.excluded_columns == ["x"]


def test_a_forecast_expands_into_targets_and_features():
    made = make(
        "forecast",
        data={"time_column": "t", "horizon": 2, "forecast_lags": [1]},
    )
    assert made.data.value_column == "y"
    assert made.data.target == "y__h1"
    assert made.data.targets == ["y__h1", "y__h2"]
    assert [n.name for n in made.feature_graph.nodes][:1] == ["y__lag1"]
    again = config.Config.model_validate(made.model_dump(mode="json"))
    assert again.data.targets == made.data.targets
    assert len(again.feature_graph.nodes) == len(made.feature_graph.nodes)


@pytest.mark.parametrize(
    ("task", "changes", "message"),
    [
        (
            "auto",
            {"validation": {"strategy": "group"}},
            "requires data.group_column",
        ),
        (
            "auto",
            {"validation": {"strategy": "temporal"}},
            "'temporal' requires",
        ),
        ("auto", {"validation": {"strategy": "rolling"}}, "'rolling' requires"),
        (
            "auto",
            {"validation": {"strategy": "random", "horizon_rows": 2}},
            "apply to the rolling strategy",
        ),
        (
            "auto",
            {"data": {"feature_delays": {"x": 1}}},
            "requires data.time_column",
        ),
        (
            "auto",
            {"data": {"time_column": "t", "feature_delays": {"x": 0}}},
            "at least 1 row",
        ),
        (
            "regression",
            {"validation": {"strategy": "stratified"}},
            "stratified validation is undefined for regression",
        ),
        ("survival", {}, "survival tasks need data.event_column"),
        ("auto", {"data": {"event_column": "e"}}, "survival tasks need"),
        ("ordinal", {}, "ordinal tasks need data.ordinal_order"),
        (
            "ordinal",
            {"data": {"ordinal_order": ["a", "a", "b"]}},
            "must not repeat",
        ),
        (
            "ordinal",
            {"data": {"ordinal_order": ["a", "b"]}},
            "at least three levels",
        ),
        (
            "auto",
            {"data": {"memory_budget_gb": 1.0, "time_column": "t"}},
            "streaming a file larger than memory",
        ),
        (
            "auto",
            {"data": {"training": "stream"}},
            "needs data.memory_budget_gb",
        ),
        (
            "auto",
            {
                "data": {"weight_column": "w", "sensitive_columns": ["s"]},
                "fairness": {"mitigation": "reweighing"},
            },
            "cannot be combined with data.weight_column",
        ),
        (
            "auto",
            {"fairness": {"mitigation": "group_thresholds"}},
            "needs a column listed in data.sensitive_columns",
        ),
        (
            "auto",
            {
                "data": {"sensitive_columns": ["s"]},
                "fairness": {"mitigation": "group_thresholds", "column": "z"},
            },
            "needs a column listed",
        ),
        (
            "forecast",
            {"data": {"horizon": 2}},
            "need data.horizon and data.time_column",
        ),
        ("auto", {"data": {"horizon": 2}}, "applies to forecast tasks only"),
        ("multilabel", {}, "need data.targets"),
        ("multilabel", {"data": {"targets": ["y"]}}, "at least two distinct"),
        (
            "multilabel",
            {"data": {"targets": ["y", "y"]}},
            "at least two distinct",
        ),
        (
            "multilabel",
            {"data": {"targets": ["z", "y"]}},
            "must start with data.target",
        ),
        ("uplift", {}, "uplift tasks need data.treatment_column"),
        ("ranking", {}, "ranking tasks need data.query_column"),
        ("auto", {"data": {"group_column": "y"}}, "group_column must differ"),
        ("auto", {"data": {"time_column": "y"}}, "time_column must differ"),
        ("auto", {"data": {"id_columns": ["y"]}}, "cannot be an id column"),
        ("auto", {"data": {"excluded_columns": ["y"]}}, "cannot be excluded"),
        (
            "auto",
            {"data": {"monotonic_constraints": {"x": 2}}},
            "must be \\+1 or -1",
        ),
        (
            "auto",
            {"data": {"monotonic_constraints": {"y": 1}}},
            "monotonic constraint",
        ),
        (
            "auto",
            {"data": {"sensitive_columns": ["y"]}},
            "cannot be a sensitive",
        ),
        (
            "auto",
            {"data": {"unavailable_at_prediction": ["y"]}},
            "cannot be unavailable at prediction",
        ),
        ("auto", {"validation": {"embargo_rows": 3}}, "embargo_rows requires"),
    ],
)
def test_inconsistent_configs_are_refused(task, changes, message):
    refused(message, task, **changes)


def test_a_survival_event_equal_to_the_target_is_refused():
    refused("event_column must differ", "survival", data={"event_column": "y"})


def test_a_node_reading_a_delayed_column_directly_is_refused():
    refused(
        r"read a delayed column directly",
        data={"time_column": "t", "feature_delays": {"x": 1}},
        feature_graph={
            "nodes": [{"name": "z", "op": "log1p", "inputs": ["x"]}]
        },
    )


def test_consistent_special_tasks_are_accepted():
    assert make("survival", data={"event_column": "e"}).data.event_column == "e"
    assert make("ordinal", data={"ordinal_order": ["a", "b", "c"]})
    assert make("multilabel", data={"targets": ["y", "z"]}).data.targets == [
        "y",
        "z",
    ]
    assert make("uplift", data={"treatment_column": "t"})
    assert make(
        "ranking", data={"query_column": "q"}, validation={"strategy": "group"}
    )
    assert make(
        data={"sensitive_columns": ["s"], "weight_column": "fairness_weight"},
        fairness={"mitigation": "reweighing"},
    )
    assert make(data={"memory_budget_gb": 1.0, "training": "stream"})
    assert make(data={"time_column": "t"}, validation={"embargo_rows": 3})
    assert make(validation={"strategy": "auto", "window_rows": 2})
