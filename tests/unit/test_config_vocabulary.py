"""The configuration's single-word keys, CLI flags and gate variables (5, F7).

A configuration file is written with the approved single-word keys and read
in either spelling; its hash is taken on the old (version-1) layout, so the
hash of a run does not change. Every renamed CLI flag and `gates.toml`
variable keeps its old spelling for one release, with a warning.
"""

import json
import pathlib

import pytest

from benchmarks import registry
from scmle import cli
from scmle.cli import command
from scmle.cli import training
from scmle.contracts import tasks

ROOT = pathlib.Path(__file__).resolve().parents[2]

OLD = {
    "data": {
        "uri": "d.csv",
        "target": "y",
        "time_column": "t",
        "feature_delays": {"x": 2},
        "sensitive_columns": ["s"],
    },
    "task": "regression",
    "validation": {
        "strategy": "rolling",
        "horizon_rows": 5,
        "window_rows": 20,
        "embargo_rows": 1,
    },
    "resources": {
        "memory_limit_gb": 4.0,
        "cpu_limit": 2,
        "time_budget_seconds": 600,
    },
    "models": {"exclude_models": ["m"]},
    "decision": {"false_positive_cost": 2.0},
    "leakage_policy": "block",
}

NEW = {
    "data": {
        "uri": "d.csv",
        "target": "y",
        "time": "t",
        "delays": {"x": 2},
        "sensitive": ["s"],
    },
    "task": "regression",
    "validation": {
        "strategy": "rolling",
        "rolling": {"horizon": 5, "window": 20},
        "embargo": 1,
    },
    "resources": {"limits": {"memory": 4.0, "cpu": 2}, "budget": 600},
    "models": {"exclude": {"models": ["m"]}},
    "decision": {"costs": {"fp": 2.0}},
    "leakage": "block",
}

# The hashes the configurations above had before the rename.
HASH = "c7c20496a54618ee"
FORECAST_HASH = "f634083b6ad69498"


def test_an_old_and_a_new_configuration_are_the_same():
    old = tasks.Config.model_validate(OLD)
    new = tasks.Config.model_validate(NEW)
    assert old == new
    assert new.data.time_column == "t"
    assert new.validation.horizon_rows == 5
    assert new.resources.memory_limit_gb == 4.0
    assert new.models.exclude_models == ["m"]
    assert new.decision.false_positive_cost == 2.0
    assert new.leakage_policy == "block"


def test_the_hash_is_anchored_at_the_old_layout():
    assert tasks.Config.model_validate(OLD).config_hash() == HASH
    assert tasks.Config.model_validate(NEW).config_hash() == HASH
    forecast = tasks.Config(
        task="forecast",
        data={
            "uri": "d.csv",
            "target": "y",
            "time": "t",
            "horizon": 2,
            "forecast": {"lags": [1]},
        },
    )
    assert forecast.config_hash() == FORECAST_HASH
    assert forecast.data.forecast_lags == [1]
    assert forecast.data.targets == ["y__h1", "y__h2"]


def test_a_configuration_is_written_with_single_words():
    dumped = json.loads(tasks.Config.model_validate(OLD).canonize())
    assert "derivation" in dumped
    assert dumped["leakage"] == "block"
    assert dumped["data"]["time"] == "t"
    assert dumped["data"]["delays"] == {"x": 2}
    assert dumped["data"]["forecast"] == {
        "lags": [],
        "windows": [],
        "calendar": [],
    }
    assert dumped["validation"]["rolling"] == {"horizon": 5, "window": 20}
    assert dumped["resources"]["limits"] == {"memory": 4.0, "cpu": 2}
    assert dumped["models"]["exclude"] == {"models": ["m"], "families": []}
    assert dumped["decision"]["costs"] == {"fp": 2.0, "fn": 1.0}
    assert dumped["search"]["leaders"] == 2
    assert dumped["strategies"]["leaders"] == 2
    assert dumped["problem"]["criteria"] == {}
    words = []

    def collect(layout):
        for key, value in layout.items():
            words.append(key)
            if isinstance(value, dict) and key not in ("delays", "criteria"):
                collect(value)

    collect({k: v for k, v in dumped.items() if k != "derivation"})
    assert [w for w in words if "_" in w] == []
    again = tasks.Config.model_validate_json(json.dumps(dumped))
    assert again == tasks.Config.model_validate(OLD)


def test_the_renamed_paths_cover_every_section():
    renamed = tasks.renamed()
    assert len(renamed) == 70
    assert renamed["leakage_policy"] == "leakage"
    assert renamed["feature_graph"] == "derivation"
    assert renamed["data.group_column"] == "data.group"
    assert renamed["validation.horizon_rows"] == "validation.rolling.horizon"
    assert renamed["resources.time_budget_seconds"] == "resources.budget"
    assert renamed["selection.max_model_mb"] == "selection.size"
    assert renamed["search.top_models"] == "search.leaders"
    assert renamed["strategies.top_models"] == "strategies.leaders"
    assert renamed["decision.false_negative_cost"] == "decision.costs.fn"
    assert renamed["fairness.max_accuracy_cost"] == "fairness.cost"
    assert renamed["problem.success_criteria"] == "problem.criteria"


def parse(*argv):
    """Parses a `plan` command line."""
    return cli.build().parse_args(["plan", "d.csv", "--target", "y", *argv])


NEW_FLAGS = [
    ("--budget", "60", "budget", 60),
    ("--memory", "1.5", "memory", 1.5),
    ("--identifiers", "a", "identifiers", ["a"]),
    ("--group", "g", "group", "g"),
    ("--time", "t", "time", "t"),
    ("--weight", "w", "weight", "w"),
    ("--event", "e", "event", "e"),
    ("--query", "q", "query", "q"),
    ("--treatment", "x", "treatment", "x"),
    ("--baseline", "b", "baseline", "b"),
    ("--comparator", "selected_model", "comparator", "selected_model"),
    ("--draft", "f.json", "draft", "f.json"),
    ("--propensity", "p", "propensity", "p"),
    ("--action", "a", "action", "a"),
    ("--ingest", "i.json", "ingest", "i.json"),
    ("--period", "7", "period", 7),
    ("--order", "a", "order", ["a"]),
    ("--sensitive", "s", "sensitive", ["s"]),
    ("--delay", "x=2", "delay", ["x=2"]),
    ("--unavailable", "u", "unavailable", ["u"]),
    ("--test", "0.3", "test", 0.3),
    ("--embargo", "2", "embargo", 2),
    ("--rolling-horizon", "5", "rolling_horizon", 5),
    ("--rolling-window", "9", "rolling_window", 9),
    ("--leaders", "1", "leaders", 1),
    ("--configurations", "4", "configurations", 4),
    ("--leakage", "block", "leakage", "block"),
    ("--attestations", "a.json", "attestations", "a.json"),
]

OLD_FLAGS = {
    "--budget": "--time-budget",
    "--memory": "--memory-budget-gb",
    "--identifiers": "--id-columns",
    "--group": "--group-column",
    "--time": "--time-column",
    "--weight": "--weight-column",
    "--event": "--event-column",
    "--query": "--query-column",
    "--treatment": "--treatment-column",
    "--baseline": "--baseline-column",
    "--comparator": "--baseline-subject",
    "--draft": "--framing-draft",
    "--propensity": "--propensity-column",
    "--action": "--logged-action-column",
    "--ingest": "--ingest-plan",
    "--period": "--seasonal-period",
    "--order": "--ordinal-order",
    "--sensitive": "--sensitive-columns",
    "--delay": "--feature-delay",
    "--unavailable": "--unavailable-at-prediction",
    "--test": "--test-fraction",
    "--embargo": "--embargo-rows",
    "--rolling-horizon": "--horizon-rows",
    "--rolling-window": "--window-rows",
    "--leaders": "--top-models",
    "--configurations": "--configs-per-model",
    "--leakage": "--leakage-policy",
    "--attestations": "--leakage-attestations",
}


@pytest.mark.parametrize(("flag", "value", "dest", "parsed"), NEW_FLAGS)
def test_a_renamed_flag_parses(flag, value, dest, parsed):
    assert getattr(parse(flag, value), dest) == parsed


@pytest.mark.parametrize(("flag", "value", "dest", "parsed"), NEW_FLAGS)
def test_an_old_flag_still_works_and_warns(flag, value, dest, parsed):
    with pytest.warns(FutureWarning, match=f"{OLD_FLAGS[flag]}.*{flag}"):
        args = parse(OLD_FLAGS[flag], value)
    assert getattr(args, dest) == parsed


def test_the_old_boolean_flag_still_works_and_warns():
    assert parse("--temporal").temporal is True
    assert parse().temporal is False
    with pytest.warns(FutureWarning, match="--time-as-feature"):
        assert parse("--time-as-feature").temporal is True


def test_an_old_flag_warns_once():
    parser = command.Parser()
    command.option(parser, "--group", "--group-column")
    with pytest.warns(FutureWarning) as caught:
        args = parser.parse_args(["--group-column", "a", "--group-column", "b"])
    assert len(caught) == 1
    assert args.group == "b"
    assert parser.parse_args([]).group is None


def test_old_flags_are_hidden_from_the_help():
    parser = command.Parser()
    training.Run().configure(parser)
    text = parser.format_help()
    assert "--group" in text
    for old in [*OLD_FLAGS.values(), "--time-as-feature"]:
        assert old not in text, old


def test_the_defaults_survive_the_aliases():
    args = parse()
    assert args.budget == 1800
    assert args.test == pytest.approx(0.2)
    assert args.leaders == 2
    assert args.configurations == 6
    assert args.leakage == "warn"
    assert args.sensitive == []
    assert args.delay == []
    assert args.identifiers == []


def test_an_old_flag_parses_like_the_new_one():
    with pytest.warns(FutureWarning):
        old = parse("--time-budget", "60", "--horizon-rows", "3")
    assert old == parse("--budget", "60", "--rolling-horizon", "3")


def test_the_gate_registry_uses_the_new_variables():
    renamed = tasks.renamed()
    for gate in registry.load().values():
        stale = [v for v in gate.variables if v in renamed]
        assert stale == [], gate.name


def test_an_old_gate_variable_is_read_with_a_warning():
    table = {
        "site": "scmle.a:b",
        "error": "BlockedError",
        "claim": "C1",
        "passes": "p",
        "blocks": "b",
        "override": "none",
        "variables": ["data.time_column", "validation.folds"],
    }
    with pytest.warns(FutureWarning, match=r"data\.time_column.*data\.time"):
        gate = registry.parse("x", table)
    assert gate.variables == ("data.time", "validation.folds")


def test_the_cli_page_names_the_new_flags():
    text = (ROOT / "docs" / "CLI.md").read_text(encoding="utf-8")
    rows = "\n".join(line for line in text.splitlines() if "|" in line)
    assert "`--budget S`" in rows
    for old in OLD_FLAGS.values():
        assert old not in rows, old


def test_a_field_name_written_into_a_dump_wins():
    dumped = tasks.Config.model_validate(OLD).model_dump()
    dumped["resources"]["time_budget_seconds"] = 250
    dumped["validation"]["horizon_rows"] = 7
    again = tasks.Config.model_validate(dumped)
    assert again.resources.time_budget_seconds == 250
    assert again.resources.memory_limit_gb == 4.0
    assert again.validation.horizon_rows == 7
    assert again.validation.window_rows == 20
