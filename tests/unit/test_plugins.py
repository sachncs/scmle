"""One plugin registry: enrol, withdraw, find and discover every kind."""

from importlib import metadata
import threading

import pytest

from scmle.contracts import errors
from scmle.runtime import plugins


class Item:
    def __init__(self, name):
        self.name = name


def test_every_kind_has_its_entry_point_group():
    assert plugins.GROUPS == {
        plugins.Kind.ESTIMATORS: "scmle.models",
        plugins.Kind.METRICS: "scmle.metrics",
        plugins.Kind.READERS: "scmle.readers",
        plugins.Kind.DETECTORS: "scmle.detectors",
        plugins.Kind.STAGES: "scmle.stages",
        plugins.Kind.TRANSFORMERS: "scmle.transformers",
    }


@pytest.mark.parametrize("kind", list(plugins.Kind))
def test_every_kind_enrols_and_discovers_through_its_group(kind, monkeypatch):
    registry = plugins.Registry()
    item = Item("one")
    registry.enrol(kind, "one", item)
    assert registry.find(kind, "one") is item
    assert all(
        registry.entries(other) == {} for other in plugins.Kind if other != kind
    )
    seen = []

    def points(group):
        seen.append(group)
        return [Point("one", item)]

    monkeypatch.setattr(metadata, "entry_points", points)
    assert registry.discover(kind).items == (item,)
    assert seen == [plugins.GROUPS[kind]]


def test_a_string_kind_still_works_and_warns():
    registry = plugins.Registry()
    item = Item("old")
    with pytest.warns(DeprecationWarning, match="plugins.Kind"):
        registry.enrol("metrics", "old", item)
    assert registry.find(plugins.Kind.METRICS, "old") is item
    with pytest.warns(DeprecationWarning, match="plugins.Kind"):
        registry.withdraw("metrics", "old")
    assert registry.entries(plugins.Kind.METRICS) == {}


def test_enrol_find_and_withdraw_one_kind_at_a_time():
    registry = plugins.Registry()
    item = Item("x")
    registry.enrol(plugins.Kind.METRICS, "x", item)
    assert registry.find(plugins.Kind.METRICS, "x") is item
    assert registry.entries(plugins.Kind.METRICS) == {"x": item}
    assert registry.entries(plugins.Kind.READERS) == {}
    with pytest.raises(errors.ConfigError, match="metric 'x' is already"):
        registry.enrol(plugins.Kind.METRICS, "x", Item("x"))
    other = Item("x")
    registry.enrol(plugins.Kind.METRICS, "x", other, replace=True)
    assert registry.find(plugins.Kind.METRICS, "x") is other
    registry.withdraw(plugins.Kind.METRICS, "x")
    registry.withdraw(plugins.Kind.METRICS, "x")
    with pytest.raises(errors.ConfigError, match=r"unknown metric 'x'"):
        registry.find(plugins.Kind.METRICS, "x")


def test_entries_are_a_copy():
    registry = plugins.Registry()
    registry.enrol(plugins.Kind.STAGES, "s", Item("s"))
    registry.entries(plugins.Kind.STAGES).clear()
    assert set(registry.entries(plugins.Kind.STAGES)) == {"s"}


def test_concurrent_enrolment_keeps_every_name_and_refuses_duplicates():
    registry = plugins.Registry()
    refused = []

    def enrol(index):
        for name in (f"n{index}", "shared"):
            try:
                registry.enrol(plugins.Kind.DETECTORS, name, Item(name))
            except errors.ConfigError:
                refused.append(name)

    threads = [threading.Thread(target=enrol, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(registry.entries(plugins.Kind.DETECTORS)) == 17
    assert refused == ["shared"] * 15


class Point:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def load(self):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def test_discovery_loads_items_vets_them_and_keeps_failures(monkeypatch):
    good = Item("good")
    seen = []

    def points(group):
        seen.append(group)
        return [
            Point("one", good),
            Point("many", lambda: [Item("a"), Item("b")]),
            Point("broken", ImportError("missing")),
            Point("bad", Item("bad")),
        ]

    def vet(item):
        if item.name == "bad":
            raise errors.ConfigError("does not conform")

    monkeypatch.setattr(metadata, "entry_points", points)
    found = plugins.Registry().discover(plugins.Kind.READERS, vet)
    assert seen == ["scmle.readers"]
    assert [item.name for item in found.items] == ["good", "a", "b"]
    assert set(found.failures) == {"broken", "bad"}
    assert "missing" in found.failures["broken"]
    assert "does not conform" in found.failures["bad"]


def test_the_shared_registry_holds_user_estimators():
    from sklearn import naive_bayes

    from scmle.models import custom

    try:
        adapter = custom.register(
            "plugin_nb", classifier=naive_bayes.GaussianNB
        )
        found = plugins.REGISTRY.find(plugins.Kind.ESTIMATORS, "plugin_nb")
        assert found is adapter
    finally:
        custom.unregister("plugin_nb")
    assert "plugin_nb" not in plugins.REGISTRY.entries(plugins.Kind.ESTIMATORS)


def test_the_old_registry_calls_still_work_and_warn():
    from scmle.models import registry

    adapter = Item("old_call")
    with pytest.warns(DeprecationWarning, match="plugins.REGISTRY"):
        registry.adopt(adapter)
    with pytest.warns(DeprecationWarning, match="registry.extensions"):
        assert registry.extensions()["old_call"] is adapter
    with pytest.warns(DeprecationWarning, match="registry.deregister"):
        registry.deregister("old_call")
    assert "old_call" not in plugins.REGISTRY.entries(plugins.Kind.ESTIMATORS)


def test_a_user_metric_is_enrolled_in_the_shared_registry():
    from scmle.evaluation import metrics

    try:
        metric = metrics.register(
            "plugin_metric", lambda truth, guess: 0.0, greater_is_better=True
        )
        assert plugins.REGISTRY.find(plugins.Kind.METRICS, "plugin_metric") is (
            metric
        )
    finally:
        metrics.CATALOG.unregister("plugin_metric")
        plugins.REGISTRY.withdraw(plugins.Kind.METRICS, "plugin_metric")
