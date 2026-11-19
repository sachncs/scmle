"""Claim C14 evidence: refusal rate and precision of end-to-end runs."""

import pytest

from benchmarks.evidence import custody
from benchmarks.evidence import leakage
from benchmarks.evidence import refusal
from scmle.contracts import errors


def group(
    trials: int, refusals: int, gates: dict | None = None, failures: int = 0
) -> dict:
    return {
        "trials": trials,
        "models": trials - refusals - failures,
        "refusals": refusals,
        "failures": failures,
        "unfinished": 0,
        "gates": gates or {},
        "deciding": {"trials": trials, "hits": refusals + failures},
    }


def document(
    real: dict,
    simulated: dict | None = None,
    refused: int = 0,
    justified: int = 0,
) -> dict:
    return {
        "settings": {"budget": refusal.BUDGET},
        "counts": {
            "real": real,
            "simulated": simulated or group(0, 0),
            "planted": group(refused, refused),
            "precision": {"refusals": refused, "justified": justified},
        },
    }


def test_too_few_real_trials_is_not_measured_and_says_how_many_more():
    value, verdict = refusal.summary(document(group(20, 0)))

    assert verdict == "not measured"
    assert "39 more real validation trials that return a model" in value


def test_simulated_trials_alone_never_pass_the_real_criterion():
    value, verdict = refusal.summary(
        document(group(0, 0), group(200, 0), refused=40, justified=40)
    )

    assert verdict == "not measured"
    assert "not the real criterion" in value


def test_enough_clean_real_trials_and_justified_refusals_pass():
    verdict = refusal.summary(document(group(59, 0), refused=29, justified=29))[
        1
    ]

    assert verdict == "pass"


def test_too_many_real_refusals_fail_once_the_minimum_is_reached():
    # Enforces docs/preregistration/c14.md:43.
    value, verdict = refusal.summary(
        document(group(60, 6, {"task.auto": 6}), refused=29, justified=29)
    )

    assert verdict == "fail"
    assert "task.auto" in value


def test_unjustified_refusals_fail_the_precision():
    # Enforces docs/preregistration/c14.md:43.
    verdict = refusal.summary(document(group(59, 0), refused=30, justified=20))[
        1
    ]

    assert verdict == "fail"


def test_a_crash_counts_against_the_rate_not_as_a_refusal_free_trial():
    # Enforces docs/preregistration/c14.md:36.
    # Enforces docs/preregistration/c14.md:66.
    value, verdict = refusal.summary(
        document(group(60, 0, failures=6), refused=29, justified=29)
    )

    assert verdict == "fail"
    assert "6 failed" in value


def test_only_validation_items_decide_the_rate():
    tasks = refusal.catalog()
    splits = refusal.splits()
    held = next(
        n
        for n, t in tasks.items()
        if t.group == "real" and splits.get(n) == "validation"
    )
    sealed = next(
        n
        for n, t in tasks.items()
        if t.group == "real" and splits.get(n) == "acceptance"
    )
    records = [
        {"name": held, "outcome": "failed"},
        {"name": sealed, "outcome": "model"},
        {"name": sealed, "outcome": "refused", "gate": "task.auto"},
    ]

    deciding = refusal.tally(records, tasks)["real"]["deciding"]

    assert deciding == {"trials": 1, "hits": 1}


def test_an_unfinished_validation_run_counts_against_the_rate():
    # Enforces docs/preregistration/c14.md:36.
    # Enforces docs/preregistration/c14.md:66.
    tasks = refusal.catalog()
    splits = refusal.splits()
    held = next(
        n
        for n, t in tasks.items()
        if t.group == "real" and splits.get(n) == "validation"
    )

    counts = refusal.tally([{"name": held, "outcome": "unfinished"}], tasks)

    assert counts["real"]["deciding"] == {"trials": 1, "hits": 1}


def test_the_stored_rate_is_not_measured_below_the_validation_minimum():
    value, verdict = refusal.summary()

    assert verdict != "pass"
    assert "validation" in value


def test_a_document_without_counts_is_refused():
    with pytest.raises(errors.InputError):
        refusal.summary({"settings": {}})


def test_the_rubric_counts_only_planted_refusals_by_a_named_gate():
    tasks = refusal.catalog()
    planted = next(t for t in tasks.values() if t.group == "planted")
    real = next(t for t in tasks.values() if t.group == "real")
    records = [
        {
            "name": planted.name,
            "outcome": "refused",
            "gate": planted.expected[0],
        },
        {"name": real.name, "outcome": "refused", "gate": "task.auto"},
        {"name": real.name, "outcome": "unfinished"},
    ]

    counts = refusal.tally(records, tasks)

    assert counts["precision"]["refusals"] == 2
    assert counts["precision"]["justified"] == 1
    assert counts["real"]["trials"] == 1
    assert counts["real"]["unfinished"] == 1
    assert counts["real"]["gates"] == {"task.auto": 1}


def test_a_refusal_is_traced_to_its_registered_gate():
    sites = {f"{__name__}:test_a_refusal_is_traced_to_its_registered_gate": "x"}
    with pytest.raises(errors.BlockedError) as caught:
        raise errors.BlockedError("planted")

    assert refusal.locate(caught.value, sites)[0] == "x"
    assert refusal.locate(caught.value, {})[0] == "unregistered"


def test_every_simulated_kind_builds_a_table_with_its_target():
    for kind in (
        "binary",
        "multiclass",
        "regression",
        "survival",
        "ordinal",
        "ranking",
        "multilabel",
        "multioutput",
        "forecast",
        "uplift",
    ):
        frame, target, options = refusal.simulate(kind, 60, 0, "mixed")
        names = [target] if isinstance(target, str) else target
        assert set(names) <= set(frame.columns)
        assert "task" in options


def test_planted_tasks_name_registered_gates():
    sites = set(refusal.registry().values())
    for task in refusal.catalog().values():
        assert set(task.expected) <= sites
        assert bool(task.expected) == (task.group == "planted")


def test_the_catalog_has_enough_sourced_real_tasks_for_the_rate():
    variables = refusal.acceptance.load()
    rule = variables["refusal_rate_max"].rule
    real = [t for t in refusal.catalog().values() if t.group == "real"]
    # validation and acceptance each need the rule's minimum on their own.
    assert len(real) >= 2 * rule.minimum(variables["alpha"].value)
    assert set(refusal.SOURCES) == set(refusal.EXTRA) | set(refusal.SUITE)
    assert all(source[2] for source in refusal.SOURCES.values())


def test_every_openml_table_is_pinned_and_none_is_a_listed_copy():
    keys = [
        custody.identify(t)
        for t in refusal.catalog().values()
        if t.group == "real"
    ]
    used = {int(k.split(":")[1]) for k in keys if k.startswith("openml:")}
    screened = {
        case.table.source
        for case in leakage.catalog().values()
        if isinstance(case.table.source, int)
    }
    assert used | screened == set(refusal.PINS)
    assert all(len(digest) == 64 for digest in refusal.PINS.values())
    assert not used & set(refusal.COPIES)
    assert all(reason for reason in refusal.COPIES.values())


def test_a_table_whose_file_differs_from_its_pin_is_refused(monkeypatch):
    identifier = next(i for i in refusal.EXTRA if refusal.cached(i))
    monkeypatch.setitem(refusal.PINS, identifier, "0" * 64)
    with pytest.raises(errors.InputError, match="pin"):
        refusal.fetch(identifier)


def test_the_stored_measurements_are_for_the_current_code():
    assert refusal.problems() == []
