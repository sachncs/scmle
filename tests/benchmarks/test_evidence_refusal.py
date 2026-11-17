"""Claim C14 evidence: refusal rate and precision of end-to-end runs."""

import pytest

from benchmarks.evidence import refusal
from scmle.contracts import errors


def group(trials: int, refusals: int, gates: dict | None = None) -> dict:
    return {
        "trials": trials,
        "models": trials - refusals,
        "refusals": refusals,
        "failures": 0,
        "unfinished": 0,
        "gates": gates or {},
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
    assert "39 more refusal-free real trials" in value


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
    value, verdict = refusal.summary(
        document(group(60, 6, {"task.auto": 6}), refused=29, justified=29)
    )

    assert verdict == "fail"
    assert "task.auto" in value


def test_unjustified_refusals_fail_the_precision():
    verdict = refusal.summary(document(group(59, 0), refused=30, justified=20))[
        1
    ]

    assert verdict == "fail"


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
    assert len(real) >= rule.minimum(variables["alpha"].value)
    assert set(refusal.SOURCES) == set(refusal.EXTRA)


def test_the_stored_measurements_are_for_the_current_code():
    assert refusal.problems() == []
