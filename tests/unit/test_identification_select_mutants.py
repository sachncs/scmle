"""Exact records and wording of the identification selector."""

import pytest

import scmle.data.admissibility.dag
from scmle.data.charter import causal
from scmle.data.identification import select

COVARIATES = "unconfoundedness_given:age"
ASSUMPTIONS = (COVARIATES, "positivity", "no_interference")
NOTE = "any summary printed next to this report is associational, not causal"


def block(**changes) -> causal.Question:
    values = {
        "treatment": "t",
        "outcome": "y",
        "estimand": "ATE",
        "assumptions": ASSUMPTIONS,
    }
    values.update(changes)
    return causal.Question(**values)


def graph(*edges):
    return scmle.data.admissibility.dag.Dag.build(source="test", edges=edges)


def test_the_registry_pins_what_every_strategy_needs():
    table = {
        name: (m.name, m.estimands, m.assumptions, m.design, m.family)
        for name, m in select.STRATEGIES.items()
    }
    assert table == {
        "randomized_diff": (
            "randomized_diff",
            ("ATE",),
            ("randomized_assignment", "no_interference"),
            (),
            "difference in means with a recorded assignment mechanism",
        ),
        "backdoor_aipw": (
            "backdoor_aipw",
            ("ATE", "ATT"),
            ("unconfoundedness_given", "positivity", "no_interference"),
            (),
            "cross-fitted augmented IPW under back-door adjustment",
        ),
        "did": (
            "did",
            ("ATT_DID",),
            ("parallel_trends", "no_anticipation", "no_interference"),
            ("unit", "time", "adoption"),
            "group-time difference in differences",
        ),
        "iv": (
            "iv",
            ("LATE",),
            (
                "instrument_relevance",
                "exclusion",
                "independence",
                "monotonicity",
            ),
            ("instrument",),
            "just-identified two-stage least squares",
        ),
        "rd": (
            "rd",
            ("LATE_RD",),
            ("continuity_at_cutoff", "no_manipulation"),
            ("running_variable", "cutoff"),
            "local linear regression discontinuity",
        ),
        "synth": (
            "synth",
            ("ATT_SC",),
            ("synthetic_hull", "no_interference"),
            ("unit", "time"),
            "synthetic control with placebo-in-space inference",
        ),
    }
    assert list(select.STRATEGIES) == [
        "randomized_diff",
        "backdoor_aipw",
        "did",
        "iv",
        "rd",
        "synth",
    ]
    assert select.IDENTIFIED == "identified"
    assert select.NOT_IDENTIFIED == "not_identified"
    assert select.ASSOCIATIONAL == NOTE


def test_a_verdict_is_recorded_under_exact_keys():
    verdict = select.Verdict(
        "iv", False, ("a",), ("b", "c"), ("because",), checked=True
    )
    assert verdict.to_dict() == {
        "strategy": "iv",
        "eligible": False,
        "missing_assumptions": ["a"],
        "missing_design": ["b", "c"],
        "reasons": ["because"],
        "adjustment_checked_by_dag": True,
    }
    bare = select.Verdict("rd", True)
    assert bare.missing_assumptions == ()
    assert bare.missing_design == ()
    assert bare.reasons == ()
    assert bare.checked is False


def test_the_plan_record_of_an_identified_run_has_no_refusal_fields():
    plan = select.select(block())
    record = plan.to_dict()
    assert set(record) == {
        "status",
        "primary",
        "multiverse",
        "order",
        "order_chartered",
        "strategies",
    }
    assert record["status"] == "identified"
    assert record["primary"] == "backdoor_aipw"
    assert record["multiverse"] == []
    assert record["order"] == list(select.STRATEGIES)
    assert record["order_chartered"] is False
    assert record["strategies"] == [v.to_dict() for v in plan.verdicts]
    assert [s["strategy"] for s in record["strategies"]] == list(
        select.STRATEGIES
    )


def test_the_plan_record_of_a_refusal_adds_the_estimate_action_and_note():
    plan = select.select(block(assumptions=("positivity",)))
    record = plan.to_dict()
    assert record["status"] == "not_identified"
    assert record["primary"] is None
    assert record["estimate"] is None
    assert record["action"] == "refuse_claim"
    assert record["note"] == NOTE


def test_looking_up_an_unconsidered_strategy_raises_with_its_name():
    plan = select.select(block(strategies=("backdoor_aipw",)))
    with pytest.raises(KeyError) as raised:
        plan.verdict("iv")
    assert raised.value.args == ("iv",)
    assert plan.verdict("backdoor_aipw").strategy == "backdoor_aipw"


def test_an_estimand_mismatch_names_the_estimand_and_what_is_answered():
    verdict = select.select(block(strategies=("iv",))).verdict("iv")
    assert verdict.eligible is False
    assert verdict.reasons == (
        "estimand ATE is not answered by iv (answers LATE)",
    )
    both = select.select(block(strategies=("backdoor_aipw", "did")))
    assert both.verdict("did").reasons == (
        "estimand ATE is not answered by did (answers ATT_DID)",
    )
    att = select.select(block(estimand="ATT")).verdict("backdoor_aipw")
    assert att.eligible


def test_missing_pieces_are_listed_in_the_declared_order_with_no_reason():
    spec = block(
        estimand="LATE",
        strategies=("iv",),
        assumptions=("exclusion", "monotonicity"),
    )
    verdict = select.select(spec).verdict("iv")
    assert verdict.missing_assumptions == (
        "instrument_relevance",
        "independence",
    )
    assert verdict.missing_design == ("instrument",)
    assert verdict.reasons == ()
    assert verdict.checked is False
    assert verdict.eligible is False


def test_a_design_field_that_is_declared_is_not_missing():
    spec = block(
        estimand="ATT_DID",
        strategies=("did",),
        assumptions=("parallel_trends", "no_anticipation", "no_interference"),
        unit="id",
        time="week",
    )
    verdict = select.select(spec).verdict("did")
    assert verdict.missing_design == ("adoption",)
    full = block(
        estimand="ATT_DID",
        strategies=("did",),
        assumptions=("parallel_trends", "no_anticipation", "no_interference"),
        unit="id",
        time="week",
        adoption="first",
    )
    assert select.select(full).verdict("did").eligible


def test_covariates_set_to_none_still_attest_unconfoundedness():
    spec = block(
        assumptions=(
            "unconfoundedness_given:none",
            "positivity",
            "no_interference",
        )
    )
    assert select.select(spec).verdict("backdoor_aipw").eligible


def test_the_adjustment_check_is_skipped_while_anything_else_is_missing():
    # a DAG that would reject the set, but an assumption is already missing
    bad = graph(("age", "t"), ("age", "y"), ("t", "y"), ("u", "t"), ("u", "y"))
    spec = block(assumptions=(COVARIATES, "positivity"), dag="dag.json")
    verdict = select.select(spec, bad).verdict("backdoor_aipw")
    assert verdict.missing_assumptions == ("no_interference",)
    assert verdict.reasons == ()
    assert verdict.checked is False
    # an estimand mismatch stops the check too
    other = block(estimand="LATE", strategies=("backdoor_aipw",), dag="d.json")
    mismatch = select.select(other, bad).verdict("backdoor_aipw")
    assert len(mismatch.reasons) == 1
    assert mismatch.reasons[0].startswith("estimand LATE")
    assert mismatch.checked is False


def test_a_declared_dag_that_was_not_supplied_blocks_the_strategy():
    declared = block(dag="dag.json")
    verdict = select.select(declared).verdict("backdoor_aipw")
    assert verdict.eligible is False
    assert verdict.reasons == ("a DAG is declared but was not supplied",)
    assert verdict.checked is False
    assert select.check_adjustment(declared, None) == (
        ["a DAG is declared but was not supplied"],
        False,
    )
    assert select.check_adjustment(block(), None) == ([], False)


def test_an_invalid_adjustment_set_is_refused_with_the_graph_wording():
    bad = graph(("age", "t"), ("age", "y"), ("t", "y"), ("u", "t"), ("u", "y"))
    reasons, checked = select.check_adjustment(block(), bad)
    assert reasons == [
        (
            "the declared adjustment set is not a valid back-door set for "
            "the declared DAG"
        )
    ]
    assert checked is False


def test_a_valid_adjustment_set_is_verified():
    good = graph(("age", "t"), ("age", "y"), ("t", "y"))
    assert select.check_adjustment(block(), good) == ([], True)
    plan = select.select(block(), good)
    assert plan.primary == "backdoor_aipw"
    assert plan.verdict("backdoor_aipw").reasons == ()
    assert plan.to_dict()["strategies"][1]["adjustment_checked_by_dag"] is True


def test_a_graph_that_cannot_judge_the_set_gives_its_error_as_the_reason():
    # `age` is not a node of this graph
    other = graph(("z", "t"), ("t", "y"))
    reasons, checked = select.check_adjustment(block(), other)
    assert reasons == [
        "the declared DAG cannot check the set: not nodes of the graph: ['age']"
    ]
    assert checked is False
    verdict = select.select(block(), other).verdict("backdoor_aipw")
    assert verdict.eligible is False
    assert verdict.reasons == tuple(reasons)


def test_the_adjustment_set_names_are_read_from_the_covariate_assumption():
    chosen = block(
        assumptions=(
            "unconfoundedness_given:age, z ,",
            "positivity",
            "no_interference",
        )
    )
    good = graph(("age", "t"), ("age", "y"), ("t", "y"), ("z", "y"))
    assert select.check_adjustment(chosen, good) == ([], True)
    none = block(
        assumptions=(
            "unconfoundedness_given:none",
            "positivity",
            "no_interference",
        )
    )
    open_graph = graph(("t", "y"))
    assert select.check_adjustment(none, open_graph) == ([], True)


def test_an_unknown_strategy_is_a_complete_ineligible_verdict():
    plan = select.select(block(strategies=("magic", "backdoor_aipw")))
    magic = plan.verdict("magic")
    assert magic == select.Verdict(
        "magic", eligible=False, reasons=("unknown strategy magic",)
    )
    assert plan.primary == "backdoor_aipw"
    assert plan.multiverse == ()


def test_a_chartered_order_is_kept_and_the_first_eligible_leads():
    spec = block(
        strategies=("did", "randomized_diff", "backdoor_aipw"),
        assumptions=(*ASSUMPTIONS, "randomized_assignment"),
    )
    plan = select.select(spec)
    assert plan.chartered_order is True
    assert plan.order == ("did", "randomized_diff", "backdoor_aipw")
    assert plan.primary == "randomized_diff"
    assert plan.multiverse == ("backdoor_aipw",)
    assert [v.strategy for v in plan.verdicts] == list(plan.order)
    assert plan.status == "identified"
    assert plan.to_dict()["order_chartered"] is True
    assert plan.to_dict()["multiverse"] == ["backdoor_aipw"]


def test_the_refusal_report_lists_what_each_strategy_lacks():
    spec = block(
        estimand="LATE",
        strategies=("iv", "rd"),
        assumptions=("exclusion",),
    )
    plan = select.select(spec)
    assert select.refusal(plan, spec) == {
        "status": "not_identified",
        "estimand": "LATE",
        "treatment": "t",
        "outcome": "y",
        "estimate": None,
        "action": "refuse_claim",
        "missing_per_strategy": {
            "iv": {
                "missing_assumptions": [
                    "instrument_relevance",
                    "independence",
                    "monotonicity",
                ],
                "missing_design": ["instrument"],
                "reasons": [],
            },
            "rd": {
                "missing_assumptions": [
                    "continuity_at_cutoff",
                    "no_manipulation",
                ],
                "missing_design": ["running_variable", "cutoff"],
                "reasons": [
                    "estimand LATE is not answered by rd (answers LATE_RD)"
                ],
            },
        },
        "note": NOTE,
    }


def test_a_mismatch_lists_every_estimand_the_strategy_answers():
    spec = block(estimand="LATE", strategies=("backdoor_aipw",))
    verdict = select.select(spec).verdict("backdoor_aipw")
    assert verdict.reasons == (
        "estimand LATE is not answered by backdoor_aipw (answers ATE, ATT)",
    )
