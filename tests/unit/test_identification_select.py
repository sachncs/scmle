"""The selector is a data-blind function of the charter and refuses."""

import ast
import inspect

import numpy as np
import pytest

from scmle.data.admissibility import dag as dag_module
from scmle.data.charter import causal
from scmle.data.identification import select

COVARIATES = "unconfoundedness_given:age,region"
FIELDS = {
    "unit": "id",
    "time": "week",
    "adoption": "first",
    "instrument": "z",
    "running_variable": "r",
    "cutoff": 0.0,
}


def block(**changes) -> causal.CausalSpec:
    values = {
        "treatment": "t",
        "outcome": "y",
        "estimand": "ATE",
        "assumptions": (COVARIATES, "positivity", "no_interference"),
    }
    values.update(changes)
    return causal.CausalSpec(**values)


def test_the_first_eligible_strategy_is_primary_and_the_rest_the_multiverse():
    spec = block(
        strategies=("did", "backdoor_aipw", "randomized_diff"),
        assumptions=(
            COVARIATES,
            "positivity",
            "no_interference",
            "randomized_assignment",
        ),
    )
    plan = select.select(spec)
    assert plan.status == select.IDENTIFIED
    assert plan.primary == "backdoor_aipw"
    assert plan.multiverse == ("randomized_diff",)
    did = plan.verdict("did")
    assert not did.eligible and "estimand ATE" in did.reasons[0]


def test_a_missing_assumption_or_design_field_makes_a_strategy_ineligible():
    spec = block(
        estimand="LATE",
        strategies=("iv",),
        assumptions=("instrument_relevance", "exclusion"),
    )
    plan = select.select(spec)
    verdict = plan.verdict("iv")
    assert verdict.missing_assumptions == ("independence", "monotonicity")
    assert verdict.missing_design == ("instrument",)
    assert plan.status == select.NOT_IDENTIFIED
    report = select.refusal(plan, spec)
    assert report["estimate"] is None
    assert report["action"] == "refuse_claim"
    assert "associational" in report["note"]
    assert plan.to_dict()["estimate"] is None


def test_unconfoundedness_without_covariates_is_missing():
    spec = block(assumptions=("positivity", "no_interference"))
    verdict = select.select(spec).verdict("backdoor_aipw")
    assert verdict.missing_assumptions == ("unconfoundedness_given:<columns>",)


def test_an_unlisted_strategy_name_is_ineligible():
    plan = select.select(block(strategies=("magic",)))
    assert plan.status == select.NOT_IDENTIFIED
    assert plan.verdict("magic").reasons == ("unknown strategy magic",)


def test_no_chartered_order_uses_the_registry_order_and_says_so():
    plan = select.select(block())
    assert not plan.chartered_order
    assert plan.order == tuple(select.STRATEGIES)
    assert plan.to_dict()["order_chartered"] is False


def test_a_declared_dag_checks_the_adjustment_set():
    confounded = dag_module.Dag.build(
        source="test",
        edges=(("age", "t"), ("age", "y"), ("t", "y"), ("u", "t"), ("u", "y")),
        latent=("u",),
    )
    spec = block(
        assumptions=(
            "unconfoundedness_given:age",
            "positivity",
            "no_interference",
        )
    )
    plan = select.select(spec, confounded)
    assert plan.status == select.NOT_IDENTIFIED
    reason = plan.verdict("backdoor_aipw").reasons[0]
    assert "not a valid back-door set" in reason
    open_graph = dag_module.Dag.build(
        source="test",
        edges=(("age", "t"), ("age", "y"), ("t", "y")),
    )
    good = select.select(spec, open_graph)
    assert good.primary == "backdoor_aipw"
    assert good.verdict("backdoor_aipw").checked
    assert not select.select(spec).verdict("backdoor_aipw").checked
    declared = block(dag="dag.json", assumptions=spec.assumptions)
    assert not select.select(declared).verdict("backdoor_aipw").eligible


def random_spec(rng: np.random.Generator, eligible: bool) -> causal.CausalSpec:
    """A charter that matches one strategy, or one that matches none."""
    names = list(select.STRATEGIES)
    name = names[int(rng.integers(len(names)))]
    strategy = select.STRATEGIES[name]
    words = [
        COVARIATES if w == "unconfoundedness_given" else w
        for w in strategy.assumptions
    ]
    fields = {k: v for k, v in FIELDS.items() if k in strategy.design}
    if not eligible:
        pool = [("assumption", i) for i in range(len(words))] + [
            ("design", name) for name in strategy.design
        ]
        kind, index = pool[int(rng.integers(len(pool)))]
        if kind == "assumption":
            words.pop(index)
        else:
            fields.pop(index)
    return causal.CausalSpec(
        treatment="t",
        outcome="y",
        estimand=strategy.estimands[0],
        assumptions=tuple(words),
        strategies=(name,),
        **fields,
    )


def test_ineligible_charters_all_refuse_and_eligible_ones_do_not():
    rng = np.random.default_rng(10)
    for _ in range(200):
        plan = select.select(random_spec(rng, eligible=False))
        assert plan.status == select.NOT_IDENTIFIED
        assert plan.to_dict()["estimate"] is None
    for _ in range(200):
        plan = select.select(random_spec(rng, eligible=True))
        assert plan.status == select.IDENTIFIED


def test_the_selector_takes_no_data_and_imports_no_estimator():
    signature = inspect.signature(select.select)
    assert list(signature.parameters) == ["spec", "dag"]
    tree = ast.parse(inspect.getsource(select))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{a.name}" for a in node.names)
    for banned in ("numpy", "pandas", "scipy", "sklearn"):
        assert not any(m.startswith(banned) for m in imported)
    assert not any(m.startswith("scmle.data.identification") for m in imported)


@pytest.mark.parametrize("estimand", causal.CAUSAL_ESTIMANDS)
def test_every_estimand_has_a_strategy_that_answers_it(estimand):
    assert any(estimand in s.estimands for s in select.STRATEGIES.values())
