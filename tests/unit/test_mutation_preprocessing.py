"""The preprocessing helpers' options, front end, wrapping and messages."""

from sklearn import pipeline

from scmle.contracts import data
from scmle.contracts import tasks
from scmle.data import feature_graph
import scmle.models.registry
from scmle.preparation import multitarget
from scmle.preparation import preprocessing

PLAN = data.Blueprint(("x", "y"), ())


class Assembler:
    """A family that assembles its own model: the parameters it was given."""

    name = "assembler"

    def assemble(self, plan, task, seed, params):
        return params


def test_ordinal_levels_join_the_given_parameters() -> None:
    plan = data.Blueprint(("x",), (), ordinal_levels=3)
    found = preprocessing.instantiate(
        plan, Assembler(), tasks.Task.ORDINAL, 0, {"C": 1.0}
    )
    assert found == {"C": 1.0, "levels": 3}


def test_the_options_carry_the_plan_embedding_model() -> None:
    plan = data.Blueprint(("x",), (), embedding_model="local/encoder")
    adapter = scmle.models.registry.Catalog.stock().get("ridge")
    options = preprocessing.default_options(plan, adapter)
    assert options.embedding_model == "local/encoder"


def test_derived_features_run_before_the_column_transformer() -> None:
    node = tasks.Node(name="z", op="square", inputs=["x"])
    plan = data.Blueprint(("x",), (), derived=(node,))
    adapter = scmle.models.registry.Catalog.stock().get("ridge")
    front = preprocessing.frontend(
        plan, preprocessing.default_options(plan, adapter)
    )
    assert isinstance(front, pipeline.Pipeline)
    assert [step[0] for step in front.steps] == ["graph", "prep"]
    assert isinstance(front.steps[0][1], feature_graph.Transformer)
    assert front.steps[0][1].nodes == (node,)


def test_several_numeric_targets_are_fitted_as_multioutput() -> None:
    plan = data.Blueprint(("x",), (), target_columns=("a", "b"))
    found = preprocessing.wrap("model", plan, tasks.Task.REGRESSION)
    assert type(found) is multitarget.Multioutput


def test_missing_indicators_out_of_plan_order_are_named() -> None:
    plan = data.Blueprint(
        ("x", "y"), (), monotonic={"x": 1}, informative_missing=("y",)
    )
    assert preprocessing.misalignment(plan, False, "indicator") == (
        "missing indicators move the flagged numeric columns ahead of the "
        "others"
    )
    assert preprocessing.misalignment(PLAN, False, "indicator") is None
