import numpy
import pandas
import pytest

from scmle.data.framing import baseline
from scmle.data.framing import costs
from scmle.data.framing import draft

HUMAN = (costs.CostSource.HUMAN,) * 4
MODEL = costs.DecisionCostModel(
    0.0, 5.0, 1.0, 0.0, 0.1, 0.4, HUMAN, costs.CostSource.HUMAN
)


def test_framing_reads_only_the_rows_it_is_given():
    """Rows outside the sample cannot influence the draft."""
    rng = numpy.random.default_rng(0)
    x = pandas.DataFrame(
        {
            "id": numpy.arange(200),
            "as_of": pandas.date_range("2024-01-01", periods=200, freq="h"),
            "y": (rng.random(200) < 0.3).astype(int),
        }
    )
    source = draft.SourceDescription("u://x", tuple(x.columns), ("id",))
    request = draft.Request("predict target y", source)
    first = draft.frame(request, x, cost_model=MODEL)
    sealed = pandas.concat([x, x.assign(y=1 - x["y"])], ignore_index=True)
    assert sealed.shape[0] == 400  # the sealed rows exist but are not passed
    second = draft.frame(request, x, cost_model=MODEL)
    assert first.draft_hash == second.draft_hash


def test_baseline_claim_cannot_be_evaluated_twice_or_unregistered():
    rule = {"r": 1}
    guard = baseline.Evaluator(baseline.register(MODEL, rule, 0.05))
    y = numpy.array([1.0, 0.0] * 40)
    a = numpy.array([1.0, 0.0] * 40)
    b = numpy.zeros(80)
    clusters = numpy.arange(80) % 25
    guard.evaluate(y, a, b, clusters, rule=rule, model=MODEL)
    with pytest.raises(baseline.BaselineError):
        guard.evaluate(y, a, b, clusters, rule=rule, model=MODEL)
    fresh = baseline.Evaluator(baseline.register(MODEL, rule, 0.05))
    with pytest.raises(baseline.BaselineError):
        fresh.evaluate(y, a, b, clusters, rule={"r": 2}, model=MODEL)
    assert not fresh.used
