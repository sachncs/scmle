import numpy as np
import pandas as pd
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
    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        {
            "id": np.arange(200),
            "as_of": pd.date_range("2024-01-01", periods=200, freq="h"),
            "y": (rng.random(200) < 0.3).astype(int),
        }
    )
    source = draft.SourceDescription("u://x", tuple(x.columns), ("id",))
    request = draft.FramingRequest("predict target y", source)
    first = draft.frame(request, x, cost_model=MODEL)
    sealed = pd.concat([x, x.assign(y=1 - x["y"])], ignore_index=True)
    assert sealed.shape[0] == 400  # the sealed rows exist but are not passed
    second = draft.frame(request, x, cost_model=MODEL)
    assert first.draft_hash == second.draft_hash


def test_baseline_claim_cannot_be_evaluated_twice_or_unregistered():
    rule = {"r": 1}
    guard = baseline.BaselineGuard(baseline.register(MODEL, rule, 0.05))
    y = np.array([1.0, 0.0] * 40)
    a = np.array([1.0, 0.0] * 40)
    b = np.zeros(80)
    clusters = np.arange(80) % 25
    guard.evaluate(y, a, b, clusters, rule=rule, model=MODEL)
    with pytest.raises(baseline.BaselineError):
        guard.evaluate(y, a, b, clusters, rule=rule, model=MODEL)
    fresh = baseline.BaselineGuard(baseline.register(MODEL, rule, 0.05))
    with pytest.raises(baseline.BaselineError):
        fresh.evaluate(y, a, b, clusters, rule={"r": 2}, model=MODEL)
    assert not fresh.used
