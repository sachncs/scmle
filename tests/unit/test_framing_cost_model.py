import math

import numpy as np
import pytest

from scmle.data.charter import contracts
from scmle.data.framing import costs

HUMAN = (costs.CostSource.HUMAN,) * 4


def model(c00=0.0, c01=5.0, c10=1.0, c11=0.0, lo=0.05, hi=0.5):
    return costs.DecisionCostModel(
        c00, c01, c10, c11, lo, hi, HUMAN, costs.CostSource.HUMAN
    )


def reference_nb(y, a, t):
    """Independent loop implementation of the Vickers-Elkin definition."""
    n = len(y)
    tp = fp = 0
    for yi, ai in zip(y, a, strict=True):
        if ai == 1 and yi == 1:
            tp += 1
        elif ai == 1 and yi == 0:
            fp += 1
    return tp / n - fp / n * (t / (1 - t))


def test_identity_over_random_cost_matrices():
    worst = 0.0
    for draw in range(10_000):
        rng = np.random.default_rng(draw)
        c00, c11 = rng.normal(size=2)
        c_fp, c_fn = rng.uniform(0.1, 5.0, size=2)
        m = costs.DecisionCostModel(
            c00,
            c11 + c_fn,
            c00 + c_fp,
            c11,
            0.001,
            0.999,
            HUMAN,
            costs.CostSource.HUMAN,
        )
        y = (rng.random(40) < rng.uniform(0.05, 0.95)).astype(float)
        a = (rng.random(40) < 0.5).astype(float)
        direct = costs.expected_cost(y, a, m)
        via = costs.cost_constant(y, m) - m.c_fn * costs.net_benefit(
            y, a, m.threshold
        )
        worst = max(worst, abs(direct - via))
    assert worst <= 1e-9


def test_net_benefit_fixed_inputs_match_reference_and_hand_value():
    y = [1, 1, 0, 0, 1]
    a = [1, 0, 1, 0, 1]
    assert costs.net_benefit(y, a, 0.25) == pytest.approx(1 / 3)
    for t in (0.05, 0.1, 0.25, 0.5, 0.9):
        assert costs.net_benefit(y, a, t) == pytest.approx(
            reference_nb(y, a, t), abs=1e-9
        )
        assert costs.net_benefit_all(y, t) == pytest.approx(
            reference_nb(y, [1] * 5, t), abs=1e-9
        )
    assert costs.net_benefit_none() == 0.0


def test_decision_curve_rethresholds_and_matches_reference():
    rng = np.random.default_rng(2)
    y = (rng.random(200) < 0.3).astype(int)
    scores = rng.random(200)
    grid = [0.1, 0.2, 0.4]
    curve = costs.decision_curve(y, scores, grid)
    for t, got in zip(grid, curve["model"], strict=True):
        rule = [int(s >= t) for s in scores]
        assert got == pytest.approx(reference_nb(list(y), rule, t), abs=1e-9)
    assert len(costs.net_benefit_curve(y, scores > 0.5, grid)["all"]) == 3


def test_dominated_action_is_refused():
    with pytest.raises(costs.DominatedActionError):
        model(c10=0.0)  # c_FP = 0
    with pytest.raises(costs.DominatedActionError):
        model(c01=-1.0)  # c_FN < 0


def test_range_must_contain_the_implied_threshold():
    assert model().threshold == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match="outside"):
        model(lo=0.3, hi=0.5)
    with pytest.raises(ValueError, match="0 < t_lo"):
        model(lo=0.0)


def test_round_trip_and_unverified_entries():
    m = model()
    assert costs.DecisionCostModel.from_dict(m.to_dict()) == m
    assert m.unverified_entries == ()
    data = costs.DecisionCostModel(
        0,
        5,
        1,
        0,
        0.05,
        0.5,
        (costs.CostSource.DATA, *HUMAN[1:]),
        costs.CostSource.TEMPLATE,
    )
    assert data.unverified_entries == ("c_00",)
    assert data.model_hash != m.model_hash
    with pytest.raises(ValueError):
        costs.DecisionCostModel.from_dict({**m.to_dict(), "extra": 1})


@pytest.mark.parametrize("bad", [[0, 2], [], [[0, 1]]])
def test_vectors_are_validated(bad):
    with pytest.raises(ValueError):
        costs.net_benefit(bad, bad, 0.2)
    with pytest.raises(ValueError):
        costs.net_benefit([0, 1], [0, 1, 1], 0.2)
    with pytest.raises(ValueError):
        costs.odds(1.0)
    assert math.isclose(costs.odds(0.5), 1.0)


def test_charter_cost_model_round_trip_and_old_charters_keep_their_hash():
    record = contracts.template()
    record["question"] = "q"
    record["estimand"].update(
        population="p",
        target="y",
        prediction_moment="m",
        summary_measure="nb",
        intercurrent_events=[{"event": "none", "strategy": "not applicable"}],
    )
    record["unit"] = "u"
    record["deployment_population"] = "d"
    old = contracts.AnalysisCharter.from_dict(record)
    assert "cost_model" not in old.to_dict()
    assert old.cost_model is None
    new = contracts.AnalysisCharter.from_dict(
        {**record, "cost_model": model().to_dict()}
    )
    assert new.cost_model == model()
    assert new.charter_hash != old.charter_hash
    again = contracts.AnalysisCharter.from_dict(new.to_dict())
    assert again.charter_hash == new.charter_hash
    with pytest.raises(ValueError):
        contracts.AnalysisCharter.from_dict({**record, "bogus": 1})
