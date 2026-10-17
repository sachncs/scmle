"""Missing-value flags are planned only when gaps carry signal."""

import json

import numpy as np
import pandas as pd
import support

from mml.contracts import tasks


def frame(informative, rows=1500, seed=0):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {"x": rng.normal(size=rows), "gappy": rng.normal(size=rows)}
    )
    data["y"] = (data.x + rng.normal(0, 1.0, rows) > 0).astype(int)
    if informative:
        gaps = (data.y == 1) & (rng.random(rows) < 0.6)
    else:
        gaps = rng.random(rows) < 0.25
    data.loc[gaps, "gappy"] = np.nan
    return data


def run(informative, tmp_path):
    result = support.make_project(
        frame(informative),
        "y",
        time_budget=300,
        output_dir=tmp_path,
        strategies=tasks.StrategySpec(names=["missing_indicators"]),
        models=tasks.ModelSpec(exclude_families=["deep_learning"]),
    ).run()
    log = json.loads(
        (result.run_dir / "diagnostics/decision_log.json").read_text()
    )
    return result, [
        o
        for o in log["outcomes"]
        if o["technique"] == "missing_indicators@logistic_regression"
    ]


def test_informative_gaps_are_reported_and_the_flags_are_not_ruled_out(
    tmp_path,
):
    result, outcomes = run(True, tmp_path)

    assert result.profile.feature_plan.informative_missing == ("gappy",)
    assert any(
        "missingness of 'gappy'" in finding
        for finding in result.profile.quality_findings
    )
    assert not any("associated" in o.get("reason", "") for o in outcomes)


def test_random_gaps_do_not_earn_missing_value_flags(tmp_path):
    result, outcomes = run(False, tmp_path)

    assert result.profile.feature_plan.informative_missing == ()
    assert outcomes and all(
        o["action"] == "reject" and "associated with the target" in o["reason"]
        for o in outcomes
    )
