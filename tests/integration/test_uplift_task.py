"""Treatment effects are estimated, ranked and caveated."""

import numpy
import pandas
import pytest
import support

from scmle.contracts import errors


def trial(rows=5000, seed=0, confounded=False):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {
            "age": rng.normal(size=rows),
            "tenure": rng.normal(size=rows),
            "noise": rng.normal(size=rows),
        }
    )
    effect = 0.35 * (frame.age > 0)
    if confounded:
        arm = ((frame.tenure + rng.normal(0, 0.3, rows)) > 0).astype(int)
    else:
        arm = rng.integers(0, 2, rows)
    frame["treated"] = arm
    frame["bought"] = (rng.random(rows) < 0.25 + arm * effect).astype(int)
    frame.attrs["effect"] = effect
    return frame, effect


def run(tmp_path, frame):
    return support.make_project(
        frame,
        "bought",
        task="uplift",
        treatment_column="treated",
        time_budget=300,
        output_dir=tmp_path,
    ).run()


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    frame = trial()[0]
    return run(tmp_path_factory.mktemp("uplift"), frame)


def test_estimated_uplift_tracks_the_true_effect_in_a_randomized_trial(result):
    fresh, effect = trial(3000, seed=7)

    estimated = result.bundle.predict(fresh.drop(columns=["bought"]))

    assert numpy.corrcoef(estimated, effect)[0, 1] > 0.5
    assert result.evaluation["test_metrics"]["qini"] > 0.01


def test_the_treatment_is_never_a_feature(result):
    assert "treated" not in result.bundle.features


def test_the_report_states_the_assumption_and_the_balance_check(result):
    report = result.evaluation["uplift"]

    assert "unmeasured confounding" in report["assumption"]
    assert report["propensity"]["randomized"]
    assert report["qini_curve"]["rows"]
    assert any(
        "treatment assignment" in f for f in result.profile.quality_findings
    )


def test_confounded_assignment_is_flagged(tmp_path):
    frame = trial(confounded=True)[0]

    result = run(tmp_path, frame)

    assert not result.evaluation["uplift"]["propensity"]["randomized"]
    assert any("not random" in f for f in result.profile.quality_findings)


def test_a_non_binary_treatment_is_refused(tmp_path):
    frame = trial(500)[0]
    frame["treated"] = frame["treated"] * 2

    with pytest.raises(errors.BlockedError, match=r"0 \(control\) or 1"):
        run(tmp_path, frame)


def test_uplift_needs_a_treatment_column_and_others_forbid_one(tmp_path):
    frame = trial(200)[0]
    with pytest.raises(ValueError, match="treatment_column"):
        support.make_project(
            frame, "bought", task="uplift", output_dir=tmp_path
        )
    with pytest.raises(ValueError, match="treatment_column"):
        support.make_project(
            frame,
            "bought",
            task="binary_classification",
            treatment_column="treated",
            output_dir=tmp_path,
        )
