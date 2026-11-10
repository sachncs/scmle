"""Cross-validation uncertainty accounts for fold dependence."""

import pytest

from scmle.evaluation import confidence


def test_corrected_error_is_wider_than_the_naive_one():
    scores = [0.80, 0.82, 0.78, 0.81, 0.79]
    result = confidence.cv_interval(scores, 400, 100, "random")
    assert result["corrected_standard_error"] > result["naive_standard_error"]
    low, high = result["interval"]
    assert low < result["mean"] < high
    assert result["mean"] == pytest.approx(0.80)
    ratio = result["corrected_standard_error"] / result["naive_standard_error"]
    assert ratio == pytest.approx((1 + 5 * 100 / 400) ** 0.5)


def test_temporal_validation_is_flagged_as_optimistic():
    scores = [0.7, 0.72, 0.69]
    note = confidence.cv_interval(scores, 300, 100, "temporal")["note"]
    assert "optimistic" in note


def test_a_single_fold_has_no_interval():
    assert (
        confidence.cv_interval([0.9], 100, 10, "random")["available"] is False
    )


LEADER = [0.80, 0.82, 0.78, 0.81, 0.79]
RIVALS = [[0.70, 0.90, 0.72, 0.88, 0.71], [0.90, 0.69, 0.91, 0.70, 0.89]]


def test_one_candidate_leaves_the_interval_centred_on_the_mean():
    plain = confidence.cv_interval(LEADER, 400, 100, "random")
    alone = confidence.cv_interval(LEADER, 400, 100, "random", [LEADER])
    assert plain["interval"] == alone["interval"]
    assert alone["selection_optimism"] == 0.0 and alone["candidates"] == 1
    assert confidence.selection_optimism(LEADER, [], True) == 0.0


def test_the_optimism_is_the_gap_to_choosing_without_the_scored_fold():
    # for each fold, pick the best mean over the other four folds and take
    # its score on the held fold
    every = [LEADER, *RIVALS]
    honest = []
    for fold in range(5):
        others = [
            sum(s for i, s in enumerate(scores) if i != fold) / 4
            for scores in every
        ]
        honest.append(every[others.index(max(others))][fold])
    expected = max(sum(LEADER) / 5 - sum(honest) / 5, 0.0)
    assert expected > 0
    assert confidence.selection_optimism(LEADER, every, True) == (
        pytest.approx(expected)
    )


def test_the_interval_moves_by_the_optimism_and_keeps_its_width():
    every = [LEADER, *RIVALS]
    plain = confidence.cv_interval(LEADER, 400, 100, "random")
    moved = confidence.cv_interval(LEADER, 400, 100, "random", every, True)
    shift = moved["selection_optimism"]
    assert shift > 0 and moved["candidates"] == 3
    assert moved["interval"][0] == pytest.approx(plain["interval"][0] - shift)
    assert moved["interval"][1] == pytest.approx(plain["interval"][1] - shift)
    assert moved["mean"] == plain["mean"]
    assert "chosen among 3 candidates" in moved["note"]


def test_a_loss_is_moved_the_other_way():
    losses = [0.20, 0.18, 0.22, 0.19, 0.21]
    rivals = [[1 - s for s in scores] for scores in RIVALS]
    every = [losses, *rivals]
    gain = confidence.cv_interval(
        LEADER, 400, 100, "random", [LEADER, *RIVALS], True
    )
    loss = confidence.cv_interval(losses, 400, 100, "random", every, False)
    plain = confidence.cv_interval(losses, 400, 100, "random")
    assert gain["selection_optimism"] > 0
    assert loss["selection_optimism"] > 0
    assert loss["interval"][0] > plain["interval"][0]


def test_candidates_with_another_number_of_folds_are_ignored():
    odd = [0.5, 0.9, 0.5]
    assert confidence.selection_optimism(LEADER, [LEADER, odd], True) == 0.0


def test_runs_report_the_interval(binary_frame, tmp_path):
    import support

    result = support.make_project(binary_frame, "y", output_dir=tmp_path).run()
    record = result.evaluation["cv_confidence"]
    assert record["available"] and record["folds"] == 5
    assert record["interval"][0] <= result.evaluation["selected_cv_score"]
    finished = [t for t in result.trials if t.status.value == "ok"]
    assert record["candidates"] == len(finished) > 1
    assert record["selection_optimism"] >= 0.0
