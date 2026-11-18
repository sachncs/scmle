"""Behaviour of each module of the continuous evaluation, one at a time."""

import hashlib
import json

import numpy
import pandas
import pytest

from scmle.contracts import errors
from scmle.deploy import release_state
from scmle.governance import deployment_ledger
from scmle.operations.continuous import evaluation
from scmle.operations.continuous import feeding
from scmle.operations.continuous import policies
from scmle.operations.continuous import promotion
from scmle.operations.continuous import records
from scmle.operations.continuous import retraining
from tests import support_continuous
from tests import support_release
from tests import support_runtime


def build(tmp_path, **kwargs):
    loop = support_continuous.make_loop(tmp_path)
    return loop, support_continuous.evaluation(tmp_path, loop, **kwargs)


# ---- records ---------------------------------------------------------------


def test_record_file_names_are_the_documented_ones():
    assert records.HISTORY_FILE == "training_history.jsonl"
    assert records.LABELS_FILE == "retained_outcomes.jsonl"
    assert records.DATASET_DIRECTORY == "datasets"


def test_serialize_sorts_columns_and_drops_the_index():
    frame = pandas.DataFrame({"b": [1, 2], "a": ["x", "y"]}, index=[5, 6])
    assert records.serialize(frame) == b"a,b\nx,1\ny,2\n"


def test_a_dataset_version_is_the_sha256_of_its_canonical_csv():
    frame = pandas.DataFrame({"b": [1], "a": [2]})
    expected = hashlib.sha256(b"a,b\n2,1\n").hexdigest()
    assert records.dataset_version(frame) == "sha256:" + expected
    assert records.dataset_version(frame[["a", "b"]]) == "sha256:" + expected


def test_load_dataset_reads_a_preserved_copy_and_checks_it(tmp_path):
    frame = pandas.DataFrame({"a": [0.1, 2.5], "b": ["x", "y"]})
    version = records.dataset_version(frame)
    folder = tmp_path / "datasets"
    folder.mkdir()
    path = folder / (version.replace(":", "-") + ".csv")
    path.write_bytes(records.serialize(frame))
    loaded = records.load_dataset(tmp_path, version)
    assert records.dataset_version(loaded) == version
    path.write_text("a,b\n0.1,x\n2.5,z\n")
    with pytest.raises(errors.InputError, match=f"stored copy of {version}"):
        records.load_dataset(tmp_path, version)


def test_a_new_history_is_empty(tmp_path):
    history = records.TrainingHistory(tmp_path / "h.jsonl")
    assert history.rows() == []
    assert history.select("training") == []


def test_history_rows_chain_from_a_zero_hash(tmp_path):
    history = records.TrainingHistory(tmp_path / "h.jsonl")
    first = history.record("training", 12.34567, model="m", when=object)
    second = history.record("review", 13.0)
    assert first["seq"] == 0
    assert first["at"] == 12.346
    assert first["prev"] == "0" * 64
    assert first["facts"]["model"] == "m"
    assert first["facts"]["when"] == str(object)
    content = {k: first[k] for k in ("seq", "at", "kind", "facts", "prev")}
    assert (
        first["hash"]
        == hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()
    )
    assert second["seq"] == 1
    assert second["prev"] == first["hash"]
    assert history.rows() == [first, second]
    assert history.select("review") == [second]


@pytest.mark.parametrize(
    "edit",
    [
        lambda rows: rows[1].update(seq=5),
        lambda rows: rows[1].update(prev="1" * 64),
        lambda rows: rows[0].update(at=99.0),
    ],
)
def test_an_edited_history_is_broken(tmp_path, edit):
    history = records.TrainingHistory(tmp_path / "h.jsonl")
    history.record("a", 1.0)
    history.record("b", 2.0)
    rows = history.rows()
    edit(rows)
    history.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(
        errors.InputError, match="training history broken at row"
    ):
        history.rows()


# ---- promotion -------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"kind": "other"}, "kind must be operational or business"),
        ({"better": "more"}, "better must be higher or lower"),
        ({"margin": -0.1}, "margin must be finite and not negative"),
        ({"margin": float("inf")}, "margin must be finite and not negative"),
    ],
)
def test_an_indicator_refuses_bad_choices(changes, message):
    values = {"name": "m", "kind": "business", "better": "higher"}
    values.update(changes)
    with pytest.raises(errors.InputError, match=message):
        promotion.Indicator(**values)


def test_an_indicator_accepts_a_zero_margin():
    made = promotion.Indicator("m", "operational", "lower")
    assert made.margin == 0.0
    assert made.primary is False


def test_contrast_brackets_a_clear_shift():
    before = numpy.zeros(50)
    after = numpy.ones(50)
    assert promotion.contrast(before, after, 0.05, 200, 0) == (1.0, 1.0)
    low, high = promotion.contrast(
        numpy.arange(50.0), numpy.arange(50.0) + 10, 0.05, 500, 1
    )
    assert low < 10.0 < high


def spec(name="m", better="higher", margin=0.0, primary=True):
    return promotion.Indicator(name, "business", better, margin, primary)


def test_judge_needs_enough_samples_in_each_period():
    result = promotion.judge(
        spec(), numpy.ones(29), numpy.ones(40), 0.05, 30, 100, 0
    )
    assert result == {
        "metric": "m",
        "kind": "business",
        "verdict": "inconclusive",
        "reason": "fewer than 30 observations in a period",
        "samples_before": 29,
        "samples_after": 40,
        "interval": None,
    }


@pytest.mark.parametrize(
    ("better", "shift", "margin", "verdict"),
    [
        ("higher", 1.0, 0.0, "improved"),
        ("higher", -1.0, 0.0, "worse"),
        ("lower", -1.0, 0.0, "improved"),
        ("lower", 1.0, 0.0, "worse"),
        ("higher", 1.0, 1.0, "inconclusive"),
        ("higher", -1.0, 1.0, "inconclusive"),
        ("higher", 0.0, 0.0, "inconclusive"),
    ],
)
def test_judge_reads_the_interval_against_the_margin(
    better, shift, margin, verdict
):
    old = numpy.zeros(30)
    new = numpy.zeros(30) + shift
    result = promotion.judge(
        spec(better=better, margin=margin), old, new, 0.05, 30, 100, 0
    )
    assert result["verdict"] == verdict
    assert result["interval"] == [shift, shift]
    assert result["confidence"] == 0.95
    assert result["descriptive_difference"] == shift
    assert result["samples_before"] == 30


def test_post_promotion_check_splits_alpha_and_ranks_the_verdicts():
    specs = [spec("a"), spec("b", primary=False)]
    before = {"a": [0.0] * 30, "b": [0.0] * 30}
    result = promotion.post_promotion_check(
        specs, before, {"a": [1.0] * 30, "b": [0.0] * 30}, resamples=50
    )
    assert result["overall"] == "helped"
    assert [m["confidence"] for m in result["metrics"]] == [0.975, 0.975]
    assert "joint confidence 95% over 2 metrics" in result["statement"]
    harmed = promotion.post_promotion_check(
        specs, before, {"a": [1.0] * 30, "b": [-1.0] * 30}, resamples=50
    )
    assert harmed["overall"] == "harmed"
    no_primary = promotion.post_promotion_check(
        [spec("b", primary=False)], before, {"b": [1.0] * 30}, resamples=50
    )
    assert no_primary["overall"] == "inconclusive"


def test_post_promotion_check_needs_metrics_and_values():
    with pytest.raises(errors.InputError, match="name at least one metric"):
        promotion.post_promotion_check([], {}, {})
    with pytest.raises(errors.InputError, match="no values for m in both"):
        promotion.post_promotion_check([spec()], {"m": [1.0]}, {})


def test_reserve_is_a_stable_keyed_hash():
    units = [f"u{i}" for i in range(2000)]
    held = promotion.reserve(units, 0.2, "rel-1")
    assert held.dtype == bool
    assert numpy.array_equal(held, promotion.reserve(units, 0.2, "rel-1"))
    assert not numpy.array_equal(held, promotion.reserve(units, 0.2, "rel-2"))
    assert 0.17 < held.mean() < 0.23
    draw = (
        int.from_bytes(hashlib.sha256(b"rel-1:u0").digest()[:8], "big")
        / 2.0**64
    )
    assert held[0] == (draw < 0.2)
    for bad in (0.0, 1.0, -0.5):
        with pytest.raises(errors.InputError, match="strictly between 0 and 1"):
            promotion.reserve(units, bad, "s")


def test_balance_tests_the_observed_split():
    specs = [spec("a")]
    fair = promotion.balance(
        specs, {"a": [0.0] * 50}, {"a": [0.0] * 50}, 0.5, 0.001
    )
    assert fair["holdback_units"] == 50
    assert fair["promoted_units"] == 50
    assert fair["p_value"] == 1.0
    assert fair["flag"] is False
    skewed = promotion.balance(
        specs, {"a": [0.0] * 10}, {"a": [0.0] * 90}, 0.5, 0.001
    )
    assert skewed["flag"] is True
    empty = promotion.balance([], {}, {}, 0.5, 0.001)
    assert (empty["holdback_units"], empty["p_value"]) == (0, 1.0)


def test_attribute_says_whether_the_split_matched():
    matched = promotion.attribute(
        "helped", 0.05, 2, {"flag": False, "p_value": 0.5}
    )
    assert matched.startswith("Overall: helped.")
    assert "joint confidence 95% over 2 metrics" in matched
    assert "attributable to the model" in matched
    flagged = promotion.attribute(
        "inconclusive", 0.05, 1, {"flag": True, "p_value": 0.0001}
    )
    assert "(p = 0.0001)" in flagged
    assert flagged.endswith("metrics that were not selected.")


def test_holdback_check_caps_a_mismatched_split():
    specs = [spec("a")]
    result = promotion.holdback_check(
        specs,
        {"a": [0.0] * 30},
        {"a": [1.0] * 30},
        expected_fraction=0.5,
        resamples=50,
    )
    assert result["overall"] == "helped"
    assert result["design"] == "concurrent_holdback"
    assert result["ratio_check"]["flag"] is False
    capped = promotion.holdback_check(
        specs,
        {"a": [0.0] * 30},
        {"a": [1.0] * 300},
        expected_fraction=0.5,
        resamples=50,
    )
    assert capped["overall"] == "inconclusive"
    assert capped["metrics"][0]["verdict"] == "improved"
    with pytest.raises(errors.InputError, match="expected_fraction must lie"):
        promotion.holdback_check(specs, {}, {}, expected_fraction=1.0)


# ---- feeding ---------------------------------------------------------------


def test_feed_turns_capped_costs_into_losses_and_retains_them(tmp_path):
    loop, ev = build(tmp_path)
    assert ev.feed({"x": 5.0}, 10.0) == 0
    line = json.loads((tmp_path / "ops" / records.LABELS_FILE).read_text())
    assert line == {
        "at": 1000.0,
        "batch": loop.batch,
        "kind": "business_outcomes",
        "values": {"x": 5.0},
    }


@pytest.mark.parametrize(
    ("outcomes", "cap"), [({"x": 1.0}, 0.0), ({"x": -1.0}, 1.0)]
)
def test_feed_refuses_a_bad_cap_or_a_negative_outcome(tmp_path, outcomes, cap):
    ignored, ev = build(tmp_path)
    with pytest.raises(errors.InputError, match="cap must be positive"):
        ev.feed(outcomes, cap)


def test_absorbed_labels_match_pending_cases(tmp_path):
    loop, ev = build(tmp_path)
    frame = pandas.DataFrame({"case_id": ["a", "b", "c"]})
    loop.observe(frame, numpy.zeros(3), ["approve"] * 3)
    matched = ev.absorb({"a": 1, "b": 0, "c": 1}, {"a": 1, "b": 1, "z": 0})
    assert matched == 2
    kept = json.loads((tmp_path / "ops" / records.LABELS_FILE).read_text())
    assert kept["kind"] == "labels"
    assert kept["values"] == {"a": 1, "b": 1, "z": 0}


def test_record_training_preserves_the_dataset_once(tmp_path):
    ignored, ev = build(tmp_path)
    frame = pandas.DataFrame({"b": [1.5, 2.5], "a": ["x", "y"]})
    row = ev.record_training("m", frame, {"auc": 0.9}, "first")
    version = records.dataset_version(frame)
    assert row["kind"] == "training"
    assert row["facts"] == {
        "model": "m",
        "dataset_version": version,
        "rows": 2,
        "columns": ["a", "b"],
        "metrics": {"auc": 0.9},
        "reason": "first",
        "preserved": True,
    }
    saved = tmp_path / "ops" / "datasets" / (version.replace(":", "-") + ".csv")
    assert saved.read_bytes() == records.serialize(frame)
    ev.record_training("m", frame, {}, "again")
    assert len(list(saved.parent.iterdir())) == 1
    other = ev.record_training("m", frame.head(1), {}, "x", preserve=False)
    assert other["facts"]["preserved"] is False
    assert len(list(saved.parent.iterdir())) == 1


def test_a_fresh_loop_has_no_nominations(tmp_path):
    ignored, ev = build(tmp_path)
    assert ev.nominations() == []
    assert ev.advance() is not None


# ---- retraining ------------------------------------------------------------


def test_a_quiet_window_is_reviewed_without_a_proposal(tmp_path):
    loop, ev = build(tmp_path)
    decision = ev.review(support_continuous.loss_signals(0.05))
    assert decision.action == "none"
    assert decision.escalation_id is None
    assert decision.challenger is None
    rows = ev.history.select("retrain_review")
    assert len(rows) == 1
    assert rows[0]["facts"]["batch"] == loop.batch
    assert ev.find() is None


def test_a_breached_window_recommends_and_is_found_pending(tmp_path):
    loop, ev = build(tmp_path)
    decision = ev.review(support_continuous.loss_signals(0.4))
    assert decision.action == "recommend"
    assert decision.escalation_id == ev.find()
    assert ev.last_proposal_batch == loop.batch
    assert ev.history.select("retrain_decision")[0]["facts"]["action"] == (
        "recommend"
    )
    again = support_continuous.evaluation(tmp_path, loop)
    assert again.last_proposal_batch == loop.batch


def test_train_needs_a_hook_and_an_empty_queue(tmp_path):
    ev = build(tmp_path)[1]
    assert ev.train(("r",)) is None
    names = iter(["c1", None])
    loop2, hooked = build(tmp_path / "two", retrain=lambda: next(names))
    assert hooked.train(("why",)) == "c1"
    row = support_continuous.kinds(loop2, deployment_ledger.RETRAIN)[0]["body"]
    assert row == {
        "reason": "operations retraining policy",
        "mode": "recommend",
        "batch": loop2.batch,
        "challenger": "c1",
        "reasons": ["why"],
    }
    assert "c1" in loop2.ledger.registered
    assert hooked.train(("why",)) is None


def test_a_hook_that_starts_nothing_trains_nothing(tmp_path):
    loop, ev = build(tmp_path, retrain=lambda: None)
    assert ev.train(("r",)) is None
    assert not support_continuous.kinds(loop, deployment_ledger.RETRAIN)


def test_an_unanswered_retrain_cannot_run(tmp_path):
    ignored, ev = build(tmp_path, retrain=lambda: "c1")
    with pytest.raises(errors.InputError, match="esc-9 has no approving reply"):
        ev.run_approved_retrain("esc-9")


def test_retraining_builds_on_feeding():
    assert issubclass(retraining.Retraining, feeding.Feeding)
    assert issubclass(evaluation.ContinuousEvaluation, retraining.Retraining)
    assert policies.TRIGGER == "trigger"


# ---- evaluation ------------------------------------------------------------


def test_a_replacement_dumps_its_outcome():
    plain = evaluation.Replacement(False, ("no",))
    assert plain.to_dict() == {
        "done": False,
        "reasons": ["no"],
        "outcome": None,
    }
    outcome = release_state.Outcome(True, (), version=2, revision="r")
    made = evaluation.Replacement(True, (), outcome)
    assert made.to_dict()["outcome"] == outcome.to_dict()


def test_a_replacement_without_a_proposal_is_refused_and_recorded(tmp_path):
    ignored, ev = build(tmp_path)
    release, plane, store = support_continuous.release_world(tmp_path)
    good = support_continuous.candidate_for(store)
    refused = ev.replace("c1", release, plane, good, support_release.KEY)
    assert refused.done is False
    assert refused.reasons == ("the loop has no promotion proposal for 'c1'",)
    assert refused.outcome is None
    row = ev.history.select("replacement_refused")[0]
    assert row["facts"]["challenger"] == "c1"
    assert plane.calls == []


def test_an_unfavourable_comparison_is_a_reason(tmp_path):
    ignored, ev = build(tmp_path)
    release, plane, store = support_continuous.release_world(tmp_path)
    good = support_continuous.candidate_for(store)

    class Kept:
        decision = "keep"

    refused = ev.replace(
        "c1", release, plane, good, support_release.KEY, comparison=Kept()
    )
    assert "the offline comparison does not favour it" in refused.reasons


def test_a_proposed_challenger_is_replaced_after_confirmation(tmp_path):
    ignored, ev = support_continuous.winning_challenger(tmp_path)
    release, plane, store = support_continuous.release_world(tmp_path)
    release.promote(
        support_continuous.MODEL,
        1,
        plane,
        "staging",
        support_runtime.OLD,
        signing_key=support_release.KEY,
    )
    good = support_continuous.candidate_for(store)
    done = ev.replace("challenger-1", release, plane, good, support_release.KEY)
    assert done.done is True
    assert done.outcome.version == 2
    row = ev.history.select("replacement")[0]["facts"]
    assert row["environment"] == "staging"
    assert row["outcome"]["done"] is True


def test_followup_records_the_post_promotion_check(tmp_path):
    ignored, ev = build(tmp_path)
    result = ev.followup(
        [spec("a")], {"a": [0.0] * 30}, {"a": [1.0] * 30}, resamples=20
    )
    assert result["overall"] == "helped"
    row = ev.history.select("post_promotion_check")[0]
    assert row["facts"]["overall"] == "helped"
    assert row["at"] == 1000.0
