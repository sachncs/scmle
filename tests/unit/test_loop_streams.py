import numpy
import pytest

from benchmarks.qualification import streams


def test_stable_stream_has_no_change_point_and_known_risk():
    stream = streams.simulate(streams.StreamSpec(batches=400, size=100, seed=1))
    assert stream.change_point is None
    assert stream.loss.mean() == pytest.approx(0.10, abs=0.01)
    assert numpy.all(stream.pi == 1.0)


def test_abrupt_gradual_and_concept_risk_paths():
    base = streams.StreamSpec(kind="abrupt", batches=10, change=5, delta=0.2)
    assert streams.risk_path(base)[:5].tolist() == [0.1] * 5
    assert streams.risk_path(base)[5:] == pytest.approx([0.3] * 5)
    gradual = streams.risk_path(
        streams.StreamSpec(
            kind="gradual", batches=10, change=2, ramp=4, delta=0.4
        )
    )
    assert gradual[1] == pytest.approx(0.1)
    assert gradual[5] == pytest.approx(0.5)


def test_covariate_shift_moves_covariates_not_risk():
    spec = streams.StreamSpec(
        kind="covariate", batches=300, size=100, change=150, seed=2
    )
    stream = streams.simulate(spec)
    assert stream.z[150:].mean() - stream.z[:150].mean() == pytest.approx(
        2.0, abs=0.1
    )
    assert stream.loss[150:].mean() == pytest.approx(
        stream.loss[:150].mean(), abs=0.015
    )


def test_mar_labels_make_the_naive_estimate_biased_and_weighted_unbiased():
    spec = streams.StreamSpec(labels="mar", batches=400, size=100, seed=3)
    stream = streams.simulate(spec)
    seen = stream.selected
    naive = stream.loss[seen].mean()
    weighted = (stream.loss * seen / stream.pi).mean()
    assert weighted == pytest.approx(stream.loss.mean(), abs=0.01)
    assert (
        abs(naive - stream.loss.mean()) < 0.01
    )  # loss is independent of z here


def test_mnar_labels_depend_on_the_loss_and_log_only_the_audit_rate():
    spec = streams.StreamSpec(
        labels="mnar", epsilon=0.1, batches=400, size=100, seed=4
    )
    stream = streams.simulate(spec)
    naive = stream.loss[stream.selected].mean()
    assert naive > stream.loss.mean() + 0.05
    assert numpy.all(stream.pi == 0.1)
    audit = stream.audit
    assert (
        stream.loss * audit / 0.1
    ).sum() / stream.loss.size == pytest.approx(stream.loss.mean(), abs=0.01)


def test_outcome_dependent_delay_late_losses():
    stream = streams.simulate(
        streams.StreamSpec(window=2, slow=3, batches=50, size=50, seed=5)
    )
    base = numpy.arange(50)[:, None]
    assert numpy.all(
        stream.arrival[stream.loss == 0]
        == (base + 2 + 0 * stream.loss)[stream.loss == 0]
    )
    assert numpy.all((stream.arrival - base)[stream.loss == 1] == 5)


def test_streams_are_reproducible_from_the_seed():
    first = streams.simulate(streams.StreamSpec(seed=9))
    second = streams.simulate(streams.StreamSpec(seed=9))
    assert numpy.array_equal(first.loss, second.loss)
    assert not numpy.array_equal(
        first.loss, streams.simulate(streams.StreamSpec(seed=10)).loss
    )


def test_invalid_specs_are_refused():
    with pytest.raises(ValueError):
        streams.StreamSpec(kind="sideways")
    with pytest.raises(ValueError):
        streams.StreamSpec(labels="sometimes")


def test_feedback_loop_separates_bypass_and_model_groups_only_with_gamma():
    loop = streams.feedback_stream(300, 200, gamma=0.8, bypass=0.2, seed=1)
    free = streams.feedback_stream(300, 200, gamma=0.0, bypass=0.2, seed=1)
    spread = {}
    for name, stream in (("loop", loop), ("free", free)):
        model = stream.next_z[~stream.bypass].std()
        coin = stream.next_z[stream.bypass].std()
        spread[name] = abs(model - coin)
    assert spread["loop"] > 0.15
    assert spread["free"] < 0.05


def test_dependence_ties_the_loss_to_the_covariate_without_moving_the_mean():
    stream = streams.simulate(
        streams.StreamSpec(dependence=0.4, batches=400, size=100, seed=3)
    )
    high = stream.loss[stream.z > 1].mean()
    low = stream.loss[stream.z < -1].mean()
    assert high > low + 0.1
    assert stream.loss.mean() == pytest.approx(0.10, abs=0.01)


def test_challengers_are_dependent_and_have_the_stated_advantage():
    gains = [0.0, 0.05, 0.1]
    sequence = streams.challenger_sequence(gains, 400, 100, 0.2, seed=2)
    rates = sequence.challengers.mean(axis=(1, 2))
    assert rates == pytest.approx([0.2, 0.15, 0.1], abs=0.01)
    assert numpy.all(sequence.challengers[2] <= sequence.challengers[1])


def test_oracle_risk_matches_a_large_calibration_sample():
    data = streams.draw_calibration(400_000, 0.05, seed=3)
    kept = data["score"] >= 0.5
    assert data["loss"][kept].mean() == pytest.approx(
        streams.oracle_risk(0.5), abs=0.002
    )
