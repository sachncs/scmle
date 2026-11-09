import numpy
import pandas
import pytest

from benchmarks.qualification import streams
from scmle.monitoring import labels


def batch_estimates(stream, now_offset=0, use="logged"):
    estimates = []
    for index in range(stream.spec.batches):
        seen = stream.selected[index]
        loss = numpy.where(seen, stream.loss[index], numpy.nan)
        estimates.append(
            labels.estimate_batch(
                loss,
                seen,
                stream.pi[index] if use == "logged" else None,
                stream.audit[index] if stream.spec.epsilon else None,
                stream.spec.epsilon or stream.spec.floor,
            )
        )
    return estimates


def test_maturity_is_decided_by_prediction_time():
    assert labels.is_matured(batch=3, now=5, window=2)
    assert not labels.is_matured(batch=3, now=4, window=2)
    with pytest.raises(ValueError):
        labels.is_matured(0, 1, -1)


def test_complete_labels_use_the_plain_mean():
    estimate = labels.estimate_batch(
        numpy.array([0.0, 1.0, 1.0, 0.0]), numpy.ones(4, bool), None, None, 0.0
    )
    assert estimate.mechanism is labels.Mechanism.COMPLETE
    assert estimate.mean == pytest.approx(0.5)


def test_mar_with_logged_propensity_is_unbiased_while_the_naive_mean_is_not():
    # Loss depends on the covariate here, so selection on the covariate biases.
    rng = numpy.random.default_rng(0)
    cases = 400_000
    z = rng.standard_normal(cases)
    loss = (rng.random(cases) < numpy.where(z > 0, 0.3, 0.05)).astype(float)
    pi = numpy.where(z > 0, 0.9, 0.15)
    seen = rng.random(cases) < pi
    estimate = labels.estimate_batch(
        numpy.where(seen, loss, numpy.nan), seen, pi, None, 0.15
    )
    assert estimate.mechanism is labels.Mechanism.LOGGED_PROPENSITY
    assert estimate.mean == pytest.approx(loss.mean(), abs=0.004)
    assert abs(loss[seen].mean() - loss.mean()) > 0.05
    assert 0.0 <= estimate.scaled <= 1.0


def test_mnar_without_a_slice_is_unverified_and_blocks_action():
    stream = streams.simulate(
        streams.StreamSpec(
            labels="mnar", batches=5, size=200, epsilon=0.0, seed=1
        )
    )
    seen = stream.selected[0]
    estimate = labels.estimate_batch(
        numpy.where(seen, stream.loss[0], numpy.nan), seen, None, None, 0.0
    )
    assert estimate.mechanism is labels.Mechanism.UNVERIFIED
    assert estimate.mean is None and estimate.scaled is None
    assert not labels.permits_action(estimate.mechanism)


def test_mnar_with_an_audit_slice_is_estimated_from_the_slice():
    spec = streams.StreamSpec(
        labels="mnar", epsilon=0.1, batches=300, size=200, seed=2
    )
    stream = streams.simulate(spec)
    estimates = batch_estimates(stream, use="audit")
    assert {e.mechanism for e in estimates} == {labels.Mechanism.AUDIT_SLICE}
    mean = numpy.mean([e.mean for e in estimates])
    assert mean == pytest.approx(stream.loss.mean(), abs=0.01)
    assert labels.stream_mechanism(estimates) is labels.Mechanism.AUDIT_SLICE


def test_a_propensity_below_the_declared_floor_is_not_trusted():
    seen = numpy.array([True, False, True, False])
    mechanism = labels.classify(
        seen, numpy.array([0.5, 0.01, 0.5, 0.5]), None, epsilon=0.1
    )
    assert mechanism is labels.Mechanism.UNVERIFIED


def test_estimate_validates_inputs():
    seen = numpy.ones(3, bool)
    with pytest.raises(ValueError, match="aligned"):
        labels.estimate_batch(numpy.zeros(2), seen, None, None, 0.1)
    with pytest.raises(ValueError, match="within"):
        labels.estimate_batch(
            numpy.array([0.0, 2.0, 0.0]), seen, None, None, 0.1
        )
    with pytest.raises(ValueError, match="propensity"):
        labels.estimate_batch(numpy.zeros(3), seen, numpy.ones(2), None, 0.1)


def test_slow_losses_make_arrival_based_estimates_biased_but_maturity_not():
    spec = streams.StreamSpec(
        batches=300, size=100, window=2, slow=6, base=0.2, seed=3
    )
    stream = streams.simulate(spec)
    now = 150
    index = numpy.arange(spec.batches)[:, None] + numpy.zeros_like(stream.loss)
    recent = (index > now - 8) & (index <= now)
    arrived = stream.arrival <= now
    by_arrival = stream.loss[recent & arrived].mean()
    old = index <= now - 8
    by_prediction = stream.loss[old].mean()
    assert by_arrival < 0.2 - 0.04
    assert by_prediction == pytest.approx(0.2, abs=0.01)


def test_randomization_check_flags_a_slice_at_the_wrong_rate():
    rng = numpy.random.default_rng(0)
    assert labels.check_randomization(rng.random(20_000) < 0.05, 0.05).passed
    assert not labels.check_randomization(
        rng.random(20_000) < 0.08, 0.05
    ).passed
    with pytest.raises(ValueError):
        labels.check_randomization(numpy.zeros(5, bool), 1.0)


def test_bypass_test_finds_a_loop_and_stays_quiet_without_one():
    def groups(stream):
        frame = pandas.DataFrame({"z": stream.next_z.reshape(-1)})
        flags = stream.bypass.reshape(-1)
        return frame[flags].iloc[:300], frame[~flags].iloc[:300]

    loop = streams.feedback_stream(100, 100, gamma=0.8, bypass=0.3, seed=1)
    free = streams.feedback_stream(100, 100, gamma=0.0, bypass=0.3, seed=1)
    found = labels.bypass_test(*groups(loop)[::-1], seed=0)
    quiet = labels.bypass_test(*groups(free)[::-1], seed=0)
    assert labels.feedback_verdict(0.3, found) == "loop_detected"
    assert labels.feedback_verdict(0.3, quiet) == "no_loop_detected"
    assert labels.feedback_verdict(0.0, None) == labels.UNVERIFIED
