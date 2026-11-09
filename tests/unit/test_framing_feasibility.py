import numpy
import pytest

from benchmarks.qualification import framing_certify
from scmle.data.framing import baseline
from scmle.data.framing import feasibility


def test_intraclass_recovers_known_rho():
    rng = numpy.random.default_rng(6)
    g, m, rho = 400, 10, 0.3
    effects = rng.normal(0, rho**0.5, g)
    d = numpy.repeat(effects, m) + rng.normal(0, (1 - rho) ** 0.5, g * m)
    got, size = feasibility.intraclass(d, numpy.repeat(numpy.arange(g), m))
    assert got == pytest.approx(rho, abs=0.05)
    assert size == m


def test_verdict_away_from_boundary_matches_the_oracle():
    runs = 60
    for ratio in framing_certify.FEASIBILITY_RATIOS:
        agree = sum(
            result[0] == result[1]
            for result in (
                framing_certify.feasibility_run(
                    framing_certify.DEV_SEED_BASE + s, ratio
                )
                for s in range(runs)
            )
        )
        assert agree / runs >= 0.9


def test_near_the_boundary_the_report_says_inconclusive():
    verdicts = [
        framing_certify.feasibility_run(framing_certify.DEV_SEED_BASE + s, 1.0)[
            0
        ]
        for s in range(40)
    ]
    assert feasibility.INCONCLUSIVE in verdicts
    y, a, b, group = framing_certify.world(
        framing_certify.DEV_SEED_BASE, 60, 0.85, 0.12
    )
    d = baseline.paired_difference(y, a, b, 0.1)
    report = feasibility.assess(
        d,
        group,
        delta_min=0.04,
        n_confirmation=10**6,
        alpha=0.05,
        k_max=1,
        power=0.8,
    )
    assert report.verdict == "feasible"
    assert report.s_d_interval[0] < report.s_d < report.s_d_interval[1]
    assert report.to_dict()["s_d_interval"][0] == report.s_d_interval[0]


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        feasibility.assess(
            [1.0, 1.0, 1.0],
            [0, 1, 2],
            delta_min=0.1,
            n_confirmation=100,
            alpha=0.05,
            k_max=1,
            power=0.8,
        )
    with pytest.raises(ValueError):
        feasibility.assess(
            [1.0, 2.0, 4.0],
            [0, 1, 2],
            delta_min=0.0,
            n_confirmation=100,
            alpha=0.05,
            k_max=1,
            power=0.8,
        )
