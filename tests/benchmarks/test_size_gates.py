"""Every size gate uses one rule, and its margin is the one they need.

A size gate measures a test of nominal level `L` on `n` null replicates.
Its ceiling is `L (1 + size_margin)`; the margin in `acceptance.toml` is
the largest that any registered gate needs for a correctly sized test to
pass with chance `power`, rounded up to hundredths.
"""

import math

import pytest

from benchmarks import acceptance
from benchmarks.qualification import f6_certify
from benchmarks.qualification import f10_certify
from benchmarks.qualification import f11_certify
from benchmarks.qualification import framing_certify
from benchmarks.qualification import subgroup_certify
from benchmarks.qualification import subgroup_extended
from scmle.assurance.robustness import shift

VARIABLES = acceptance.load()
ALPHA = VARIABLES["alpha"].value
POWER = VARIABLES["power"].value
MARGIN = VARIABLES["size_margin"].value

# (level, null replicates, the ceiling the gate decides with)
GATES = {
    **{
        f"f10 {name}": (
            f10_certify.ALPHA_FALSIFY,
            f10_certify.RUNNERS[name][1],
            f10_certify.FALSIFY,
        )
        for name in f10_certify.ALARM_GATES
    },
    "f10 size_synth": (
        f10_certify.SYNTH_LEVEL,
        f10_certify.RUNNERS["size_synth"][1],
        f10_certify.SYNTH_SIZE,
    ),
    "subgroup holm": (
        1 - subgroup_certify.LEVEL,
        subgroup_certify.NULL_REPLICATES,
        subgroup_certify.FAMILYWISE,
    ),
    "subgroup extended": (
        1 - subgroup_certify.LEVEL,
        subgroup_extended.NULL_REPLICATES,
        subgroup_certify.FAMILYWISE,
    ),
    "framing size": (
        framing_certify.ALPHA,
        framing_certify.RUNS,
        framing_certify.SIZE,
    ),
    "f6 shift": (
        shift.ALPHA,
        f6_certify.SHIFT_NULL,
        acceptance.Ceiling(f6_certify.SHIFT_POLICY.alarms),
    ),
    "f11 A4": (
        f11_certify.A4_LEVEL,
        f11_certify.RUNNERS["A4"][1],
        f11_certify.FALSE_USEFUL,
    ),
    "f11 B9 promote": (
        f11_certify.ALPHA_PROMOTE,
        f11_certify.RUNNERS["B9"][1],
        acceptance.sized(f11_certify.ALPHA_PROMOTE),
    ),
    "f11 B9 rollback": (
        f11_certify.ALPHA_ROLLBACK,
        f11_certify.RUNNERS["B9"][1],
        acceptance.sized(f11_certify.ALPHA_ROLLBACK),
    ),
}


def test_the_margin_is_the_largest_need_rounded_up_to_hundredths():
    needs = [
        acceptance.margin(level, trials, ALPHA, POWER)
        for level, trials, rule in GATES.values()
    ]
    assert math.ceil(max(needs) * 100) / 100 == MARGIN


@pytest.mark.parametrize("name", sorted(GATES))
def test_every_size_gate_uses_the_size_rule(name):
    level, trials, rule = GATES[name]
    assert trials > 0
    assert rule.bound == pytest.approx(level * (1 + MARGIN))


@pytest.mark.parametrize("name", sorted(GATES))
def test_a_correctly_sized_test_passes_with_power(name):
    level, trials, rule = GATES[name]
    assert rule.chance(level, trials, ALPHA) >= POWER


@pytest.mark.parametrize("name", sorted(GATES))
def test_a_test_of_twice_its_level_fails(name):
    level, trials, rule = GATES[name]
    assert rule.chance(2 * level, trials, ALPHA) <= 0.01
