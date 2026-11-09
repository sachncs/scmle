"""Core qualification records: validation and JSON round trips."""

import json
import math

import pytest

from scmle.contracts import qualification


def make_check(**overrides):
    values = {
        "check_id": "missingness.age",
        "null_hypothesis": "missingness of age is independent of the target",
        "statistic": 12.5,
        "p_value": 0.01,
        "interval": None,
        "effect": 0.2,
        "n": 500,
        "decision": qualification.Decision.WARN,
        "set_id": qualification.SetId.EXPLORATION,
        "seed_path": (7, "missingness.age", 0),
        "tier": qualification.Tier.UNVERIFIED,
    }
    values.update(overrides)
    return qualification.Check(**values)


def test_check_round_trips_through_json():
    check = make_check()
    restored = qualification.Check.from_dict(
        json.loads(json.dumps(check.to_dict()))
    )
    assert restored == check
    assert restored.seed_path == (7, "missingness.age", 0)


def test_check_accepts_an_interval_instead_of_a_p_value():
    check = make_check(p_value=None, interval=(0.1, 0.4))
    assert qualification.Check.from_dict(check.to_dict()) == check


def test_check_needs_a_p_value_or_an_interval():
    with pytest.raises(ValueError, match="p_value or interval"):
        make_check(p_value=None, interval=None)


@pytest.mark.parametrize("p_value", [math.nan, -0.01, 1.01, math.inf])
def test_check_rejects_invalid_p_values(p_value):
    with pytest.raises(ValueError, match="p_value"):
        make_check(p_value=p_value)


@pytest.mark.parametrize("field", ["statistic", "effect"])
def test_check_rejects_non_finite_numbers(field):
    with pytest.raises(ValueError, match=field):
        make_check(**{field: math.nan})


def test_check_rejects_inverted_or_non_finite_intervals():
    with pytest.raises(ValueError, match="interval"):
        make_check(interval=(0.5, 0.1))
    with pytest.raises(ValueError, match="interval"):
        make_check(interval=(0.1, math.inf))


def test_check_rejects_negative_n_and_blank_text():
    with pytest.raises(ValueError, match="n must"):
        make_check(n=-1)
    with pytest.raises(ValueError, match="check_id"):
        make_check(check_id="  ")
    with pytest.raises(ValueError, match="null_hypothesis"):
        make_check(null_hypothesis="")


@pytest.mark.parametrize(
    "set_id",
    [
        qualification.SetId.EXPLORATION,
        qualification.SetId.FORWARD,
        qualification.SetId.ALL,
    ],
)
def test_confirmed_tier_requires_the_confirmation_set(set_id):
    with pytest.raises(ValueError, match="confirmed"):
        make_check(set_id=set_id, tier=qualification.Tier.CONFIRMED)


def test_confirmed_tier_is_allowed_on_the_confirmation_set():
    check = make_check(
        set_id=qualification.SetId.CONFIRMATION,
        tier=qualification.Tier.CONFIRMED,
    )
    assert check.tier is qualification.Tier.CONFIRMED


def test_unknown_enum_values_are_rejected_when_loading():
    record = make_check().to_dict()
    record["decision"] = "maybe"
    with pytest.raises(ValueError):
        qualification.Check.from_dict(record)


def test_loading_rejects_unknown_and_missing_fields():
    record = make_check().to_dict()
    with pytest.raises(ValueError, match="unexpected"):
        qualification.Check.from_dict({**record, "extra": 1})
    del record["effect"]
    with pytest.raises(ValueError, match="missing"):
        qualification.Check.from_dict(record)


def make_claim(**overrides):
    values = {
        "claim_id": "claim.age-effect",
        "statement": "age is associated with churn",
        "check_ids": ("assoc.age",),
        "adjusted_p_value": 0.003,
        "effect": 0.3,
        "tier": qualification.Tier.CONFIRMED,
        "assumptions": ("confirmation set opened once",),
    }
    values.update(overrides)
    return qualification.Claim(**values)


def test_claim_round_trips_through_json():
    claim = make_claim()
    restored = qualification.Claim.from_dict(
        json.loads(json.dumps(claim.to_dict()))
    )
    assert restored == claim


def test_claim_needs_checks_and_a_p_value_to_be_confirmed():
    with pytest.raises(ValueError, match="check_ids"):
        make_claim(check_ids=())
    with pytest.raises(ValueError, match="confirmed"):
        make_claim(adjusted_p_value=None)
    unconfirmed = make_claim(
        adjusted_p_value=None, tier=qualification.Tier.UNVERIFIED
    )
    assert unconfirmed.adjusted_p_value is None


@pytest.mark.parametrize("p_value", [math.nan, -0.1, 1.5])
def test_claim_rejects_invalid_adjusted_p_values(p_value):
    with pytest.raises(ValueError, match="adjusted_p_value"):
        make_claim(adjusted_p_value=p_value)


def make_certificate(**overrides):
    values = {
        "detector_id": "label_noise.confident_learning",
        "defect_type": "label_flip",
        "planted": 100,
        "detected": 80,
        "clean_rows": 1000,
        "false_alarms": 20,
        "seed_path": (7, "certificate.label_noise", 0),
    }
    values.update(overrides)
    return qualification.Certificate(**values)


def test_certificate_rates_and_round_trip():
    certificate = make_certificate()
    assert certificate.recall == pytest.approx(0.8)
    assert certificate.false_alarm_rate == pytest.approx(0.02)
    restored = qualification.Certificate.from_dict(
        json.loads(json.dumps(certificate.to_dict()))
    )
    assert restored == certificate


def test_certificate_wilson_lower_bound_matches_a_known_value():
    # Wilson 95% interval for 8 successes in 10 trials: (0.4902, 0.9433).
    certificate = make_certificate(planted=10, detected=8)
    assert certificate.recall_lower_bound(0.95) == pytest.approx(
        0.4902, abs=1e-3
    )


def test_certificate_lower_bound_is_below_the_point_estimate():
    certificate = make_certificate()
    assert certificate.recall_lower_bound() < certificate.recall


def test_certificate_edge_cases():
    perfect = make_certificate(planted=50, detected=50, false_alarms=0)
    assert perfect.recall_lower_bound() < 1.0
    assert perfect.false_alarm_rate == 0.0


def test_certificate_rejects_impossible_counts():
    with pytest.raises(ValueError, match="detected"):
        make_certificate(detected=101)
    with pytest.raises(ValueError, match="planted"):
        make_certificate(planted=0, detected=0)
    with pytest.raises(ValueError, match="false_alarms"):
        make_certificate(false_alarms=1001)
    with pytest.raises(ValueError, match="clean_rows"):
        make_certificate(clean_rows=0, false_alarms=0)


def test_lower_bound_rejects_bad_confidence():
    with pytest.raises(ValueError, match="confidence"):
        make_certificate().recall_lower_bound(1.0)
