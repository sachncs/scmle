"""The cost report's sums, its currency key and its refusals."""

import pytest

from scmle.contracts import errors
from scmle.operations import cost

RATES = cost.Rates(
    vcpu_second=0.5, gib_second=0.25, currency="EUR", source="price page"
)


def test_one_version_adds_its_records_from_zero() -> None:
    usages = [
        cost.Usage("api", "v1", vcpu_seconds=2.0, gib_seconds=4.0),
        cost.Usage("api", "v1", vcpu_seconds=2.0, gib_seconds=4.0),
    ]
    found = cost.report(usages, RATES)
    assert found["currency"] == "EUR"
    assert found["services"]["api"]["versions"]["v1"] == {
        "compute": 2.0,
        "memory": 2.0,
        "requests": 0.0,
        "egress": 0.0,
        "total": 4.0,
    }
    assert found["total"] == 4.0


def test_a_fractional_rate_is_accepted() -> None:
    assert cost.Rates(0.5, 0.25, 0.5, 0.5, source="page").per_gb_egress == 0.5


def test_a_negative_usage_is_refused_by_name() -> None:
    with pytest.raises(
        errors.InputError, match=r"^egress_gb must be finite and not negative$"
    ):
        cost.Usage("api", "v1", 1.0, 1.0, egress_gb=-1.0)


def test_the_grand_total_is_rounded_to_six_places() -> None:
    # Near 1e8 a float's spacing is about 1.5e-8, so the sum of these
    # six-place lines carries a seventh place that the total must drop.
    rates = cost.Rates(vcpu_second=1.0, gib_second=0.0, source="price page")
    spends = (80532392.454159, 94798404.423626, 93790615.02491)
    usages = [
        cost.Usage(f"api{index}", "v1", vcpu_seconds=spend, gib_seconds=0.0)
        for index, spend in enumerate(spends)
    ]
    assert cost.report(usages, rates)["total"] == 269121411.902695
