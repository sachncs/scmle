"""The feasibility verdict's reasons, word for word."""

from scmle.assurance.charter import feasibility


def test_the_infeasible_reason_names_the_normal_approximation() -> None:
    rows = feasibility.MIN_CONFIRMATION_ROWS - 1
    report = feasibility.assess(
        n_confirmation=rows, min_effect=0.3, alpha=0.05, k_max=5, power=0.8
    )
    assert report.reason == (
        f"{rows} confirmation rows is below the "
        f"{feasibility.MIN_CONFIRMATION_ROWS} needed for a normal "
        "approximation"
    )
