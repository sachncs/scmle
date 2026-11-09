"""Shared fixtures."""

import pathlib

import numpy
import pandas
import pytest


@pytest.fixture
def binary_frame():
    rng = numpy.random.default_rng(0)
    rows = 500
    frame = pandas.DataFrame(
        {
            "a": rng.normal(size=rows),
            "b": rng.normal(size=rows),
            "c": rng.choice(list("xyz"), rows),
        }
    )
    frame.loc[::13, "a"] = numpy.nan
    signal = frame.a.fillna(0) + 0.5 * frame.b + (frame.c == "x")
    noise = rng.normal(scale=0.5, size=rows)
    frame["y"] = (signal + noise > 0.5).astype(int)
    return frame


@pytest.fixture
def multiclass_frame():
    rng = numpy.random.default_rng(1)
    rows = 600
    frame = pandas.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    frame["y"] = pandas.cut(
        frame.a + frame.b, [-9, -0.7, 0.7, 9], labels=["lo", "mid", "hi"]
    ).astype(str)
    return frame


@pytest.fixture
def regression_frame():
    rng = numpy.random.default_rng(2)
    rows = 500
    frame = pandas.DataFrame(
        {
            "a": rng.normal(size=rows),
            "b": rng.normal(size=rows),
            "c": rng.choice(list("pq"), rows),
        }
    )
    frame["y"] = (
        3 * frame.a
        - 2 * frame.b
        + (frame.c == "p")
        + rng.normal(scale=0.3, size=rows)
    )
    return frame


@pytest.fixture
def imbalanced_frame():
    rng = numpy.random.default_rng(3)
    rows = 900
    frame = pandas.DataFrame(
        {
            "spend": rng.exponential(5.0, rows),
            "age": rng.normal(40, 10, rows),
            "visits": rng.exponential(2.0, rows),
        }
    )
    logit = 0.25 * frame.spend + 0.05 * (frame.age - 40) - 3.2
    probability = 1 / (1 + numpy.exp(-logit))
    frame["y"] = (rng.random(rows) < probability).astype(int)
    return frame


# mutmut rewrites the sources in its scope (), and a certificate
# records the hash of the source it certified. Under mutmut every one of these
# tests would report the certificate stale, which is the instrument working,
# not a regression, so they are skipped there and run everywhere else.
HASH_SENSITIVE = frozenset(
    (
        "tests/benchmarks/test_f5_certification.py::test_check_command_passes_on_committed_files",
        "tests/benchmarks/test_f5_certification.py::test_committed_certificates_are_current_and_consistent",
        "tests/benchmarks/test_f5_certification.py::test_gating_lists_only_passing_outcomes",
        "tests/benchmarks/test_framing_certify.py::test_check_framing_is_wired_into_the_harness",
        "tests/benchmarks/test_framing_certify.py::test_committed_framing_files_are_current",
        "tests/benchmarks/test_framing_certify.py::test_selective_label_certificate_is_stale_when_code_changes",
        "tests/benchmarks/test_qualification_f4_certify.py::test_every_required_defect_type_has_an_outcome_on_file",
        "tests/benchmarks/test_qualification_f4_certify.py::test_measurements_agree_with_the_certificate",
        "tests/benchmarks/test_qualification_f4_certify.py::test_the_measured_verdicts_are_the_ones_the_gates_rely_on",
        "tests/benchmarks/test_qualification_f4_certify.py::test_the_stored_certificates_are_for_the_current_code",
        "tests/benchmarks/test_qualification_f6_certify.py::test_an_uncertified_check_is_descriptive_not_an_error",
        "tests/benchmarks/test_qualification_f6_certify.py::test_every_required_defect_type_has_an_outcome_on_file",
        "tests/benchmarks/test_qualification_f6_certify.py::test_measurements_agree_with_the_certificates",
        "tests/benchmarks/test_qualification_f6_certify.py::test_the_stored_certificates_are_for_the_current_code",
        "tests/benchmarks/test_qualification_screen_certify.py::test_every_leak_type_of_every_mechanism_has_an_outcome_on_file",
        "tests/benchmarks/test_qualification_screen_certify.py::test_measurements_agree_with_the_certificate",
        "tests/benchmarks/test_qualification_screen_certify.py::test_the_stored_certificate_is_for_the_current_code",
        "tests/benchmarks/test_qualification_uncertainty_certify.py::test_certificates_match_the_current_code",
        "tests/benchmarks/test_qualification_uncertainty_certify.py::test_missing_and_stale_certificates_are_reported",
        "tests/unit/test_admissibility_attestations.py::test_an_attestation_for_an_unflagged_column_is_unused_and_says_so",
        "tests/unit/test_admissibility_attestations.py::test_an_attestation_for_another_charter_does_not_count",
        "tests/unit/test_admissibility_attestations.py::test_an_expired_attestation_does_not_count",
        "tests/unit/test_admissibility_attestations.py::test_without_an_attestation_the_certified_suspect_is_blocked",
        "tests/unit/test_leakage_screen.py::test_a_mechanism_without_a_passing_type_stays_descriptive",
        "tests/unit/test_leakage_screen.py::test_only_the_certified_target_and_settings_may_gate",
        "tests/unit/test_leakage_screen.py::test_the_shipped_certificate_is_current_for_the_screen_code",
        "tests/unit/test_leakage_screen.py::test_with_the_shipped_certificate_a_suspect_blocks",
        "tests/unit/test_robustness_qualify.py::test_the_shipped_certificates_are_found_from_a_checkout",
    )
)


def under_mutation() -> bool:
    """Whether mutmut has instrumented the sources the certificates hash."""
    from scmle.data.detectors import sentinels

    text = pathlib.Path(sentinels.__file__).read_text(encoding="utf-8")
    return "__mutmut_orig" in text


def pytest_collection_modifyitems(items):
    if not under_mutation():
        return
    skip = pytest.mark.skip(reason="mutmut instruments the hashed sources")
    for item in items:
        if item.nodeid in HASH_SENSITIVE:
            item.add_marker(skip)
