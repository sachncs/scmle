"""Detector certificates: Wilson bounds, code hashes and gating rules."""

import json
import textwrap

import pytest
from scipy import stats

from scmle.contracts import qualification
from scmle.data.detectors import certificates

HASH_A = "a" * 64
HASH_B = "b" * 64
CORPUS = "c" * 64
POLICY = certificates.CertificationPolicy(
    recall_floor=0.8, false_alarm_ceiling=0.05, confidence=0.95
)


def measured(defect="label_flip", planted=200, detected=190, fa=2, clean=1000):
    return qualification.Certificate(
        detector_id="d1",
        defect_type=defect,
        planted=planted,
        detected=detected,
        clean_rows=clean,
        false_alarms=fa,
        seed_path=(0, "inject.label_flip", 0),
    )


def certified(code_hash=HASH_A, results=None):
    return certificates.certify(
        "d1",
        code_hash,
        CORPUS,
        (0, 1, 2),
        [measured()] if results is None else results,
        POLICY,
    )


@pytest.mark.parametrize("confidence", [0.8, 0.9, 0.95, 0.99])
@pytest.mark.parametrize(
    ("successes", "n"),
    [(0, 10), (1, 10), (5, 10), (9, 10), (10, 10), (37, 120), (1, 1), (0, 1)],
)
def test_wilson_interval_matches_scipy(successes, n, confidence):
    low, high = certificates.wilson_interval(successes, n, confidence)
    reference = stats.binomtest(successes, n).proportion_ci(
        confidence_level=confidence, method="wilson"
    )
    assert low == pytest.approx(reference.low, abs=1e-12)
    assert high == pytest.approx(reference.high, abs=1e-12)
    assert 0.0 <= low <= successes / n <= high <= 1.0


def test_wilson_interval_agrees_with_the_certificate_lower_bound():
    cert = measured(planted=57, detected=49)
    low = certificates.wilson_interval(49, 57, 0.95)[0]
    assert low == pytest.approx(cert.recall_lower_bound(0.95), abs=1e-12)


@pytest.mark.parametrize(
    ("successes", "n", "confidence"),
    [(1, 0, 0.95), (-1, 5, 0.95), (6, 5, 0.95), (1, 5, 0.0), (1, 5, 1.0)],
)
def test_wilson_interval_rejects_invalid_input(successes, n, confidence):
    with pytest.raises(ValueError):
        certificates.wilson_interval(successes, n, confidence)


def test_code_hash_ignores_comments_docstrings_and_blank_lines():
    plain = "def f(x):\n    return x + 1\n"
    decorated = textwrap.dedent(
        '''
        def f(x):
            """Adds one."""

            # a comment
            return x + 1  # trailing comment
        '''
    )
    assert certificates.hash_source(plain) == certificates.hash_source(
        decorated
    )


@pytest.mark.parametrize(
    "changed",
    [
        "def f(x):\n    return x + 2\n",
        "def f(x):\n    return x - 1\n",
        "def g(x):\n    return x + 1\n",
        "def f(x):\n    return x + 1.0\n",
    ],
)
def test_code_hash_changes_with_the_logic(changed):
    original = "def f(x):\n    return x + 1\n"
    assert certificates.hash_source(original) != certificates.hash_source(
        changed
    )


def test_code_hash_keeps_a_docstring_that_shares_a_line_with_code():
    one = 'def f():\n    "doc"; return 1\n'
    two = 'def f():\n    "doc"; return 2\n'
    assert certificates.hash_source(one) != certificates.hash_source(two)


def test_code_hash_of_objects_depends_on_order_and_content():
    def first() -> int:
        return 1

    def second() -> int:
        return 2

    forward = certificates.code_hash(first, second)
    assert forward == certificates.code_hash(first, second)
    assert forward != certificates.code_hash(second, first)
    assert forward != certificates.code_hash(first)
    assert len(forward) == 64


def test_corpus_hash_is_order_independent_and_needs_parts():
    assert certificates.corpus_hash([HASH_A, HASH_B]) == (
        certificates.corpus_hash([HASH_B, HASH_A])
    )
    assert certificates.corpus_hash([HASH_A]) != certificates.corpus_hash(
        [HASH_A, HASH_B]
    )
    with pytest.raises(ValueError):
        certificates.corpus_hash([])


def test_certify_passes_only_when_both_bounds_hold():
    good = certified()
    assert good.certified_defect_types == ("label_flip",)
    outcome = good.outcomes[0]
    assert outcome.recall_interval[0] >= POLICY.recall_floor
    assert outcome.false_alarm_interval[1] <= POLICY.false_alarm_ceiling


def test_low_recall_lower_bound_fails_even_when_the_point_estimate_passes():
    # 8/10 = 0.8 meets the floor as a point estimate; the interval does not.
    weak = certified(results=[measured(planted=10, detected=8, fa=0)])
    assert weak.certified_defect_types == ()
    assert weak.outcomes[0].passed is False


def test_high_false_alarm_upper_bound_fails():
    noisy = certified(results=[measured(fa=60)])
    assert noisy.certified_defect_types == ()


def test_certify_rejects_empty_duplicate_and_bad_inputs():
    with pytest.raises(ValueError):
        certified(results=[])
    with pytest.raises(ValueError):
        certified(results=[measured(), measured()])
    with pytest.raises(ValueError):
        certificates.certify("d1", "short", CORPUS, (0,), [measured()], POLICY)
    with pytest.raises(ValueError):
        certificates.certify("d1", HASH_A, CORPUS, (), [measured()], POLICY)
    with pytest.raises(ValueError):
        certificates.CertificationPolicy(1.5, 0.05)
    with pytest.raises(ValueError):
        certificates.CertificationPolicy(0.8, 0.05, confidence=1.0)


def test_store_round_trips_with_stable_bytes(tmp_path):
    store = certificates.CertificateStore([certified()])
    path = tmp_path / "detector_certificates.json"
    store.save(path)
    first = path.read_bytes()
    loaded = certificates.CertificateStore.load(path)
    loaded.save(path)
    assert path.read_bytes() == first
    assert loaded.certificates == store.certificates
    assert json.loads(first)["schema_version"] == certificates.SCHEMA_VERSION


def test_store_rejects_tampered_pass_flags_and_unknown_fields(tmp_path):
    store = certificates.CertificateStore(
        [certified(results=[measured(planted=10, detected=8, fa=0)])]
    )
    document = store.to_dict()
    document["certificates"][0]["outcomes"][0]["passed"] = True
    with pytest.raises(certificates.CertificateError, match="passed"):
        certificates.CertificateStore.from_dict(document)

    document = store.to_dict()
    document["certificates"][0]["outcomes"][0]["recall_interval"] = [0.9, 1.0]
    with pytest.raises(certificates.CertificateError, match="interval"):
        certificates.CertificateStore.from_dict(document)

    document = store.to_dict()
    document["certificates"][0]["extra"] = 1
    with pytest.raises(ValueError, match="unexpected"):
        certificates.CertificateStore.from_dict(document)

    document = store.to_dict()
    document["schema_version"] = 99
    with pytest.raises(certificates.CertificateError, match="schema"):
        certificates.CertificateStore.from_dict(document)


def test_a_current_certified_detector_may_gate_for_that_defect_only():
    store = certificates.CertificateStore([certified()])
    assert store.state("d1", HASH_A) is certificates.CertificateState.CURRENT
    assert (
        store.status("d1", HASH_A, "label_flip")
        is certificates.GateStatus.GATING
    )
    assert (
        store.status("d1", HASH_A, "sentinel")
        is certificates.GateStatus.DESCRIPTIVE
    )
    assert (
        store.mark_gating("d1", HASH_A, "label_flip")
        is certificates.GateStatus.GATING
    )
    with pytest.raises(certificates.CertificateError, match="sentinel"):
        store.mark_gating("d1", HASH_A, "sentinel")


def test_a_code_change_makes_the_certificate_stale_and_blocks_gating():
    store = certificates.CertificateStore([certified(HASH_A)])
    assert store.state("d1", HASH_B) is certificates.CertificateState.STALE
    assert (
        store.status("d1", HASH_B, "label_flip")
        is certificates.GateStatus.DESCRIPTIVE
    )
    with pytest.raises(certificates.StaleCertificateError):
        store.mark_gating("d1", HASH_B, "label_flip")


def test_a_detector_without_a_certificate_cannot_be_marked_gating():
    store = certificates.CertificateStore()
    assert store.state("d1", HASH_A) is certificates.CertificateState.MISSING
    with pytest.raises(certificates.CertificateError, match="no certificate"):
        store.mark_gating("d1", HASH_A, "label_flip")


def test_a_failed_outcome_does_not_certify_gating():
    weak = certified(results=[measured(planted=10, detected=8, fa=0)])
    store = certificates.CertificateStore([weak])
    assert (
        store.status("d1", HASH_A, "label_flip")
        is certificates.GateStatus.DESCRIPTIVE
    )
    with pytest.raises(certificates.CertificateError):
        store.mark_gating("d1", HASH_A, "label_flip")


def test_add_replaces_the_entry_with_the_same_key():
    first = certified()
    second = certified(results=[measured(detected=180)])
    store = certificates.CertificateStore([first])
    store.add(second)
    assert len(store.certificates) == 1
    assert store.certificates[0] == second
    other = certified(HASH_B)
    store.add(other)
    assert len(store.certificates) == 2


def test_descriptive_status_demotes_blocking_decisions():
    descriptive = certificates.GateStatus.DESCRIPTIVE
    gating = certificates.GateStatus.GATING
    decision = qualification.Decision
    assert certificates.demote(decision.BLOCK, descriptive) is decision.WARN
    assert (
        certificates.demote(decision.QUARANTINE, descriptive) is decision.WARN
    )
    assert certificates.demote(decision.WARN, descriptive) is decision.WARN
    assert certificates.demote(decision.PASS, descriptive) is decision.PASS
    assert certificates.demote(decision.BLOCK, gating) is decision.BLOCK


def test_audit_reports_missing_and_stale_gating_detectors():
    store = certificates.CertificateStore([certified(HASH_A)])
    report = certificates.audit(
        store,
        {
            "d1": (HASH_B, "label_flip"),
            "d2": (HASH_A, "label_flip"),
        },
    )
    assert report.stale == ("d1",)
    assert report.missing == ("d2",)
    assert report.uncertified == ()
    assert not report.ok

    fine = certificates.audit(store, {"d1": (HASH_A, "label_flip")})
    assert fine.ok

    uncertified = certificates.audit(store, {"d1": (HASH_A, "sentinel")})
    assert uncertified.uncertified == ("d1",)
    assert certificates.audit(store, {}).ok
