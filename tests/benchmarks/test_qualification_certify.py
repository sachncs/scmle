"""Certification runs and the smoke gate built from the whole harness."""

import json

import pandas as pd
import pytest

from benchmarks.qualification import __main__ as cli
from benchmarks.qualification import certify
from benchmarks.qualification import inject
from benchmarks.qualification import smoke
from benchmarks.qualification import standins
from scmle.data.detectors import certificates

POLICY = certificates.CertificationPolicy(
    recall_floor=0.5, false_alarm_ceiling=0.1
)


def small_frame(rows: int = 120) -> pd.DataFrame:
    return standins.demo_frame().head(rows).reset_index(drop=True)


def duplicate_plan() -> certify.DefectPlan:
    return certify.DefectPlan(
        "duplicate_exact",
        lambda frame, seed: inject.inject_duplicates(frame, 0.1, seed),
    )


def test_the_frame_digest_follows_the_data():
    frame = small_frame()
    assert certify.frame_digest(frame) == certify.frame_digest(frame.copy())
    changed = frame.copy()
    changed.iloc[0, 0] += 1.0
    assert certify.frame_digest(changed) != certify.frame_digest(frame)
    assert len(certify.frame_digest(frame)) == 64


def test_a_detector_that_finds_the_plants_is_certified():
    frame = small_frame()
    result = certify.certify_detector(
        "dup",
        standins.duplicate_standin,
        certificates.code_hash(standins.duplicate_standin),
        frame,
        [duplicate_plan()],
        seeds=(0, 1),
        policy=POLICY,
    )
    outcome = result.outcomes[0]
    assert outcome.certificate.planted == 2 * round(0.1 * len(frame))
    assert outcome.certificate.recall == 1.0
    assert outcome.certificate.false_alarms == 0
    # Clean rows are counted once even with two seeds (no pseudo-replication).
    assert outcome.certificate.clean_rows == len(frame)
    assert result.certified_defect_types == ("duplicate_exact",)
    assert result.seeds == (0, 1)
    assert result.corpus_hash == certificates.corpus_hash(
        [certify.frame_digest(frame)]
    )


def test_a_detector_that_finds_nothing_is_not_certified():
    def silent(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(False, index=frame.index)

    result = certify.certify_detector(
        "silent",
        silent,
        certificates.code_hash(silent),
        small_frame(),
        [duplicate_plan()],
        seeds=(0,),
        policy=POLICY,
    )
    assert result.outcomes[0].certificate.recall == 0.0
    assert result.certified_defect_types == ()


def test_a_detector_that_flags_everything_is_not_certified():
    def alarmist(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(True, index=frame.index)

    result = certify.certify_detector(
        "alarmist",
        alarmist,
        certificates.code_hash(alarmist),
        small_frame(),
        [duplicate_plan()],
        seeds=(0,),
        policy=POLICY,
    )
    outcome = result.outcomes[0]
    assert outcome.certificate.recall == 1.0
    assert outcome.certificate.false_alarm_rate == 1.0
    assert outcome.passed is False


@pytest.mark.parametrize(
    "bad",
    [
        lambda frame: pd.Series(True, index=frame.index[:-1]),
        lambda frame: pd.Series(1, index=frame.index),
        lambda frame: [True] * len(frame),
    ],
)
def test_a_malformed_detector_output_is_refused(bad):
    with pytest.raises(ValueError, match="flags"):
        certify.certify_detector(
            "bad",
            bad,
            "a" * 64,
            small_frame(),
            [duplicate_plan()],
            seeds=(0,),
            policy=POLICY,
        )


def test_certification_needs_plans_seeds_and_a_unique_defect_type():
    args = ("d", standins.duplicate_standin, "a" * 64, small_frame())
    with pytest.raises(ValueError, match="plan"):
        certify.certify_detector(*args, [], seeds=(0,), policy=POLICY)
    with pytest.raises(ValueError, match="seed"):
        certify.certify_detector(
            *args, [duplicate_plan()], seeds=(), policy=POLICY
        )
    with pytest.raises(ValueError, match="defect type"):
        certify.certify_detector(
            *args,
            [duplicate_plan(), duplicate_plan()],
            seeds=(0,),
            policy=POLICY,
        )


def test_certification_is_reproducible():
    first = standins.certify_all(seeds=(0, 1))
    second = standins.certify_all(seeds=(0, 1))
    assert first.certificates == second.certificates


def test_every_standin_is_certified_with_its_current_code_hash():
    store = standins.certify_all(seeds=(0, 1, 2))
    for name, standin in standins.STANDINS.items():
        digest = standin.code_hash()
        assert (
            store.state(name, digest) is certificates.CertificateState.CURRENT
        )
        assert (
            store.status(name, digest, standin.plan.defect_type)
            is certificates.GateStatus.GATING
        )
        assert standin.code_hash() == standin.code_hash()


def test_editing_a_standin_makes_its_certificate_stale():
    store = standins.certify_all(seeds=(0,))
    name = "standin-duplicate"
    changed = "f" * 64
    assert store.state(name, changed) is certificates.CertificateState.STALE
    with pytest.raises(certificates.StaleCertificateError):
        store.mark_gating(name, changed, "duplicate_exact")


def test_the_committed_sample_is_current_for_the_standin_code():
    store = certificates.CertificateStore.load(standins.SAMPLE_PATH)
    assert {c.detector_id for c in store.certificates} == set(standins.STANDINS)
    problems = standins.sample_problems(store)
    assert problems == []


def test_a_sample_with_an_old_code_hash_is_reported():
    store = certificates.CertificateStore.load(standins.SAMPLE_PATH)
    document = store.to_dict()
    document["certificates"][0]["code_hash"] = "0" * 64
    stale = certificates.CertificateStore.from_dict(document)
    assert standins.sample_problems(stale)


def test_the_smoke_gate_passes_and_lists_every_check():
    report = smoke.run_smoke()
    names = {check.name for check in report.checks}
    assert report.ok, [c for c in report.checks if not c.ok]
    assert {
        "injector masks are exact",
        "mask checker detects a corrupted mask",
        "null size inside the band",
        "biased test falls outside the band",
        "oracles load and agree",
        "corpus manifest is valid",
        "stand-in certificates",
        "stale certificate demotes gating",
    } <= names


def test_the_smoke_gate_can_fail():
    report = smoke.run_smoke(sabotage="size")
    assert not report.ok
    failed = {c.name for c in report.checks if not c.ok}
    assert "null size inside the band" in failed


def test_the_command_line_runs_smoke_check_and_writes_a_sample(
    tmp_path, capsys
):
    assert cli.main(["--smoke"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True
    assert cli.main(["--check"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    target = tmp_path / "sample.json"
    assert cli.main(["--write-sample", str(target)]) == 0
    capsys.readouterr()
    written = certificates.CertificateStore.load(target)
    assert standins.sample_problems(written) == []
    with pytest.raises(SystemExit):
        cli.main([])


def test_check_fails_when_a_gating_detector_has_no_certificate(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setitem(cli.GATING, "ghost", ("a" * 64, "label_flip"))
    assert cli.main(["--check"]) == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is False
    assert printed["missing"] == ["ghost"]
