"""Pins how the robustness checks are wired, and what each stage is handed."""

import numpy

from scmle.contracts import qualification
from scmle.data.charter import contracts
from scmle.data.detectors import certificates
from scmle.data.robustness import coverage
from scmle.data.robustness import multiverse
from scmle.data.robustness import qualify
from scmle.data.robustness import shift
from scmle.data.robustness import stability
from tests.unit.test_robustness_qualify import RICH
from tests.unit.test_robustness_qualify import run
from tests.unit.test_robustness_qualify import world


def certified_store(tmp_path, *, stability_too=True, shift_too=True):
    """A certificate store issued for the code that is running now."""
    policy = certificates.Standard(0.8, 0.05, 0.95)
    issued = []
    wanted = []
    if stability_too:
        wanted.append(
            (
                stability.DETECTOR_ID,
                stability.code_digest(),
                stability.DEFECT_TYPES,
            )
        )
    if shift_too:
        wanted += [
            (detector, shift.code_digest(), (shift.DEFECT_TYPE,))
            for detector in (shift.C2ST_ID, shift.MMD_ID)
        ]
    for detector, digest, defects in wanted:
        proofs = [
            qualification.Certificate(
                detector, defect, 300, 300, 500, 0, (0, "c", 0)
            )
            for defect in defects
        ]
        issued.append(
            certificates.certify(
                detector, digest, "c" * 64, (0,), proofs, policy
            )
        )
    certificates.Store(issued).save(tmp_path / qualify.F6_STORE)


def test_the_constants_are_the_documented_ones():
    assert qualify.F6_STORE == "f6_certificates.json"
    assert qualify.BUDGET_SECONDS == 60.0
    assert qualify.MAX_FINDINGS == 50
    assert qualify.MAX_BOOTSTRAP == 100
    assert qualify.MAX_CURVE_PERMUTATIONS == 200
    assert qualify.MIN_PERMUTATIONS == 99
    assert qualify.MAX_PERMUTATIONS == 499
    assert qualify.MAX_REFERENCE_ROWS == 20000


def test_a_missing_store_gives_a_note_naming_the_file(tmp_path):
    frame, codes = world()
    found = run(frame, codes, tmp_path)
    assert found.store_note == (
        f"f6_certificates.json not found at {tmp_path / 'f6_certificates.json'}"
        ": detectors are descriptive"
    )
    assert found.mmd_gate is certificates.GateStatus.DESCRIPTIVE


def test_an_unreadable_store_is_descriptive_with_the_reason(tmp_path):
    (tmp_path / qualify.F6_STORE).write_text("not json", encoding="utf-8")
    frame, codes = world()
    found = run(frame, codes, tmp_path)
    assert found.store_note.startswith("f6_certificates.json unreadable (")
    assert found.store_note.endswith("): detectors are descriptive")
    assert found.stability_gate is certificates.GateStatus.DESCRIPTIVE


def test_a_current_store_lets_every_check_gate(tmp_path):
    certified_store(tmp_path)
    frame, codes = world()
    found = run(frame, codes, tmp_path)
    assert found.store_note == ""
    assert found.stability_gate is certificates.GateStatus.GATING
    assert found.c2st_gate is certificates.GateStatus.GATING
    assert found.mmd_gate is certificates.GateStatus.GATING
    assert found.rule_note == "" and found.rule == stability.DEFAULT_RULE


def test_each_gate_follows_its_own_certificate(tmp_path):
    certified_store(tmp_path, stability_too=False)
    frame, codes = world()
    only_shift = run(frame, codes, tmp_path)
    assert only_shift.stability_gate is certificates.GateStatus.DESCRIPTIVE
    assert only_shift.c2st_gate is certificates.GateStatus.GATING
    assert only_shift.mmd_gate is certificates.GateStatus.GATING
    certified_store(tmp_path, shift_too=False)
    only_stability = run(frame, codes, tmp_path)
    assert only_stability.stability_gate is certificates.GateStatus.GATING
    assert only_stability.c2st_gate is certificates.GateStatus.DESCRIPTIVE
    assert only_stability.mmd_gate is certificates.GateStatus.DESCRIPTIVE


def test_another_rule_is_descriptive_and_says_so(tmp_path):
    certified_store(tmp_path)
    frame, codes = world()
    rule = stability.Thresholds(0.7, 0.5, 0.8)
    found = run(frame, codes, tmp_path, rule=rule)
    assert found.rule == rule
    assert found.stability_gate is certificates.GateStatus.DESCRIPTIVE
    assert found.rule_note == (
        "the chartered stability thresholds differ from the certified rule: "
        "the verdicts are descriptive"
    )
    assert found.c2st_gate is certificates.GateStatus.GATING


def test_the_reasons_stability_was_not_run_are_worded_exactly(tmp_path):
    frame, codes = world()
    many = run(frame, codes, tmp_path, multiclass=True)
    assert many.stability_note == (
        "the target has more than two classes: findings use Kruskal-Wallis "
        "and the rank-correlation stability estimator does not apply"
    )
    empty = run(frame, codes, tmp_path, survivors=[])
    assert empty.stability_note == "no finding survived exploration"
    few = run(
        frame,
        codes,
        tmp_path,
        choices=(contracts.Choice("imputation", ("median", "mice"), "median"),),
    )
    assert few.stability_note == (
        "the declared choices give 1 specification(s) this estimator can "
        "vary; at least 4 are needed, so no finding can be called stable"
    )
    assert few.findings == ()


def test_a_multiclass_target_wins_over_the_other_reasons(tmp_path):
    frame, codes = world()
    both = run(frame, codes, tmp_path, multiclass=True, survivors=[])
    assert both.stability_note.startswith("the target has more than two")
    assert both.grid is not None


def test_exactly_the_minimum_number_of_specifications_is_enough(tmp_path):
    frame, codes = world()
    four = (
        contracts.Choice("imputation", ("median", "complete_case"), "median"),
        contracts.Choice("outlier_rule", ("none", "winsorize"), "none"),
    )
    found = run(frame, codes, tmp_path, choices=four)
    assert len(found.grid.specs) == 4
    assert found.stability_note == "" and len(found.findings) == 1


def test_the_applicable_choices_and_dropped_options_are_reported(tmp_path):
    frame, codes = world()
    found = run(frame, codes, tmp_path)
    applicable, notes = stability.applicable_choices(RICH)
    assert found.applicable == applicable and found.notes == notes
    assert isinstance(found.notes, tuple)
    assert found.grid.total == 18


class Capture:
    """Stands in for `assess_finding` and records what it is given."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return f"verdict-{len(self.calls)}"


def test_each_survivor_is_assessed_with_its_own_slice_and_the_shared_budget(
    tmp_path, monkeypatch
):
    capture = Capture()
    monkeypatch.setattr(stability, "assess_finding", capture)
    frame, codes = world()
    strata = numpy.arange(len(frame)) % 5
    found = run(
        frame,
        codes,
        tmp_path,
        survivors=["a", "b"],
        seed=11,
        permutations=1000,
        strata=strata,
        budget_seconds=10.0,
    )
    assert found.findings == ("verdict-1", "verdict-2")
    (first, kwargs), (second, other) = capture.calls
    assert first[0] == "a" and second[0] == "b"
    assert first[1].tolist() == frame["a"].tolist()
    assert second[1].tolist() == frame["b"].tolist()
    assert first[1].dtype == float and first[2].dtype == float
    assert first[2].tolist() == codes.tolist()
    assert first[3] is found.grid and first[4] == 11
    assert len(first) == 5
    assert kwargs == other
    assert kwargs["bootstrap"] == 100
    assert kwargs["curve_permutations"] == 200
    assert kwargs["budget_seconds"] == 5.0
    assert kwargs["strata"] is strata
    assert kwargs["gate"] is certificates.GateStatus.DESCRIPTIVE
    assert kwargs["rule"] == stability.DEFAULT_RULE
    assert set(kwargs) == {
        "bootstrap",
        "strata",
        "curve_permutations",
        "budget_seconds",
        "gate",
        "rule",
    }


def test_the_curve_uses_the_charter_permutations_when_fewer_than_the_cap(
    tmp_path, monkeypatch
):
    capture = Capture()
    monkeypatch.setattr(stability, "assess_finding", capture)
    frame, codes = world()
    run(frame, codes, tmp_path, permutations=150)
    assert capture.calls[0][1]["curve_permutations"] == 150
    run(frame, codes, tmp_path, permutations=200)
    assert capture.calls[1][1]["curve_permutations"] == 200
    run(frame, codes, tmp_path, permutations=201)
    assert capture.calls[2][1]["curve_permutations"] == 200


def test_the_default_budget_is_shared_equally_and_the_rule_is_passed_on(
    tmp_path, monkeypatch
):
    capture = Capture()
    monkeypatch.setattr(stability, "assess_finding", capture)
    frame, codes = world()
    rule = stability.Thresholds(0.9, 0.4, 0.7)
    qualify.run(
        frame,
        codes,
        survivors=["a", "b", "g"][:2],
        multiclass=False,
        choices=RICH,
        seed=1,
        permutations=20,
        directory=tmp_path,
        rule=rule,
    )
    assert capture.calls[0][1]["budget_seconds"] == 30.0
    assert capture.calls[0][1]["rule"] == rule
    assert capture.calls[0][1]["strata"] is None


def test_a_cap_on_the_findings_keeps_the_first_ones_in_order(
    tmp_path, monkeypatch
):
    capture = Capture()
    monkeypatch.setattr(stability, "assess_finding", capture)
    monkeypatch.setattr(qualify, "MAX_FINDINGS", 2)
    frame, codes = world()
    frame["c"] = frame["b"]
    found = run(frame, codes, tmp_path, survivors=["b", "a", "c"])
    assert [call[0][0] for call in capture.calls] == ["b", "a"]
    assert found.notes[-1] == "only the first 2 of 3 findings were assessed"
    assert capture.calls[0][1]["budget_seconds"] == 30.0
    exact = run(frame, codes, tmp_path, survivors=["b", "a"])
    assert not any("were assessed" in note for note in exact.notes)


def test_no_assessment_runs_when_a_reason_stops_it(tmp_path, monkeypatch):
    capture = Capture()
    monkeypatch.setattr(stability, "assess_finding", capture)
    frame, codes = world()
    run(frame, codes, tmp_path, multiclass=True)
    run(frame, codes, tmp_path, survivors=[])
    assert capture.calls == []


class Shifts:
    def __init__(self):
        self.calls = []

    def __call__(self, sample, reference, seed, **options):
        self.calls.append((sample, reference, seed, options))
        return "shift-result"


def test_the_shift_check_gets_the_candidates_the_reference_and_the_gates(
    tmp_path, monkeypatch
):
    certified_store(tmp_path)
    capture = Shifts()
    monkeypatch.setattr(shift, "assess", capture)
    frame, codes = world()
    reference = frame.copy()
    found = run(
        frame,
        codes,
        tmp_path,
        candidates=["b", "ghost", "a"],
        reference=reference,
        seed=4,
        permutations=250,
    )
    assert found.shift == "shift-result"
    ((sample, given, seed, options),) = capture.calls
    assert list(sample.columns) == ["b", "a"]
    assert given is not None and len(given) == len(frame)
    assert seed == 4
    assert options == {
        "permutations": 250,
        "c2st_gate": certificates.GateStatus.GATING,
        "mmd_gate": certificates.GateStatus.GATING,
    }


def test_the_shift_permutations_are_clamped_to_the_documented_range(
    tmp_path, monkeypatch
):
    capture = Shifts()
    monkeypatch.setattr(shift, "assess", capture)
    frame, codes = world()
    for asked, wanted in (
        (20, 99),
        (99, 99),
        (300, 300),
        (499, 499),
        (900, 499),
    ):
        run(frame, codes, tmp_path, permutations=asked)
        assert capture.calls[-1][3]["permutations"] == wanted
    assert capture.calls[0][1] is None


def test_the_sample_and_the_reference_are_cut_to_the_row_limit(
    tmp_path, monkeypatch
):
    capture = Shifts()
    monkeypatch.setattr(shift, "assess", capture)
    monkeypatch.setattr(qualify, "MAX_REFERENCE_ROWS", 50)
    frame, codes = world(rows=120)
    run(frame, codes, tmp_path, reference=frame.iloc[::-1])
    ((sample, given, seed, options),) = capture.calls
    assert seed == 1 and set(options) == {
        "permutations",
        "c2st_gate",
        "mmd_gate",
    }
    assert len(sample) == 50 and len(given) == 50
    assert sample.index.tolist() == list(range(50))
    assert given.index.tolist() == list(range(119, 69, -1))


def test_the_default_candidates_are_none(tmp_path, monkeypatch):
    capture = Shifts()
    monkeypatch.setattr(shift, "assess", capture)
    frame, codes = world()
    qualify.run(
        frame,
        codes,
        survivors=[],
        multiclass=False,
        choices=RICH,
        seed=1,
        permutations=20,
        directory=tmp_path,
    )
    assert capture.calls[0][0].shape == (len(frame), 0)


def test_subgroup_rows_are_reported_for_the_named_columns_only(tmp_path):
    frame, codes = world()
    found = run(frame, codes, tmp_path, sensitive=["missing", "g", "a"][:2])
    assert {row.level for row in found.coverage} == {"u", "v"}
    assert found.coverage == tuple(coverage.subgroup_coverage(frame["g"], None))
    nothing = run(frame, codes, tmp_path)
    assert nothing.coverage == ()


def test_a_reference_without_the_column_leaves_the_deployment_share_unknown(
    tmp_path,
):
    frame, codes = world()
    reference = frame.drop(columns=["g"])
    found = run(frame, codes, tmp_path, sensitive=["g"], reference=reference)
    assert {row.status.value for row in found.coverage} == {"unverifiable"}
    full = frame.copy()
    full["g"] = "u"
    shares = coverage.shares(full["g"])
    expected = coverage.subgroup_coverage(frame["g"], shares)
    found = run(frame, codes, tmp_path, sensitive=["g"], reference=full)
    assert found.coverage == tuple(expected)


def test_the_coverage_shares_come_from_the_whole_reference(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(qualify, "MAX_REFERENCE_ROWS", 10)
    frame, codes = world(rows=100)
    reference = frame.copy()
    reference["g"] = ["u"] * 50 + ["v"] * 50
    found = run(frame, codes, tmp_path, sensitive=["g"], reference=reference)
    assert found.coverage == tuple(
        coverage.subgroup_coverage(frame["g"], coverage.shares(reference["g"]))
    )


def test_the_forward_check_says_why_it_was_not_run(tmp_path):
    frame, codes = world()
    plain = run(frame, codes, tmp_path)
    assert plain.forward.reason == (
        "the data are not temporal: no forward check"
    )
    timed = run(frame, codes, tmp_path, temporal=True)
    assert timed.forward.reason == (
        "F is the final test set, which only the final evaluation opens; the "
        "forward check was not run in this stage"
    )
    assert plain.forward.findings == () and plain.forward.opened is False


def test_the_fragile_names_keep_the_order_of_the_findings():
    def finding(name, status):
        return stability.Stability(
            name=name,
            status=status,
            reason="r",
            headline=None,
            numbers={},
            specs=(),
            skipped=(),
            attribution={},
            curve=None,
            check=None,
        )

    names = ("x", "y", "z")
    statuses = (stability.FRAGILE, stability.STABLE, stability.FRAGILE)
    pieces = tuple(finding(n, s) for n, s in zip(names, statuses, strict=True))
    holder = qualify.Robustness(
        grid=None,
        applicable=(),
        findings=pieces,
        stability_note="",
        notes=(),
        shift=None,
        coverage=(),
        forward=None,
        stability_gate=certificates.GateStatus.DESCRIPTIVE,
        c2st_gate=certificates.GateStatus.DESCRIPTIVE,
        mmd_gate=certificates.GateStatus.DESCRIPTIVE,
        store_note="",
    )
    assert holder.fragile == ("x", "z")
    assert holder.rule == stability.DEFAULT_RULE and holder.rule_note == ""


def test_the_charter_seed_orders_the_specification_subset(tmp_path):
    frame, codes = world()
    applicable = stability.applicable_choices(RICH)[0]
    first = run(frame, codes, tmp_path, seed=1)
    second = run(frame, codes, tmp_path, seed=2)
    assert first.grid.specs == multiverse.enumerate_grid(applicable, 1).specs
    assert second.grid.specs == multiverse.enumerate_grid(applicable, 2).specs
    assert first.grid.specs != second.grid.specs


def test_a_single_survivor_gets_the_whole_budget(tmp_path, monkeypatch):
    capture = Capture()
    monkeypatch.setattr(stability, "assess_finding", capture)
    frame, codes = world()
    run(frame, codes, tmp_path, budget_seconds=8.0)
    assert capture.calls[0][1]["budget_seconds"] == 8.0


def test_integer_columns_and_codes_are_handed_over_as_floats(
    tmp_path, monkeypatch
):
    capture = Capture()
    monkeypatch.setattr(stability, "assess_finding", capture)
    frame, codes = world()
    frame["n"] = numpy.arange(len(frame))
    run(frame, codes.round().astype(int), tmp_path, survivors=["n"])
    args = capture.calls[0][0]
    assert args[1].dtype == numpy.float64 and args[2].dtype == numpy.float64
