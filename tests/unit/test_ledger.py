"""The hash-chained hypothesis ledger: order, replay and tamper evidence."""

from fractions import Fraction
import json

import numpy
import pandas
import pytest

from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.data.ledger import bivariate
from scmle.data.ledger import online_fdr
import scmle.data.ledger.ledger
from tests.unit.test_charter_contracts import make_charter


def data(rows=300, seed=0):
    rng = numpy.random.default_rng(seed)
    features = pandas.DataFrame(
        {
            "signal": rng.normal(size=rows),
            "other": rng.normal(size=rows),
            "noise": rng.normal(size=rows),
        }
    )
    target = pandas.Series(
        features["signal"] * 0.9
        + features["other"] * 0.0
        + rng.normal(size=rows)
    )
    return features, target


def spec(column, name=None):
    return scmle.data.ledger.ledger.HypothesisSpec.create(
        name=name or f"{column}~y",
        family="bivariate",
        statistic="spearman",
        null=f"no monotone association between {column} and y",
        column=column,
    )


TEN_PERCENT = Fraction(1, 10)


def fresh(path=None, rule=online_fdr.ELOND, alpha=TEN_PERCENT):
    config = online_fdr.RuleConfig(rule, alpha)
    return scmle.data.ledger.ledger.Ledger.create(
        path,
        charter_hash="c" * 64,
        config=config,
        alpha_confirm=Fraction(1, 20),
        seed=7,
    )


def explore(ledger, features, target):
    handles = [ledger.register(spec(c)) for c in features.columns]
    return [ledger.test(h, features, target) for h in handles]


def test_a_registered_hypothesis_is_tested_and_recorded_with_wealth():
    features, target = data()
    ledger = fresh()
    h = ledger.register(spec("signal"))
    before = ledger.wealth
    result = ledger.test(h, features, target)
    assert (
        result.rejected and result.decision == scmle.data.ledger.ledger.REJECT
    )
    assert result.wealth_before == before == Fraction(1, 10)
    assert result.wealth_after == ledger.wealth == Fraction(1, 20)
    assert result.level == Fraction(1, 10) * Fraction(1, 2)
    assert ledger.decision(h) is result
    check = result.check
    assert check.set_id is qualification.SetId.EXPLORATION
    assert check.tier is qualification.Tier.UNVERIFIED
    assert check.decision is qualification.Decision.WARN
    assert check.p_value < 1e-10
    assert check.seed_path[0] == 7
    assert check.check_id == "h000001:X"


def test_a_retained_hypothesis_passes():
    features, target = data()
    ledger = fresh()
    h = ledger.register(spec("noise"))
    result = ledger.test(h, features, target)
    assert result.decision == scmle.data.ledger.ledger.RETAIN
    assert result.check.decision is qualification.Decision.PASS


def test_evaluating_before_registering_is_refused_in_code():
    features, target = data()
    ledger = fresh()
    ghost = scmle.data.ledger.ledger.Hypothesis("h000001", spec("signal"))
    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="registration must precede"
    ):
        ledger.test(ghost, features, target)
    other = fresh()
    foreign = other.register(spec("signal"))
    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="not registered"
    ):
        ledger.test(foreign, features, target)
    outcome = bivariate.Outcome(0.01, 0.2, 0.2, 100)
    with pytest.raises(scmle.data.ledger.ledger.LedgerError):
        ledger.record(ghost, outcome)
    assert len(ledger.rows) == 1  # only the open row


def test_a_hypothesis_is_tested_once_and_names_are_unique():
    features, target = data()
    ledger = fresh()
    h = ledger.register(spec("signal"))
    ledger.test(h, features, target)
    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="already tested"
    ):
        ledger.test(h, features, target)
    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="registered"
    ):
        ledger.register(spec("signal"))
    with pytest.raises(scmle.data.ledger.ledger.LedgerError, match="no result"):
        ledger.decision(ledger.register(spec("noise")))


def test_only_exploration_hypotheses_can_be_registered():
    ledger = fresh()
    on_c = scmle.data.ledger.ledger.HypothesisSpec.create(
        name="x",
        family="f",
        statistic="spearman",
        null="n",
        column="a",
        set_id=qualification.SetId.CONFIRMATION,
    )
    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="exploration"
    ):
        ledger.register(on_c)


def test_spec_validation_and_round_trip():
    s = spec("signal")
    assert scmle.data.ledger.ledger.HypothesisSpec.from_dict(s.to_dict()) == s
    assert len(s.code_hash) == 64
    with pytest.raises(ValueError, match="direction"):
        scmle.data.ledger.ledger.HypothesisSpec.create(
            name="n", family="f", statistic="spearman", null="x", direction="up"
        )
    with pytest.raises(ValueError, match="name"):
        scmle.data.ledger.ledger.HypothesisSpec.create(
            name=" ", family="f", statistic="spearman", null="x"
        )
    record = s.to_dict()
    record["extra"] = 1
    with pytest.raises(ValueError, match="unexpected"):
        scmle.data.ledger.ledger.HypothesisSpec.from_dict(record)


def test_the_file_reloads_replays_and_continues(tmp_path):
    features, target = data()
    path = tmp_path / "hypothesis_ledger.jsonl"
    first = fresh(path)
    explore(first, features, target)
    again = scmle.data.ledger.ledger.Ledger.load(path)
    assert again.head == first.head
    assert again.wealth == first.wealth
    assert {k: v.decision for k, v in again.results.items()} == {
        k: v.decision for k, v in first.results.items()
    }
    extra = again.register(spec("signal", name="signal~y second"))
    assert extra.hypothesis_id == "h000004"
    scmle.data.ledger.ledger.Ledger.verify_file(path)


def test_two_identical_runs_write_identical_bytes(tmp_path):
    features, target = data()
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    explore(fresh(a), features, target)
    explore(fresh(b), features, target)
    assert a.read_bytes() == b.read_bytes()


def lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def write(path, rows):
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def make_file(tmp_path):
    features, target = data()
    path = tmp_path / "hypothesis_ledger.jsonl"
    explore(fresh(path), features, target)
    return path


def test_editing_a_line_breaks_the_chain(tmp_path):
    path = make_file(tmp_path)
    rows = lines(path)
    record = json.loads(rows[4])
    record["body"]["evidence"]["p"] = 0.5
    rows[4] = json.dumps(record, sort_keys=True)
    write(path, rows)
    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="chain broken at row 4"
    ):
        scmle.data.ledger.ledger.Ledger.load(path)


def test_deleting_or_reordering_a_line_breaks_the_chain(tmp_path):
    path = make_file(tmp_path)
    rows = lines(path)
    write(path, rows[:2] + rows[3:])
    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="chain broken"
    ):
        scmle.data.ledger.ledger.Ledger.load(path)
    swapped = rows[:]
    swapped[2], swapped[3] = swapped[3], swapped[2]
    write(path, swapped)
    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="chain broken"
    ):
        scmle.data.ledger.ledger.Ledger.load(path)


def test_truncating_the_tail_is_caught_only_against_a_recorded_head(tmp_path):
    path = make_file(tmp_path)
    head = scmle.data.ledger.ledger.Ledger.load(path).head
    write(path, lines(path)[:-1])
    scmle.data.ledger.ledger.Ledger.load(path)  # a valid prefix on its own
    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="head does not match"
    ):
        scmle.data.ledger.ledger.Ledger.load(path, expected_head=head)


def test_a_rechained_forgery_is_caught_by_replay(tmp_path):
    path = make_file(tmp_path)
    rows = [json.loads(r) for r in lines(path)]
    # Flip the first result to a rejection of a null hypothesis and rechain.
    target = next(
        i
        for i, r in enumerate(rows)
        if r["kind"] == "result" and not r["body"]["rejected"]
    )
    rows[target]["body"]["rejected"] = True
    prev = rows[target - 1]["hash"]
    out = []
    for row in rows[target:]:
        row["prev"] = prev
        row["hash"] = scmle.data.ledger.ledger.digest(
            {k: row[k] for k in ("seq", "kind", "body", "prev")}
        )
        prev = row["hash"]
        out.append(row)
    write(path, [json.dumps(r, sort_keys=True) for r in rows[:target] + out])
    with pytest.raises(scmle.data.ledger.ledger.LedgerError, match="replay"):
        scmle.data.ledger.ledger.Ledger.load(path)


def test_a_result_before_its_registration_is_rejected_on_load(tmp_path):
    path = make_file(tmp_path)
    rows = [json.loads(r) for r in lines(path)]
    # Move the first result in front of the registrations and rechain.
    result_at = next(i for i, r in enumerate(rows) if r["kind"] == "result")
    moved = rows.pop(result_at)
    rows.insert(1, moved)
    prev = rows[0]["hash"]
    for index, row in enumerate(rows[1:], start=1):
        row["seq"] = index
        row["prev"] = prev
        row["hash"] = scmle.data.ledger.ledger.digest(
            {k: row[k] for k in ("seq", "kind", "body", "prev")}
        )
        prev = row["hash"]
    write(path, [json.dumps(r, sort_keys=True) for r in rows])
    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="unregistered"
    ):
        scmle.data.ledger.ledger.Ledger.load(path)


def test_garbage_and_missing_files_are_ledger_errors(tmp_path):
    path = tmp_path / "x.jsonl"
    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="does not exist"
    ):
        scmle.data.ledger.ledger.Ledger.load(path)
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(scmle.data.ledger.ledger.LedgerError, match="not JSON"):
        scmle.data.ledger.ledger.Ledger.load(path)
    path.write_text("", encoding="utf-8")
    with pytest.raises(scmle.data.ledger.ledger.LedgerError, match="empty"):
        scmle.data.ledger.ledger.Ledger.load(path)
    path.write_text('{"seq": 0}\n', encoding="utf-8")
    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="wrong fields"
    ):
        scmle.data.ledger.ledger.Ledger.load(path)
    assert isinstance(
        scmle.data.ledger.ledger.LedgerError("x"), errors.ArtifactError
    )


def test_create_refuses_to_overwrite(tmp_path):
    path = make_file(tmp_path)
    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="already exists"
    ):
        fresh(path)


def test_exhausted_wealth_logs_not_tested_without_computing(tmp_path):
    features, target = data()
    ledger = fresh(rule=online_fdr.ELOND, alpha=Fraction(0))
    results = explore(ledger, features, target)
    assert [r.decision for r in results] == [
        scmle.data.ledger.ledger.NOT_TESTED
    ] * 3
    assert all(r.check is None for r in results)
    assert "wealth exhausted" in results[0].reason
    assert ledger.survivors() == ()
    ledger.verify()


def test_survivors_freeze_before_confirmation_and_block_new_hypotheses():
    features, target = data()
    ledger = fresh()
    explore(ledger, features, target)
    ledger.record_control({"observed": 1})
    frozen = ledger.freeze()
    assert frozen.hypothesis_ids == ("h000001",)
    assert ledger.freeze() is frozen
    with pytest.raises(scmle.data.ledger.ledger.LedgerError, match="frozen"):
        ledger.register(spec("signal", name="late"))
    with pytest.raises(scmle.data.ledger.ledger.LedgerError, match="frozen"):
        ledger.record_control({"observed": 2})
    ledger.seal({"outcome": "confirmed"})
    assert ledger.confirmation["survivors_sha256"] == frozen.sha256
    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="no row may follow"
    ):
        ledger.freeze() and ledger.record_control({})
    ledger.verify()


def test_confirmation_needs_frozen_survivors():
    ledger = fresh()
    with pytest.raises(scmle.data.ledger.ledger.LedgerError, match="freeze"):
        ledger.seal({})


def test_open_creates_then_reuses_and_rejects_a_different_charter(tmp_path):
    charter = make_charter()
    first = scmle.data.ledger.ledger.Ledger.open(tmp_path, charter)
    assert (tmp_path / scmle.data.ledger.ledger.FILE_NAME).exists()
    assert first.config.alpha == charter.error_budget.exploration.initial
    assert first.header["alpha_confirm"] == str(
        charter.error_budget.confirmation.alpha_confirm
    )
    assert first.header["guarantee"]["dependence"] == "arbitrary"
    again = scmle.data.ledger.ledger.Ledger.open(tmp_path, charter)
    assert again.head == first.head
    other = make_charter(question="A different question?")
    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="different charter"
    ):
        scmle.data.ledger.ledger.Ledger.open(tmp_path, other)
    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="different rule"
    ):
        scmle.data.ledger.ledger.Ledger.open(
            tmp_path, charter, rule=online_fdr.SAFFRON
        )


def test_each_rule_runs_through_the_ledger():
    features, target = data()
    for rule in online_fdr.RULE_NAMES:
        ledger = fresh(rule=rule)
        results = explore(ledger, features, target)
        assert results[0].rejected
        ledger.verify()


def test_in_memory_verify_replays_the_rows():
    features, target = data()
    ledger = fresh()
    explore(ledger, features, target)
    ledger.verify()


def rechain(rows):
    """Re-hashes a list of row dicts so that only semantics can fail."""
    previous = scmle.data.ledger.ledger.GENESIS
    out = []
    for index, row in enumerate(rows):
        row["seq"] = index
        row["prev"] = previous
        row["hash"] = scmle.data.ledger.ledger.digest(
            {k: row[k] for k in ("seq", "kind", "body", "prev")}
        )
        previous = row["hash"]
        out.append(json.dumps(row, sort_keys=True))
    return out


def forged(tmp_path, edit):
    folder = tmp_path / f"case{len(list(tmp_path.iterdir()))}"
    folder.mkdir()
    path = make_file(folder)
    rows = [json.loads(r) for r in lines(path)]
    edit(rows)
    write(path, rechain(rows))
    return path


def test_header_forgeries_are_rejected(tmp_path):
    def wrong_version(rows):
        rows[0]["body"]["ledger_version"] = 99

    with pytest.raises(scmle.data.ledger.ledger.LedgerError, match="version"):
        scmle.data.ledger.ledger.Ledger.load(forged(tmp_path, wrong_version))

    def bad_rule(rows):
        rows[0]["body"]["rule"]["rule"] = "bonferroni"

    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="invalid rule"
    ):
        scmle.data.ledger.ledger.Ledger.load(forged(tmp_path, bad_rule))


def test_row_order_and_kind_forgeries_are_rejected(tmp_path):
    def second_open(rows):
        rows.insert(1, json.loads(json.dumps(rows[0])))

    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="exactly one"
    ):
        scmle.data.ledger.ledger.Ledger.load(forged(tmp_path, second_open))

    def starts_with_register(rows):
        rows.pop(0)

    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="first ledger row"
    ):
        scmle.data.ledger.ledger.Ledger.load(
            forged(tmp_path, starts_with_register)
        )

    def unknown(rows):
        rows[1]["kind"] = "mystery"

    with pytest.raises(scmle.data.ledger.ledger.LedgerError, match="unknown"):
        scmle.data.ledger.ledger.Ledger.load(forged(tmp_path, unknown))


def test_registration_forgeries_are_rejected(tmp_path):
    def skip_id(rows):
        rows[1]["body"]["hypothesis_id"] = "h000007"

    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="sequential"
    ):
        scmle.data.ledger.ledger.Ledger.load(forged(tmp_path, skip_id))

    def duplicate_name(rows):
        rows[2]["body"]["spec"]["name"] = rows[1]["body"]["spec"]["name"]

    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="registered"
    ):
        scmle.data.ledger.ledger.Ledger.load(forged(tmp_path, duplicate_name))

    def extra_field(rows):
        rows[1]["body"]["spec"]["bogus"] = 1

    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="invalid registration"
    ):
        scmle.data.ledger.ledger.Ledger.load(forged(tmp_path, extra_field))


def first_result(rows):
    return next(r for r in rows if r["kind"] == "result")


def test_result_forgeries_are_rejected(tmp_path):
    def double(rows):
        again = json.loads(json.dumps(first_result(rows)))
        rows.append(again)

    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="already tested"
    ):
        scmle.data.ledger.ledger.Ledger.load(forged(tmp_path, double))

    def wrong_level(rows):
        first_result(rows)["body"]["level"] = "1/3"

    with pytest.raises(scmle.data.ledger.ledger.LedgerError, match="replay"):
        scmle.data.ledger.ledger.Ledger.load(forged(tmp_path, wrong_level))

    def wrong_wealth(rows):
        first_result(rows)["body"]["wealth_after"] = "9/10"

    with pytest.raises(scmle.data.ledger.ledger.LedgerError, match="replay"):
        scmle.data.ledger.ledger.Ledger.load(forged(tmp_path, wrong_wealth))

    def bad_e(rows):
        first_result(rows)["body"]["evidence"]["e"] *= 1.5

    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match=r"replay|calibrated"
    ):
        scmle.data.ledger.ledger.Ledger.load(forged(tmp_path, bad_e))

    def bad_decision(rows):
        first_result(rows)["body"]["decision"] = "retain"

    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="invalid decision"
    ):
        scmle.data.ledger.ledger.Ledger.load(forged(tmp_path, bad_decision))

    def missing_field(rows):
        del first_result(rows)["body"]["level"]

    with pytest.raises(scmle.data.ledger.ledger.LedgerError, match="malformed"):
        scmle.data.ledger.ledger.Ledger.load(forged(tmp_path, missing_field))


def test_survivor_and_confirmation_forgeries_are_rejected(tmp_path):
    features, target = data()
    path = tmp_path / "hypothesis_ledger.jsonl"
    ledger = fresh(path)
    explore(ledger, features, target)
    ledger.freeze()
    ledger.seal({"outcome": "confirmed"})
    base = [json.loads(r) for r in lines(path)]

    def edit(change):
        rows = json.loads(json.dumps(base))
        change(rows)
        write(path, rechain(rows))
        return path

    def other_ids(rows):
        next(r for r in rows if r["kind"] == "survivors")["body"]["ids"] = []

    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="survivors differ"
    ):
        scmle.data.ledger.ledger.Ledger.load(edit(other_ids))

    def other_hash(rows):
        next(r for r in rows if r["kind"] == "survivors")["body"]["sha256"] = (
            "0"
        )

    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="hash does not match"
    ):
        scmle.data.ledger.ledger.Ledger.load(edit(other_hash))

    def other_survivors(rows):
        next(r for r in rows if r["kind"] == "confirmation")["body"][
            "survivors_sha256"
        ] = "0"

    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="different survivors"
    ):
        scmle.data.ledger.ledger.Ledger.load(edit(other_survivors))

    def row_after_confirmation(rows):
        rows.append(json.loads(json.dumps(rows[1])))

    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="no row may follow"
    ):
        scmle.data.ledger.ledger.Ledger.load(edit(row_after_confirmation))

    def confirmation_without_survivors(rows):
        rows[:] = [r for r in rows if r["kind"] != "survivors"]

    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="needs frozen survivors"
    ):
        scmle.data.ledger.ledger.Ledger.load(
            edit(confirmation_without_survivors)
        )

    def result_after_survivors(rows):
        survivors_at = next(
            i for i, r in enumerate(rows) if r["kind"] == "survivors"
        )
        rows.insert(survivors_at + 1, json.loads(json.dumps(rows[1])))

    with pytest.raises(scmle.data.ledger.ledger.LedgerError, match="frozen"):
        scmle.data.ledger.ledger.Ledger.load(edit(result_after_survivors))


def test_a_not_tested_row_cannot_hide_a_rejection(tmp_path):
    features, target = data()
    path = tmp_path / "hypothesis_ledger.jsonl"
    explore(fresh(path), features, target)
    rows = [json.loads(r) for r in lines(path)]
    first = first_result(rows)
    assert first["body"]["rejected"] is True
    first["body"]["decision"] = scmle.data.ledger.ledger.NOT_TESTED
    write(path, rechain(rows))
    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="invalid decision"
    ):
        scmle.data.ledger.ledger.Ledger.load(path)


def test_the_ledger_reports_state_properties():
    features, target = data()
    ledger = fresh()
    assert ledger.head == ledger.rows[-1]["hash"]
    assert ledger.frozen is False
    explore(ledger, features, target)
    ledger.freeze()
    assert ledger.frozen is True
    assert (
        scmle.data.ledger.ledger.Ledger(None).head
        == scmle.data.ledger.ledger.GENESIS
    )
    with pytest.raises(
        scmle.data.ledger.ledger.LedgerError, match="no open row"
    ):
        scmle.data.ledger.ledger.Ledger(None).require_rule()
