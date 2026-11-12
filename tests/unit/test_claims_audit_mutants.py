"""Targeted tests for claims-audit behaviours the main suite left unpinned."""

import json
import shutil
import types

import pytest

from scmle.artifacts import claims_audit
from scmle.contracts import errors
from scmle.core import decision_reports
from scmle.data.charter import sealing
from scmle.data.custody import sealed
from scmle.data.ledger import confirm
from scmle.data.ledger import governed
import scmle.data.ledger.ledger
from scmle.data.splitting import splitters
from scmle.evaluation import coverage_confirm
from scmle.evaluation import coverage_run
from tests.unit.test_charter_causal import causal_charter
from tests.unit.test_charter_contracts import make_charter
from tests.unit.test_claims_audit import edit_claims
from tests.unit.test_claims_audit import FOUR
from tests.unit.test_claims_audit import KEY
from tests.unit.test_claims_audit import make_run
from tests.unit.test_claims_audit import register_decision_files
from tests.unit.test_ledger_governed import build

SHA = "a" * 64
CHAIN = "ledger chain verifies and decisions replay"
CUSTODY = "custody log: selection frozen before C opened, C opened once"
DECISION = "decision files: hashes match the custody log"
COVERAGE = "coverage claim: registered before the rows were read"
CHARTER = "charter verifies and matches the ledger"


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    folder = tmp_path_factory.mktemp("template")
    make_run(folder)
    return folder


@pytest.fixture
def run(template, tmp_path):
    target = tmp_path / "run"
    shutil.copytree(template, target)
    return target


def finding(name, status, detail):
    return claims_audit.Finding(name, status, detail)


def test_verification_serializes_every_field():
    items = (
        finding("a", claims_audit.PASS, "x"),
        finding("b", claims_audit.SKIPPED, "y"),
        finding("c", claims_audit.FAIL, "z"),
    )
    result = claims_audit.Verification("some/dir", items)
    assert result.to_dict() == {
        "run_dir": "some/dir",
        "ok": False,
        "passed": 1,
        "findings": [
            {"name": "a", "status": "pass", "detail": "x"},
            {"name": "b", "status": "skipped", "detail": "y"},
            {"name": "c", "status": "fail", "detail": "z"},
        ],
    }
    assert claims_audit.Verification("d", items[:2]).ok is True


def test_read_claims_refuses_unreadable_and_foreign_documents(tmp_path):
    with pytest.raises(errors.ArtifactError, match=r"cannot read .*claims"):
        claims_audit.read_claims(tmp_path)
    path = tmp_path / confirm.CLAIMS_FILE
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(errors.ArtifactError, match=r"^cannot read "):
        claims_audit.read_claims(tmp_path)
    path.write_text(json.dumps({"other": 1}), encoding="utf-8")
    with pytest.raises(
        errors.ArtifactError, match=r"is not a claims document$"
    ):
        claims_audit.read_claims(tmp_path)
    path.write_text(json.dumps([{"claims": 1}]), encoding="utf-8")
    with pytest.raises(errors.ArtifactError, match=r"is not a claims document"):
        claims_audit.read_claims(tmp_path)
    path.write_text(json.dumps({"claims": []}), encoding="utf-8")
    assert claims_audit.read_claims(tmp_path) == {"claims": []}


def test_read_optional_returns_only_json_objects(tmp_path):
    path = tmp_path / "x.json"
    assert claims_audit.read_optional(path) is None
    path.write_text("[1]", encoding="utf-8")
    assert claims_audit.read_optional(path) is None
    path.write_text("{bad", encoding="utf-8")
    assert claims_audit.read_optional(path) is None
    path.write_text('{"a": 1}', encoding="utf-8")
    assert claims_audit.read_optional(path) == {"a": 1}


def test_column_of_names_the_feature_or_none(run):
    ledger = claims_audit.read_ledger(run)
    claim_id = next(iter(ledger.hypotheses))
    assert claims_audit.column_of(ledger, claim_id) == (
        ledger.hypotheses[claim_id].spec.column
    )
    assert claims_audit.column_of(ledger, "h-unknown") is None


def test_list_and_show_work_when_the_claims_file_has_no_withdrawals(run):
    edit_claims(run, lambda d: d.pop("stage_two"))
    rows = claims_audit.list_claims(run)
    assert [r["withdrawn"] for r in rows] == [False, False]
    shown = claims_audit.show_claim(run, rows[0]["claim_id"])
    assert shown["withdrawn"] is None

    def hollow(document):
        document["stage_two"] = {}

    edit_claims(run, hollow)
    assert [r["withdrawn"] for r in claims_audit.list_claims(run)] == [
        False,
        False,
    ]
    assert (
        claims_audit.show_claim(run, rows[0]["claim_id"])["withdrawn"] is None
    )


def test_show_claim_reports_every_recorded_part(run):
    document = json.loads((run / confirm.CLAIMS_FILE).read_text())
    ledger = claims_audit.read_ledger(run)
    checks_seen = set()
    for claim in document["claims"]:
        claim_id = claim["claim_id"]
        shown = claims_audit.show_claim(run, claim_id)
        result = ledger.results[claim_id]
        expected_check = (
            None if result.check is None else result.check.to_dict()
        )
        checks_seen.add(expected_check is None)
        assert shown["claim"] == claim
        assert shown["hypothesis"] == ledger.hypotheses[claim_id].spec.to_dict()
        assert shown["exploration"] == {
            "decision": result.decision,
            "level": str(result.level),
            "check": expected_check,
        }
        assert shown["outcome"] == document["outcome"]
        assert set(shown) == {
            "claim",
            "hypothesis",
            "exploration",
            "confirmation_check",
            "withdrawn",
            "outcome",
        }
    assert checks_seen


def test_show_claim_names_the_known_claims_when_unknown(run):
    document = json.loads((run / confirm.CLAIMS_FILE).read_text())
    known = ", ".join(c["claim_id"] for c in document["claims"])
    with pytest.raises(errors.ArtifactError) as caught:
        claims_audit.show_claim(run, "zz")
    assert str(caught.value) == f"no claim 'zz'; the run has: {known}"
    edit_claims(run, lambda d: d.update(claims=[]))
    with pytest.raises(errors.ArtifactError) as empty:
        claims_audit.show_claim(run, "zz")
    assert str(empty.value) == "no claim 'zz'; the run has: none"


def test_show_claim_needs_both_a_hypothesis_and_a_result(run, monkeypatch):
    edit_claims(
        run,
        lambda d: d.update(
            claims=[{"claim_id": "h1", "tier": "confirmed"}], outcome="o"
        ),
    )
    thin = types.SimpleNamespace(
        claims={}, hypotheses={"h1": object()}, results={}
    )
    monkeypatch.setattr(claims_audit, "read_ledger", lambda folder: thin)
    with pytest.raises(errors.ArtifactError) as caught:
        claims_audit.show_claim(run, "h1")
    assert str(caught.value) == "claim 'h1' is not in the sealed ledger"
    other = types.SimpleNamespace(
        claims={}, hypotheses={}, results={"h1": object()}
    )
    monkeypatch.setattr(claims_audit, "read_ledger", lambda folder: other)
    with pytest.raises(errors.ArtifactError, match=r"not in the sealed ledger"):
        claims_audit.show_claim(run, "h1")


def registered_claim_run(folder, baseline_report=True, causal_report=True):
    features, target = build(rows=500, seed=0, effect=0.7)
    cfeatures, ctarget = build(rows=400, seed=9, effect=0.7)
    log = sealed.OpenLog(folder / claims_audit.CUSTODY_FILE, KEY)
    source = sealed.SealedSet(
        "C",
        splitters.ResidentTestSet(
            cfeatures, ctarget, tuple(range(len(ctarget)))
        ),
        log,
    )
    charter = make_charter()
    sealing.write_charter(
        folder / claims_audit.CHARTER_FILE, charter, signing_key=KEY
    )

    documents = {}
    if baseline_report:
        documents[claims_audit.BASELINE_FILE] = {
            "claim": {"tier": "attested"},
            "status": "identified",
            "registration_hash": SHA,
        }
    if causal_report:
        documents[claims_audit.CAUSAL_FILE] = {
            "tier": "confirmed",
            "status": "estimated",
        }

    def produce(frame, outcome):
        return confirm.ClaimRun({"base": "done", "cause": "held"}, documents)

    runner = confirm.ClaimRunner(
        (
            confirm.RegisteredClaim("base", "baseline", "a baseline", SHA),
            confirm.RegisteredClaim(
                "cause", "causal_primary", "a cause", "b" * 64
            ),
        ),
        produce,
    )
    result = governed.run_governed(
        charter=charter,
        features=features,
        target=target,
        classification=False,
        source=source,
        run_dir=folder,
        claim_runner=runner,
    )
    for name, body in documents.items():
        (folder / name).write_text(json.dumps(body), encoding="utf-8")
    return result


def test_registered_claims_are_listed_with_report_tier_and_status(tmp_path):
    registered_claim_run(tmp_path)
    rows = claims_audit.list_claims(tmp_path)
    by_id = {r["claim_id"]: r for r in rows}
    assert by_id["base"] == {
        "claim_id": "base",
        "feature": None,
        "kind": "baseline",
        "tier": "attested",
        "status": "done",
        "identification": "identified",
        "adjusted_p_value": None,
        "effect": None,
        "withdrawn": False,
    }
    assert by_id["cause"] == {
        "claim_id": "cause",
        "feature": None,
        "kind": "causal_primary",
        "tier": "confirmed",
        "status": "held",
        "identification": "estimated",
        "adjusted_p_value": None,
        "effect": None,
        "withdrawn": False,
    }
    assert [r["claim_id"] for r in rows[-2:]] == ["base", "cause"]
    shown = claims_audit.show_claim(tmp_path, "base")
    assert shown["registered"]["kind"] == "baseline"
    assert shown["status"]["status"] == "done"
    assert shown["outcome"] == "confirmed"
    assert claims_audit.verify(tmp_path, KEY).ok


def test_registered_claims_without_reports_fall_back_to_unverified(tmp_path):
    registered_claim_run(tmp_path, baseline_report=False, causal_report=False)
    ledger = claims_audit.read_ledger(tmp_path)
    for row in claims_audit.registered_rows(tmp_path, ledger):
        assert row["tier"] == "unverified"
        assert row["identification"] is None
        assert row["status"] in ("done", "held")
    stripped = types.SimpleNamespace(
        claims={"x": {"kind": "baseline"}}, confirmation=None
    )
    only = claims_audit.registered_rows(tmp_path, stripped)
    assert only[0]["status"] == "not_stated"
    assert only[0]["tier"] == "unverified"
    empty = types.SimpleNamespace(
        claims={"x": {"kind": "other"}}, confirmation={}
    )
    assert claims_audit.registered_rows(tmp_path, empty)[0]["status"] == (
        "not_stated"
    )
    thin = types.SimpleNamespace(
        claims={"x": {"kind": "other"}},
        confirmation={"claims": {"x": {"status": "ok"}}},
    )
    assert claims_audit.registered_rows(tmp_path, thin)[0]["status"] == "ok"
    (tmp_path / claims_audit.BASELINE_FILE).write_text("[1]", encoding="utf-8")
    assert claims_audit.registered_rows(tmp_path, stripped)[0]["tier"] == (
        "unverified"
    )


def test_ledger_head_findings_are_exact(run):
    ledger = claims_audit.read_ledger(run)
    document = claims_audit.read_claims(run)
    assert claims_audit.check_ledger_head(document, ledger) == finding(
        "claims file names the ledger head",
        "pass",
        f"head {ledger.head[:12]}",
    )
    document["ledger_head"] = "0" * 64
    assert claims_audit.check_ledger_head(document, ledger) == finding(
        "claims file names the ledger head",
        "fail",
        "confirmed_claims.json names a different ledger head",
    )


def charter_namespace(rules, signed=True):
    return types.SimpleNamespace(
        charter_hash="h" * 40,
        signed=signed,
        charter=types.SimpleNamespace(rules=rules),
    )


def test_charter_findings_describe_the_signature_state(run):
    ledger = claims_audit.read_ledger(run)
    signed = claims_audit.check_charter(run, ledger, KEY)
    assert signed[0].name == CHARTER
    assert signed[0].detail.endswith("(signature verified)")
    assert signed[0].detail[:12] == ledger.header["charter_hash"][:12]
    assert signed[0].detail[12:13] == " "
    unsigned = claims_audit.check_charter(run, ledger, None)
    assert unsigned[0].detail.endswith("(hash only, not signed)")
    assert unsigned[1] == finding(
        "chartered rules were used", "skipped", "the charter fixes no rules"
    )
    failed = claims_audit.check_charter(run, ledger, "another-key")
    assert len(failed) == 1
    assert failed[0].name == CHARTER
    assert failed[0].status == "fail"
    assert failed[0].detail != "None"
    with pytest.raises(errors.ScmleError) as caught:
        sealing.read_verified(
            run / claims_audit.CHARTER_FILE, signing_key="another-key"
        )
    assert failed[0].detail == str(caught.value)


def test_chartered_rules_compare_both_ledger_and_confirmation_rules(
    tmp_path, monkeypatch
):
    (tmp_path / claims_audit.CHARTER_FILE).write_text("{}", encoding="utf-8")
    rules = types.SimpleNamespace(ledger_rule="lord++", confirmation_rule="bh")
    monkeypatch.setattr(
        sealing,
        "read_verified",
        lambda path, signing_key=None: charter_namespace(rules),
    )

    def ledger_with(used, confirmed):
        return types.SimpleNamespace(
            header={"charter_hash": "h" * 40, "rule": {"rule": used}},
            confirmation=confirmed,
        )

    step = "chartered rules were used"
    same = claims_audit.check_charter(
        tmp_path, ledger_with("lord++", {"rule": "bh"}), None
    )
    assert same[1] == finding(step, "pass", "lord++ and bh")
    assert claims_audit.check_charter(
        tmp_path, ledger_with("lord++", None), None
    )[1] == finding(step, "pass", "lord++ and bh")
    assert (
        claims_audit.check_charter(
            tmp_path, ledger_with("lord++", {"other": 1}), None
        )[1].status
        == "pass"
    )
    wrong_confirm = claims_audit.check_charter(
        tmp_path, ledger_with("lord++", {"rule": "holm"}), None
    )
    assert wrong_confirm[1] == finding(
        step,
        "fail",
        "ledger rule 'lord++' / confirmation rule 'holm' differ from the "
        "charter's 'lord++' / 'bh'",
    )
    wrong_ledger = claims_audit.check_charter(
        tmp_path, ledger_with("bh", {"rule": "bh"}), None
    )
    assert wrong_ledger[1].status == "fail"
    assert wrong_ledger[1].detail.startswith("ledger rule 'bh' / ")
    other_hash = types.SimpleNamespace(
        header={"charter_hash": "z", "rule": {"rule": "x"}}, confirmation=None
    )
    assert claims_audit.check_charter(tmp_path, other_hash, None) == [
        finding(CHARTER, "fail", "the ledger names another charter hash")
    ]


def test_header_mismatches_are_named_in_a_fixed_order():
    survivors = types.SimpleNamespace(sha256="s", hypothesis_ids=("h1",))
    row = {"outcome": "o", "rule": "r", "alpha_confirm": "a"}
    document = {
        "survivors_sha256": "s",
        "outcome": "o",
        "rule": "r",
        "alpha_confirm": "a",
    }
    assert claims_audit.ledger_header_mismatches(document, row, survivors) == []
    assert claims_audit.ledger_header_mismatches(
        {"survivors_sha256": "x"}, {}, survivors
    ) == ["survivor hash"]
    assert claims_audit.ledger_header_mismatches(
        {"survivors_sha256": "x", "outcome": "p", "rule": "q"},
        row,
        survivors,
    ) == ["survivor hash", "outcome", "rule", "alpha_confirm"]


def make_sealed_ledger(confirmed=None, outcome="confirmed", stage_two=None):
    row = {
        "outcome": outcome,
        "rule": "holm",
        "alpha_confirm": "1/20",
        "stage_two": stage_two,
    }
    if confirmed is not None:
        row["confirmed"] = confirmed
    survivors = types.SimpleNamespace(sha256="s", hypothesis_ids=("h1", "h2"))
    return types.SimpleNamespace(confirmation=row, survivor_set=survivors)


def make_claims_document(**changes):
    document = {
        "survivors_sha256": "s",
        "outcome": "confirmed",
        "rule": "holm",
        "alpha_confirm": "1/20",
        "stage_two": None,
        "claims": [
            {"claim_id": "h1", "tier": "confirmed"},
            {"claim_id": "h2", "tier": "unverified"},
        ],
    }
    document.update(changes)
    return document


def test_claims_against_ledger_findings_are_exact():
    name = "claims agree with the sealed ledger"
    ledger = make_sealed_ledger(confirmed=["h1"])
    assert claims_audit.check_claims_against_ledger(
        make_claims_document(), ledger
    ) == finding(name, "pass", "1 confirmed of 2 claims")
    unsealed = types.SimpleNamespace(confirmation=None, survivor_set=None)
    assert claims_audit.check_claims_against_ledger(
        make_claims_document(), unsealed
    ) == finding(name, "fail", "the ledger is not sealed")
    half = types.SimpleNamespace(confirmation={}, survivor_set=None)
    assert (
        claims_audit.check_claims_against_ledger(
            make_claims_document(), half
        ).status
        == "fail"
    )
    wrong_set = make_claims_document(
        claims=[
            {"claim_id": "h1", "tier": "unverified"},
            {"claim_id": "h2", "tier": "unverified"},
        ],
        stage_two={"x": 1},
    )
    assert claims_audit.check_claims_against_ledger(
        wrong_set, ledger
    ) == finding(
        name,
        "fail",
        "differs from the ledger: confirmed set, second-stage accounting",
    )
    odd_ids = make_claims_document(claims=[{"claim_id": "h9"}])
    assert claims_audit.check_claims_against_ledger(
        odd_ids, make_sealed_ledger(confirmed=[])
    ).detail == (
        "differs from the ledger: claim ids differ from the frozen survivors"
    )


def test_claims_against_ledger_when_there_were_no_survivors():
    name = "claims agree with the sealed ledger"
    none = confirm.NO_SURVIVORS
    ledger = make_sealed_ledger(outcome=none)
    document = make_claims_document(outcome=none, claims=[])
    assert claims_audit.check_claims_against_ledger(
        document, ledger
    ) == finding(name, "pass", "0 confirmed of 0 claims")
    listed = make_claims_document(
        outcome=none, claims=[{"claim_id": "h1", "tier": "unverified"}]
    )
    assert claims_audit.check_claims_against_ledger(listed, ledger).detail == (
        "differs from the ledger: claim ids differ from the frozen survivors"
    )


def replay_document(adjusted, p_value=1e-6, tier="confirmed"):
    return {
        "outcome": confirm.CONFIRMED,
        "rule": "holm",
        "alpha_confirm": "1/20",
        "claims": [
            {"claim_id": "h1", "tier": tier, "adjusted_p_value": adjusted}
        ],
        "checks": [{"check_id": "h1:C", "p_value": p_value}],
    }


def test_replay_tolerates_tiny_absolute_differences_only():
    name = "confirmation replays from the p-values on C"
    assert claims_audit.check_replay(replay_document(1e-6 + 5e-13)) == finding(
        name, "pass", "1 hypotheses, rule holm"
    )
    assert claims_audit.check_replay(replay_document(1e-6 + 5e-6)) == finding(
        name, "fail", "h1: adjusted p differs"
    )
    assert claims_audit.check_replay(replay_document(None)) == finding(
        name, "fail", "h1: adjusted p differs"
    )
    assert claims_audit.check_replay(
        replay_document(1e-6, tier="unverified")
    ) == finding(name, "fail", "h1: tier differs")
    broken = replay_document(1e-6)
    del broken["checks"]
    assert claims_audit.check_replay(broken) == finding(
        name, "fail", "cannot replay: KeyError('checks')"
    )
    skipped = claims_audit.check_replay({"outcome": "none"})
    assert skipped.status == "skipped"
    assert skipped.name == name
    assert skipped.detail == "outcome is 'none'"


def test_stage_two_record_problems_are_named_in_order():
    survivors = types.SimpleNamespace(sha256="s", hypothesis_ids=("h1", "h2"))
    good = {"family": [], "family_sha256": "d", "survivors_sha256": "s"}
    assert (
        claims_audit.stage_two_record_problems(
            good, survivors, {"h1", "h2"}, [], "d"
        )
        == []
    )
    bad = {"family": ["h1"], "family_sha256": "x", "survivors_sha256": "y"}
    assert claims_audit.stage_two_record_problems(
        bad, survivors, {"h1", "h9"}, [], "d"
    ) == [
        "a withdrawn id is not a survivor",
        "family is not survivors minus withdrawn",
        "family hash",
        "survivor hash",
    ]


def stage_two_ledger():
    survivors = types.SimpleNamespace(
        sha256="s", hypothesis_ids=("h1", "h2", "h3")
    )
    return types.SimpleNamespace(survivor_set=survivors)


def stage_two_document(**changes):
    family = ["h1", "h2"]
    digest = scmle.data.ledger.ledger.digest(
        {"survivors_sha256": "s", "family": family}
    )
    document = {
        "stage_two": {
            "withdrawn": {"h3": "why"},
            "family": family,
            "family_sha256": digest,
            "survivors_sha256": "s",
        },
        "checks": [{"check_id": "h1:C"}, {"check_id": "h2:C"}],
        "claims": [
            {"claim_id": "h1", "tier": "confirmed"},
            {"claim_id": "h3", "tier": "unverified"},
        ],
    }
    document.update(changes)
    return document


def test_stage_two_findings_are_exact():
    name = "second-stage accounting is consistent"
    ledger = stage_two_ledger()
    assert claims_audit.check_stage_two(stage_two_document(), ledger) == (
        finding(name, "pass", "2 tested, 1 withdrawn before C opened")
    )
    assert claims_audit.check_stage_two({}, ledger) == finding(
        name, "skipped", "no withdrawal rule ran"
    )
    unfrozen = types.SimpleNamespace(survivor_set=None)
    assert claims_audit.check_stage_two(stage_two_document(), unfrozen) == (
        finding(name, "fail", "the ledger has no frozen survivors")
    )
    tested = stage_two_document(
        checks=[{"check_id": "h1:C"}, {"check_id": "h3:C"}]
    )
    assert claims_audit.check_stage_two(tested, ledger).detail == (
        "a withdrawn survivor was tested on C; "
        "a check on C is outside the family"
    )
    outside = stage_two_document(checks=[{"check_id": "h7:C"}])
    assert claims_audit.check_stage_two(outside, ledger).detail == (
        "a check on C is outside the family"
    )
    confirmed = stage_two_document(
        claims=[{"claim_id": "h3", "tier": "confirmed"}]
    )
    assert claims_audit.check_stage_two(confirmed, ledger) == finding(
        name, "fail", "a withdrawn survivor is confirmed"
    )
    bad_hash = stage_two_document()
    bad_hash["stage_two"]["family_sha256"] = "x"
    assert claims_audit.check_stage_two(bad_hash, ledger).detail == (
        "family hash"
    )


def test_custody_open_finding_judges_how_often_c_was_opened():
    name = "n"
    confirmed = {"outcome": confirm.CONFIRMED}
    other = {"outcome": "none"}
    assert claims_audit.custody_open_finding(name, confirmed, 2) == finding(
        name, "fail", "C was opened 2 times"
    )
    assert claims_audit.custody_open_finding(name, other, 2) == finding(
        name, "fail", "C was opened 2 times"
    )
    assert claims_audit.custody_open_finding(name, confirmed, 0) == finding(
        name, "fail", "outcome is confirmed but C has no open event"
    )
    assert claims_audit.custody_open_finding(name, other, 0) == finding(
        name, "pass", "C not opened (outcome none)"
    )
    assert claims_audit.custody_open_finding(name, confirmed, 1) is None
    assert claims_audit.custody_open_finding(name, other, 1) is None


def test_frozen_selection_problems_are_named():
    frozen = {"event": "selection_frozen", "detail": {"survivors_sha256": "s"}}
    opened = {"event": "open"}
    document = {"survivors_sha256": "s"}
    names = ["selection_frozen", "open"]
    assert (
        claims_audit.frozen_selection_problem([frozen, opened], names, document)
        is None
    )
    missing = "no single selection_frozen before the open"
    assert (
        claims_audit.frozen_selection_problem([opened], ["open"], document)
        == missing
    )
    assert (
        claims_audit.frozen_selection_problem(
            [frozen, frozen, opened],
            ["selection_frozen", "selection_frozen", "open"],
            document,
        )
        == missing
    )
    assert (
        claims_audit.frozen_selection_problem(
            [opened, frozen], ["open", "selection_frozen"], document
        )
        == missing
    )
    assert (
        claims_audit.frozen_selection_problem(
            [frozen, opened], names, {"survivors_sha256": "other"}
        )
        == "frozen survivor hash differs from the claims"
    )
    bare = {"event": "selection_frozen"}
    assert (
        claims_audit.frozen_selection_problem([bare, opened], names, {}) is None
    )
    with_family = dict(document, stage_two={"family_sha256": "f"})
    assert (
        claims_audit.frozen_selection_problem(
            [frozen, opened], names, with_family
        )
        == "frozen family hash differs from the claims"
    )
    matching = {
        "event": "selection_frozen",
        "detail": {"survivors_sha256": "s", "family_sha256": "f"},
    }
    assert (
        claims_audit.frozen_selection_problem(
            [matching, opened], names, with_family
        )
        is None
    )


def write_custody(folder, events, key=KEY):
    log = sealed.OpenLog(folder / claims_audit.CUSTODY_FILE, key)
    for set_id, event, detail in events:
        log.append(set_id, event, "tester", detail)
    return log


def test_custody_findings_are_exact_for_hand_made_logs(tmp_path):
    document = {"outcome": confirm.CONFIRMED, "survivors_sha256": "s"}
    assert claims_audit.check_custody(tmp_path, document, KEY) == finding(
        CUSTODY, "skipped", "custody_log.jsonl is not in the run"
    )
    write_custody(tmp_path, [("C", "sealed", {})])
    assert claims_audit.check_custody(tmp_path, document, KEY) == finding(
        CUSTODY, "fail", "outcome is confirmed but C has no open event"
    )
    other = {"outcome": "none", "survivors_sha256": "s"}
    assert claims_audit.check_custody(tmp_path, other, KEY) == finding(
        CUSTODY, "pass", "C not opened (outcome none)"
    )
    write_custody(
        tmp_path,
        [
            ("C", "open", {}),
            ("C", "selection_frozen", {"survivors_sha256": "s"}),
        ],
    )
    assert claims_audit.check_custody(tmp_path, document, KEY) == finding(
        CUSTODY, "fail", "no single selection_frozen before the open"
    )


def test_custody_pass_counts_entries_and_a_wrong_key_fails(tmp_path):
    events = [
        ("C", "selection_frozen", {"survivors_sha256": "s"}),
        ("C", "open", {}),
        ("final_test", "sealed", {}),
    ]
    log = write_custody(tmp_path, events)
    document = {"outcome": confirm.CONFIRMED, "survivors_sha256": "s"}
    assert claims_audit.check_custody(tmp_path, document, KEY) == finding(
        CUSTODY, "pass", f"{len(log.entries)} entries, chain verified"
    )
    with pytest.raises(errors.ScmleError) as caught:
        sealed.OpenLog(tmp_path / claims_audit.CUSTODY_FILE, "wrong-key")
    bad = claims_audit.check_custody(tmp_path, document, "wrong-key")
    assert bad == finding(CUSTODY, "fail", str(caught.value))
    decision = claims_audit.check_decision_files(tmp_path, "wrong-key")
    assert decision == finding(DECISION, "fail", str(caught.value))
    coverage = tmp_path / coverage_run.CLAIM_FILE
    coverage.write_text("{}", encoding="utf-8")
    assert claims_audit.check_coverage_claim(tmp_path, "wrong-key") == finding(
        COVERAGE, "fail", str(caught.value)
    )
    assert claims_audit.verify(tmp_path, "wrong-key").ok is False


def test_decision_files_without_a_registration_are_an_older_run(tmp_path):
    assert claims_audit.check_decision_files(tmp_path, KEY) is None
    write_custody(tmp_path, [("C", "sealed", {})])
    assert claims_audit.check_decision_files(tmp_path, KEY) is None
    (tmp_path / decision_reports.DECISION_FILES[0]).write_text(
        "{}", encoding="utf-8"
    )
    assert claims_audit.check_decision_files(tmp_path, KEY) == finding(
        DECISION,
        "skipped",
        "the run registered no decision file hashes (older run)",
    )


def test_decision_files_registration_counts_and_joins_problems(tmp_path):
    folder = tmp_path
    names = decision_reports.DECISION_FILES
    for name in names[:3]:
        (folder / name).write_text(json.dumps({"n": name}), encoding="utf-8")
    digests = decision_reports.file_digests(folder)
    write_custody(
        folder,
        [
            (
                "final_test",
                decision_reports.REGISTRATION_EVENT,
                {"files": digests},
            )
        ],
    )
    assert claims_audit.check_decision_files(folder, KEY) == finding(
        DECISION, "pass", "3 file(s) hash to the registration"
    )
    (folder / names[0]).write_text("{}", encoding="utf-8")
    (folder / names[1]).unlink()
    failed = claims_audit.check_decision_files(folder, KEY)
    assert failed == finding(
        DECISION,
        "fail",
        f"{names[0]}: differs from the registered hash; {names[1]}: missing",
    )
    write_custody(
        folder,
        [
            (
                "final_test",
                decision_reports.REGISTRATION_EVENT,
                {"files": digests},
            )
        ],
    )
    assert claims_audit.check_decision_files(folder, KEY) == finding(
        DECISION, "fail", "the decision files were registered twice"
    )


def test_a_decision_registration_without_files_registers_nothing(tmp_path):
    write_custody(
        tmp_path, [("final_test", decision_reports.REGISTRATION_EVENT, {})]
    )
    assert claims_audit.check_decision_files(tmp_path, KEY) == finding(
        DECISION, "pass", "0 file(s) hash to the registration"
    )


def claim_registration(folder, claim, registrations=1):
    (folder / coverage_run.CLAIM_FILE).write_text(
        json.dumps(claim), encoding="utf-8"
    )
    digest = scmle.data.ledger.ledger.digest(claim)
    events = [
        ("final_test", coverage_run.REGISTRATION_EVENT, {"sha256": digest})
    ] * registrations
    write_custody(folder, events)
    return digest


def test_coverage_claim_findings_are_exact(tmp_path):
    assert claims_audit.check_coverage_claim(tmp_path, KEY) is None
    claim = {"levels": [0.9]}
    digest = claim_registration(tmp_path, claim)
    assert claims_audit.check_coverage_claim(tmp_path, KEY) == finding(
        COVERAGE, "pass", "claim hash registered; the report names it"
    )
    report = tmp_path / coverage_run.REPORT_FILE
    report.write_text(json.dumps({"claim_sha256": digest}), encoding="utf-8")
    assert claims_audit.check_coverage_claim(tmp_path, KEY).status == "pass"
    report.write_text(json.dumps({"claim_sha256": "x"}), encoding="utf-8")
    assert claims_audit.check_coverage_claim(tmp_path, KEY) == finding(
        COVERAGE, "fail", "coverage_report.json: names another claim"
    )
    report.write_text("{bad", encoding="utf-8")
    assert claims_audit.check_coverage_claim(tmp_path, KEY).detail == (
        "coverage_report.json: names another claim"
    )
    (tmp_path / coverage_run.CLAIM_FILE).write_text(
        json.dumps({"levels": [0.5]}), encoding="utf-8"
    )
    assert claims_audit.check_coverage_claim(tmp_path, KEY).detail == (
        "coverage_claim.json: differs from the registered hash; "
        "coverage_report.json: names another claim"
    )


def test_coverage_claim_registration_counts(tmp_path):
    claim_registration(tmp_path, {"levels": [0.9]}, registrations=2)
    assert claims_audit.check_coverage_claim(tmp_path, KEY) == finding(
        COVERAGE, "fail", "2 registrations in the custody log"
    )
    other = tmp_path / "other"
    other.mkdir()
    (other / coverage_run.CLAIM_FILE).write_text("{}", encoding="utf-8")
    write_custody(other, [("C", "sealed", {})])
    assert claims_audit.check_coverage_claim(other, KEY) == finding(
        COVERAGE, "fail", "0 registrations in the custody log (no registration)"
    )
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / coverage_run.CLAIM_FILE).write_text("{}", encoding="utf-8")
    assert claims_audit.check_coverage_claim(bare, KEY) == finding(
        COVERAGE, "skipped", "custody_log.jsonl is not in the run"
    )


def test_a_coverage_report_without_its_claim_file_fails(tmp_path):
    (tmp_path / coverage_run.REPORT_FILE).write_text("{}", encoding="utf-8")
    write_custody(tmp_path, [("C", "sealed", {})])
    found = claims_audit.check_coverage_claim(tmp_path, KEY)
    assert found.status == "fail"
    assert found.detail.startswith("coverage_claim.json: missing or unreadable")
    two = found.detail.split("; ")
    assert two[1] == "0 registrations in the custody log (no registration)"


def test_coverage_confirmation_is_skipped_for_an_unevaluated_claim(tmp_path):
    name = "coverage claim on C: registered, recomputed, status matches"
    ledger = types.SimpleNamespace(claims={})
    assert claims_audit.check_coverage_confirmation(tmp_path, ledger) is None
    document = tmp_path / coverage_confirm.DOCUMENT
    document.write_text(
        json.dumps({"status": confirm.NOT_EVALUATED}), encoding="utf-8"
    )
    assert claims_audit.check_coverage_confirmation(
        tmp_path, ledger
    ) == finding(name, "skipped", "the claim could not be tested on C")
    document.write_text(json.dumps({"status": "x"}), encoding="utf-8")
    assert claims_audit.check_coverage_confirmation(
        tmp_path, ledger
    ) == finding(name, "fail", "the claim is not registered in the ledger")


def test_control_findings_are_exact(run):
    ledger = claims_audit.read_ledger(run)
    name = "negative control matches the ledger"
    assert claims_audit.check_control(run, ledger) == finding(
        name, "pass", "hash equal"
    )
    path = run / claims_audit.CONTROL_FILE
    path.write_text(json.dumps({"changed": 1}), encoding="utf-8")
    assert claims_audit.check_control(run, ledger) == finding(
        name, "fail", "the file differs from the ledger's control row"
    )
    path.unlink()
    assert claims_audit.check_control(run, ledger) == finding(
        name, "skipped", "no control file or no control row"
    )


def test_attestations_are_skipped_without_both_files(run, tmp_path):
    name = "causal attestations verify"
    expected = finding(name, "skipped", "no causal claims or charter file")
    assert claims_audit.check_attestations(run, KEY) == expected
    (run / claims_audit.CAUSAL_FILE).write_text("{}", encoding="utf-8")
    (run / claims_audit.CHARTER_FILE).unlink()
    assert claims_audit.check_attestations(run, KEY) == expected
    other = tmp_path / "other"
    other.mkdir()
    (other / claims_audit.CHARTER_FILE).write_text("{}", encoding="utf-8")
    assert claims_audit.check_attestations(other, KEY) == expected


def test_verify_reports_the_chain_step_exactly(run, tmp_path):
    result = claims_audit.verify(run, KEY)
    ledger = claims_audit.read_ledger(run)
    assert result.run_dir == str(run)
    assert result.findings[0] == finding(
        CHAIN, "pass", f"{len(ledger.rows)} rows, head {ledger.head[:12]}"
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(scmle.data.ledger.ledger.LedgerError) as caught:
        scmle.data.ledger.ledger.Ledger.load(
            empty / scmle.data.ledger.ledger.FILE_NAME
        )
    assert claims_audit.verify(empty, KEY) == claims_audit.Verification(
        str(empty), (finding(CHAIN, "fail", str(caught.value)),)
    )


def test_verify_stops_after_the_chain_when_claims_are_unreadable(run):
    (run / confirm.CLAIMS_FILE).unlink()
    result = claims_audit.verify(run, KEY)
    assert [f.name for f in result.findings] == [CHAIN, "claims file reads"]
    assert result.findings[1].status == "fail"
    assert result.findings[1].detail.startswith("cannot read ")


def test_show_claim_reports_a_missing_or_present_check(run, monkeypatch):
    edit_claims(
        run,
        lambda d: d.update(
            claims=[{"claim_id": "h1", "tier": "confirmed"}], outcome="o"
        ),
    )

    def ledger_with(check):
        result = types.SimpleNamespace(decision="d", level=0.5, check=check)
        spec = types.SimpleNamespace(to_dict=lambda: {"column": "c"})
        return types.SimpleNamespace(
            claims={},
            hypotheses={"h1": types.SimpleNamespace(spec=spec)},
            results={"h1": result},
        )

    monkeypatch.setattr(
        claims_audit, "read_ledger", lambda folder: ledger_with(None)
    )
    bare = claims_audit.show_claim(run, "h1")
    assert bare["exploration"] == {
        "decision": "d",
        "level": "0.5",
        "check": None,
    }
    assert bare["hypothesis"] == {"column": "c"}
    held = types.SimpleNamespace(to_dict=lambda: {"p": 1})
    monkeypatch.setattr(
        claims_audit, "read_ledger", lambda folder: ledger_with(held)
    )
    assert claims_audit.show_claim(run, "h1")["exploration"]["check"] == {
        "p": 1
    }


def test_registered_rows_of_an_unknown_kind_have_no_identification(tmp_path):
    ledger = types.SimpleNamespace(
        claims={"x": {"kind": "other"}}, confirmation=None
    )
    row = claims_audit.registered_rows(tmp_path, ledger)[0]
    assert row["identification"] is None
    assert row["tier"] == "unverified"


def test_baseline_report_problems_name_the_mismatched_registration(tmp_path):
    ledger = types.SimpleNamespace(
        claims={
            "base": {"kind": "baseline", "sha256": SHA},
            "cause": {"kind": "causal_primary", "sha256": "b" * 64},
        }
    )
    assert claims_audit.baseline_report_problems(tmp_path, ledger) == []
    report = tmp_path / claims_audit.BASELINE_FILE
    report.write_text(json.dumps({"registration_hash": SHA}), encoding="utf-8")
    assert claims_audit.baseline_report_problems(tmp_path, ledger) == []
    report.write_text(json.dumps({"registration_hash": "x"}), encoding="utf-8")
    assert claims_audit.baseline_report_problems(tmp_path, ledger) == [
        "base: report names another registration"
    ]


def test_registered_claim_step_is_exact(tmp_path):
    name = "registered claims: files hash to the sealed ledger"
    registered_claim_run(tmp_path)
    ledger = claims_audit.read_ledger(tmp_path)
    document = claims_audit.read_claims(tmp_path)
    assert claims_audit.check_registered(tmp_path, document, ledger) == (
        finding(name, "pass", "2 claim(s), 2 file(s) bound")
    )
    bare = tmp_path / "bare"
    bare.mkdir()
    registered_claim_run(bare, baseline_report=False, causal_report=False)
    plain = claims_audit.check_registered(
        bare,
        claims_audit.read_claims(bare),
        claims_audit.read_ledger(bare),
    )
    assert plain == finding(
        name, "fail", "attachment hashes differ from the claims file"
    )


def test_registered_claim_step_names_each_problem(tmp_path):
    name = "registered claims: files hash to the sealed ledger"
    ledger = types.SimpleNamespace(
        confirmation={"claims": {}, "attachments": {}},
        claims={"x": {"kind": "other", "sha256": "s"}},
    )
    document = {"registered_claims": {}, "attachments": {}}
    assert claims_audit.check_registered(tmp_path, document, ledger) == (
        finding(name, "fail", "x: registration hash differs")
    )
    skewed = {"registered_claims": {"y": 1}, "attachments": {"f": "h"}}
    assert claims_audit.check_registered(tmp_path, skewed, ledger) == finding(
        name,
        "fail",
        "claims file and confirmation row differ; "
        "x: registration hash differs; "
        "attachment hashes differ from the claims file",
    )
    stated = types.SimpleNamespace(
        confirmation={
            "claims": {"x": {"sha256": "other"}},
            "attachments": {"a.json": "h"},
        },
        claims={"x": {"kind": "other", "sha256": "s"}},
    )
    both = {"registered_claims": {"x": {"sha256": "other"}}, "attachments": {}}
    assert claims_audit.check_registered(tmp_path, both, stated).detail == (
        "x: registration hash differs; a.json: missing; "
        "attachment hashes differ from the claims file"
    )


def test_a_registered_coverage_claim_without_a_document_fails(tmp_path):
    name = "coverage claim on C: registered, recomputed, status matches"
    ledger = types.SimpleNamespace(
        claims={coverage_confirm.CLAIM_ID: {"sha256": "s"}}
    )
    (tmp_path / coverage_confirm.DOCUMENT).write_text(
        json.dumps({"status": confirm.NOT_EVALUATED}), encoding="utf-8"
    )
    assert claims_audit.check_coverage_confirmation(
        tmp_path, ledger
    ) == finding(name, "fail", "a registered claim has no evaluation document")


def test_attestations_of_a_causal_charter_without_attestors(run):
    name = "causal attestations verify"
    path = run / claims_audit.CHARTER_FILE
    sealing.write_charter(path, causal_charter(), signing_key=KEY)
    (run / claims_audit.CAUSAL_FILE).write_text(
        json.dumps({"tier": "unverified"}), encoding="utf-8"
    )
    assert claims_audit.check_attestations(run, KEY) == finding(
        name, "pass", "0 attestation(s), none required"
    )
    bare = json.loads(path.read_text(encoding="utf-8"))["charter"]
    path.write_text(json.dumps(bare), encoding="utf-8")
    assert claims_audit.check_attestations(run, KEY) == finding(
        name, "pass", "0 attestation(s), none required"
    )
    (run / claims_audit.CAUSAL_FILE).write_text(
        json.dumps({"tier": "attested"}), encoding="utf-8"
    )
    raised = claims_audit.check_attestations(run, KEY)
    assert raised == finding(
        name,
        "fail",
        "tier above unverified but the charter names no attestors; "
        "tier above unverified with no accepted attestation",
    )


def test_attestations_of_a_charter_without_a_causal_block(run):
    name = "causal attestations verify"
    (run / claims_audit.CAUSAL_FILE).write_text("{}", encoding="utf-8")
    assert claims_audit.check_attestations(run, KEY) == finding(
        name, "skipped", "the charter has no causal block"
    )
    path = run / claims_audit.CHARTER_FILE
    path.write_text(json.dumps({"charter": {"question": 1}}), encoding="utf-8")
    broken = claims_audit.check_attestations(run, KEY)
    assert broken.name == name
    assert broken.status == "fail"


def test_verify_passes_its_key_to_every_log_reading_step(run):
    register_decision_files(run, FOUR)
    claim_registration(run, {"levels": [0.9]})
    (run / coverage_confirm.DOCUMENT).write_text(
        json.dumps({"status": confirm.NOT_EVALUATED}), encoding="utf-8"
    )
    good = {f.name: f.status for f in claims_audit.verify(run, KEY).findings}
    assert good[DECISION] == "pass"
    assert good[COVERAGE] == "pass"
    assert good[CUSTODY] == "pass"
    assert (
        good["coverage claim on C: registered, recomputed, status matches"]
        == "skipped"
    )
    wrong = {
        f.name: f.status for f in claims_audit.verify(run, "wrong-key").findings
    }
    assert wrong[DECISION] == "fail"
    assert wrong[COVERAGE] == "fail"
    assert wrong[CUSTODY] == "fail"


def test_verify_names_its_directory_when_the_claims_cannot_be_read(run):
    (run / confirm.CLAIMS_FILE).unlink()
    assert claims_audit.verify(run, KEY).run_dir == str(run)
