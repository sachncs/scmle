"""The production claim is refused unless both providers' evidence is whole."""

import dataclasses
import json
import pathlib
import shutil

import pytest

from scmle.operations import claim
from scmle.operations import recovery
from scripts import check_production_claim as cli
from tests import support_claim as sc
from tests import support_release as sr

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "plan" / "evidence" / "phase-17"


def verdict(root):
    return claim.check_claim(root, sc.PIPELINE_KEY, sc.REVIEWER_KEY)


def edit(folder, kind, change):
    """Rewrites a record's body with the pipeline key, leaving the sign-off."""
    record = json.loads((folder / f"{kind}.json").read_text())
    body = change(record["body"]) or record["body"]
    sc.write(folder, kind, record["provider"], body, commit=record["commit"])


@pytest.fixture
def tree(tmp_path):
    return sc.build(tmp_path / "phase-17")


def test_a_complete_signed_tree_is_the_only_thing_that_allows_the_claim(tree):
    found = verdict(tree)
    assert found.allowed and not found.problems
    assert found.to_dict()["claim"] == claim.CLAIM


def test_the_committed_evidence_folders_do_not_allow_the_claim():
    found = claim.check_claim(EVIDENCE, "k" * 12, "r" * 12)
    assert not found.allowed
    for provider in claim.PROVIDERS:
        assert found.providers[provider]


@pytest.mark.parametrize("provider", claim.PROVIDERS)
def test_a_missing_provider_folder_refuses(tree, provider):
    shutil.rmtree(tree / provider)
    found = verdict(tree)
    assert not found.allowed
    assert any("does not exist" in p for p in found.providers[provider])
    other = next(p for p in claim.PROVIDERS if p != provider)
    assert not found.providers[other]


@pytest.mark.parametrize("provider", claim.PROVIDERS)
@pytest.mark.parametrize("kind", [*claim.KINDS, claim.SIGNOFF])
def test_each_missing_record_refuses_on_that_provider_only(
    tree, provider, kind
):
    (tree / provider / f"{kind}.json").unlink()
    found = verdict(tree)
    assert not found.allowed
    assert any(f"{kind}.json is missing" in p for p in found.problems)
    assert all(p.startswith(provider) for p in found.problems)


def test_a_record_that_is_not_json_or_not_signed_refuses(tree):
    folder = tree / "aws"
    (folder / "recovery.json").write_text("{not json")
    (folder / "acceptance.json").write_text(json.dumps({"kind": "acceptance"}))
    found = verdict(tree)
    text = " ".join(found.problems)
    assert "recovery.json is not readable JSON" in text
    assert "acceptance.json is not a signed record" in text


def test_a_changed_body_or_a_wrong_key_breaks_the_signature(tree):
    path = tree / "gcp" / "observability.json"
    record = json.loads(path.read_text())
    record["body"]["alerts_delivered"] = True
    record["body"]["extra"] = 1
    path.write_text(json.dumps(record))
    assert any("signature does not verify" in p for p in verdict(tree).problems)
    other = claim.check_claim(tree, "a-different-pipeline-key", sc.REVIEWER_KEY)
    assert not other.allowed


def test_records_from_production_or_another_provider_or_commit_refuse(tree):
    folder = tree / "aws"
    sc.write(
        folder, "acceptance", "aws", sc.acceptance(), environment="production"
    )
    sc.write(folder, "observability", "gcp", sc.observability())
    sc.write(
        folder,
        "continuous-evaluation",
        "aws",
        sc.continuous(),
        commit="b" * 40,
    )
    text = " ".join(verdict(tree).problems)
    assert "environment is 'production'" in text
    assert "it is for 'gcp'" in text
    assert "different commits" in text


def test_every_acceptance_criterion_must_have_passed_with_evidence(tree):
    def fail(body):
        body["items"][claim.ACCEPTANCE[0]]["status"] = "failed"
        body["items"][claim.ACCEPTANCE[1]]["evidence"] = []
        del body["items"][claim.ACCEPTANCE[2]]

    edit(tree / "aws", "acceptance", fail)
    text = " ".join(verdict(tree).problems)
    assert f"{claim.ACCEPTANCE[0]} did not pass" in text
    assert f"{claim.ACCEPTANCE[1]} cites no evidence" in text
    assert f"{claim.ACCEPTANCE[2]} did not pass" in text


def test_the_deployment_record_must_verify_match_and_be_independent(tree):
    staged = sr.staging_record()
    cases = {
        "signature": dataclasses.replace(staged, provider="aws", approver="x"),
        "provider": staged.signed(sc.PIPELINE_KEY),
        "production": dataclasses.replace(
            staged, provider="gcp", environment="production"
        ).signed(sc.PIPELINE_KEY),
        "approver": dataclasses.replace(
            staged, provider="gcp", approver=staged.requested_by
        ).signed(sc.PIPELINE_KEY),
        "commit": dataclasses.replace(
            staged, provider="gcp", source_commit="c" * 40
        ).signed(sc.PIPELINE_KEY),
    }
    expected = {
        "signature": "signature does not verify",
        "provider": "another provider",
        "production": "not for staging",
        "approver": "approver is its requester",
        "commit": "another commit",
    }
    for name, record in cases.items():
        fresh = sc.build(tree.parent / f"copy-{name}")
        folder = fresh / ("aws" if name == "signature" else "gcp")
        if name == "provider":
            folder = fresh / "gcp"
        sc.write(
            folder,
            "deployment",
            folder.name,
            {"record": record.to_dict()},
        )
        sc.sign_off(folder, folder.name)
        text = " ".join(verdict(fresh).problems)
        assert expected[name] in text, name
    garbage = sc.build(tree.parent / "garbage")
    sc.write(garbage / "aws", "deployment", "aws", {"record": {"x": 1}})
    assert "no valid deployment record" in " ".join(verdict(garbage).problems)


def test_recovery_must_cover_every_fault_in_staging_within_the_objectives(
    tree,
):
    def spoil(body):
        results = body["results"]
        results[0]["status"] = "not_run"
        results[1]["environment"] = "local subprocess"
        results[2]["measurements"][0]["rto_seconds"] = 9999.0
        results[3]["measurements"][0]["rpo_lost"] = 1
        results[4]["measurements"][0]["rto_seconds"] = None
        del results[5]

    edit(tree / "aws", "recovery", spoil)
    text = " ".join(verdict(tree).problems)
    assert "is 'not_run'" in text
    assert "did not run in aws staging" in text
    assert "RTO misses" in text and "RPO misses" in text
    assert f"{recovery.CATALOGUE[5].id} has no result" in text


def test_a_component_that_was_never_measured_refuses(tree):
    def strip(body):
        for result in body["results"]:
            result["measurements"] = [
                m for m in result["measurements"] if m["component"] != "ledger"
            ]

    edit(tree / "gcp", "recovery", strip)
    assert "ledger was never measured" in " ".join(verdict(tree).problems)


def test_observability_needs_both_classifications_alerts_and_operator_actions(
    tree,
):
    def spoil(body):
        body["injected_failure"]["verdict"] = "mixed"
        body["controlled_drift"]["verdict"] = "none"
        body["alerts_delivered"] = False
        body["operator_actions"] = ["incident_acknowledged"]
        del body["telemetry_retained"]

    edit(tree / "aws", "observability", spoil)
    text = " ".join(verdict(tree).problems)
    assert "not classified as infrastructure" in text
    assert "not classified as data quality or mixed" in text
    assert "alerts_delivered is not true" in text
    assert "telemetry_retained is not true" in text
    assert "no incident_resolved operator action" in text


def test_continuous_evaluation_needs_a_proposal_a_gate_and_a_confirmed_runtime(
    tree,
):
    edit(
        tree / "gcp",
        "continuous-evaluation",
        lambda body: {"proposal": "none", "gate_allowed": False},
    )
    text = " ".join(verdict(tree).problems)
    for part in (
        "no retraining proposal",
        "no escalation was opened",
        "did not admit the model",
        "runtime did not confirm",
        "no post-promotion check",
    ):
        assert part in text


def test_the_signoff_must_be_independent_signed_by_the_reviewer_and_current(
    tree,
):
    folder = tree / "aws"
    sc.sign_off(folder, "aws", independent=False)
    assert "not recorded as independent" in " ".join(verdict(tree).problems)
    requester = sr.staging_record().requested_by
    sc.sign_off(folder, "aws", reviewer=requester)
    assert "requested the release" in " ".join(verdict(tree).problems)
    sc.sign_off(folder, "aws", reviewer="  ")
    assert "names no reviewer" in " ".join(verdict(tree).problems)
    sc.sign_off(folder, "aws")
    assert verdict(tree).allowed
    edit(folder, "observability", lambda body: body.update(note="edited"))
    assert "does not cover the current observability.json" in " ".join(
        verdict(tree).problems
    )
    sc.write(folder, "signoff", "aws", {"reviewer": "x"}, key=sc.PIPELINE_KEY)
    assert "signoff.json: the signature does not verify" in " ".join(
        verdict(tree).problems
    )


def test_the_pipeline_cannot_sign_for_the_reviewer(tree):
    with pytest.raises(ValueError, match="differ"):
        claim.check_claim(tree, sc.PIPELINE_KEY, sc.PIPELINE_KEY)
    with pytest.raises(ValueError, match="required"):
        claim.check_claim(tree, "", sc.REVIEWER_KEY)


def test_the_fixed_claim_text_is_in_the_releasing_guide():
    assert claim.claim_text_problems(ROOT / "docs" / "RELEASING.md") == []


def test_a_document_without_the_claim_is_refused(tmp_path):
    other = tmp_path / "x.md"
    other.write_text("nothing here")
    assert claim.claim_text_problems(other)
    other.write_text(
        f"{claim.CLAIM_START} a different sentence {claim.CLAIM_END}"
    )
    assert "does not hold" in claim.claim_text_problems(other)[0]
    other.write_text(f"{claim.CLAIM_START}\n> {claim.CLAIM}\n{claim.CLAIM_END}")
    assert claim.claim_text_problems(other) == []


def test_the_command_exits_zero_only_for_a_complete_tree(
    tree, tmp_path, capsys
):
    pipeline = tmp_path / "pipeline.key"
    reviewer = tmp_path / "reviewer.key"
    pipeline.write_text(sc.PIPELINE_KEY)
    reviewer.write_text(sc.REVIEWER_KEY)
    arguments = [
        "--root",
        str(tree),
        "--key-file",
        str(pipeline),
        "--reviewer-key-file",
        str(reviewer),
        "--releasing",
        str(ROOT / "docs" / "RELEASING.md"),
    ]
    assert cli.main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["allowed"] is True
    (tree / "aws" / "recovery.json").unlink()
    assert cli.main(arguments) == 1
    assert json.loads(capsys.readouterr().out)["allowed"] is False
    empty = tmp_path / "empty.key"
    empty.write_text("")
    assert (
        cli.main(
            [
                *arguments[:-6],
                "--key-file",
                str(empty),
                "--reviewer-key-file",
                str(reviewer),
            ]
        )
        == 2
    )
    capsys.readouterr()
    assert cli.main(["--key-file", "missing", "--reviewer-key-file", "x"]) == 2
