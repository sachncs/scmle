"""Pins the approval record, the registry index, promotion and the audit log."""

import dataclasses
import json
import pathlib

import pytest

from scmle import durability
from scmle.artifacts import bundle
from scmle.contracts import errors
from scmle.governance import never_list
from scmle.governance import registry
from tests.support_release import KEY
from tests.support_runtime import approval_for
from tests.support_runtime import make_registry
from tests.unit.test_lifecycle_extras import bundle_of
from tests.unit.test_lifecycle_extras import traffic


@pytest.fixture(scope="module")
def trained():
    return bundle_of("logistic_regression", traffic(300))


@pytest.fixture(scope="module")
def other():
    return bundle_of("decision_tree", traffic(200, seed=9))


def review(**changes):
    values = {
        "model": "m",
        "version": 1,
        "content_id": "c" * 64,
        "reviewer": "r",
        "decision": registry.PromotionDecision.APPROVED,
        "reason": "ok",
        "evidence": ("e1", "e2"),
        "approved_at": "2026-01-01T00:00:00+00:00",
    }
    values.update(changes)
    return registry.PromotionApproval(**values)


def test_the_constants_and_decisions_are_the_documented_ones():
    assert registry.INDEX_FILE == "registry.json"
    assert registry.AUDIT_FILE == "audit.jsonl"
    assert registry.CHAMPION == "champion"
    assert registry.CHALLENGER == "challenger"
    assert registry.PREVIOUS == "previous"
    assert registry.PromotionDecision.APPROVED.value == "approved"
    assert registry.PromotionDecision.REJECTED.value == "rejected"


@pytest.mark.parametrize(
    "name", ["model", "content_id", "reviewer", "reason", "approved_at"]
)
def test_a_blank_or_non_text_field_names_itself(name):
    for bad in ("", "   ", None, 5):
        with pytest.raises(ValueError, match=rf"^{name} must not be empty$"):
            review(**{name: bad})


def test_the_version_must_be_positive():
    with pytest.raises(ValueError, match=r"^version must be positive$"):
        review(version=0)
    assert review(version=1).version == 1


def test_a_decision_given_as_text_is_converted_and_a_bad_one_is_refused():
    converted = review(decision="rejected")
    assert converted.decision is registry.PromotionDecision.REJECTED
    with pytest.raises(ValueError):
        review(decision="maybe")


def test_evidence_must_be_present_and_every_reference_non_blank():
    for bad in ((), ("ok", ""), ("  ",)):
        with pytest.raises(
            ValueError, match=r"^evidence must contain non-empty references$"
        ):
            review(evidence=bad)


def test_independence_must_be_a_bool_and_the_signature_text():
    with pytest.raises(TypeError, match=r"^independent must be bool$"):
        review(independent=1)
    assert review(independent=False).independent is False
    for bad in ("", "  ", 5):
        with pytest.raises(
            ValueError, match=r"^signature must be non-empty text$"
        ):
            review(signature=bad)
    assert review(signature=None).signature is None
    assert review(signature="s").signature == "s"


def test_the_signed_fields_are_exactly_the_documented_ones():
    approval = review()
    assert approval.unsigned_dict() == {
        "model": "m",
        "version": 1,
        "content_id": "c" * 64,
        "reviewer": "r",
        "decision": "approved",
        "reason": "ok",
        "evidence": ["e1", "e2"],
        "approved_at": "2026-01-01T00:00:00+00:00",
        "independent": True,
    }
    assert "signature" not in approval.to_dict()
    signed = approval.sign(KEY)
    assert signed.to_dict() == {
        **approval.unsigned_dict(),
        "signature": (signed.signature),
    }
    assert signed.signature and approval.signature is None


def test_a_signature_verifies_only_with_the_same_key_and_content(monkeypatch):
    monkeypatch.delenv("SCMLE_SIGNING_KEY", raising=False)
    signed = review().sign(KEY)
    assert signed.verify(KEY) is True
    assert signed.verify(KEY.encode()) is True
    assert signed.verify("another-key") is False
    assert review().verify(KEY) is False
    assert signed.verify() is False
    assert signed.verify("") is False
    tampered = dataclasses.replace(signed, reason="changed")
    assert tampered.verify(KEY) is False
    monkeypatch.setenv("SCMLE_SIGNING_KEY", KEY)
    assert signed.verify() is True
    with pytest.raises(ValueError):
        review().sign(b"")


def test_each_signed_field_is_covered_by_the_signature():
    signed = review().sign(KEY)
    for change in (
        {"model": "n"},
        {"version": 2},
        {"content_id": "d" * 64},
        {"reviewer": "s"},
        {"decision": registry.PromotionDecision.REJECTED},
        {"reason": "no"},
        {"evidence": ("e1",)},
        {"approved_at": "2027-01-01T00:00:00+00:00"},
        {"independent": False},
    ):
        assert dataclasses.replace(signed, **change).verify(KEY) is False


def test_validate_target_checks_identity_decision_independence_and_signature():
    good = review().sign(KEY)
    good.validate_target("m", 1, "c" * 64, KEY)
    for args in (("x", 1, "c" * 64), ("m", 2, "c" * 64), ("m", 1, "d" * 64)):
        with pytest.raises(
            ValueError,
            match=r"^promotion approval does not match bundle identity$",
        ):
            good.validate_target(*args, KEY)
    rejected = review(decision="rejected").sign(KEY)
    with pytest.raises(
        ValueError, match=r"^promotion was rejected by the reviewer$"
    ):
        rejected.validate_target("m", 1, "c" * 64, KEY)
    dependent = review(independent=False).sign(KEY)
    with pytest.raises(
        ValueError,
        match=r"^champion promotion requires independent review$",
    ):
        dependent.validate_target("m", 1, "c" * 64, KEY)
    with pytest.raises(
        ValueError, match=r"^champion promotion requires a valid signature$"
    ):
        review().validate_target("m", 1, "c" * 64, KEY)
    with pytest.raises(
        ValueError, match=r"^champion promotion requires a valid signature$"
    ):
        good.validate_target("m", 1, "c" * 64, "wrong")


def test_a_version_record_defaults_to_an_unsigned_unaddressed_one():
    record = registry.VersionRecord("m", 1, "t", {}, {}, "r", "m/v1")
    assert record.content_id == "" and record.signed is False


def test_the_content_id_is_the_hash_of_the_manifest(trained, tmp_path):
    root = trained.save(tmp_path / "b", signing_key=KEY)
    expected = bundle.file_sha256(root / bundle.MANIFEST_FILE)
    assert registry.content_id_of(root) == expected
    assert registry.content_id_of(str(root)) == expected


def test_a_new_store_creates_its_root_and_has_an_empty_index(tmp_path):
    store = registry.ModelRegistryStore(tmp_path / "a" / "b")
    assert store.root == tmp_path / "a" / "b" and store.root.is_dir()
    assert store.guard is None and store.scope == "stream"
    assert store.lock is durability.path_lock(store.root)
    assert store.read_index() == {"models": {}}
    assert not (store.root / registry.INDEX_FILE).exists()


def test_the_index_is_written_as_indented_json_and_read_back(tmp_path):
    store = registry.ModelRegistryStore(tmp_path)
    index = {"models": {"m": {"versions": {}, "aliases": {}}}}
    store.write_index(index)
    text = (tmp_path / registry.INDEX_FILE).read_text(encoding="utf-8")
    assert text == json.dumps(index, indent=2)
    assert store.read_index() == index


def test_audit_records_are_appended_one_json_line_each(tmp_path):
    store = registry.ModelRegistryStore(tmp_path)
    store.audit("first", model="m", path=pathlib.Path("p"))
    store.audit("second")
    lines = (tmp_path / registry.AUDIT_FILE).read_text(encoding="utf-8")
    assert lines.endswith("\n") and lines.count("\n") == 2
    first, second = (json.loads(line) for line in lines.splitlines())
    assert first["action"] == "first" and first["model"] == "m"
    assert first["path"] == "p" and first["timestamp"]
    assert second["action"] == "second" and set(second) == {
        "timestamp",
        "action",
    }


def test_attaching_a_guard_sets_it_and_its_scope(tmp_path):
    store = registry.ModelRegistryStore(tmp_path)
    sentinel = object()
    store.attach_guard(sentinel, "scope-1")
    assert store.guard is sentinel and store.scope == "scope-1"
    store.attach_guard(None, "s2")
    assert store.guard is None and store.scope == "s2"
    guarded = registry.ModelRegistryStore(tmp_path, sentinel, "s3")
    assert guarded.guard is sentinel and guarded.scope == "s3"


def test_registering_records_the_version_the_index_and_the_audit(
    trained, tmp_path
):
    root = trained.save(tmp_path / "b", signing_key=KEY)
    store = registry.ModelRegistryStore(tmp_path / "reg")
    record = store.register(
        "m",
        root,
        metrics={"auc": 0.9},
        config={"seed": 1},
        run_id="run-7",
        signing_key=KEY,
    )
    assert record.model == "m" and record.version == 1
    assert record.metrics == {"auc": 0.9} and record.config == {"seed": 1}
    assert record.run_id == "run-7" and record.path == "m/v1"
    assert record.content_id == registry.content_id_of(root)
    assert record.signed is True and record.registered_at
    assert (store.root / "m" / "v1" / bundle.MANIFEST_FILE).is_file()
    index = store.read_index()
    assert index["models"]["m"] == {
        "versions": {"1": dataclasses.asdict(record)},
        "aliases": {},
    }
    (line,) = (store.root / registry.AUDIT_FILE).read_text().splitlines()
    audit = json.loads(line)
    assert audit["action"] == "register" and audit["model"] == "m"
    assert audit["version"] == 1 and audit["run_id"] == "run-7"


def test_an_unsigned_bundle_needs_to_be_vouched_for_and_is_marked_so(
    trained, tmp_path, monkeypatch
):
    monkeypatch.delenv("SCMLE_SIGNING_KEY", raising=False)
    root = trained.save(tmp_path / "b")
    store = registry.ModelRegistryStore(tmp_path / "reg")
    with pytest.raises(errors.ArtifactError):
        store.register("m", root)
    assert store.read_index() == {"models": {}}
    record = store.register("m", root, trusted=True)
    assert record.signed is False
    assert record.metrics == {} and record.config == {} and record.run_id == ""


def test_the_same_content_is_not_registered_twice_and_new_content_is(
    trained, other, tmp_path
):
    first = trained.save(tmp_path / "one", signing_key=KEY)
    second = other.save(tmp_path / "two", signing_key=KEY)
    store = registry.ModelRegistryStore(tmp_path / "reg")
    one = store.register("m", first, signing_key=KEY)
    again = store.register("m", first, signing_key=KEY, run_id="later")
    assert again == one
    assert len((store.root / registry.AUDIT_FILE).read_text().splitlines()) == 1
    two = store.register("m", second, signing_key=KEY)
    assert two.version == 2 and two.path == "m/v2"
    elsewhere = store.register("n", first, signing_key=KEY)
    assert elsewhere.version == 1 and elsewhere.path == "n/v1"


def test_the_next_version_follows_the_highest_not_the_count(trained, tmp_path):
    root = trained.save(tmp_path / "b", signing_key=KEY)
    store = make_registry(tmp_path / "reg", versions=(2, 5))
    record = store.register("churn", root, signing_key=KEY)
    assert record.version == 6 and record.path == "churn/v6"
    assert (store.root / "churn" / "v6").is_dir()


def test_a_tampered_bundle_is_refused_and_leaves_no_trace(trained, tmp_path):
    root = trained.save(tmp_path / "b", signing_key=KEY)
    (root / bundle.MODEL_FILE).write_bytes(b"x")
    store = registry.ModelRegistryStore(tmp_path / "reg")
    with pytest.raises(errors.ArtifactError):
        store.register("m", root, signing_key=KEY)
    assert not (store.root / "m").exists()
    assert not (store.root / registry.AUDIT_FILE).exists()


def test_versions_come_back_in_numeric_order_and_unknown_models_are_named(
    tmp_path,
):
    store = make_registry(tmp_path / "reg", versions=(10, 2, 9))
    assert [v.version for v in store.versions("churn")] == [2, 9, 10]
    with pytest.raises(errors.ArtifactError, match=r"^unknown model 'zzz'$"):
        store.versions("zzz")


def test_aliases_are_integers_and_empty_for_an_unknown_model(tmp_path):
    store = make_registry(tmp_path / "reg")
    assert store.aliases("nobody") == {}
    assert store.aliases("churn") == {}
    index = store.read_index()
    index["models"]["churn"]["aliases"] = {"champion": "2", "challenger": 3}
    store.write_index(index)
    assert store.aliases("churn") == {"champion": 2, "challenger": 3}


def test_a_staging_alias_moves_without_approval_and_leaves_no_previous(
    tmp_path,
):
    store = make_registry(tmp_path / "reg")
    store.promote("churn", 2, registry.CHALLENGER)
    store.promote("churn", 3, registry.CHALLENGER)
    assert store.aliases("churn") == {"challenger": 3}
    (first, second) = (
        json.loads(line)
        for line in (store.root / registry.AUDIT_FILE).read_text().splitlines()
    )
    assert first["action"] == "promote" and first["model"] == "churn"
    assert first["alias"] == "challenger" and first["version"] == 2
    assert first["previous"] is None and first["approval_record"] is None
    assert second["previous"] == 2


def test_the_champion_needs_an_approval_that_names_the_exact_bundle(tmp_path):
    store = make_registry(tmp_path / "reg")
    with pytest.raises(
        errors.ArtifactError,
        match=r"^champion promotion requires independent approval$",
    ):
        store.promote("churn", 1)
    with pytest.raises(
        errors.ArtifactError,
        match=r"^promotion approval does not match bundle identity$",
    ):
        store.promote("churn", 1, approval=approval_for(2), signing_key=KEY)
    with pytest.raises(
        errors.ArtifactError,
        match=r"^champion promotion requires a valid signature$",
    ):
        store.promote("churn", 1, approval=approval_for(1), signing_key="wrong")
    assert store.aliases("churn") == {}


def test_an_unregistered_target_is_named_in_the_refusal(tmp_path):
    store = make_registry(tmp_path / "reg")
    with pytest.raises(
        errors.ArtifactError, match=r"^churn v9 is not registered$"
    ):
        store.promote("churn", 9, registry.CHALLENGER)
    with pytest.raises(
        errors.ArtifactError, match=r"^ghost v1 is not registered$"
    ):
        store.promote("ghost", 1, registry.CHALLENGER)


def test_a_type_error_in_an_approval_is_reported_as_an_artifact_error(tmp_path):
    store = make_registry(tmp_path / "reg")

    class Broken:
        def validate_target(self, *args):
            raise TypeError("odd approval")

    with pytest.raises(errors.ArtifactError, match=r"^odd approval$"):
        store.check_promotion("churn", 1, "champion", Broken(), KEY)


def test_check_promotion_returns_the_entry_and_reads_the_index_when_not_given(
    tmp_path,
):
    store = make_registry(tmp_path / "reg")
    entry = store.check_promotion("churn", 1, "challenger", None, None)
    assert set(entry) == {"versions", "aliases"}
    index = {"models": {"churn": {"versions": {}, "aliases": {}}}}
    with pytest.raises(errors.ArtifactError):
        store.check_promotion("churn", 1, "challenger", None, None, index)
    good = store.check_promotion(
        "churn", 2, "champion", approval_for(2), KEY, store.read_index()
    )
    assert "2" in good["versions"]


def test_promoting_the_champion_records_the_previous_one_and_the_approval(
    tmp_path,
):
    store = make_registry(tmp_path / "reg")
    store.promote("churn", 1, approval=approval_for(1), signing_key=KEY)
    assert store.aliases("churn") == {"champion": 1}
    store.promote("churn", 2, approval=approval_for(2), signing_key=KEY)
    assert store.aliases("churn") == {"champion": 2, "previous": 1}
    store.promote("churn", 2, approval=approval_for(2), signing_key=KEY)
    assert store.aliases("churn") == {"champion": 2, "previous": 1}
    lines = [
        json.loads(line)
        for line in (store.root / registry.AUDIT_FILE).read_text().splitlines()
    ]
    assert [(r["version"], r["previous"]) for r in lines] == [
        (1, None),
        (2, 1),
        (2, 2),
    ]
    assert lines[1]["approval_record"] == approval_for(2).to_dict()
    assert lines[1]["alias"] == "champion"


def test_a_staging_move_to_the_champion_version_does_not_set_previous(tmp_path):
    store = make_registry(tmp_path / "reg")
    store.promote("churn", 1, "previous")
    store.promote("churn", 2, "previous")
    store.promote("churn", 3, "other")
    assert store.aliases("churn") == {"previous": 2, "other": 3}


class Recorder:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def require(self, action, detail, scope, *rest):
        self.calls.append((action, detail, scope, rest))
        if self.error is not None:
            raise self.error


def test_a_guard_is_asked_for_the_champion_move_with_the_exact_request(
    tmp_path,
):
    store = make_registry(tmp_path / "reg")
    guard = Recorder()
    store.attach_guard(guard, "scope-9")
    store.promote("churn", 2, approval=approval_for(2), signing_key=KEY)
    assert guard.calls == [
        (
            never_list.NeverAction.MOVE_CHAMPION,
            {"model": "churn", "version": 2},
            "scope-9",
            (),
        )
    ]
    store.promote(
        "churn",
        3,
        approval=approval_for(3),
        signing_key=KEY,
        guard_detail={"rollback": True},
    )
    assert guard.calls[1][1] == {"rollback": True}
    store.promote("churn", 1, registry.CHALLENGER)
    assert len(guard.calls) == 2


def test_a_guard_that_refuses_stops_the_move(tmp_path):
    store = make_registry(tmp_path / "reg")
    store.attach_guard(Recorder(error=RuntimeError("no attestation")), "s")
    with pytest.raises(RuntimeError, match="no attestation"):
        store.promote("churn", 1, approval=approval_for(1), signing_key=KEY)
    assert store.aliases("churn") == {}
    assert not (store.root / registry.AUDIT_FILE).exists()


def test_a_refused_request_does_not_spend_an_attestation(tmp_path):
    store = make_registry(tmp_path / "reg")
    guard = Recorder()
    store.attach_guard(guard, "s")
    with pytest.raises(errors.ArtifactError):
        store.promote("churn", 1)
    with pytest.raises(errors.ArtifactError):
        store.promote("churn", 9, approval=approval_for(1), signing_key=KEY)
    assert guard.calls == []


def test_a_version_is_resolved_by_number_digits_or_alias(tmp_path):
    store = make_registry(tmp_path / "reg")
    store.promote("churn", 3, "challenger")
    assert store.resolve("churn", 2).version == 2
    assert store.resolve("churn", "2").version == 2
    assert store.resolve("churn", "challenger").version == 3
    with pytest.raises(
        errors.ArtifactError, match=r"^churn has no alias 'champion'$"
    ):
        store.resolve("churn")
    with pytest.raises(
        errors.ArtifactError, match=r"^churn has no alias 'nope'$"
    ):
        store.resolve("churn", "nope")
    with pytest.raises(
        errors.ArtifactError, match=r"^churn v7 is not registered$"
    ):
        store.resolve("churn", 7)
    with pytest.raises(errors.ArtifactError, match=r"^unknown model 'z'$"):
        store.resolve("z", 1)


def test_an_alias_pointing_at_a_removed_version_is_reported(tmp_path):
    store = make_registry(tmp_path / "reg")
    index = store.read_index()
    index["models"]["churn"]["aliases"]["champion"] = 8
    store.write_index(index)
    with pytest.raises(
        errors.ArtifactError, match=r"^churn v8 is not registered$"
    ):
        store.resolve("churn")


def test_a_registered_bundle_loads_by_alias_and_version(
    trained, tmp_path, monkeypatch
):
    monkeypatch.delenv("SCMLE_SIGNING_KEY", raising=False)
    root = trained.save(tmp_path / "b", signing_key=KEY)
    store = registry.ModelRegistryStore(tmp_path / "reg")
    record = store.register("m", root, signing_key=KEY)
    store.promote(
        "m",
        1,
        approval=review(content_id=record.content_id).sign(KEY),
        signing_key=KEY,
    )
    by_alias = store.load("m", signing_key=KEY)
    by_version = store.load("m", 1, signing_key=KEY)
    rows = traffic(20, seed=4).drop(columns="y")
    assert (by_alias.predict(rows) == trained.predict(rows)).all()
    assert (by_version.predict(rows) == trained.predict(rows)).all()
    with pytest.raises(errors.ArtifactError):
        store.load("m", 1)
    with pytest.raises(errors.ArtifactError, match=r"^m v4 is not registered$"):
        store.load("m", 4, signing_key=KEY)


def test_a_bundle_registered_as_vouched_loads_as_vouched(
    trained, tmp_path, monkeypatch
):
    monkeypatch.delenv("SCMLE_SIGNING_KEY", raising=False)
    monkeypatch.delenv("SCMLE_ENV", raising=False)
    root = trained.save(tmp_path / "b")
    store = registry.ModelRegistryStore(tmp_path / "reg")
    store.register("m", root, trusted=True)
    rows = traffic(20, seed=4).drop(columns="y")
    assert (store.load("m", 1).predict(rows) == trained.predict(rows)).all()
    monkeypatch.setenv("SCMLE_ENV", "production")
    with pytest.raises(errors.ArtifactError):
        store.load("m", 1)


def test_the_move_is_checked_again_under_the_lock_for_the_named_alias(tmp_path):
    store = make_registry(tmp_path / "reg")
    with pytest.raises(
        errors.ArtifactError,
        match=r"^champion promotion requires independent approval$",
    ):
        store.promote_locked("churn", 1, registry.CHAMPION, None, None)
    assert store.aliases("churn") == {}
    store.promote_locked("churn", 1, registry.CHALLENGER, None, None)
    assert store.aliases("churn") == {registry.CHALLENGER: 1}


def test_the_index_is_read_as_utf8(tmp_path, monkeypatch):
    store = registry.ModelRegistryStore(tmp_path)
    store.write_index({"models": {"é": {"versions": {}, "aliases": {}}}})
    seen = []
    real = pathlib.Path.read_text

    def spy(self, encoding=None, errors=None, newline=None):
        seen.append(encoding)
        return real(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(pathlib.Path, "read_text", spy)
    assert "é" in store.read_index()["models"]
    assert seen == ["utf-8"]
