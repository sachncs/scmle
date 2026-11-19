"""Claim C7 protocols: the lease and journal models, checked and replayed."""

import dataclasses
import json

import pytest

from benchmarks.evidence import protocols
from scmle.assurance.custody import journal
from scmle.contracts import errors

SMALL = {"horizon": 2, "beats": 0, "crashes": 0}
# The shortest path to a valid holder refused: a stalled `a` stores an
# already-lapsed lease, `b` takes it over (fence 2), writes and releases
# it (the lock is deleted), then takes it afresh at fence 1 and is refused
# by its own record.
REFUSED = (
    "tick",
    "tick",
    "a create",
    "b create",
    "b read",
    "b takeover",
    "b fetch",
    "b hold",
    "b put",
    "b release",
    "b drop",
    "b create",
    "b fetch",
)
REUSED = (
    "a create",
    "a fetch",
    "a hold",
    "a put",
    "a release",
    "a drop",
    "a create",
)


class Counter(protocols.Model):
    """Counts up or down from zero; three is forbidden."""

    name = "counter"
    links = ()

    def start(self):
        yield 0

    def moves(self, state):
        if state < 5:
            yield "up", state + 1
        if state > 0:
            yield "down", state - 1

    def invariants(self):
        return {"small": lambda s: s < 3, "natural": lambda s: s >= 0}

    def replay(self, path, folder):
        return []

    def scope(self):
        return {}


def test_explore_gives_the_shortest_counterexample_of_each_invariant():
    verdict = protocols.explore(Counter())
    assert verdict.states == 6
    assert verdict.transitions == 10
    assert verdict.depth == 5
    assert list(verdict.broken) == ["small"]
    assert verdict.broken["small"].labels == ("up", "up", "up")
    assert verdict.broken["small"].states == (0, 1, 2, 3)
    assert [path.states[-1] for path in verdict.samples] == [5, 4, 3, 2, 1, 0]


def test_explore_refuses_a_scope_past_its_limit():
    with pytest.raises(protocols.BoundError, match="more than 4 states"):
        protocols.explore(Counter(), limit=4)


def test_tokens_read_as_lines_with_cut_and_lone_lines_unreadable():
    tokens = (
        ("line", 0),
        ("part", 1),
        ("end", 0),
        ("line", 2),
        ("part", 3),
        ("line", 4),
        ("part", 5),
    )
    assert protocols.lines(tokens) == [0, None, 2, None, None]
    assert protocols.lines((("end", 0),)) == [None]


def test_the_full_lease_scopes_are_explored_whole():
    distinct, shared = protocols.models()[:2]
    assert protocols.explore(distinct).states == 474902
    assert protocols.explore(shared).states == 515120


def test_the_lease_reuses_fence_one_and_refuses_a_valid_holder():
    verdict = protocols.explore(protocols.Lease(("a", "b"), **SMALL))
    assert verdict.states == 11405
    assert list(verdict.broken) == ["unique", "admitted"]
    assert verdict.broken["unique"].labels == REUSED
    assert verdict.broken["admitted"].labels == REFUSED
    final = verdict.broken["admitted"].states[-1]
    assert final.record == protocols.Record(fence=2, writer=1, number=0)
    assert final.lock == protocols.Lock("b", 1, 4, 1)
    assert final.workers[1].pc == "release"


def test_the_real_executors_follow_the_lease_counterexamples(tmp_path):
    model = protocols.Lease(("a", "b"), **SMALL)
    verdict = protocols.explore(model)
    for name, path in verdict.broken.items():
        assert model.replay(path, tmp_path / name) == [], name


def test_shared_owner_names_let_two_holders_write_interleaved(tmp_path):
    model = protocols.Lease(("executor", "executor"), **SMALL)
    verdict = protocols.explore(model)
    assert list(verdict.broken) == ["unique", "exclusive", "admitted"]
    path = verdict.broken["exclusive"]
    assert path.labels == (
        "tick",
        "tick",
        "a create",
        "a fetch",
        "a hold",
        "b create",
        "b read",
        "b takeover",
        "b fetch",
        "a put",
        "b hold",
        "b put",
        "b release",
        "b drop",
        "b create",
        "a release",
        "a drop",
        "a create",
        "a fetch",
        "a hold",
        "a put",
        "b fetch",
        "b hold",
        "b put",
    )
    # `a drop` deletes the lease `b` just took: same owner, fence and
    # expiry, so the same content and the same store version.
    assert path.labels[16] == "a drop"
    assert path.states[16].lock == protocols.Lock("executor", 1, 4, 1)
    assert path.states[17].lock is None
    assert model.replay(path, tmp_path) == []


def test_a_removed_guard_is_caught_and_the_real_code_refuses_its_trace(
    tmp_path,
):
    mutant = protocols.Lease(
        ("a", "b"), guards=frozenset({"stored", "hold"}), **SMALL
    )
    assert mutant.name == "lease, distinct owners, without version"
    verdict = protocols.explore(mutant)
    assert list(verdict.broken) == ["unique", "fenced", "exclusive", "admitted"]
    path = verdict.broken["fenced"]
    assert path.labels == (
        "tick",
        "tick",
        "a create",
        "a fetch",
        "a hold",
        "b create",
        "b read",
        "b takeover",
        "b fetch",
        "b hold",
        "b put",
        "a put",
    )
    assert mutant.replay(path, tmp_path) == [
        "step 12: record [2, 1], model [1, 0]"
    ]


def test_a_crash_and_a_renewal_replay_against_the_real_code(tmp_path):
    model = protocols.Lease(("a", "b"), horizon=2, rounds=1)
    verdict = protocols.explore(model)
    assert verdict.states == 7487
    labels = [label for path in verdict.samples for label in path.labels]
    assert {"a crash", "a beat", "a renew", "tick"} <= set(labels)
    for number, path in enumerate(verdict.samples):
        assert model.replay(path, tmp_path / str(number)) == [], path.labels


def test_a_cut_custody_line_loses_every_record(tmp_path):
    model = protocols.Custody("power")
    verdict = protocols.explore(model)
    assert (verdict.states, list(verdict.broken)) == (465, ["kept"])
    path = verdict.broken["kept"]
    assert path.labels == (
        "recover",
        "prepare",
        "write",
        "crash, 1 of 1 tokens kept, the last cut",
        "recover",
    )
    assert model.replay(path, tmp_path / "replay") == []
    # The same trace by hand: a synced entry, a second one cut by the
    # crash; the reader then refuses the whole journal, untyped.
    real = journal.Journal(tmp_path / "custody" / "run.jsonl", "run")
    real.start("config")
    real.decide("g0", True, "reviewer", "kept")
    text = real.path.read_bytes()
    first = text.index(b"\n") + 1
    real.path.write_bytes(text[: first + (len(text) - first) // 2])
    with pytest.raises(json.JSONDecodeError):
        real.read()


def test_custody_survives_kills_and_erasure_survives_both_crashes():
    found = {
        model.name: (verdict.states, list(verdict.broken))
        for model in protocols.models()[2:]
        for verdict in [protocols.explore(model)]
    }
    assert found == {
        "custody journal, kill crashes": (417, []),
        "custody journal, power crashes": (465, ["kept"]),
        "erasure journal, kill crashes": (121, []),
        "erasure journal, power crashes": (399, []),
    }


class Again(protocols.Erasure):
    """An erasure caller that performs every step again on resume."""

    kind = "again"

    def resume(self, found):
        return frozenset()


def test_a_caller_that_appends_again_breaks_single(tmp_path):
    model = Again("kill")
    verdict = protocols.explore(model)
    assert list(verdict.broken) == ["single"]
    path = verdict.broken["single"]
    assert path.labels == (
        "recover",
        "prepare",
        "write",
        "crash, 1 of 1 tokens kept",
        "recover",
        "prepare",
        "write",
    )
    assert protocols.twice(path.states[-1].tokens) == [0]
    # The real journal takes the second line; its reader keys it away.
    assert model.replay(path, tmp_path) == []


def test_the_custody_caller_decides_every_gate_again_on_resume():
    model = protocols.Custody("kill")
    assert "single" not in model.invariants()
    state = next(iter(model.start()))
    for label in ("recover", "prepare", "write", "sync", "ack"):
        state = dict(model.moves(state))[label]
    crashed = dict(model.moves(state))["crash, 1 of 1 tokens kept"]
    resumed = model.step(crashed)
    assert (resumed.pc, resumed.next, resumed.skip) == (
        "prepare",
        0,
        frozenset(),
    )
    erasure = protocols.Erasure("kill")
    kept = dataclasses.replace(crashed, skip=frozenset())
    assert erasure.step(kept).next == 1
    assert erasure.step(kept).skip == frozenset({0})


def test_confront_keys_only_a_distinct_journal(tmp_path):
    doubled = dataclasses.replace(
        next(iter(protocols.Custody("kill").start())),
        tokens=(("line", 0), ("line", 0)),
        exists=True,
    )
    custody = protocols.Custody("kill")
    real = custody.open(tmp_path / "custody")
    custody.append(real, 0)
    assert custody.confront(real, doubled, "prepare", 3) == [
        "step 3 (prepare): read [0], model [0, 0]"
    ]
    erasure = protocols.Erasure("kill")
    ledger = erasure.open(tmp_path / "erasure")
    erasure.append(ledger, 0)
    assert erasure.confront(ledger, doubled, "prepare", 3) == []


def test_the_journals_replay_every_sample_against_the_real_code(tmp_path):
    for model in protocols.models()[2:]:
        verdict = protocols.explore(model)
        for number, path in enumerate(verdict.samples):
            folder = tmp_path / f"{model.kind}-{model.crash}-{number}"
            assert model.replay(path, folder) == [], (model.name, path.labels)


def test_an_unknown_crash_kind_is_refused():
    with pytest.raises(errors.InputError, match="crash is one of"):
        protocols.Erasure("flood")
    assert protocols.Journal.CRASHES == ("kill", "power")


def test_every_link_resolves_and_a_moved_one_does_not():
    for model in protocols.models():
        assert all(link.resolve() for link in model.links), model.name
    assert {link.action for link in protocols.Lease.links} == {
        *protocols.Lease.ACTIONS,
        "expiry",
    }
    for model in protocols.models()[2:]:
        actions = {link.action for link in model.links}
        assert actions == set(protocols.Journal.ACTIONS), model.name
    moved = protocols.Link("x", protocols.LEASES + ".hold", "no such text")
    gone = protocols.Link("x", "scmle.nowhere:Thing", "text")
    assert not moved.resolve()
    assert not gone.resolve()


def test_the_models_cover_both_owner_namings_and_both_crash_kinds():
    assert [model.name for model in protocols.models()] == [
        "lease, distinct owners",
        "lease, shared owners",
        "custody journal, kill crashes",
        "custody journal, power crashes",
        "erasure journal, kill crashes",
        "erasure journal, power crashes",
    ]
    assert protocols.models()[0].scope() == {
        "owners": ["a", "b"],
        "guards": ["hold", "stored", "version"],
        "ticks": 2,
        "horizon": 3,
        "rounds": 2,
        "writes": 1,
        "beats": 1,
        "crashes": 1,
    }


def record(holds=True, diverged=(), linked=True):
    outcome = {"holds": holds}
    if not holds:
        outcome.update(counterexample=["tick"], diverged=list(diverged))
    return {
        "model": "toy",
        "invariants": {"safe": outcome},
        "samples": {"paths": 1, "steps": 1, "diverged": []},
        "links": {"a -> m:f": linked},
    }


def test_the_verdict_needs_faithful_models_and_no_counterexample():
    assert protocols.decide([record()]) == "pass"
    assert protocols.decide([record(), record(holds=False)]) == "fail"
    diverging = record(holds=False, diverged=["step 1: lock"])
    assert protocols.decide([diverging]) == "not measured"
    assert protocols.decide([record(linked=False)]) == "not measured"
    sampled = record()
    sampled["samples"]["diverged"] = [["step 2: record"]]
    assert protocols.decide([sampled]) == "not measured"


def test_a_stored_document_decides_only_for_this_code():
    document = {
        "code": protocols.digest(),
        "complete": True,
        "models": [record(), {**record(holds=False), "model": "lease"}],
        "verdict": "fail",
    }
    assert protocols.problems(document) == []
    assert protocols.summary(document) == (
        "toy: all hold; lease: safe fail",
        "fail",
    )
    stale = {**document, "code": "0", "complete": False}
    assert protocols.problems(stale) == [
        "protocols: measured on other code",
        "protocols: not every model was checked",
    ]
    assert protocols.summary(stale)[1] == "not measured"


def test_the_cli_checks_and_shows(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(protocols, "PATH", tmp_path / "protocols.json")
    assert protocols.main(["--check"]) == 1
    assert json.loads(capsys.readouterr().out) == {
        "ok": False,
        "problems": ["protocols: protocols.json is missing"],
    }
    assert protocols.summary() == ("no measurement", "not measured")
    shown = ["--show", "--model", "erasure journal, kill crashes"]
    assert protocols.main(shown) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["complete"] is False
    assert document["verdict"] == "not measured"
    assert document["models"][0]["invariants"] == {
        "kept": {"holds": True},
        "single": {"holds": True},
    }
    assert document["models"][0]["samples"]["diverged"] == []
    assert not (tmp_path / "protocols.json").exists()
    with pytest.raises(SystemExit):
        protocols.main(["--show", "--model", "nothing"])
    with pytest.raises(SystemExit):
        protocols.main(["--write", "--model", "erasure journal, kill crashes"])


def cheap():
    return (
        protocols.Lease(("a", "b"), **SMALL),
        protocols.Custody("kill"),
        protocols.Custody("power"),
        protocols.Erasure("kill"),
        protocols.Erasure("power"),
    )


def test_a_complete_measurement_decides_and_is_written(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(protocols, "models", cheap)
    document = protocols.measure()
    assert document["complete"] is True
    assert document["verdict"] == "fail"
    assert [record["model"] for record in document["models"]] == [
        model.name for model in cheap()
    ]
    assert all(
        record["samples"]["diverged"] == [] for record in document["models"]
    )
    monkeypatch.setattr(protocols, "PATH", tmp_path / "protocols.json")
    assert protocols.main(["--write"]) == 0
    written = json.loads((tmp_path / "protocols.json").read_text())
    assert json.loads(capsys.readouterr().out) == written
    assert written["verdict"] == "fail"
    assert protocols.main(["--check"]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "problems": []}
