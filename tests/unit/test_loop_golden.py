"""Snapshots of the closed loop's ledger and state (mutation pins).

Each scenario replays a fixed simulated stream and snapshots every ledger
row, the saved state and the artifacts. See `tests/unit/golden.py` and
`tests/unit/MUTANTS_loop.md`. Regenerate with SCMLE_WRITE_GOLDEN=1 only after
a reviewed change of behaviour.
"""

import json
from pathlib import Path
import types

import numpy
import pandas
import pytest

from benchmarks.qualification import streams
from scmle.governance import deployment_ledger
from scmle.governance import loop as loop_module
from scmle.governance import never_list
from tests.unit import golden
from tests.unit import test_loop

FIXTURE = Path(__file__).parents[1] / "fixtures" / "loop_golden.json"


def record(loop, extra=None):
    directory = loop.directory
    files = {}
    for name in (
        loop_module.STATE_FILE,
        loop_module.RISK_COVERAGE_FILE,
        loop_module.LABEL_MECHANISM_FILE,
    ):
        path = directory / name
        if path.exists():
            files[name] = json.loads(path.read_text(encoding="utf-8"))
    state = files.get(loop_module.STATE_FILE, {})
    state.pop("ledger_head", None)
    return {
        "rows": [
            {"kind": row["kind"], "body": row["body"]}
            for row in loop.ledger.rows
        ],
        "files": files,
        "tau": loop.tau,
        "halted": loop.halted,
        "mechanism": loop.mechanism.value,
        "extra": extra,
    }


def scenario_stable(tmp_path):
    loop = test_loop.open_loop(tmp_path)
    result = test_loop.feed(
        loop, streams.simulate(test_loop.spec(batches=40, seed=3, base=0.10))
    )
    return record(loop, [result.status, result.events])


def scenario_abrupt(tmp_path):
    loop = test_loop.open_loop(
        tmp_path, hooks=loop_module.Hooks(retrain=lambda: "challenger-1")
    )
    stream = streams.simulate(
        test_loop.spec(kind="abrupt", batches=60, change=25, delta=0.25, seed=4)
    )
    result = test_loop.feed(loop, stream)
    return record(loop, [result.status, result.events])


def scenario_uncertified(tmp_path):
    loop = test_loop.open_loop(
        tmp_path, test_loop.make_policy(alarm_gates_retrain=False)
    )
    stream = streams.simulate(
        test_loop.spec(kind="abrupt", batches=60, change=25, delta=0.3, seed=4)
    )
    test_loop.feed(loop, stream)
    return record(loop)


def scenario_promotion(tmp_path):
    loop = test_loop.open_loop(
        tmp_path,
        test_loop.make_policy(
            reference_risk=0.2,
            tolerance=0.3,
            challenger_budget=100,
            reply_timeout=1000,
        ),
    )
    stream, sequence = test_loop.paired([0.0, 0.12])
    loop.register_challenger("null")
    loop.register_challenger("better")
    test_loop.feed(
        loop,
        stream,
        challengers={
            "null": sequence.challengers[0],
            "better": sequence.challengers[1],
        },
    )
    return record(loop)


def scenario_rollback(tmp_path):
    loop = test_loop.open_loop(
        tmp_path,
        test_loop.make_policy(
            reference_risk=0.2, tolerance=0.3, reply_timeout=1000
        ),
    )
    sequence = streams.challenger_sequence([-0.1], 120, 100, 0.2, 1)
    stream = streams.simulate(test_loop.spec(batches=120, size=100, seed=1))
    object.__setattr__(stream, "loss", sequence.champion)
    loop.watch_rollback("previous")
    test_loop.feed(
        loop, stream, challengers={"previous": sequence.challengers[0]}
    )
    return record(loop)


def scenario_halt(tmp_path):
    loop = test_loop.open_loop(
        tmp_path,
        test_loop.make_policy(reference_risk=0.5, tolerance=0.4, r_star=0.05),
    )
    stream = streams.simulate(test_loop.spec(base=0.3, batches=50, seed=8))
    test_loop.feed(loop, stream)
    routed = loop.observe(
        test_loop.batch_frame(stream, 0), numpy.zeros(50), ["a"] * 50
    )
    bad = streams.draw_calibration(30, 0.05, seed=1)
    loop.calibrate(bad["score"], bad["loss"])
    good = streams.draw_calibration(3000, 0.05, seed=2)
    result = loop.calibrate(good["score"], good["loss"])
    return record(loop, [routed.routing, result])


def scenario_routing(tmp_path):
    loop = test_loop.open_loop(
        tmp_path,
        test_loop.make_policy(
            high_impact_actions=("approve",), coverage_target=0.9
        ),
    )
    stream = streams.simulate(test_loop.spec(batches=30, seed=9))
    routed = []
    for index in range(30):
        frame = test_loop.batch_frame(
            stream,
            index,
            novel=numpy.arange(50) < 20,
            set_size=numpy.arange(50) % 4,
        )
        routed.append(
            loop.observe(frame, numpy.zeros(50), ["approve"] * 25 + ["x"] * 25)
        )
        ids = list(frame["case_id"])
        loop.mature(
            dict.fromkeys(ids, 0.0),
            covered={i: float(n % 2) for n, i in enumerate(ids)},
        )
        loop.step()
    return record(loop, routed[:2])


def scenario_mnar(tmp_path):
    loop = test_loop.open_loop(
        tmp_path, hooks=loop_module.Hooks(retrain=lambda: "x")
    )
    config = test_loop.spec(
        kind="abrupt",
        labels="mnar",
        batches=50,
        change=15,
        delta=0.3,
        epsilon=0.0,
        seed=6,
    )
    stream = streams.simulate(config)
    for index in range(config.batches):
        frame = test_loop.batch_frame(stream, index).drop(
            columns=["propensity", "audit"]
        )
        loop.observe(frame, numpy.zeros(50), ["a"] * 50)
        arrived = {
            f"b{b}-{p}": stream.loss[b, p]
            for b in range(index + 1)
            for p in numpy.flatnonzero(
                (stream.arrival[b] == index) & stream.selected[b]
            )
        }
        loop.mature(arrived)
        loop.step()
    return record(loop)


def scenario_replies(tmp_path):
    loop = test_loop.open_loop(tmp_path, test_loop.make_policy(reply_timeout=3))
    registry = types.SimpleNamespace(promote=lambda *a, **k: "moved")
    outcomes = []
    for version in (3, 4):
        try:
            loop.move_champion(registry, "m", version, None)
        except Exception as error:
            outcomes.append(type(error).__name__)
    first, second = loop.pending_escalations()
    outcomes.append(loop.reply(first.id, test_loop.reply_to(first)))
    outcomes.append(
        loop.reply(second.id, test_loop.reply_to(second, decision="reject"))
    )
    outcomes.append(loop.move_champion(registry, "m", 3, None))
    with pytest.raises(never_list.AttestationRequired):
        loop.move_champion(registry, "m", 4, None)
    try:
        loop.amend_policy(r_star=0.9)
    except Exception as error:
        outcomes.append(type(error).__name__)
    outcomes.append(loop.amend_policy(r_star=0.04))
    try:
        loop.rewrite_ledger(lambda: None, {"op": "truncate"})
    except Exception as error:
        outcomes.append(type(error).__name__)
    test_loop.feed(
        loop, streams.simulate(test_loop.spec(batches=6, seed=1)), 0, 6
    )
    return record(loop, outcomes)


def scenario_wiring(tmp_path):
    loop = test_loop.open_loop(tmp_path, test_loop.make_policy(bypass_rate=0.3))
    stream = streams.feedback_stream(100, 100, gamma=0.8, bypass=0.3, seed=1)
    frame = pandas.DataFrame({"z": stream.next_z.reshape(-1)})
    flags = stream.bypass.reshape(-1)
    check = loop.check_feedback(
        frame[flags].iloc[:300], frame[~flags].iloc[:300]
    )
    return record(loop, check)


SCENARIOS = {
    "stable": scenario_stable,
    "abrupt": scenario_abrupt,
    "uncertified": scenario_uncertified,
    "promotion": scenario_promotion,
    "rollback": scenario_rollback,
    "halt": scenario_halt,
    "routing": scenario_routing,
    "mnar": scenario_mnar,
    "replies": scenario_replies,
    "wiring": scenario_wiring,
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_loop_reproduces_its_recorded_ledger_and_state(name, tmp_path):
    expected, actual = golden.check(
        FIXTURE, name, lambda: SCENARIOS[name](tmp_path)
    )
    assert actual == expected
    assert deployment_ledger.DeploymentLedger.load(
        tmp_path / "deploy" / deployment_ledger.FILE_NAME
    )
