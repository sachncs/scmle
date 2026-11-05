"""The whole local demo runs natively, including the rollback drill."""

import json

from scmle.deploy import demo


def test_the_native_demo_passes_every_step(tmp_path, capsys):
    transcript = demo.run(tmp_path / "demo")
    capsys.readouterr()
    names = [step.name for step in transcript.steps]

    assert transcript.passed, json.dumps(transcript.to_dict(), default=str)
    assert names == [
        "train and evaluate",
        "register and package",
        "object store round trip",
        "serve under production rules",
        "predict (online and batch)",
        "golden-record parity",
        "corrupted and unsigned bundles never become ready",
        "monitor and rollback drill",
        "measure and ship metrics",
    ]
    drill = next(
        step for step in transcript.steps if step.name.startswith("monitor")
    )
    assert drill.detail["champion_before_rollback"] == 2
    assert drill.detail["champion_after_rollback"] == 1
    assert drill.detail["monitor_recommendation"] == "retrain"
