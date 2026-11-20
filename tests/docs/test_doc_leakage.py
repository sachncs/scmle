"""Guarantees of docs/LEAKAGE.md that no other test enforced."""

import inspect
import json

import numpy
import pandas
import pytest

from benchmarks.qualification import screen_certify
from scmle import cli
from scmle.api import project
from scmle.assurance.admissibility import attestations
from scmle.assurance.admissibility import plants
from scmle.assurance.admissibility import screen
from scmle.contracts import tasks


def table(rows=240, seed=0):
    """A binary table with one column that copies the target.

    Args:
        rows: Number of rows.
        seed: Seed of the random generator.

    Returns:
        A frame with four noise columns, the target `y` and its copy `leak`.
    """
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame({f"x{i}": rng.normal(size=rows) for i in range(4)})
    frame["y"] = (frame["x0"] + rng.normal(size=rows) > 0).astype(int)
    frame["leak"] = frame["y"] * 1.0
    return frame


def test_every_entry_point_defaults_to_the_warn_policy():
    """Enforces docs/LEAKAGE.md:228.

    What `warn` does with a flagged column (it trains and labels the score)
    is enforced by tests/leakage/test_leakage.py, which runs it end to end.

    Also enforces website/src/learn/leakage.html:68-69.
    """
    signature = inspect.signature(project.Project)
    parser = cli.build()
    run = parser.parse_args(["run", "data.csv", "--target", "y"])

    assert signature.parameters["leakage_policy"].default == "warn"
    assert tasks.Config.model_fields["leakage_policy"].default == "warn"
    assert run.leakage == "warn"


@pytest.mark.parametrize(
    "structure", [{"group_column": "g"}, {"time_column": "t"}]
)
def test_the_probe_screens_plants_with_the_real_screens_settings(
    monkeypatch, structure
):
    """Enforces docs/LEAKAGE.md:238.

    Also enforces website/src/learn/leakage.html:69-70.
    """
    calls = []
    original = screen.screen
    signature = inspect.signature(original)

    def recording(*args, **kwargs):
        calls.append(signature.bind(*args, **kwargs).arguments)
        return original(*args, **kwargs)

    # The digest traces package code, which the recording wrapper is not.
    digest = screen.fingerprint()
    monkeypatch.setattr(screen, "fingerprint", lambda: digest)
    monkeypatch.setattr(screen, "screen", recording)
    frame = table(rows=200).assign(
        g=numpy.arange(200) % 20, t=numpy.arange(200)[::-1]
    )
    features = ["x0", "x1", "leak"]

    report = screen.screen_table(
        frame,
        "y",
        "classification",
        features,
        3,
        time_budget_seconds=60.0,
        **structure,
    )

    real, planted = calls
    assert list(planted["features"]) == [*features, *plants.DEFECTS]
    assert planted["cfg"] == real["cfg"]
    assert planted["block_size"] == real["block_size"]
    # One share of the budget covers both scans; `screen_table` applies it
    # before each one, so neither is handed a budget of its own.
    budget = "time_budget_seconds"
    assert planted.get(budget) is real.get(budget) is None
    if "group_column" in structure:
        assert planted["groups"].equals(real["groups"])
    else:
        assert real["block_size"] and planted["groups"] is None
    assert list(planted["frame"].index) == list(real["frame"].index)
    assert set(report.probe) == set(plants.KINDS)


def test_an_attestation_that_expires_before_its_date_is_refused(tmp_path):
    """Enforces docs/LEAKAGE.md:351.

    Also enforces website/src/learn/leakage.html:128.
    Also enforces website/src/learn/when-things-go-wrong.html:150.
    """
    record = {
        "column": "worst area",
        "name_or_id": "A. Pathologist",
        "role": "data owner",
        "statement": "available at prediction",
        "evidence": "imaging protocol v2",
        "date": "2026-11-05",
        "expires": "2026-11-04",
        "scope": "charter-hash",
        "lead_seconds": 0,
    }
    path = tmp_path / "att.json"
    path.write_text(
        json.dumps({"format_version": 1, "attestations": [record]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expires must not precede date"):
        attestations.load(path)
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "attestations": [{**record, "expires": "2026-11-05"}],
            }
        ),
        encoding="utf-8",
    )
    assert len(attestations.load(path)) == 1


@pytest.mark.parametrize("kind", screen_certify.KINDS)
def test_no_null_world_of_the_certification_was_flagged(kind):
    """Enforces docs/LEAKAGE.md:435 (0 of 1000 null worlds flagged).

    Also enforces website/src/learn/leakage.html:174.
    """
    document = json.loads(
        screen_certify.MEASUREMENTS_PATH.read_text(encoding="utf-8")
    )[kind]
    (cert,) = screen.load().current(
        screen.DETECTORS[kind], screen.fingerprint()
    )

    # The certificate counts false alarms over every world (null, grouped
    # and planted); the null worlds alone are the claim here.
    assert document["null_false_alarms"] == 0
    assert document["null_replicates"] == 1000
    for outcome in cert.outcomes:
        assert outcome.certificate.alarms == document["suspect_false_alarms"]
        assert outcome.certificate.clean == document["clean_worlds"]
