"""A charter run reads the declared prediction time and column availability."""

import numpy
import pandas

from scmle.assurance.admissibility import qualify
from scmle.assurance.admissibility import screen
from scmle.assurance.detectors import certificates


def table(rows=400, seed=0):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {
            "a": rng.normal(size=rows),
            "b": rng.normal(size=rows),
            "duration": rng.exponential(size=rows),
            "closed": pandas.Timestamp("2020-01-01")
            + pandas.to_timedelta(rng.integers(0, 60, rows), unit="D"),
        }
    )
    target = pandas.Series(
        (frame.a + rng.normal(size=rows) > 0).astype(int), name="y"
    )
    return frame, target


def qualified(tmp_path, **declarations):
    features, target = table()
    return qualify.run(
        features,
        target,
        kind="predictive",
        classification=True,
        seed=0,
        directory=tmp_path,
        **declarations,
    )


def test_a_column_declared_known_after_the_outcome_blocks_a_charter_run(
    tmp_path,
):
    found = qualified(tmp_path, known={"duration": "outcome"})

    assert found.leakage is not None
    blocking = {f.feature for f in found.leakage.blocking}
    assert "duration" in blocking
    assert "duration" in found.quarantined
    assert found.leakage.gates[screen.DECLARED] is (
        certificates.Authority.GATING
    )
    assert any("known only after the outcome" in n for n in found.notes)


def test_a_date_after_the_declared_prediction_time_blocks_a_charter_run(
    tmp_path,
):
    found = qualified(tmp_path, cutoff="2020-01-31")

    assert found.leakage is not None
    assert "closed" in {f.feature for f in found.leakage.blocking}
    assert "closed" in found.quarantined
    assert any("prediction time 2020-01-31" in n for n in found.notes)


def test_without_declarations_the_charter_run_does_not_block_them(tmp_path):
    found = qualified(tmp_path)

    assert {"duration", "closed"}.isdisjoint(found.quarantined)


def test_a_column_known_at_prediction_is_not_blocked(tmp_path):
    found = qualified(
        tmp_path, cutoff="2021-01-01", known={"duration": "prediction"}
    )

    assert {"duration", "closed"}.isdisjoint(found.quarantined)
