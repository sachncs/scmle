"""Column availability, enumerated tables and small tables in the screen."""

import numpy
import pandas
import pydantic
import pytest

from scmle.assurance.admissibility import leakage
from scmle.assurance.admissibility import screen
from scmle.assurance.admissibility.screen import fitting
from scmle.assurance.admissibility.screen import records
from scmle.assurance.admissibility.screen import settings
from scmle.assurance.detectors import certificates
from scmle.contracts import errors
from scmle.contracts import tasks
from tests import support


def config(**changes):
    values = {"skill": 0.9, "ceiling": 0.5}
    values.update(changes)
    return leakage.Config(**values)


def calls(seed=0, rows=200):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {"age": rng.normal(size=rows), "y": rng.integers(0, 2, rows)}
    )
    frame["duration"] = rng.exponential(size=rows)
    return frame


# ---- the declaration on the data --------------------------------------------


def test_availability_defaults_to_none_and_is_not_stored_when_absent():
    plain = tasks.Dataset(uri="x", target="y")
    assert plain.known == {}
    assert "known" not in plain.model_dump(mode="json")
    declared = tasks.Dataset(uri="x", target="y", known={"d": "outcome"})
    assert declared.model_dump(mode="json")["known"] == {"d": "outcome"}
    assert tasks.Dataset.model_validate(declared.model_dump()) == declared
    with pytest.raises(pydantic.ValidationError, match="known"):
        tasks.Dataset(uri="x", target="y", known={"d": "later"})


def test_an_absent_declaration_keeps_the_stored_configuration_hash():
    plain = tasks.Config(data=tasks.Dataset(uri="x", target="y"))
    stored = plain.model_dump(mode="json")
    assert "known" not in stored["data"]
    assert tasks.Config.model_validate(stored).config_hash() == (
        plain.config_hash()
    )
    told = tasks.Config(
        data=tasks.Dataset(uri="x", target="y", known={"d": "prediction"})
    )
    assert told.config_hash() != plain.config_hash()


# ---- the declared check -----------------------------------------------------


def test_a_column_declared_known_after_the_outcome_is_declared():
    found = records.declared(
        ["age", "duration"], {"duration": "outcome", "age": "prediction"}
    )
    assert [f.feature for f in found] == ["duration"]
    (late,) = found
    assert late.mechanism == settings.DECLARED
    assert late.statistic == 1.0 and late.lower == 1.0
    assert late.suspect and late.blocking and late.significant
    assert records.declared(["age"], {"duration": "outcome"}) == []
    assert records.declared(["age"], None) == []


def test_an_unknown_moment_is_refused():
    with pytest.raises(errors.InputError, match="known"):
        records.declared(["age"], {"age": "soon"})


def test_a_run_blocks_a_column_declared_known_after_the_outcome():
    frame = calls()
    names = ["age", "duration"]
    report = screen.screen_table(
        frame,
        "y",
        "classification",
        names,
        0,
        known={"duration": "outcome"},
    )
    assert [f.feature for f in report.blocking] == ["duration"]
    assert report.gates[settings.DECLARED] is certificates.Authority.GATING
    assert report.ceilings[settings.DECLARED] == settings.LATE
    assert any("after the outcome" in s for s in report.sentences())
    assert report.to_dict()["blocking"] == ["duration"]
    plain = screen.screen_table(frame, "y", "classification", names, 0)
    assert settings.DECLARED not in plain.gates
    assert plain.blocking == ()
    known = screen.screen_table(
        frame, "y", "classification", names, 0, known={"age": "prediction"}
    )
    assert known.blocking == ()


def test_a_declined_screen_still_reads_the_declaration():
    frame = calls(rows=300)
    for k in range(30):
        frame[f"x{k}"] = frame["y"] + 0.05 * numpy.random.default_rng(k).normal(
            size=300
        )
    names = [c for c in frame.columns if c != "y"]
    report = screen.screen_table(
        frame,
        "y",
        "classification",
        names,
        0,
        time_budget_seconds=1,
        known={"duration": "outcome"},
    )
    assert report.unchecked
    assert "duration" in [f.feature for f in report.blocking]


# ---- duplicates in a table that enumerates its space ------------------------


def grid():
    levels = [numpy.arange(k) for k in (3, 3, 2, 4)]
    mesh = numpy.meshgrid(*levels, indexing="ij")
    frame = pandas.DataFrame({f"a{i}": m.ravel() for i, m in enumerate(mesh)})
    frame["y"] = (frame.a0 == frame.a1) | (frame.a3 == 0)
    return frame.astype({"y": "int64"})


def test_a_table_whose_space_fits_its_rows_raises_no_duplication():
    space = grid()
    names = ["a0", "a1", "a2", "a3"]
    assert records.enumerated(space, names)
    drawn = space.sample(30, random_state=0)
    designed = pandas.concat([space, drawn]).reset_index(drop=True)
    assert records.enumerated(designed, names)
    assert records.duplication(designed, names, "y", config()) is None
    # A split keeps some cells out of the development rows: still a space
    # no larger than the rows.
    assert records.enumerated(designed.sample(frac=0.8, random_state=0), names)
    assert not records.enumerated(space.head(40), names)


def test_a_sample_with_unique_values_is_not_an_enumeration():
    rng = numpy.random.default_rng(1)
    frame = pandas.DataFrame(
        {"x": rng.normal(size=300), "y": rng.integers(0, 2, 300)}
    )
    assert not records.enumerated(frame, ["x"])
    copied = pandas.concat([frame, frame.sample(90, random_state=1)])
    found = records.duplication(copied, ["x"], "y", config())
    assert found is not None and found.bound > settings.NATURAL


def test_unhashable_cells_are_enumerated_by_their_text():
    frame = pandas.DataFrame(
        {"x": [[1], [2], [1], [2]], "z": [0, 0, 1, 1], "y": [0, 1, 0, 1]}
    )
    assert records.enumerated(frame, ["x", "z"])
    assert not records.enumerated(frame, ["x"])
    assert not records.enumerated(frame, [])


# ---- a small table is screened, with low power said -------------------------


def test_fewer_rows_than_the_minimum_are_screened_with_a_note():
    y = numpy.r_[numpy.zeros(20, "int64"), numpy.ones(20, "int64")]
    assert fitting.enough(y, "classification")
    assert not fitting.enough(numpy.r_[y, [2]], "classification")
    rng = numpy.random.default_rng(3)
    frame = pandas.DataFrame({"x": rng.normal(size=60), "y": [0, 1] * 30})
    frame["leak"] = frame["y"] + 0.0
    report = screen.screen(
        frame, "y", "classification", config(permutations=199)
    )
    assert report.unchecked == ()
    assert ("leak", settings.SKILL) in {
        (f.feature, f.mechanism) for f in report.suspects
    }
    assert any("low power" in note for note in report.notes)


# ---- the declaration through the project API -------------------------------


def test_a_project_declaring_a_column_known_after_the_outcome_is_blocked(
    tmp_path,
):
    frame = calls(rows=400)
    project = support.make_project(
        frame,
        "y",
        known={"duration": "outcome"},
        leakage_policy="block",
        output_dir=tmp_path,
    )
    assert project.spec.data.known == {"duration": "outcome"}
    with pytest.raises(errors.BlockedError, match="duration"):
        project.run()
    assert support.make_project(frame, "y").spec.data.known == {}
