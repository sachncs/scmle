"""Missing role columns and missing run directories give clear errors."""

import numpy
import pandas
import pytest

from scmle import cli
import scmle.api.project
from scmle.contracts import errors


def frame():
    generator = numpy.random.default_rng(0)
    return pandas.DataFrame(
        {
            "a": generator.normal(size=120),
            "b": generator.normal(size=120),
            "y": numpy.tile([0, 1], 60),
        }
    )


@pytest.mark.parametrize(
    ("role", "option"),
    [
        ("group", "group_column"),
        ("time", "time_column"),
        ("weight", "weight_column"),
    ],
)
def test_a_missing_role_column_names_the_role(tmp_path, role, option):
    project = scmle.api.project.Project(
        frame(), "y", output_dir=str(tmp_path), **{option: "nosuch"}
    )
    with pytest.raises(errors.DataError, match=f"{role} column 'nosuch'"):
        project.plan()


def test_resuming_a_missing_run_directory_is_an_artifact_error(
    tmp_path, caplog
):
    assert cli.main(["resume", str(tmp_path / "nope")]) == 1
    assert "run directory not found" in caplog.text
