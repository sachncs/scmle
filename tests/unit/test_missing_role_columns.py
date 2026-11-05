"""A group, time or weight column that is not in the data is a clear error."""

import numpy as np
import pandas as pd
import pytest

import scmle
from scmle.contracts import errors


def frame():
    generator = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "a": generator.normal(size=120),
            "b": generator.normal(size=120),
            "y": np.tile([0, 1], 60),
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
    project = scmle.MLProject(
        frame(), "y", output_dir=str(tmp_path), **{option: "nosuch"}
    )
    with pytest.raises(errors.DataError, match=f"{role} column 'nosuch'"):
        project.plan()
