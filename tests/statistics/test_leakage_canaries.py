"""Leakage canaries: plants the screen must flag every time.

`tests/leakage` plants a numeric copy of the target, a noisy copy and a value
recorded for one class. These canaries cover the gaps next to them: a copy
of the target stored as text, a negated and rescaled copy, and a statistic
computed after the split, on all rows, with the labels of the held-out rows
inside it (a deep tree fitted on the whole table, the predictions of which
reproduce the target). Each is planted in 12 differently seeded small tables
and must be a suspect in every one, with no suspect among the honest columns.
A leave-in target encoding of a high-cardinality key is a partial leak; the
screen is documented to flag only near-copies, so the check there is that the
column is significant, which is what the documentation promises.
"""

import numpy
import pandas
import pytest
from sklearn import tree
import support

from scmle.contracts import errors
from scmle.data.admissibility import leakage
from scmle.data.admissibility import screen

SEEDS = range(12)
CONFIG = leakage.Config(
    skill_ceiling=0.95,
    association_ceiling=0.9,
    permutations=199,
    bootstrap=100,
)


def table(seed, rows=400):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame({f"x{i}": rng.normal(size=rows) for i in range(4)})
    frame["y"] = rng.integers(0, 2, rows)
    return frame, rng


def suspect_names(frame):
    report = screen.screen(frame, "y", "classification", CONFIG)
    return {finding.feature for finding in report.suspects}


@pytest.mark.parametrize("seed", SEEDS)
def test_a_copy_of_the_target_stored_as_text_is_a_suspect(seed):
    frame, ignored = table(seed)
    frame["label_text"] = numpy.where(frame.y == 1, "yes", "no")
    assert suspect_names(frame) == {"label_text"}


@pytest.mark.parametrize("seed", SEEDS)
def test_a_negated_and_rescaled_copy_of_the_target_is_a_suspect(seed):
    frame, ignored = table(seed)
    frame["score"] = -5.0 * frame.y + 2.0
    assert suspect_names(frame) == {"score"}


@pytest.mark.parametrize("seed", SEEDS)
def test_a_statistic_fitted_on_all_rows_is_a_suspect(seed):
    frame, ignored = table(seed)
    features = frame[["x0", "x1", "x2", "x3"]].to_numpy()
    full_fit = tree.DecisionTreeClassifier(random_state=0).fit(
        features, frame.y
    )
    frame["full_fit"] = full_fit.predict_proba(features)[:, 1]
    assert suspect_names(frame) == {"full_fit"}


@pytest.mark.parametrize("seed", SEEDS)
def test_a_leave_in_target_encoding_is_at_least_significant(seed):
    frame, rng = table(seed)
    key = rng.integers(0, 200, len(frame))
    frame["encoded_key"] = (
        pandas.Series(frame.y.to_numpy()).groupby(key).transform("mean")
    ).to_numpy()
    report = screen.screen(frame, "y", "classification", CONFIG)
    found = [f for f in report.findings if f.feature == "encoded_key"]
    assert found and any(f.significant for f in found)
    assert {f.feature for f in report.suspects} <= {"encoded_key"}


def test_the_pipeline_blocks_a_target_copy_stored_as_text(
    binary_frame, tmp_path
):
    leaky = binary_frame.assign(
        label_text=numpy.where(binary_frame.y == 1, "yes", "no")
    )
    result = support.make_project(
        leaky, "y", output_dir=tmp_path / "warn", leakage_policy="warn"
    ).run()
    assert "label_text" in result.profile.blocking_leaks
    with pytest.raises(errors.BlockedError, match="leakage") as raised:
        support.make_project(
            leaky, "y", output_dir=tmp_path / "block", leakage_policy="block"
        ).run()
    assert "label_text" in raised.value.details["leakage"]["blocking"]


def test_the_pipeline_blocks_a_statistic_fitted_on_all_rows(
    binary_frame, tmp_path
):
    features = binary_frame[["a", "b"]].fillna(0.0).to_numpy()
    fitted = tree.DecisionTreeClassifier(random_state=0).fit(
        features, binary_frame.y
    )
    leaky = binary_frame.assign(full_fit=fitted.predict_proba(features)[:, 1])
    with pytest.raises(errors.BlockedError, match="leakage") as raised:
        support.make_project(
            leaky, "y", output_dir=tmp_path, leakage_policy="block"
        ).run()
    assert "full_fit" in raised.value.details["leakage"]["blocking"]
