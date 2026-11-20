"""The time column can be offered to the model, but only on request."""

import numpy
import pandas
import support


def drifting_frame(rows=700, seed=0):
    rng = numpy.random.default_rng(seed)
    stamp = numpy.arange(rows, dtype=float)
    x = rng.normal(size=rows)
    drift = (stamp / rows) * 2 - 1
    y = (x + 1.5 * drift + rng.normal(0, 0.5, rows) > 0).astype(int)
    return pandas.DataFrame({"t": stamp, "x": x, "y": y})


def run(tmp_path, **kwargs):
    return support.make_project(
        drifting_frame(),
        "y",
        budget=300,
        directory=tmp_path,
        timestamp="t",
        **kwargs,
    ).run()


def test_the_time_column_is_split_only_by_default(tmp_path):
    # Enforces docs/BENCHMARKS.md:97.
    # Enforces website/src/pages/docs/benchmarks.astro:793.
    assert "t" not in run(tmp_path).bundle.features


def test_the_time_column_becomes_a_feature_on_request(tmp_path):
    result = run(tmp_path, time_as_feature=True)

    assert "t" in result.bundle.features
    assert result.evaluation["validation"] == "temporal"
