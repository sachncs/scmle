"""The profile stage reports label noise and a gap to the reference sample."""

import numpy as np
import pandas as pd

import scml


def frame(rows=700, seed=0, flip_every=0, shift=0.0):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {"a": rng.normal(size=rows) + shift, "b": rng.normal(size=rows)}
    )
    data["y"] = (data.a - shift + data.b > 0).astype(int)
    if flip_every:
        data.loc[::flip_every, "y"] = 1 - data.loc[::flip_every, "y"]
    return data


def findings(result):
    return " | ".join(result.profile.quality_findings)


def test_noisy_labels_and_a_shifted_reference_are_reported(tmp_path):
    reference = tmp_path / "reference.parquet"
    frame(500, seed=9, shift=2.0).drop(columns="y").to_parquet(reference)

    result = scml.MLProject(
        frame(flip_every=6),
        "y",
        metric="roc_auc",
        time_budget=45,
        reference=str(reference),
    ).run()

    text = findings(result)
    assert "label noise" in text
    assert "representativeness" in text and "'a'" in text


def test_clean_data_with_a_matching_reference_reports_nothing(tmp_path):
    reference = tmp_path / "reference.parquet"
    frame(500, seed=9).drop(columns="y").to_parquet(reference)

    result = scml.MLProject(
        frame(),
        "y",
        metric="roc_auc",
        time_budget=45,
        reference=str(reference),
    ).run()

    text = findings(result)
    assert "label noise" not in text and "representativeness" not in text
