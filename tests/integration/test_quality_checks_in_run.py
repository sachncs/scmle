"""The profile stage reports curation findings before modelling."""

import json

import numpy
import pandas

import scmle.api.project


def frame(rows=700, seed=0, flip_every=0, shift=0.0):
    rng = numpy.random.default_rng(seed)
    data = pandas.DataFrame(
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

    result = scmle.api.project.Project(
        frame(flip_every=6),
        "y",
        metric="roc_auc",
        time_budget=45,
        reference=str(reference),
    ).run()

    text = findings(result)
    assert "label noise" in text
    assert "representativeness" in text and "'a'" in text
    report = json.loads(
        (result.run_dir / "curation_report.json").read_text(encoding="utf-8")
    )
    profile = json.loads(
        (result.run_dir / "dataset_profile.json").read_text(encoding="utf-8")
    )
    evidence_by_observation = {
        item["observation"]: item for item in profile["quality_evidence"]
    }
    assert set(evidence_by_observation) == {
        "label_noise",
        "representativeness",
    }
    for item in evidence_by_observation.values():
        assert item["method"]
        assert item["thresholds"]
        assert item["sample_bounds"]
        assert item["seed"] == 42
        assert item["metrics"]
        assert item["limitations"]
    assert report["schema_version"] == 1
    assert report["source_rows"] >= report["profiled_rows"]
    assert report["quality_findings"]
    assert all(
        set(finding)
        == {
            "id",
            "rule_version",
            "severity",
            "confidence",
            "affected_scope",
            "evidence",
            "remediation",
            "owner",
            "reviewer",
            "disposition",
            "blocking",
            "timestamp",
        }
        for finding in report["quality_findings"]
    )
    assert report["feature_columns"]["numeric"]


def test_clean_data_with_a_matching_reference_reports_nothing(tmp_path):
    reference = tmp_path / "reference.parquet"
    frame(500, seed=9).drop(columns="y").to_parquet(reference)

    result = scmle.api.project.Project(
        frame(),
        "y",
        metric="roc_auc",
        time_budget=45,
        reference=str(reference),
    ).run()

    text = findings(result)
    assert "label noise" not in text and "representativeness" not in text
