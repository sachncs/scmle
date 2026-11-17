"""A trained bundle traces back to its run's inputs (ML-10, claim C6)."""

import json

from benchmarks import reliability
import scmle.artifacts.bundle


def test_the_bundle_names_the_seed_configuration_data_and_split(tmp_path):
    project = reliability.fast_project(reliability.binary_data(), "y", tmp_path)
    result = project.run()
    run = result.run_dir
    info = scmle.artifacts.bundle.Bundle.load(run / "model", trusted=True).info
    evaluation = json.loads((run / "evaluation.json").read_text("utf-8"))
    dataset = json.loads((run / "dataset_manifest.json").read_text("utf-8"))

    assert info.seed == project.spec.validation.seed
    assert info.configuration == project.spec.config_hash()
    assert info.dataset == dataset["data_hash"]
    assert info.split == evaluation["split_fingerprint"]
    assert info.commit and info.threads
