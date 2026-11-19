"""Curation-only runs: signed evidence about a modality with no model."""

import json
import types

import pytest

from scmle.api import curator
from scmle.contracts import errors
from scmle.data import modalities
from scmle.pipeline.modality_lifecycle.preparers import document

KEY = "curation-key"


def adapter(count=6):
    samples = tuple(
        modalities.document.Sample(
            document_id=f"document-{index}",
            content=("alpha " if index % 2 else "beta ") * (index + 2),
            label="alpha" if index % 2 else "beta",
        )
        for index in range(count)
    )
    return document.DocumentPreparer(modalities.document.Dataset(samples))


def test_emplace_publishes_once_and_leaves_no_temporary_file(tmp_path):
    # Enforces website/src/pages/docs/extensions.astro:154.
    path = tmp_path / "deep" / "artifact.json"
    curator.emplace(path, b"first")
    assert path.read_bytes() == b"first"
    with pytest.raises(FileExistsError):
        curator.emplace(path, b"second")
    assert path.read_bytes() == b"first"
    assert [p.name for p in path.parent.iterdir()] == ["artifact.json"]


def passport(modality="document", content="h", samples=3):
    return types.SimpleNamespace(
        modality=modality, digests={"content": content}, samples=samples
    )


def test_conform_checks_modality_hash_and_count():
    # Enforces website/src/pages/docs/data.astro:199.
    curator.conform(passport(), "document", "h", 3)
    with pytest.raises(errors.InputError, match="manifest modality"):
        curator.conform(passport(modality="image"), "document", "h", 3)
    with pytest.raises(errors.InputError, match="content_hash"):
        curator.conform(passport(content="g"), "document", "h", 3)
    with pytest.raises(errors.InputError, match="sample_count"):
        curator.conform(passport(samples=4), "document", "h", 3)


def test_the_curator_checks_its_inputs(tmp_path):
    with pytest.raises(errors.KindError, match="adapter must be a Curator"):
        curator.Curator(object(), KEY)
    with pytest.raises(errors.InputError, match="signing_key must not be"):
        curator.Curator(adapter(), "")
    with pytest.raises(errors.KindError, match="manifest must be Passport"):
        curator.Curator(adapter(), KEY, manifest=passport())
    made = curator.Curator(adapter(), KEY, output_dir=str(tmp_path))
    assert made.output_dir == tmp_path
    assert made.manifest is None


def test_a_run_writes_signed_curation_evidence(tmp_path):
    # Enforces website/src/pages/docs/extensions.astro:154.
    source = adapter()
    result = curator.Curator(source, KEY, output_dir=tmp_path).run()
    path = tmp_path / "modality_curation.json"
    assert result.path == str(path)
    assert result.modality == "document"
    assert result.digests == {"dataset": source.digest}
    assert result.samples == 6
    assert result.signature
    stored = json.loads(path.read_text("utf-8"))
    assert stored == json.loads(json.dumps(result.to_dict()))
    assert path.read_text("utf-8").endswith("}\n")


def test_a_second_run_reuses_the_matching_artifact(tmp_path):
    first = curator.Curator(adapter(), KEY, output_dir=tmp_path).run()
    again = curator.Curator(adapter(), KEY, output_dir=tmp_path).run()
    assert again.digest == first.digest
    assert again.to_dict() == first.to_dict()


def test_an_artifact_of_other_data_is_not_replaced(tmp_path):
    curator.Curator(adapter(6), KEY, output_dir=tmp_path).run()
    before = (tmp_path / "modality_curation.json").read_bytes()
    with pytest.raises(errors.InputError, match="does not match current"):
        curator.Curator(adapter(8), KEY, output_dir=tmp_path).run()
    assert (tmp_path / "modality_curation.json").read_bytes() == before


def test_an_output_path_that_is_a_file_is_refused(tmp_path):
    (tmp_path / "file").write_text("x")
    with pytest.raises(errors.InputError, match="must be a directory"):
        curator.Curator(adapter(), KEY, output_dir=tmp_path / "file").run()


def test_examine_refuses_a_report_that_is_not_a_curation():
    made = curator.Curator(adapter(), KEY)
    made.adapter.curate = lambda: {"not": "a report"}
    with pytest.raises(errors.KindError, match="curate\\(\\) must return"):
        made.examine("document", made.adapter.digest, 6)


def test_examine_refuses_another_hash_or_count():
    # Enforces website/src/pages/docs/extensions.astro:167.
    made = curator.Curator(adapter(), KEY)
    with pytest.raises(errors.InputError, match="curation dataset_hash"):
        made.examine("document", "0" * 64, 6)
    with pytest.raises(errors.InputError, match="curation sample_count"):
        made.examine("document", made.adapter.digest, 7)
    payload = made.examine("document", made.adapter.digest, 6)
    assert isinstance(payload, dict)
