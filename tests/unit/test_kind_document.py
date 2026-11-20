"""The document kind: text files through the shared stages.

The kind reads a directory of class folders or a listing table into one row
per document, keys each row by its document id joined with its
near-duplicate cluster, and hands the rows to the shared stages: the same
search over model families, cross-validation, locked test set, explanations
and package as a table. The end-to-end runs register the kind and its
capabilities for the test (`register`), as `kinds.KINDS` and
`roster.enrol` will once the lead wires them.
"""

import importlib
import json

import numpy
import pandas
import pytest

from scmle.api import project
from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.contracts import tasks
import scmle.contracts.data
import scmle.data.modalities.document
from scmle.data.readers import tabular
from scmle.data.validation import columns
from scmle.pipeline import kinds
from scmle.pipeline import profiler
from scmle.pipeline import roster
from scmle.pipeline.kinds import document
from scmle.pipeline.kinds import rows
from tests import support

COMMON = tuple(f"w{index}" for index in range(300))
SIGNAL = {
    "pos": tuple(f"good{index}" for index in range(30)),
    "neg": tuple(f"bad{index}" for index in range(30)),
}


def prose(generator, label, words=60):
    """A text of common words salted with words of its label."""
    drawn = [
        *generator.choice(COMMON, words),
        *generator.choice(SIGNAL[label], 6),
    ]
    generator.shuffle(drawn)
    return " ".join(drawn)


def folders(root, count=40):
    """Class folders of `count` documents each, plus one exact copy."""
    generator = numpy.random.default_rng(0)
    for label in SIGNAL:
        (root / label).mkdir(parents=True)
        for index in range(count):
            text = prose(generator, label)
            (root / label / f"{index}.txt").write_text(text, encoding="utf-8")
    copy = (root / "pos" / "3.txt").read_text(encoding="utf-8")
    (root / "pos" / "copy.md").write_text(copy, encoding="utf-8")
    return root


def config(uri, target="label", **sections):
    return tasks.Config(
        data=tasks.Dataset(uri=str(uri), target=target, modality="document"),
        **sections,
    )


def test_the_document_kind_is_a_kind():
    assert isinstance(document.DOCUMENT, kinds.Kind)
    assert document.DOCUMENT.name == "document"


def test_class_folders_become_one_row_per_document(tmp_path):
    root = folders(tmp_path / "corpus", count=5)
    (root / "loose.txt").write_text("an unlabelled note", encoding="utf-8")
    (root / ".hidden.txt").write_text("skipped", encoding="utf-8")
    frame, stream = document.DOCUMENT.read(config(root))
    assert stream is None
    assert len(frame) == 12
    assert {document.TEXT, *document.MEASURES, "label", rows.KEY} == set(
        frame.columns
    )
    assert frame["label"].isna().sum() == 1
    assert set(frame["label"].dropna()) == {"pos", "neg"}
    assert (frame["pages"] == 1).all()
    assert (frame["characters"] == frame[document.TEXT].str.len()).all()


def test_an_exact_copy_shares_the_split_key_of_its_original(tmp_path):
    root = folders(tmp_path / "corpus", count=5)
    frame = document.DOCUMENT.read(config(root))[0]
    copy = (root / "pos" / "copy.md").read_text(encoding="utf-8")
    keys = frame.loc[frame[document.TEXT] == copy, rows.KEY]
    assert len(keys) == 2
    assert (keys == keys.iloc[0]).all()
    assert frame[rows.KEY].nunique() == len(frame) - 1


def test_a_listing_reads_files_and_texts_and_keeps_chunks_together(tmp_path):
    (tmp_path / "a.txt").write_text("first file text here", encoding="utf-8")
    listing = pandas.DataFrame(
        {
            "path": ["a.txt", None, None],
            "text": [None, "chunk one of b", "chunk two of b"],
            "document": [None, "007", "007"],
            "source": ["web", "mail", "mail"],
            "y": [1.5, 2.5, 3.5],
        }
    )
    path = tmp_path / "listing.csv"
    listing.to_csv(path, index=False)
    frame = document.DOCUMENT.read(config(path, "y"))[0]
    assert frame[document.TEXT].tolist() == [
        "first file text here",
        "chunk one of b",
        "chunk two of b",
    ]
    assert frame["source"].tolist() == ["web", "mail", "mail"]
    assert "document" not in frame.columns
    assert frame[rows.KEY].iloc[1] == frame[rows.KEY].iloc[2]
    assert frame[rows.KEY].iloc[0] != frame[rows.KEY].iloc[1]


@pytest.mark.parametrize(
    ("columns", "match"),
    [
        ({"body": ["x"], "y": [1]}, "needs a 'path' or 'text' column"),
        ({"text": ["x"], "tokens": [1], "y": [1]}, "clash"),
        ({"text": [None], "y": [1]}, "names no document file or text"),
    ],
)
def test_an_unusable_listing_is_refused_with_a_data_error(
    tmp_path, columns, match
):
    path = tmp_path / "listing.csv"
    pandas.DataFrame(columns).to_csv(path, index=False)
    with pytest.raises(errors.DataError, match=match):
        document.DOCUMENT.read(config(path, "y"))


def test_a_missing_source_or_empty_directory_is_refused(tmp_path):
    with pytest.raises(errors.DataError, match="not found"):
        document.DOCUMENT.read(config(tmp_path / "absent"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(errors.DataError, match="no document"):
        document.DOCUMENT.read(config(tmp_path / "empty"))


def test_mm16_identical_text_with_conflicting_labels_is_refused(tmp_path):
    path = tmp_path / "listing.csv"
    pandas.DataFrame(
        {
            "text": ["same words", "same words", "other words"],
            "document": ["a", "a", "b"],
            "label": ["x", "y", "x"],
        }
    ).to_csv(path, index=False)
    with pytest.raises(errors.DataError, match="conflicting labels") as failure:
        document.DOCUMENT.read(config(path))
    assert failure.value.details["conflicts"][0]["sources"] == ["a"]


def test_a_pdf_without_its_optional_reader_is_a_typed_dependency_error(
    tmp_path, monkeypatch
):
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "scan.pdf").write_bytes(b"%PDF-1.4")
    with pytest.raises(
        scmle.data.modalities.document.DependencyError, match="PDF"
    ) as failure:
        document.DOCUMENT.read(config(tmp_path))
    assert failure.value.details["dependency"] == "pypdf"


def test_a_pdf_is_read_through_pypdf(tmp_path):
    pdf = pytest.importorskip("pypdf", reason="pypdf is an optional reader")
    writer = pdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    (tmp_path / "a").mkdir()
    with (tmp_path / "a" / "blank.pdf").open("wb") as handle:
        writer.write(handle)
    with pytest.raises(
        scmle.data.modalities.document.DocumentError, match="empty"
    ):
        document.DOCUMENT.read(config(tmp_path))


def test_the_split_keeps_near_duplicates_and_drops_the_key(tmp_path):
    root = folders(tmp_path / "corpus")
    spec = config(root)
    frame = document.DOCUMENT.read(spec)[0]
    copy = (root / "pos" / "copy.md").read_text(encoding="utf-8")
    twins = set(numpy.flatnonzero(frame[document.TEXT] == copy).tolist())
    for seed in range(6):
        seeded = spec.model_copy(
            update={
                "validation": spec.validation.model_copy(update={"seed": seed})
            }
        )
        splitter = document.DOCUMENT.constrain(seeded, tasks.Task.BINARY)
        split = splitter.split(frame)
        assert split.manifest.strategy == "group"
        assert rows.KEY not in split.features.columns
        assert rows.KEY not in split.test.features.columns
        test = set(split.manifest.test)
        assert twins <= test or not twins & test
        kept = numpy.asarray(split.manifest.training)
        for fold in split.manifest.folds:
            inner = set(kept[list(fold[1])].tolist())
            assert twins <= inner or not twins & inner


def test_an_exchangeable_strategy_becomes_the_group_split_with_a_note(
    tmp_path,
):
    root = folders(tmp_path / "corpus", count=30)
    spec = config(root, validation=tasks.Validation(strategy="stratified"))
    frame = document.DOCUMENT.read(spec)[0]
    split = document.DOCUMENT.constrain(spec, tasks.Task.BINARY).split(frame)
    assert split.manifest.strategy == "group"
    assert any("replaced by the group split" in n for n in split.manifest.notes)


def test_the_survey_profiles_the_rows_and_names_the_duplicate_groups(
    tmp_path,
):
    root = folders(tmp_path / "corpus", count=20)
    (root / "neg" / "mail.txt").write_text(
        "reach me at ann.lee@example.org "
        + prose(numpy.random.default_rng(9), "neg"),
        encoding="utf-8",
    )
    spec = config(root)
    frame = document.DOCUMENT.read(spec)[0]
    manifest = tabular.manifested(frame, "label", str(root), "document")
    portrait = document.DOCUMENT.survey(
        spec, frame, manifest, tasks.Task.BINARY
    )
    assert isinstance(portrait, scmle.contracts.data.Portrait)
    assert portrait.plan.text == (document.TEXT,)
    assert rows.KEY not in portrait.statistics
    notes = " ".join(portrait.quality)
    assert "document: 2 of 42 rows in 1 cohorts" in notes
    assert "1 documents hold sensitive values" in notes


def test_the_featuriser_encodes_the_text_and_the_decoder_defers(tmp_path):
    plan = scmle.contracts.data.Blueprint(
        numeric=("characters",), categorical=(), text=(document.TEXT,)
    )
    kept = document.DOCUMENT.rows(
        [
            scmle.data.modalities.document.Sample(
                f"d{i}", f"alpha beta gamma {i}"
            )
            for i in range(8)
        ]
    )
    featuriser = document.DOCUMENT.featurise(
        plan, scmle.contracts.data.Preprocessing()
    )
    values = featuriser.fit_transform(kept)
    assert values.shape[0] == 8
    assert values.shape[1] > 1
    assert document.DOCUMENT.decode(("low", "high")) == ("low", "high")
    assert document.DOCUMENT.decode(None) is None


def test_the_featurise_cost_adds_the_document_reading_bound():
    bound = profiler.estimate(1000, 4, 2, 600.0)
    assert document.DOCUMENT.cost(1000, 4, 2, 600.0) == pytest.approx(
        bound + 1000 * document.SECONDS
    )
    with pytest.raises(errors.InputError):
        document.DOCUMENT.cost(-1, 4, 2, 600.0)


def test_the_capabilities_are_declared_for_each_task_and_family():
    declared = document.DOCUMENT.declare()
    keys = {item.key for item in declared}
    assert len(keys) == len(declared)
    assert {item.status for item in declared} == {
        capabilities.Status.EXPERIMENTAL
    }
    assert {item.key.modality for item in declared} == {
        capabilities.Modality.DOCUMENT
    }
    assert {item.key.task for item in declared} == set(rows.TASKS)
    registry = roster.enrol()
    assert {item.key for item in declared} <= set(registry.records)


def test_a_document_run_trains_tests_explains_and_packages(
    tmp_path, monkeypatch
):
    root = folders(tmp_path / "corpus")
    run = project.Project(
        str(root),
        "label",
        budget=120,
        modality="document",
        models=support.optin(tasks.Task.BINARY),
        enforcement="cooperative",
        directory=tmp_path / "runs",
    ).run()
    assert run.state == "COMPLETED"
    manifest = json.loads(
        (run.directory / "validation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["strategy"] == "group"
    profile = json.loads(
        (run.directory / "dataset_profile.json").read_text(encoding="utf-8")
    )
    assert json.dumps(profile).count(rows.KEY) == 0
    assert len(set(run.plan.candidates)) > 2
    evaluation = json.loads(
        (run.directory / "evaluation.json").read_text(encoding="utf-8")
    )
    assert evaluation["task"] == "binary_classification"
    assert (run.directory / "explanations").is_dir()


def test_a_numeric_document_label_is_a_regression(tmp_path, monkeypatch):
    generator = numpy.random.default_rng(3)
    texts, scores = [], []
    for good in generator.integers(0, 6, 80).tolist():
        words = [*generator.choice(COMMON, 50), *["good0"] * good]
        generator.shuffle(words)
        texts.append(" ".join(words))
        scores.append(good + float(generator.normal(0, 0.1)))
    path = tmp_path / "listing.csv"
    pandas.DataFrame({"text": texts, "score": scores}).to_csv(path, index=False)
    run = project.Project(
        str(path),
        "score",
        budget=120,
        modality="document",
        models=support.optin(tasks.Task.REGRESSION),
        enforcement="cooperative",
        directory=tmp_path / "runs",
    ).run()
    assert run.state == "COMPLETED"
    assert run.plan.task == "regression"


def test_a_short_corpus_keeps_its_text_column_as_text(tmp_path):
    """Under 30 documents the column profiler would call the text an id.

    The document kind's survey declares it text (`Document.texts`), so a
    corpus of 24 documents still trains on its words.
    """
    root = folders(tmp_path / "corpus", count=12)
    spec = config(root)
    frame = document.DOCUMENT.read(spec)[0]
    assert len(frame) < columns.MIN_TEXT_ROWS
    manifest = tabular.manifested(frame, "label", str(root), "document")
    portrait = document.DOCUMENT.survey(
        spec, frame, manifest, tasks.Task.BINARY
    )
    assert portrait.plan.text == (document.TEXT,)
    assert document.TEXT not in portrait.plan.dropped
