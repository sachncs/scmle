"""Tests for the dependency-light document data foundation."""

import dataclasses
import json

import pytest

from scml.data.modalities import document


@pytest.fixture
def valid_markdown() -> bytes:
    """Returns a valid UTF-8 Markdown fixture with two sections."""
    return b"# Introduction\nHello from the document.\n\n## Method\nMeasure twice.\n"


@pytest.fixture
def empty_document() -> bytes:
    """Returns an empty-content fixture."""
    return b" \n\t"


@pytest.fixture
def malformed_bytes() -> bytes:
    """Returns bytes that cannot be decoded as UTF-8."""
    return b"valid prefix\xff\xfe"


@pytest.fixture
def duplicate_dataset() -> document.DocumentDataset:
    """Returns a dataset containing one deterministic content duplicate."""
    return document.DocumentDataset(
        (
            document.DocumentSample("first", "same content"),
            document.DocumentSample("second", "same content"),
            document.DocumentSample("third", "different content"),
        )
    )


@pytest.fixture
def shifted_dataset() -> document.DocumentDataset:
    """Returns chunked documents with a shifted document in the corpus."""
    return document.DocumentDataset(
        (
            document.DocumentSample("doc-a", "alpha train chunk"),
            document.DocumentSample("doc-a", "alpha shifted chunk"),
            document.DocumentSample("doc-b", "beta train chunk"),
            document.DocumentSample("doc-c", "gamma shifted chunk"),
            document.DocumentSample("doc-d", "delta train chunk"),
            document.DocumentSample("doc-e", "epsilon test chunk"),
        )
    )


def test_reader_loads_utf8_markdown_and_builds_identity_and_provenance(
    valid_markdown: bytes,
):
    """Markdown is decoded without optional dependencies."""
    sample = document.DocumentReader().load(
        valid_markdown,
        document_id="guide-1",
        document_format=document.DocumentFormat.MARKDOWN,
    )

    assert sample.format is document.DocumentFormat.MARKDOWN
    assert sample.encoding == "utf-8"
    assert sample.size_bytes == len(valid_markdown)
    assert len(sample.content_hash) == 64
    assert {span.kind for span in sample.provenance} == {"page", "section"}
    assert [
        span.label for span in sample.provenance if span.kind == "section"
    ] == [
        "Introduction",
        "Method",
    ]


def test_plain_text_reader_infers_path_format(tmp_path):
    """A text path is inferred and retains its canonical source URI."""
    path = tmp_path / "notes.txt"
    path.write_text("A small plain-text document.", encoding="utf-8")

    sample = document.DocumentReader().load(path)

    assert sample.format is document.DocumentFormat.TEXT
    assert sample.document_id == "notes.txt"
    assert sample.source_uri == path.resolve().as_uri()


def test_empty_fixture_fails_with_structured_validation(empty_document: bytes):
    """Whitespace-only content cannot enter the immutable contract."""
    with pytest.raises(document.DocumentValidationError) as caught:
        document.DocumentReader().load(empty_document, document_id="empty")

    assert caught.value.category == "document_validation"
    assert caught.value.details == {
        "code": "empty_content",
        "document_id": "empty",
    }


def test_malformed_bytes_fixture_reports_encoding_offset(
    malformed_bytes: bytes,
):
    """Malformed bytes fail closed with the decoder offset."""
    with pytest.raises(document.DocumentValidationError) as caught:
        document.DocumentReader().load(malformed_bytes, document_id="bad")

    assert caught.value.details["code"] == "invalid_encoding"
    assert caught.value.details["encoding"] == "utf-8"
    assert caught.value.details["start"] == 12


def test_size_validation_is_bounded_and_structured():
    """Oversized input is rejected before decoding."""
    with pytest.raises(document.DocumentValidationError) as caught:
        document.DocumentReader(max_bytes=4).load(b"hello", document_id="large")

    assert caught.value.details == {
        "code": "size_limit",
        "actual_bytes": 5,
        "max_bytes": 4,
    }


def test_duplicate_fixture_is_deterministic(duplicate_dataset):
    """Identical content is grouped independently of sample order."""
    assert duplicate_dataset.duplicate_groups() == (("first", "second"),)
    reversed_dataset = document.DocumentDataset(
        tuple(reversed(duplicate_dataset.samples))
    )
    assert (
        reversed_dataset.duplicate_groups()
        == duplicate_dataset.duplicate_groups()
    )
    assert reversed_dataset.content_hash == duplicate_dataset.content_hash


def test_profile_is_bounded_and_uses_a_conservative_language_heuristic():
    """Long content is profiled within both requested bounds."""
    sample = document.DocumentSample("profiled", "hello world " * 100)

    profile = sample.profile(max_characters=32, max_tokens=4)

    assert profile.character_count == 32
    assert profile.token_count == 4
    assert profile.language == "en"
    assert profile.truncated is True


def test_shifted_fixture_splits_by_document_id_without_leakage(shifted_dataset):
    """Repeated chunks of one document stay in exactly one partition."""
    first = shifted_dataset.split(
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=17,
    )
    second = shifted_dataset.split(
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=17,
    )

    assignments = first.document_ids()
    all_partition_ids = [set(ids) for ids in assignments.values()]
    assert sum("doc-a" in ids for ids in all_partition_ids) == 1
    assert set().union(*all_partition_ids) == {
        "doc-a",
        "doc-b",
        "doc-c",
        "doc-d",
        "doc-e",
    }
    assert first == second


def test_pdf_without_optional_reader_has_structured_dependency_error():
    """PDF support fails explicitly when the optional reader is absent."""
    with pytest.raises(document.DocumentDependencyError) as caught:
        document.DocumentReader().load(
            b"%PDF-1.7", document_id="pdf", document_format="pdf"
        )

    assert caught.value.category == "document_dependency"
    assert caught.value.details == {
        "code": "missing_dependency",
        "dependency": "pypdf",
        "extra": "documents",
        "format": "pdf",
    }


def test_injected_pdf_reader_provides_page_provenance():
    """An optional parser can be supplied without adding a core dependency."""
    reader = document.DocumentReader(
        pdf_reader=lambda raw_bytes: ("first page", "second page")
    )

    sample = reader.load(b"%PDF-1.7", document_id="pdf", document_format="pdf")

    assert sample.content == "first page\fsecond page"
    assert [
        span.page_number for span in sample.provenance if span.kind == "page"
    ] == [
        1,
        2,
    ]


def test_contracts_are_immutable_and_profile_is_json_native():
    """Frozen contracts prevent mutation and profiles remain serializable."""
    sample = document.DocumentSample("immutable", "content")
    with pytest.raises(dataclasses.FrozenInstanceError):
        sample.content = "changed"  # type: ignore[misc]

    profile = sample.profile()
    assert json.loads(json.dumps(dataclasses.asdict(profile))) == {
        "document_id": "immutable",
        "character_count": 7,
        "token_count": 1,
        "language": "en",
        "truncated": False,
    }
