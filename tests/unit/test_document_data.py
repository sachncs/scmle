"""Tests for the dependency-light document data foundation."""

import dataclasses
import hashlib
import json

import pytest

from scmle.data.modalities import document


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
def duplicate_dataset() -> document.Dataset:
    """Returns a dataset containing one deterministic content duplicate."""
    return document.Dataset(
        (
            document.Sample("first", "same content"),
            document.Sample("second", "same content"),
            document.Sample("third", "different content"),
        )
    )


@pytest.fixture
def shifted_dataset() -> document.Dataset:
    """Returns chunked documents with a shifted document in the corpus."""
    return document.Dataset(
        (
            document.Sample("doc-a", "alpha train chunk"),
            document.Sample("doc-a", "alpha shifted chunk"),
            document.Sample("doc-b", "beta train chunk"),
            document.Sample("doc-c", "gamma shifted chunk"),
            document.Sample("doc-d", "delta train chunk"),
            document.Sample("doc-e", "epsilon test chunk"),
        )
    )


def test_reader_loads_utf8_markdown_and_builds_identity_and_provenance(
    valid_markdown: bytes,
):
    """Markdown is decoded without optional dependencies."""
    sample = document.Reader().load(
        valid_markdown,
        document_id="guide-1",
        document_format=document.Format.MARKDOWN,
    )

    assert sample.format is document.Format.MARKDOWN
    assert sample.encoding == "utf-8"
    assert sample.size_bytes == len(valid_markdown)
    assert len(sample.content_hash) == 64
    assert sample.source_digest == hashlib.sha256(valid_markdown).hexdigest()
    assert [citation.locator for citation in sample.citations()] == [
        "page 1",
        "Introduction",
        "Method",
    ]
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

    sample = document.Reader().load(path)

    assert sample.format is document.Format.TEXT
    assert sample.document_id == "notes.txt"
    assert sample.source_uri == path.resolve().as_uri()
    assert sample.source_digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_reader_accepts_an_explicit_layout_extractor_contract():
    """Optional layout extraction receives raw bytes and format identity."""
    observed: dict[str, object] = {}

    def layout_reader(
        raw_bytes: bytes,
        document_id: str,
        document_format: document.Format,
    ) -> document.Layout:
        observed.update(
            {
                "raw_bytes": raw_bytes,
                "document_id": document_id,
                "document_format": document_format,
            }
        )
        return document.Layout(
            source="layout://run-1",
            provider="fixture-layout",
            provider_version="1",
            elements=(document.Element("heading", 0, 7, 1),),
            source_digest=hashlib.sha256(raw_bytes).hexdigest(),
        )

    sample = document.Reader(layout_reader=layout_reader).load(
        b"Heading", document_id="doc-1"
    )

    assert observed == {
        "raw_bytes": b"Heading",
        "document_id": "doc-1",
        "document_format": document.Format.TEXT,
    }
    assert sample.layout is not None
    assert sample.layout.source_digest == sample.source_digest


@pytest.mark.parametrize(
    ("reader", "expected_error", "code"),
    (
        (
            document.Reader(
                layout_reader=lambda *arguments: (
                    generator for generator in ()
                ).throw(ImportError())
            ),
            document.DocumentDependencyError,
            "missing_dependency",
        ),
        (
            document.Reader(
                layout_reader=lambda *arguments: "invalid"  # type: ignore[return-value]
            ),
            document.DocumentValidationError,
            "layout_invalid_result",
        ),
    ),
)
def test_layout_extractor_failures_are_structured(reader, expected_error, code):
    """Optional extraction failures never become silent metadata loss."""
    with pytest.raises(expected_error) as caught:
        reader.load(b"content", document_id="doc-1")

    assert caught.value.details["code"] == code


def test_layout_source_digest_mismatch_fails_closed():
    """Layout provenance cannot be attached to a different raw source."""
    layout = document.Layout(
        source="layout://run-2",
        provider="fixture-layout",
        provider_version="1",
        elements=(document.Element("text", 0, 7, 1),),
        source_digest="0" * 64,
    )

    with pytest.raises(
        document.DocumentValidationError, match="does not match"
    ):
        document.Reader().load(b"Heading", document_id="doc-1", layout=layout)


def test_curation_reports_source_traceability_without_inventing_citations():
    """Curation exposes raw-source coverage and preserves missing identity."""
    sourced = document.Reader().load(b"source-backed", document_id="sourced")
    in_memory = document.Sample("manual", "manual content")

    report = document.Dataset((sourced, in_memory)).curate()

    assert report.source_traceability_coverage == 0.5
    assert report.citation_count == len(sourced.citations())
    assert in_memory.citations() == ()
    assert report.to_dict()["source_traceability_coverage"] == 0.5


def test_partial_source_identity_is_visible_as_a_quality_finding():
    """A URI without a raw digest cannot claim citation traceability."""
    sample = document.Sample(
        "partial",
        "content",
        source_uri="file:///tmp/partial.txt",
    )

    report = document.Dataset((sample,)).curate()

    assert [finding.finding_id for finding in report.quality_findings] == [
        "quality.document-source-traceability-incomplete",
    ]


def test_document_layout_records_ocr_and_table_provenance():
    """Curation preserves caller-owned layout extraction evidence."""
    content = "Heading\nrow one\nrow two"
    layout = document.Layout(
        source="ocr://run-4",
        provider="layout-engine",
        provider_version="3.2",
        ocr_confidence=0.93,
        elements=(
            document.Element("heading", 0, 7, 1, 0.99),
            document.Element(
                "table", 8, len(content), 1, 0.88, table_rows=2, table_columns=1
            ),
        ),
    )
    sample = document.Sample("structured", content, layout=layout)

    report = document.Dataset((sample,)).curate()

    assert sample.layout is layout
    assert layout.table_count == 1
    assert layout.to_dict()["provider_version"] == "3.2"
    assert report.layout_coverage == 1.0
    assert report.layout_providers == ("layout-engine",)
    assert report.layout_element_count == 2
    assert report.table_element_count == 1
    assert report.ocr_confidence_min == 0.93
    assert report.ocr_confidence_max == 0.93
    assert report.release_ready


def test_document_layout_confidence_and_offsets_fail_closed():
    """Missing OCR confidence is visible and offsets cannot escape content."""
    content = "OCR text"
    layout = document.Layout(
        source="ocr://run-5",
        provider="layout-engine",
        provider_version="3.2",
        elements=(document.Element("ocr", 0, len(content), 1),),
    )
    report = document.Dataset(
        (document.Sample("uncertain", content, layout=layout),)
    ).curate()

    assert [finding.finding_id for finding in report.quality_findings] == [
        "quality.document-ocr-confidence-missing",
    ]
    invalid_layout = document.Layout(
        source="ocr://run-6",
        provider="layout-engine",
        provider_version="3.2",
        elements=(document.Element("ocr", 0, 9, 1),),
    )
    with pytest.raises(document.DocumentValidationError, match="exceeds"):
        document.Sample("invalid", content, layout=invalid_layout)


def test_document_sensitive_content_is_detected_without_leaking_values():
    """Curation counts PII and redaction removes values from derived text."""
    content = "Contact alice@example.com; SSN 123-45-6789."
    sample = document.Sample("sensitive", content)
    report = document.Dataset((sample,)).curate()

    assert document.sensitive_matches(content) == {"email": 1, "ssn": 1}
    assert document.redact_sensitive_text(content) == (
        "Contact [REDACTED_EMAIL]; SSN [REDACTED_SSN]."
    )
    assert report.sensitive_match_counts == {"email": 1, "ssn": 1}
    assert report.release_ready is False
    assert [finding.finding_id for finding in report.risk_findings] == [
        "risk.document-sensitive-content",
    ]
    assert "alice@example.com" not in str(report.to_dict())
    assert "123-45-6789" not in str(report.to_dict())


def test_empty_fixture_fails_with_structured_validation(empty_document: bytes):
    """Whitespace-only content cannot enter the immutable contract."""
    with pytest.raises(document.DocumentValidationError) as caught:
        document.Reader().load(empty_document, document_id="empty")

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
        document.Reader().load(malformed_bytes, document_id="bad")

    assert caught.value.details["code"] == "invalid_encoding"
    assert caught.value.details["encoding"] == "utf-8"
    assert caught.value.details["start"] == 12


def test_size_validation_is_bounded_and_structured():
    """Oversized input is rejected before decoding."""
    with pytest.raises(document.DocumentValidationError) as caught:
        document.Reader(max_bytes=4).load(b"hello", document_id="large")

    assert caught.value.details == {
        "code": "size_limit",
        "actual_bytes": 5,
        "max_bytes": 4,
    }


def test_duplicate_fixture_is_deterministic(duplicate_dataset):
    """Identical content is grouped independently of sample order."""
    assert duplicate_dataset.duplicate_groups() == (("first", "second"),)
    reversed_dataset = document.Dataset(
        tuple(reversed(duplicate_dataset.samples))
    )
    assert (
        reversed_dataset.duplicate_groups()
        == duplicate_dataset.duplicate_groups()
    )
    assert reversed_dataset.content_hash == duplicate_dataset.content_hash


def test_profile_is_bounded_and_uses_a_conservative_language_heuristic():
    """Long content is profiled within both requested bounds."""
    sample = document.Sample("profiled", "hello world " * 100)

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
        document.Reader().load(
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
    reader = document.Reader(
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
    sample = document.Sample("immutable", "content")
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


def test_curation_reports_composition_duplicates_and_bounded_profiles(
    duplicate_dataset,
):
    """Curation makes document understanding reviewable before modelling."""
    report = duplicate_dataset.curate(max_characters=5, max_tokens=1)

    assert report.dataset_hash == duplicate_dataset.content_hash
    assert report.sample_count == 3
    assert report.document_count == 3
    assert report.duplicate_groups == (("first", "second"),)
    assert report.language_counts == {"en": 3}
    assert report.format_counts == {"text": 3}
    assert report.label_counts == {"unlabeled": 3}
    assert [finding.finding_id for finding in report.quality_findings] == [
        "quality.duplicate-document-content",
        "quality.profile-truncated",
    ]
    assert report.risk_findings == ()
    assert json.loads(json.dumps(report.to_dict())) == report.to_dict()


def test_curation_blocks_identical_content_with_conflicting_labels():
    """Duplicate content with different labels is a supervised-data risk."""
    dataset = document.Dataset(
        (
            document.Sample("positive", "same text", label="yes"),
            document.Sample("negative", "same text", label="no"),
            document.Sample("unlabeled", "other text"),
        )
    )

    report = dataset.curate()

    assert report.release_ready is False
    assert [finding.finding_id for finding in report.risk_findings] == [
        "risk.conflicting-duplicate-labels",
        "risk.mixed-label-state",
    ]
    assert report.unresolved_blocking_findings[0].affected_scope == {
        "document_ids": ["negative", "positive"]
    }


def test_curation_bounds_must_be_positive():
    """Curation cannot silently turn off the bounded inspection contract."""
    dataset = document.Dataset((document.Sample("doc", "content"),))

    with pytest.raises(ValueError, match="curation bounds"):
        dataset.curate(max_characters=0)
