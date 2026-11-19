"""Behavioural tests for the document modality: text, sample, reader, review."""

import hashlib

import pytest

from scmle.contracts import data
from scmle.contracts import errors
from scmle.data.modalities.document import dataset
from scmle.data.modalities.document import reader
from scmle.data.modalities.document import review
from scmle.data.modalities.document import sample
from scmle.data.modalities.document import text
from scmle.data.modalities.document.elements import DependencyError
from scmle.data.modalities.document.elements import DocumentError
from scmle.data.modalities.document.elements import Element
from scmle.data.modalities.document.elements import Format
from scmle.data.modalities.document.elements import Layout
from scmle.data.modalities.document.elements import Span

DIGEST = "a" * 64


def layout(confidence=None, digest=None, end=2, kind="text", provider="p"):
    element = Element(
        kind,
        0,
        end,
        1,
        table_rows=1 if kind == "table" else None,
        table_columns=1 if kind == "table" else None,
    )
    return Layout("s", provider, "1", (element,), confidence, digest)


def doc(identifier, content="hello world", **kwargs):
    return sample.Sample(identifier, content, **kwargs)


@pytest.mark.parametrize(
    ("content", "language"),
    [
        ("", "unknown"),
        ("123 !!", "unknown"),
        ("hello", "en"),
        ("नमस्ते abc", "hi"),
        ("aaaअ", "hi"),
        ("aaaaaaaअअअ", "hi"),
        ("abcéé", "en"),
        ("abéé", "unknown"),
    ],
)
def test_detect_classifies_by_script_share(content, language):
    assert text.detect(content) == language


def test_detect_thresholds():
    assert text.DEVANAGARI_MIN_SHARE == 0.25
    assert text.ASCII_LETTER_MIN_SHARE == 0.60
    assert text.detect("abcअ") == "hi"
    assert text.detect("abcdअ") == "en"
    assert text.detect("abcéé") == "en"


def test_segment_builds_pages_and_sections():
    content = "# One\nbody\f\n## Two\nmore"
    spans = text.segment(content)
    assert spans == (
        Span("page", 0, 10, "page 1", 1),
        Span("page", 11, 23, "page 2", 2),
        Span("section", 0, 12, "One", 1),
        Span("section", 12, 23, "Two", 2),
    )


def test_segment_finds_a_heading_at_the_top_of_a_page():
    spans = text.segment("# One\nbody\f## Two\nmore")
    assert [s.label for s in spans if s.kind == "section"] == ["One", "Two"]
    assert spans[-1] == Span("section", 11, 22, "Two", 2)


def test_segment_ends_a_heading_at_a_form_feed():
    spans = text.segment("# One\f## Two")
    assert [s for s in spans if s.kind == "section"] == [
        Span("section", 0, 6, "One", 1),
        Span("section", 6, 12, "Two", 2),
    ]
    assert [s.label for s in text.segment("#\fbody\n#\nnext")] == [
        "page 1",
        "page 2",
    ]


def test_segment_skips_empty_pages_and_deep_headings():
    spans = text.segment("a\f\f####### no\n   ###### six")
    assert [s.label for s in spans] == ["page 1", "page 3", "six"]
    assert spans[2].page_number == 3
    assert text.segment("    # indented") == (Span("page", 0, 14, "page 1", 1),)


def test_a_sample_derives_identity_and_provenance():
    built = doc("d", "# Title\nbody")
    assert built.content_hash == hashlib.sha256(b"# Title\nbody").hexdigest()
    assert built.size == 12
    assert built.format is Format.TEXT
    assert [s.kind for s in built.provenance] == ["page", "section"]
    assert doc("d", "é").size == 2
    assert doc("d", format="markdown").format is Format.MARKDOWN
    given = (Span("page", 0, 1, "p", 1),)
    assert doc("d", provenance=given).provenance == given


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"identifier": " "}, "empty_document_id"),
        ({"content": " \n"}, "empty_content"),
        ({"encoding": "latin-1"}, "unsupported_encoding"),
        ({"digest": DIGEST.upper()}, "invalid_source_digest"),
        ({"layout": layout(end=99)}, "layout_out_of_range"),
        (
            {"layout": layout(digest="b" * 64), "digest": DIGEST},
            "layout_source_mismatch",
        ),
    ],
)
def test_a_sample_rejects_invalid_parts(kwargs, code):
    arguments = {"identifier": "d", "content": "hello"} | kwargs
    with pytest.raises(DocumentError) as raised:
        sample.Sample(**arguments)
    assert raised.value.details["code"] == code


def test_a_sample_accepts_utf8_spellings_and_a_matching_layout():
    assert doc("d", encoding="UTF8").encoding == "UTF8"
    built = doc("d", "hello", layout=layout(end=5, digest=DIGEST))
    assert built.layout.elements[0].end == 5
    assert doc("d", layout=layout(digest="b" * 64)).digest is None


def test_citations_need_both_a_uri_and_a_digest():
    assert doc("d").citations() == ()
    assert doc("d", uri="file:///x").citations() == ()
    assert doc("d", digest=DIGEST).citations() == ()
    built = doc("d", "ab", uri="file:///x", digest=DIGEST)
    citations = built.citations()
    assert len(citations) == 1
    assert citations[0].locator == "page 1"
    assert (citations[0].start, citations[0].end) == (0, 2)
    assert citations[0].page == 1
    unlabelled = doc(
        "d",
        "ab",
        uri="u",
        digest=DIGEST,
        provenance=(Span("section", 0, 2),),
    )
    assert unlabelled.citations()[0].locator == "section:0-2"


def test_profile_bounds_characters_and_tokens():
    built = doc("d", "one two three")
    profile = built.profile()
    assert (profile.character_count, profile.token_count) == (13, 3)
    assert profile.language == "en"
    assert not profile.truncated
    cut = built.profile(max_characters=7)
    assert (cut.character_count, cut.token_count, cut.truncated) == (7, 2, True)
    capped = built.profile(max_tokens=2)
    assert (capped.token_count, capped.truncated) == (2, True)
    exact = built.profile(max_characters=13, max_tokens=3)
    assert not exact.truncated
    with pytest.raises(errors.InputError, match="bounds must be positive"):
        built.profile(max_characters=0)
    with pytest.raises(errors.InputError, match="bounds must be positive"):
        built.profile(max_tokens=0)


def test_bucketize_names_label_types():
    assert sample.bucketize(None) == "unlabeled"
    assert sample.bucketize(1) == "int:1"
    assert sample.bucketize("a") == "str:a"
    assert sample.bucketize(True) == "bool:True"


def test_a_split_lists_sorted_document_ids():
    split = sample.Split((doc("b"), doc("a"), doc("a")), (doc("c"),), ())
    assert split.index() == {
        "train": ("a", "b"),
        "validation": ("c",),
        "test": (),
    }


def test_a_dataset_hash_ignores_order():
    first, second = doc("a", "x"), doc("b", "y")
    forward = dataset.Dataset((first, second))
    assert dataset.Dataset((second, first)).digest == forward.digest
    expected = hashlib.sha256(
        "\n".join(
            sorted([f"a\0{first.content_hash}", f"b\0{second.content_hash}"])
        ).encode()
    ).hexdigest()
    assert forward.digest == expected


def test_a_dataset_rejects_empty_and_foreign_samples():
    with pytest.raises(DocumentError, match="at least one sample"):
        dataset.Dataset(())
    with pytest.raises(errors.KindError, match="only Sample"):
        dataset.Dataset(("x",))


def test_duplicate_groups_are_sorted_and_need_two_members():
    data = dataset.Dataset(
        (doc("c", "x"), doc("a", "x"), doc("b", "y"), doc("d", "z"))
    )
    assert data.group() == (("a", "c"),)
    assert len(data.profile(max_tokens=1)) == 4


def test_split_keeps_each_document_in_one_partition():
    samples = tuple(doc(f"d{i % 5}", f"c{i}") for i in range(10))
    data = dataset.Dataset(samples)
    split = data.split(0.2, 0.2, seed=1)
    ids = split.index()
    assert (len(ids["train"]), len(ids["validation"]), len(ids["test"])) == (
        3,
        1,
        1,
    )
    assert len(split.train) == 6
    assert data.split(0.2, 0.2, seed=1) == split
    with pytest.raises(errors.InputError, match="test_fraction"):
        data.split(0.0, 1.0)


def test_curate_counts_formats_labels_layouts_and_traceability():
    data = dataset.Dataset(
        (
            doc(
                "a",
                "hello there",
                label="x",
                layout=layout(0.4, kind="table"),
                uri="u",
                digest=DIGEST,
            ),
            doc("b", "bonjour", format="markdown", layout=layout(0.8)),
            doc("c", "plain words"),
        )
    )
    report = data.curate()
    assert report.samples == 3
    assert report.documents == 3
    assert report.format_counts == {"markdown": 1, "text": 2}
    assert report.label_counts == {"str:x": 1, "unlabeled": 2}
    assert report.language_counts == {"en": 3}
    assert report.layout_coverage == pytest.approx(2 / 3)
    assert report.layout_element_count == 2
    assert report.tables == 1
    assert report.ocr_confidence_min == 0.4
    assert report.ocr_confidence_max == 0.8
    assert report.traceability == pytest.approx(1 / 3)
    assert report.citations == 1
    assert [f.identifier for f in report.risk_findings] == [
        "risk.mixed-label-state"
    ]
    with pytest.raises(errors.InputError, match="curation bounds"):
        data.curate(max_characters=0)
    with pytest.raises(errors.InputError, match="curation bounds"):
        data.curate(max_tokens=0)


def test_curate_blocks_sensitive_content_and_conflicting_duplicates():
    data = dataset.Dataset(
        (
            doc("a", "mail me at a@b.com", label=1),
            doc("b", "mail me at a@b.com", label=2),
        )
    )
    report = data.curate()
    names = [f.identifier for f in report.risk_findings]
    assert names == [
        "risk.document-sensitive-content",
        "risk.conflicting-duplicate-labels",
    ]
    assert report.risk_findings[0].evidence == {"match_counts": {"email": 2}}
    assert report.risk_findings[0].scope == {"document_ids": ["a", "b"]}
    assert report.risk_findings[1].evidence == {"labels": ["int:1", "int:2"]}
    assert all(f.blocking for f in report.risk_findings)
    assert report.sensitive_match_counts == {"email": 2}
    quality = [f.identifier for f in report.quality_findings]
    assert quality == ["quality.duplicate-document-content"]


def test_inspect_lists_quality_findings_in_order():
    data = dataset.Dataset(
        (
            doc("a", "1234 5678", layout=layout(), uri="u"),
            doc("b", "one two three"),
        )
    )
    findings = data.curate(max_tokens=2).quality_findings
    assert [f.identifier for f in findings] == [
        "quality.document-ocr-confidence-missing",
        "quality.profile-truncated",
        "quality.unknown-language",
        "quality.document-source-traceability-incomplete",
    ]
    assert findings[1].evidence == {"max_characters": 100_000, "max_tokens": 2}
    assert findings[1].scope == {"document_ids": ["b"]}
    assert findings[1].severity == data_severity("INFO")
    assert findings[2].evidence == {"count": 1}


def data_severity(name):
    return getattr(data.FindingSeverity, name)


def test_review_flaw_builds_an_open_finding():
    finding = review.flaw(
        "x", data.FindingSeverity.ERROR, {"a": 1}, {"b": 2}, "fix", True
    )
    assert finding.rule == "document-curation.v1"
    assert finding.owner == "data-curation"
    assert finding.confidence == 1.0
    assert finding.reviewer is None
    assert finding.disposition == data.Disposition.OPEN
    assert finding.blocking


def test_legibility_and_traceability_stay_quiet_when_complete():
    assert review.legibility([doc("a")], [layout(0.5)]) == []
    assert review.legibility([doc("a")], []) == []
    complete = doc("a", uri="u", digest=DIGEST)
    assert review.traceability([complete, doc("b")]) == []
    half = doc("c", digest=DIGEST)
    findings = review.traceability([half])
    assert findings[0].evidence == {"sample_count": 1}


def test_layering_summarises_providers_and_confidence():
    summary = review.layering([layout(0.3, provider="z"), layout(provider="a")])
    assert summary == {
        "layout_providers": ("a", "z"),
        "layout_element_count": 2,
        "tables": 0,
        "ocr_confidence_min": 0.3,
        "ocr_confidence_max": 0.3,
    }
    empty = review.layering([])
    assert empty["ocr_confidence_min"] is None
    assert empty["ocr_confidence_max"] is None


def test_sensitivity_counts_matches_per_document():
    data = dataset.Dataset(
        (doc("a", "call 555-123-4567 or 555-123-4568"), doc("b", "clean"))
    )
    counts, documents = data.sensitivity()
    assert dict(counts) == {"phone": 2}
    assert documents == ["a"]


def test_slurp_enforces_the_size_limit(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"abcd")
    assert reader.slurp(path, 4) == b"abcd"
    with pytest.raises(DocumentError) as raised:
        reader.slurp(path, 3)
    assert raised.value.details["actual_bytes"] == 4
    assert raised.value.details["code"] == "size_limit"


def test_a_reader_needs_a_positive_limit():
    with pytest.raises(errors.InputError, match="limit"):
        reader.Reader(limit=0)
    assert reader.Reader(limit=1).limit == 1


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a.md", Format.MARKDOWN),
        ("a.MARKDOWN", Format.MARKDOWN),
        ("a.pdf", Format.PDF),
        ("a.txt", Format.TEXT),
        ("a.text", Format.TEXT),
        ("a", Format.TEXT),
    ],
)
def test_sniff_reads_the_suffix(name, expected):
    assert reader.Reader().sniff(name) is expected


def test_sniff_prefers_an_explicit_format_and_rejects_unknown_suffixes():
    assert reader.Reader().sniff("a.pdf", "markdown") is Format.MARKDOWN
    with pytest.raises(DocumentError, match="unsupported document suffix"):
        reader.Reader().sniff("a.docx")


def test_load_reads_bytes_with_a_fingerprint_identity():
    built = reader.Reader().load(b"hello", label="x")
    digest = hashlib.sha256(b"hello").hexdigest()
    assert built.identifier == f"document-{digest[:16]}"
    assert built.uri == "memory://"
    assert built.digest == digest
    assert built.format is Format.TEXT
    assert built.label == "x"
    assert reader.Reader().load(b"hi", document_id="k").identifier == "k"


def test_load_reads_a_file_and_names_it(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Hi\nthere", encoding="utf-8")
    built = reader.Reader().load(path)
    assert built.identifier == "note.md"
    assert built.format is Format.MARKDOWN
    assert built.uri == path.resolve().as_uri()
    with pytest.raises(DocumentError, match="not found"):
        reader.Reader().load(tmp_path / "missing.txt")
    with pytest.raises(DocumentError, match="not found"):
        reader.Reader().load(tmp_path)


def test_load_enforces_the_byte_limit():
    with pytest.raises(DocumentError) as raised:
        reader.Reader(limit=3).load(b"abcd")
    assert raised.value.details == {
        "code": "size_limit",
        "actual_bytes": 4,
        "max_bytes": 3,
    }
    assert reader.Reader(limit=4).load(b"abcd").size == 4


def test_unicode_reports_the_malformed_range():
    with pytest.raises(DocumentError) as raised:
        reader.Reader().unicode(b"ab\xffcd", "d")
    assert raised.value.details == {
        "code": "invalid_encoding",
        "document_id": "d",
        "encoding": "utf-8",
        "start": 2,
        "end": 3,
    }


def test_extract_joins_pdf_pages_with_form_feeds():
    pdf = reader.Reader(pdf=lambda raw: ["one", "two"])
    built = pdf.load(b"%PDF", document_format="pdf")
    assert built.content == "one\ftwo"
    assert [s.page_number for s in built.provenance] == [1, 2]
    with pytest.raises(DependencyError, match="optional document parser"):
        reader.Reader().extract(b"%PDF")


def test_extract_maps_parser_failures():
    def missing(raw):
        raise ImportError(raw)

    def broken(raw):
        raise RuntimeError(raw)

    with pytest.raises(DependencyError) as raised:
        reader.Reader(pdf=missing).extract(b"x")
    assert raised.value.details["dependency"] == "pypdf"
    with pytest.raises(DocumentError, match="could not decode"):
        reader.Reader(pdf=broken).extract(b"x")
    with pytest.raises(DocumentError, match="no valid page text"):
        reader.Reader(pdf=lambda raw: []).extract(b"x")
    with pytest.raises(DocumentError, match="no valid page text"):
        reader.Reader(pdf=lambda raw: ["a", 1]).extract(b"x")


def test_overlay_runs_the_layout_callback():
    found = layout(end=2)
    calls = []

    def capture(raw, identifier, kind):
        calls.append((raw, identifier, kind))
        return found

    built = reader.Reader(reader=capture).load(b"hi", document_id="d")
    assert built.layout is found
    assert calls == [(b"hi", "d", Format.TEXT)]
    given = layout(end=1)
    kept = reader.Reader(reader=capture).load(b"hi", layout=given)
    assert kept.layout is given
    assert reader.Reader().overlay(b"x", "d", Format.TEXT) is None
    none = reader.Reader(reader=lambda *a: None)
    assert none.overlay(b"x", "d", Format.TEXT) is None


def test_overlay_maps_callback_failures():
    def missing(*arguments):
        raise ImportError

    def broken(*arguments):
        raise RuntimeError

    with pytest.raises(DependencyError, match="layout extraction"):
        reader.Reader(reader=missing).overlay(b"x", "d", Format.TEXT)
    with pytest.raises(DocumentError, match="could not decode"):
        reader.Reader(reader=broken).overlay(b"x", "d", Format.TEXT)
    with pytest.raises(DocumentError, match="invalid contract"):
        reader.Reader(reader=lambda *a: "x").overlay(b"x", "d", Format.TEXT)


def test_fingerprint_uses_the_file_name_or_a_digest(tmp_path):
    assert reader.Reader().fingerprint(b"x", tmp_path / "f.txt") == "f.txt"
    assert reader.Reader().fingerprint(b"x", None) == (
        f"document-{hashlib.sha256(b'x').hexdigest()[:16]}"
    )
