"""The document kind: text files read into rows of the shared stages.

A document source is either a directory of class folders (each folder's
name is the label of the files in it; files at the top are unlabelled) or a
table listing one document per row: its file under `path` (relative to the
table) or its text under `text`, its id under `document` when chunks of one
document are listed apart, the target, and any other columns, which pass
through as features. UTF-8 text and Markdown are read by the standard
library; a PDF needs the optional `pypdf` library and is refused with a
typed error without it.

Each document becomes one row: its text (`TEXT`, which the shared stages
encode as TF-IDF terms fitted inside the folds), its character, token and
page counts, the pass-through columns and the target. The row's split key
(`rows.KEY`) joins its document id with its near-duplicate cluster
(`likeness.Resemblance`), so the shared splitter keeps chunks of one
document and exact or near-duplicate content on one side of the test split
and of every fold (MM-10). The shared stages then search the model
families, cross-validate, hold out the locked test set, explain, package
and serve as they do for a table.
"""

from collections.abc import Sequence
import importlib
import io
import pathlib
import re

import pandas

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.modalities import curation
from scmle.data.modalities import document
from scmle.data.modalities.document import likeness
from scmle.data.readers import streaming
from scmle.pipeline.kinds import rows

TEXT = "text"
"""The column of a document's text, in a listing and in the rows."""

PATH = "path"
"""The column of a listing naming a document's file."""

IDENTITY = "document"
"""The column of a listing naming a document's id; chunks share one."""

MEASURES = ("characters", "tokens", "pages")
"""The count columns each row gets: characters, word tokens and pages."""

SUFFIXES = frozenset({".txt", ".text", ".md", ".markdown", ".pdf"})
"""File suffixes a class folder's documents may have."""

SECONDS = 2.5e-3
"""Upper bound on the seconds reading and clustering take per document.

Observed on 2,000 documents of 2 KB (one thread, 2026-11-19): reading,
counting, shingling and signing took 0.33 ms a document, and the cost grows
linearly with the text (0.17 ms per KB). The bound covers documents up to
about 15 KB at that rate; longer corpora are dominated by TF-IDF fitting,
which the trial budget governs.
"""


def paginate(raw: bytes) -> list[str]:
    """Reads the page texts of a PDF with the optional `pypdf` library.

    Args:
        raw: The PDF file.

    Returns:
        The text of each page, in page order.

    Raises:
        ImportError: If `pypdf` is not installed; the reader turns it into
            a `DependencyError`.
    """
    pdf = importlib.import_module("pypdf")
    return [
        page.extract_text() or ""
        for page in pdf.PdfReader(io.BytesIO(raw)).pages
    ]


def gather(root: pathlib.Path) -> list[tuple[pathlib.Path, str | None]]:
    """Lists the documents of a directory of class folders.

    Args:
        root: The directory.

    Returns:
        Each document file with its label: the name of its class folder, or
        None for a file at the top. Files are sorted by path; other
        suffixes and hidden files are skipped.

    Raises:
        DataError: If the directory holds no document.
    """
    found = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUFFIXES
        and not any(
            part.startswith(".") for part in path.relative_to(root).parts
        )
    )
    if not found:
        raise errors.DataError(
            f"no document ({', '.join(sorted(SUFFIXES))}) under {root}",
            {"uri": str(root)},
        )
    return [
        (path, path.parent.name if path.parent != root else None)
        for path in found
    ]


class Document(rows.Rows):
    """Text documents read into one row each.

    The featuriser and the decoder are the table's: the rows are a table
    whose text column the shared stages encode inside the folds.

    Attributes:
        name: "document".
        reader: The bounded document reader.
        kin: The near-duplicate clustering behind the split key.
    """

    name = "document"
    seconds = SECONDS
    texts = (TEXT,)
    evidence = "tests/unit/test_kind_document.py"

    def __init__(
        self,
        reader: document.Reader | None = None,
        kin: likeness.Resemblance | None = None,
    ) -> None:
        """Binds the reader and the clustering.

        Args:
            reader: The document reader; the default reads PDFs through
                `paginate`.
            kin: The near-duplicate clustering; the default thresholds.
        """
        self.reader = reader or document.Reader(pdf_reader=paginate)
        self.kin = kin or likeness.Resemblance()

    def load(
        self, spec: tasks.Config
    ) -> tuple[list[document.Sample], pandas.DataFrame]:
        """Reads the documents and their listing.

        Args:
            spec: The run's specification.

        Returns:
            The documents, in listing order, and the listing's other
            columns with the target (one row per document).

        Raises:
            DataError: If the source is missing, a listing has neither a
                `path` nor a `text` column, a file cannot be read, or a
                listing column clashes with a column the rows get.
        """
        source = pathlib.Path(spec.data.uri)
        if not source.exists():
            raise errors.DataError(
                f"document source not found: {source}", {"uri": str(source)}
            )
        if source.is_dir():
            listed = gather(source)
            samples = [
                self.reader.load(entry[0], str(entry[0].relative_to(source)))
                for entry in listed
            ]
            extra = pandas.DataFrame(
                {spec.data.target: [entry[1] for entry in listed]}
            )
            return samples, extra
        listing = rows.catalogue(source, IDENTITY)
        if PATH not in listing.columns and TEXT not in listing.columns:
            raise errors.DataError(
                f"a document listing needs a '{PATH}' or '{TEXT}' column",
                {
                    "uri": str(source),
                    "columns": [str(c) for c in listing.columns],
                },
            )
        reserved = {*MEASURES, rows.KEY} & set(map(str, listing.columns))
        if reserved:
            raise errors.DataError(
                "listing columns clash with the document rows: "
                f"{sorted(reserved)}",
                {"uri": str(source), "reserved": sorted(reserved)},
            )
        samples = []
        for position, record in enumerate(listing.to_dict(orient="records")):
            identity = record.get(IDENTITY)
            if identity is None or pandas.isna(identity):
                identity = None
            text = record.get(TEXT)
            if (
                PATH in record
                and isinstance(record[PATH], str)
                and record[PATH].strip()
            ):
                path = source.parent / record[PATH]
                samples.append(self.reader.load(path, identity or record[PATH]))
            elif isinstance(text, str):
                samples.append(
                    document.Sample(identity or f"row-{position}", text)
                )
            else:
                raise errors.DataError(
                    f"listing row {position} names no document file or text",
                    {"uri": str(source), "row": position},
                )
        extra = listing.drop(
            columns=[c for c in (PATH, TEXT, IDENTITY) if c in listing.columns]
        )
        return samples, extra

    def rows(self, samples: Sequence[document.Sample]) -> pandas.DataFrame:
        """Turns documents into the rows a model reads.

        Serving uses it on new documents: the rows carry the text and the
        counts a packaged model was trained on, and nothing else.

        Args:
            samples: The documents.

        Returns:
            One row per document: its text and its character, token and
            page counts.
        """
        return pandas.DataFrame(
            {
                TEXT: [sample.content for sample in samples],
                MEASURES[0]: [len(sample.content) for sample in samples],
                MEASURES[1]: [
                    len(re.findall(r"\w+", sample.content))
                    for sample in samples
                ],
                MEASURES[2]: [
                    sample.content.count("\f") + 1 for sample in samples
                ],
            }
        )

    def read(
        self, spec: tasks.Config
    ) -> tuple[
        pandas.DataFrame, tuple[streaming.Feed, streaming.Totals] | None
    ]:
        """Reads the documents into rows with their split key.

        Identical content listed twice with two labels is a contradiction
        no split can hide, so it is refused (MM-16): one document id that
        repeats is compared by its content, not collapsed.

        Args:
            spec: The run's specification.

        Returns:
            The rows (see `rows`) with the listing's other columns, the
            target and the split key, and no stream.

        Raises:
            DataError: If the source cannot be read (see `load`) or
                identical content carries conflicting labels.
        """
        samples, extra = self.load(spec)
        target = spec.data.target
        labels = (
            extra[target].tolist()
            if target in extra.columns
            else [None] * len(samples)
        )
        conflicts = curation.conflicting_label_groups(
            (sample.content_hash, sample.document_id, label)
            for sample, label in zip(samples, labels, strict=True)
            if label is not None and not pandas.isna(label)
        )
        if conflicts:
            raise errors.DataError(
                f"{len(conflicts)} document text(s) carry conflicting labels",
                {"conflicts": [dict(item) for item in conflicts[:10]]},
            )
        clusters = self.kin.cluster([sample.content for sample in samples])
        key = rows.unite(
            len(samples),
            (),
            [sample.document_id for sample in samples],
            clusters,
        )
        frame = pandas.concat(
            [self.rows(samples), extra.reset_index(drop=True)], axis=1
        )
        return frame.assign(**{rows.KEY: key}), None

    def notice(self, frame: pandas.DataFrame) -> list[str]:
        """The cohorts kept together and the sensitive texts.

        Args:
            frame: The rows, with `rows.KEY` when they have one.

        Returns:
            The cohort finding (`rows.Rows.notice`), and one counting the
            documents holding sensitive values when there are any.
        """
        found = super().notice(frame)
        texts = frame[TEXT].astype(str).tolist() if TEXT in frame else []
        holding = sum(bool(document.pii(text)) for text in texts)
        if holding:
            found.append(
                f"{holding} documents hold sensitive values (email, phone or "
                "ssn patterns); govern or redact them before release"
            )
        return found

    def limits(self) -> tuple[str, ...]:
        """The boundaries the document capability states.

        Returns:
            Plain statements of what the kind does not cover.
        """
        return (
            "UTF-8 text, Markdown and PDF (with pypdf) only; no OCR",
            (
                "bag-of-words (TF-IDF) text features, not semantic "
                "embeddings unless data.text_encoding is 'embedding'"
            ),
            "a served model reads rows built by `Document.rows`",
        )


DOCUMENT = Document()
"""The document kind."""


__all__ = ["DOCUMENT", "Document", "gather", "paginate"]
