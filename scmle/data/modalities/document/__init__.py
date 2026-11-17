"""Document data contracts and dependency-light ingestion.

This module deliberately keeps parsing small and explicit. UTF-8 text
and Markdown are supported by the standard library. PDF support is an
optional injected reader so the core package never imports a document parser
implicitly or silently changes its dependency surface.
"""

from scmle.data.modalities.document.dataset import Dataset
from scmle.data.modalities.document.elements import Citation
from scmle.data.modalities.document.elements import DependencyError
from scmle.data.modalities.document.elements import DocumentError
from scmle.data.modalities.document.elements import Element
from scmle.data.modalities.document.elements import Format
from scmle.data.modalities.document.elements import Layout
from scmle.data.modalities.document.elements import pii
from scmle.data.modalities.document.elements import Profile
from scmle.data.modalities.document.elements import SENSITIVE_PATTERNS
from scmle.data.modalities.document.elements import Span
from scmle.data.modalities.document.reader import DocumentSource
from scmle.data.modalities.document.reader import LayoutReader
from scmle.data.modalities.document.reader import PdfPageReader
from scmle.data.modalities.document.reader import Reader
from scmle.data.modalities.document.report import Curation
from scmle.data.modalities.document.review import flaw
from scmle.data.modalities.document.sample import bucketize
from scmle.data.modalities.document.sample import Sample
from scmle.data.modalities.document.sample import Split
from scmle.data.modalities.document.text import ASCII_LETTER_MIN_SHARE
from scmle.data.modalities.document.text import detect
from scmle.data.modalities.document.text import DEVANAGARI_MIN_SHARE
from scmle.data.modalities.document.text import segment

__all__ = [
    "ASCII_LETTER_MIN_SHARE",
    "DEVANAGARI_MIN_SHARE",
    "SENSITIVE_PATTERNS",
    "Citation",
    "Curation",
    "Dataset",
    "DependencyError",
    "DocumentError",
    "DocumentSource",
    "Element",
    "Format",
    "Layout",
    "LayoutReader",
    "PdfPageReader",
    "Profile",
    "Reader",
    "Sample",
    "Span",
    "Split",
    "bucketize",
    "detect",
    "flaw",
    "pii",
    "segment",
]
