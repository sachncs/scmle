"""Unit tests of detector code hashing, imported by its full module name."""

import ast
import hashlib

import pytest

from scmle.assurance.detectors import hashing
from scmle.contracts import errors

DIGEST = "a" * 64


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_documentation_finds_docstring_lines_of_every_owner():
    source = (
        '"""Module."""\n'
        "def f():\n"
        '    """Two\n'
        '    lines."""\n'
        "    return 1\n"
        "class C:\n"
        '    """Class."""\n'
        "x = 'not a docstring'\n"
    )
    assert hashing.documentation(ast.parse(source)) == {1, 3, 4, 7}


def test_a_docstring_sharing_its_line_is_kept():
    source = 'def f(): """doc"""; return 1\n'
    assert hashing.documentation(ast.parse(source)) == set()


def test_a_body_starting_with_code_has_no_docstring():
    assert hashing.documentation(ast.parse("def f():\n    return 'a'\n")) == (
        set()
    )
    assert hashing.documentation(ast.parse("def f():\n    3\n")) == set()


def test_hash_source_drops_comments_docstrings_and_blanks():
    plain = "def f():\n    return 1"
    noisy = (
        "def f():  # a comment\n"
        '    """Docstring."""\n'
        "\n"
        "    # whole-line comment\n"
        "    return 1   \n"
    )
    assert hashing.hash_source(noisy) == sha(plain)
    assert hashing.hash_source(plain) == sha(plain)


def test_hash_source_dedents_and_sees_any_code_change():
    assert hashing.hash_source("    x = 1\n") == sha("x = 1")
    assert hashing.hash_source("x = 1\n") != hashing.hash_source("x = 2\n")
    assert hashing.hash_source("x = 1\n") != hashing.hash_source("x  = 1\n")


def test_hash_source_refuses_invalid_python():
    with pytest.raises(SyntaxError):
        hashing.hash_source("def (:\n")


def sample():
    return 1


def other():
    return 2


def test_code_hash_combines_the_source_hashes_in_order():
    first = hashing.hash_source("def sample():\n    return 1\n")
    second = hashing.hash_source("def other():\n    return 2\n")
    assert hashing.code_hash(sample) == sha(first)
    assert hashing.code_hash(sample, other) == sha(f"{first}\n{second}")
    assert hashing.code_hash(other, sample) != hashing.code_hash(sample, other)


def test_code_hash_needs_an_object():
    with pytest.raises(errors.InputError, match="needs at least one object"):
        hashing.code_hash()


def test_corpus_hash_ignores_order():
    a, b = "a" * 64, "b" * 64
    assert hashing.corpus_hash([b, a]) == sha(f"{a}\n{b}")
    assert hashing.corpus_hash([a, b]) == hashing.corpus_hash([b, a])


def test_corpus_hash_needs_digests():
    with pytest.raises(errors.InputError, match="needs at least one part"):
        hashing.corpus_hash([])
    with pytest.raises(errors.InputError, match="corpus part must be"):
        hashing.corpus_hash(["A" * 64])


@pytest.mark.parametrize("bad", ["a" * 63, "a" * 65, "g" * 64, "A" * 64, 5])
def test_require_digest_refuses_anything_but_lowercase_sha256(bad):
    with pytest.raises(
        errors.InputError, match="x must be a 64-character lowercase SHA-256"
    ):
        hashing.require_digest(bad, "x")


def test_require_digest_returns_a_valid_digest():
    assert hashing.require_digest(DIGEST, "x") is DIGEST
    assert hashing.DOCSTRING_OWNERS[0] is ast.Module
