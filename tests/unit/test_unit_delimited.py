"""Behavioural tests for delimited text reading and table interchange."""

import bz2
import codecs
import gzip
import lzma
import zipfile

import pandas
import pyarrow
import pytest

from scmle.contracts import errors
from scmle.data.readers import delimited
from scmle.data.readers import interchange


def test_inflate_decompresses_by_suffix(tmp_path):
    payload = b"a,b\n1,2\n"
    for suffix, pack in (
        (".gz", gzip.compress),
        (".bz2", bz2.compress),
        (".xz", lzma.compress),
        (".csv", bytes),
    ):
        path = tmp_path / f"t{suffix}"
        path.write_bytes(pack(payload))
        assert delimited.inflate(path) == payload
    upper = tmp_path / "t.GZ"
    upper.write_bytes(gzip.compress(payload))
    assert delimited.inflate(upper) == payload


def test_inflate_reads_a_single_member_zip(tmp_path):
    path = tmp_path / "t.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("folder/", "")
        archive.writestr("folder/t.csv", "a\n1\n")
    assert delimited.inflate(path) == b"a\n1\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a.csv", "a\n")
        archive.writestr("b.csv", "b\n")
    with pytest.raises(errors.InputError, match="one file, found 2"):
        delimited.inflate(path)


def test_decode_honours_marks_and_stated_encodings():
    assert delimited.decode(codecs.BOM_UTF8 + b"a") == "a"
    assert delimited.decode("ab".encode("utf-16")) == "ab"
    assert delimited.decode("ab".encode("utf-32")) == "ab"
    assert (
        delimited.decode(codecs.BOM_UTF16_BE + "a".encode("utf-16-be")) == "a"
    )
    assert delimited.decode("é".encode("latin-1"), "latin-1") == "é"
    assert delimited.decode("café".encode()) == "café"
    with pytest.raises(errors.InputError, match="cannot decode as ascii"):
        delimited.decode("é".encode(), "ascii")
    with pytest.raises(errors.InputError, match="cannot decode as nope"):
        delimited.decode(b"a", "nope")


def test_decode_falls_back_to_cp1252_only_when_plausible():
    assert delimited.decode("café, Ångström".encode("cp1252")) == (
        "café, Ångström"
    )
    assert delimited.decode("price €5".encode("cp1252")) == "price €5"
    with pytest.raises(errors.InputError, match="NUL bytes"):
        delimited.decode(b"a\x00b")
    with pytest.raises(errors.InputError, match="neither UTF-8"):
        delimited.decode("中文".encode("gbk"))
    with pytest.raises(errors.InputError, match="cannot decode as cp1252"):
        delimited.decode(b"\x81\xff")


def test_plausible_limits_runs_of_non_ascii():
    assert delimited.plausible("ção")
    assert delimited.plausible("abc")
    assert delimited.plausible("éé a éée")
    assert not delimited.plausible("éééé")
    assert not delimited.plausible("ж")
    assert delimited.plausible("ÿ")
    assert not delimited.plausible("÷")
    assert delimited.RUN == 3


def test_records_split_on_the_delimiter():
    assert list(delimited.records('a;"b;c"\n1;2\n', ";")) == [
        ["a", "b;c"],
        ["1", "2"],
    ]


@pytest.mark.parametrize(
    ("text", "delimiter"),
    [
        ("a,b\n1,2\n", ","),
        ("a;b\n1;2\n", ";"),
        ("a|b\n1|2\n", "|"),
        ("a\tb\n1\t2\n", "\t"),
        ("a\n1\n", ","),
        ("", ","),
    ],
)
def test_sniff_finds_the_one_delimiter(text, delimiter):
    assert delimited.sniff(text) == delimiter


def test_sniff_refuses_an_ambiguous_header():
    with pytest.raises(errors.InputError, match=r"\[',', ';'\]"):
        delimited.sniff("a,b;c\n")


def test_check_refuses_duplicates_and_ragged_rows():
    delimited.check("a,b\n1,2\n\n3,4\n", ",")
    delimited.check("", ",")
    delimited.check(",\n1,2\n", ",")
    with pytest.raises(
        errors.InputError, match=r"duplicate column names: \['a'\]"
    ):
        delimited.check("a,a,b\n1,2,3\n", ",")
    with pytest.raises(
        errors.InputError, match="line 3 has 3 fields, the header 2"
    ):
        delimited.check("a,b\n1,2\n1,2,3\n", ",")
    with pytest.raises(errors.InputError, match="line 2 has 1 fields"):
        delimited.check("a,b\n1\n", ",")


def test_parse_reads_decimal_commas_with_semicolons():
    frame = delimited.parse("a;b\n1,5;x\n2,25;y\n", ";")
    assert frame["a"].tolist() == [1.5, 2.25]
    frame = delimited.parse("a,b\n1.5,x\n", ",")
    assert frame["a"].tolist() == [1.5]


def test_parse_keeps_padded_identifiers_as_text():
    frame = delimited.parse("id,v\n02134,1\n2,2\n", ",")
    assert frame["id"].tolist() == ["02134", "2"]
    assert frame["v"].tolist() == [1, 2]
    signed = delimited.parse("id\n-01\n3\n", ",")
    assert signed["id"].tolist() == ["-01", "3"]
    zero = delimited.parse("id\n0\n3\n", ",")
    assert zero["id"].tolist() == [0, 3]


def test_parse_keeps_large_integers_exact():
    big = 2**53 + 1
    gap = delimited.parse(f"a,b\n{big},1\n,2\n", ",")
    assert str(gap["a"].dtype) == "Int64"
    assert gap["a"].iloc[0] == big
    assert gap["a"].isna().iloc[1]
    mixed = delimited.parse(f"a,b\n{big},1\n1.5,2\n", ",")
    assert mixed["a"].tolist() == [big, 1.5]
    assert type(mixed["a"].iloc[0]) is int
    small = delimited.parse("a,b\n1,1\n,2\n", ",")
    assert str(small["a"].dtype) == "float64"
    exact = delimited.parse(f"a,b\n{2**53},1\n,2\n", ",")
    assert str(exact["a"].dtype) == "float64"


def test_parse_refuses_ambiguous_thousands():
    with pytest.raises(errors.InputError, match="thousands separator"):
        delimited.parse("a;b\n1,500;2\n", ";")
    assert delimited.parse("a;b\n1,500;2\n2,5;3\n", ";")["a"].tolist() == [
        1.5,
        2.5,
    ]


def test_parse_leaves_text_and_booleans_alone():
    frame = delimited.parse("a,b\nTrue,x\nFalse,y\n", ",")
    assert frame["a"].tolist() == [True, False]


def test_settle_returns_the_exact_form():
    column = pandas.Series([1.0, None], name="c")
    raw = pandas.Series(["1", None])
    assert delimited.settle(column, raw) is column
    integers = pandas.Series([1, 2], name="c")
    assert delimited.settle(integers, pandas.Series(["1", "2"])) is integers
    padded = pandas.Series(["007", "1"])
    assert delimited.settle(pandas.Series([7, 1]), padded) is padded
    with pytest.raises(errors.InputError, match="column 'c' reads '1,000'"):
        delimited.settle(column, pandas.Series(["1,000", None]), ",")
    huge = pandas.Series([float(2**64), None], name="h")
    with pytest.raises(errors.InputError, match="beyond int64"):
        delimited.settle(huge, pandas.Series([str(2**64), None]))
    decimals = delimited.settle(
        pandas.Series([float(2**53 + 1), 2.5]),
        pandas.Series([str(2**53 + 1), "2,5"]),
        ",",
    )
    assert decimals.tolist() == [2**53 + 1, 2.5]


def test_tablelike_detects_in_memory_tables():
    assert interchange.tablelike(pandas.DataFrame())
    assert interchange.tablelike(pyarrow.table({"a": [1]}))

    class Collectable:
        def collect(self):
            return pandas.DataFrame({"a": [1]})

    assert interchange.tablelike(Collectable())
    assert not interchange.tablelike("data.csv")
    assert not interchange.tablelike([1, 2])
    assert interchange.ARROW_STREAM == "__arrow_c_stream__"


def test_to_pandas_converts_each_supported_form():
    frame = pandas.DataFrame({"a": [1, 2]})
    copied = interchange.frame(frame)
    assert copied.equals(frame)
    assert copied is not frame
    assert interchange.frame(pyarrow.table({"a": [1, 2]})).equals(frame)

    class Streamed:
        def __arrow_c_stream__(self, requested_schema=None):
            return pyarrow.table({"a": [1, 2]}).__arrow_c_stream__(
                requested_schema
            )

    assert interchange.frame(Streamed()).equals(frame)

    class Convertible:
        def to_pandas(self):
            return frame

    assert interchange.frame(Convertible()) is frame

    class Lazy:
        def collect(self):
            return pyarrow.table({"a": [1, 2]})

    assert interchange.frame(Lazy()).equals(frame)


def test_to_pandas_refuses_paths_and_other_objects(tmp_path):
    with pytest.raises(errors.DataError, match="dataset loader"):
        interchange.frame("data.csv")
    with pytest.raises(errors.DataError, match="dataset loader"):
        interchange.frame(tmp_path)
    with pytest.raises(errors.DataError, match="unsupported table type list"):
        interchange.frame([1])
