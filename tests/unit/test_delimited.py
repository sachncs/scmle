"""Awkward delimited files end correct or refused, never silently wrong."""

import gzip

import pandas
import pytest

from scmle.contracts import errors
from scmle.data.readers import tabular

BIG = 2**53 + 1


def write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_bytes(
        payload if isinstance(payload, bytes) else payload.encode()
    )
    return path


@pytest.mark.parametrize(
    "body", ["a;b;c\n1;2;3\n4;5;6\n", "a|b|c\n1|2|3\n4|5|6\n"]
)
def test_other_delimiters_are_detected(tmp_path, body):
    frame = tabular.read_table(write(tmp_path, "d.csv", body))
    assert list(frame.columns) == ["a", "b", "c"]
    assert frame.to_numpy().tolist() == [[1, 2, 3], [4, 5, 6]]


def test_semicolon_file_reads_decimal_commas(tmp_path):
    frame = tabular.read_table(write(tmp_path, "d.csv", "a;b\n1,5;2\n3,5;4\n"))
    assert frame["a"].tolist() == [1.5, 3.5]
    assert frame["b"].tolist() == [2, 4]


def test_semicolon_file_with_thousands_like_commas_is_refused(tmp_path):
    path = write(tmp_path, "d.csv", "a;b\n1,500;2\n3,250;4\n")
    with pytest.raises(errors.InputError, match="decimal"):
        tabular.read_table(path)


def test_ambiguous_delimiter_is_refused(tmp_path):
    path = write(tmp_path, "d.csv", "a,b;c\n1,2;3\n")
    with pytest.raises(errors.InputError, match="delimiter"):
        tabular.read_table(path)


def test_stated_delimiter_settles_ambiguity(tmp_path):
    path = write(tmp_path, "d.csv", "a,b;c\n1,2;3\n")
    loader = tabular.Loader((tabular.Csv(delimiter=";"),))
    assert list(loader.load(path).columns) == ["a,b", "c"]


@pytest.mark.parametrize(
    ("name", "body"), [("d.csv", "a,a,b\n1,2,3\n"), ("d.tsv", "a\ta\n1\t2\n")]
)
def test_duplicate_headers_are_refused(tmp_path, name, body):
    with pytest.raises(errors.InputError, match="duplicate"):
        tabular.read_table(write(tmp_path, name, body))


def test_long_integer_with_gap_stays_exact(tmp_path):
    frame = tabular.read_table(write(tmp_path, "d.csv", f"k,v\n1,{BIG}\n2,\n"))
    assert int(frame["v"].iloc[0]) == BIG
    assert pandas.isna(frame["v"].iloc[1])


def test_long_integer_beside_float_stays_exact(tmp_path):
    path = write(tmp_path, "d.csv", f"k,v\n1,{BIG}\n2,1.5\n3,\n")
    cells = tabular.read_table(path)["v"].tolist()
    assert type(cells[0]) is int and cells[0] == BIG
    assert type(cells[1]) is float and cells[1] == 1.5
    assert pandas.isna(cells[2])


def test_long_integer_beside_decimal_comma_stays_exact(tmp_path):
    path = write(tmp_path, "d.csv", f"k;v\n1;{BIG}\n2;1,5\n")
    assert tabular.read_table(path)["v"].tolist() == [BIG, 1.5]


def test_ordinary_integers_with_gap_stay_float(tmp_path):
    frame = tabular.read_table(write(tmp_path, "d.csv", "k,v\n1,1\n2,\n"))
    assert frame["v"].dtype == "float64"


def test_zero_padded_identifiers_stay_text(tmp_path):
    frame = tabular.read_table(write(tmp_path, "d.csv", "zip\n02134\n10001\n"))
    assert frame["zip"].tolist() == ["02134", "10001"]


def test_plain_zero_stays_numeric(tmp_path):
    frame = tabular.read_table(write(tmp_path, "d.csv", "v,w\n0,0.5\n1,1\n"))
    assert frame["v"].tolist() == [0, 1]
    assert frame["w"].tolist() == [0.5, 1.0]


@pytest.mark.parametrize(
    "body", ["a,b,c\n1,2,3\n4,5", "a,b,c\n1,2,3\n4,5\n6,7,8\n"]
)
def test_short_row_is_refused(tmp_path, body):
    with pytest.raises(errors.InputError, match="fields"):
        tabular.read_table(write(tmp_path, "d.csv", body))


def test_short_row_in_tsv_is_refused(tmp_path):
    with pytest.raises(errors.InputError, match="fields"):
        tabular.read_table(write(tmp_path, "d.tsv", "a\tb\n1\n"))


@pytest.mark.parametrize("encoding", ["latin-1", "utf-16", "utf-16-be"])
def test_other_encodings_are_decoded(tmp_path, encoding):
    text = "id,name\n1,café\n"
    payload = text.encode(encoding)
    if encoding == "utf-16-be":
        payload = b"\xfe\xff" + payload
    frame = tabular.read_table(write(tmp_path, "d.csv", payload))
    assert frame["name"].tolist() == ["café"]


def test_stated_encoding_is_used(tmp_path):
    path = write(tmp_path, "d.csv", "id,name\n1,Ωμέγα\n".encode("cp1253"))
    loader = tabular.Loader((tabular.Csv(encoding="cp1253"),))
    assert loader.load(path)["name"].tolist() == ["Ωμέγα"]


def test_undetectable_legacy_encoding_is_refused(tmp_path):
    path = write(
        tmp_path, "d.csv", "id,name\n1,日本語テキスト\n".encode("shift_jis")
    )
    with pytest.raises(errors.InputError, match="encoding"):
        tabular.read_table(path)


def test_utf16_without_mark_is_refused(tmp_path):
    path = write(tmp_path, "d.csv", "id,name\n1,a\n".encode("utf-16-le"))
    with pytest.raises(errors.InputError, match="encoding"):
        tabular.read_table(path)


def test_compressed_latin1_semicolon_file(tmp_path):
    payload = gzip.compress("a;b\n1;é\n".encode("latin-1"), mtime=0)
    frame = tabular.read_table(write(tmp_path, "d.csv.gz", payload))
    assert frame.to_numpy().tolist() == [[1, "é"]]


def test_truncated_gzip_is_refused(tmp_path):
    whole = gzip.compress(b"a,b\n" + b"1,2\n" * 5000, mtime=0)
    with pytest.raises(errors.DataError):
        tabular.read_table(
            write(tmp_path, "d.csv.gz", whole[: len(whole) // 2])
        )


def test_quoted_newlines_and_blank_lines_keep_working(tmp_path):
    body = 'id,note\n1,"line one\nline two"\n\n2,"a, ""b"""\n'
    frame = tabular.read_table(write(tmp_path, "d.csv", body))
    assert frame["note"].tolist() == ["line one\nline two", 'a, "b"']


def test_carriage_return_only_lines(tmp_path):
    frame = tabular.read_table(write(tmp_path, "d.csv", "a,b\r1,2\r3,4\r"))
    assert frame.to_numpy().tolist() == [[1, 2], [3, 4]]
