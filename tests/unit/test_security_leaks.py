"""The artifact scanner finds planted rows, rare categories and secrets in
every form and container a run writes, and nothing else (C9)."""

import bz2
import functools
import gzip
import hashlib
import hmac
import html
import io
import json
import sqlite3
import struct
import zipfile

import joblib
import numpy
import pandas
import pytest

from scmle import envelope
from scmle.contracts import errors
from scmle.security import leaks

NAME = "Quorvelle Ambrintide"
IDENT = "PLANT-ID-0042"
RARE = "Xylophonica"
SECRET = "sk-planted-0123456789abcdef"


def table():
    return pandas.DataFrame(
        {
            "name": [NAME, "Bexley Tarnwhistle", "Corvane Ulstrade"] * 2,
            "ident": [IDENT, "PLANT-ID-0043", "PLANT-ID-0044"] * 2,
            "colour": ["green"] * 5 + [RARE],
            "a": [0.1234567890123456, 1.0, 2.5, 3.0, 4.0, 5.0],
            "b": [0.9876543210987654, 1.0, 2.5, 3.0, 4.0, 5.0],
        }
    )


DECLARED = {"colour": ["green", RARE]}


def test_probes_split_rows_rare_levels_cells_and_secrets():
    found = leaks.probes(table(), DECLARED, [SECRET], floor=5)
    kinds = sorted((p.kind, p.column, p.text) for p in found)
    assert kinds == [
        ("category", "colour", RARE),
        ("cell", "a", "0.1234567890123456"),
        ("cell", "b", "0.9876543210987654"),
        ("row", "ident", "PLANT-ID-0042"),
        ("row", "ident", "PLANT-ID-0043"),
        ("row", "ident", "PLANT-ID-0044"),
        ("row", "name", "Bexley Tarnwhistle"),
        ("row", "name", "Corvane Ulstrade"),
        ("row", "name", NAME),
        ("secret", "", SECRET),
    ]


def test_a_common_level_is_a_row_unless_declared():
    frame = pandas.DataFrame({"colour": ["green"] * 6})
    assert leaks.probes(frame, {"colour": ["green"]}) == []
    [probe] = leaks.probes(frame, {"colour": []})
    assert (probe.kind, probe.text) == ("row", "green")
    [probe] = leaks.probes(frame, {})
    assert (probe.kind, probe.text) == ("row", "green")


def test_a_probe_has_every_text_form_and_never_shows_its_value():
    probe = leaks.Probe('Ann "O" <B>', "row", "name")
    raw = b'Ann "O" <B>'
    assert probe.forms() == {
        "utf-8": raw,
        "json": b'Ann \\"O\\" <B>',
        "html": html.escape('Ann "O" <B>').encode(),
        "utf-16": 'Ann "O" <B>'.encode("utf-16-le"),
        "utf-32": 'Ann "O" <B>'.encode("utf-32-le"),
        "sha256": hashlib.sha256(raw).hexdigest().encode(),
        "sha1": hashlib.sha1(raw).hexdigest().encode(),
        "md5": hashlib.md5(raw).hexdigest().encode(),
    }
    salt = b"s" * leaks.SALT
    assert (
        probe.label(salt)
        == (hmac.new(salt, raw, hashlib.sha256).hexdigest()[:12])
    )
    assert probe.label(salt) != hashlib.sha256(raw).hexdigest()[:12]
    assert probe.label(b"t" * leaks.SALT) != probe.label(salt)
    plain = leaks.Probe("plainvalue", "row")
    assert list(plain.forms()) == [
        "utf-8",
        "utf-16",
        "utf-32",
        "sha256",
        "sha1",
        "md5",
    ]


def test_a_number_is_searched_as_its_repr_and_its_bytes():
    probe = leaks.Probe(
        "0.1234567890123456", "cell", "a", 0, 0.1234567890123456
    )
    assert probe.forms() == {
        "repr": b"0.1234567890123456",
        "float64": struct.pack("<d", 0.1234567890123456),
    }


def test_an_integer_is_searched_as_its_text_and_its_bytes():
    probe = leaks.Probe("1234567890123", "cell", "account", 0, 1234567890123)
    assert probe.forms() == {
        "decimal": b"1234567890123",
        "int64": struct.pack("<q", 1234567890123),
        "float64": struct.pack("<d", 1234567890123.0),
    }
    wide = leaks.Probe(str(1 << 64), "cell", "account", 0, 1 << 64)
    assert list(wide.forms()) == ["decimal", "float64"]


def test_long_integer_identifiers_are_cells_and_are_found(tmp_path):
    accounts = [4000000000000 + 7919 * n for n in range(6)]
    frame = pandas.DataFrame(
        {"account": numpy.array(accounts, dtype="int64"), "age": range(6)}
    )
    found = leaks.probes(frame, {})
    assert sorted((p.kind, p.column, p.row, p.number) for p in found) == [
        ("cell", "account", row, number) for row, number in enumerate(accounts)
    ]
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "ids.json").write_text(json.dumps(accounts), encoding="utf-8")
    report = leaks.scan(folder, found)
    assert sorted((f.path, f.kind, f.row, f.form) for f in report.findings) == [
        ("ids.json", "cell", row, "decimal") for row in range(6)
    ]
    assert report.probes["cell"] == 6
    assert report.unsearchable["cell"] == 0


def test_labels_are_stable_within_a_scan_and_unrelated_across_scans(tmp_path):
    folder = written(tmp_path / "run")
    chosen = leaks.probes(table(), DECLARED, [SECRET])
    first = leaks.scan(folder, chosen)
    second = leaks.scan(folder, chosen)
    secrets = {f.label for f in first.findings if f.kind == "secret"}
    assert len(secrets) == 1
    assert secrets.isdisjoint(f.label for f in second.findings)
    plain = hashlib.sha256(SECRET.encode()).hexdigest()[:12]
    assert plain not in json.dumps(first.to_dict())


@pytest.mark.parametrize(
    ("number", "expected"),
    [
        (0.1234567890123456, True),
        (-1.2887899296505332e-05, True),
        (42.0, False),
        (0.125, False),
        (float("nan"), False),
        (float("inf"), False),
    ],
)
def test_only_precise_numbers_are_cells(number, expected):
    assert leaks.precise(number) is expected


def written(folder):
    """One planted value per container and form, and clean files."""
    folder.mkdir()
    (folder / "profile.json").write_text(
        json.dumps({"top": {NAME: 1}}), encoding="utf-8"
    )
    (folder / "clean.json").write_text(
        json.dumps({"colour": "green"}), encoding="utf-8"
    )
    joblib.dump({"categories": [RARE]}, folder / "model.joblib", compress=3)
    payload = io.BytesIO()
    joblib.dump({"id": IDENT}, payload, compress=3)
    envelope.seal(
        folder / "stage.product", {"product": "x"}, payload.getvalue(), "k" * 32
    )
    pandas.DataFrame({"who": [NAME]}).to_parquet(folder / "board.parquet")
    connection = sqlite3.connect(folder / "store.sqlite")
    connection.execute("create table t (v text)")
    connection.execute("insert into t values (?)", (IDENT,))
    connection.commit()
    connection.close()
    with zipfile.ZipFile(folder / "bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("inner.txt", SECRET)
    numpy.save(folder / "names.npy", numpy.array([NAME]))
    (folder / "hashed.txt").write_text(
        hashlib.sha256(IDENT.encode()).hexdigest(), encoding="utf-8"
    )
    (folder / "page.html.gz").write_bytes(gzip.compress(NAME.encode()))
    (folder / "log.bz2").write_bytes(bz2.compress(SECRET.encode()))
    (folder / "rows.json").write_text(
        json.dumps({"x": [0.1234567890123456, 0.9876543210987654]}),
        encoding="utf-8",
    )
    (folder / "minimum.json").write_text(
        json.dumps({"min": 0.1234567890123456}), encoding="utf-8"
    )
    return folder


def test_every_container_and_form_is_searched(tmp_path):
    folder = written(tmp_path / "run")
    report = leaks.scan(folder, leaks.probes(table(), DECLARED, [SECRET]))
    found = sorted((f.path, f.kind, f.column, f.form) for f in report.leaks())
    assert found == [
        ("board.parquet", "row", "name", "utf-8"),
        ("bundle.zip", "secret", "", "utf-8"),
        ("hashed.txt", "row", "ident", "sha256"),
        ("log.bz2", "secret", "", "utf-8"),
        ("model.joblib", "category", "colour", "utf-8"),
        ("names.npy", "row", "name", "utf-32"),
        ("page.html.gz", "row", "name", "utf-8"),
        ("profile.json", "row", "name", "utf-8"),
        ("rows.json", "row", "a,b", "cells"),
        ("stage.product", "row", "ident", "utf-8"),
        ("store.sqlite", "row", "ident", "utf-8"),
    ]
    assert report.counts() == {"row": 8, "category": 1, "secret": 2}
    assert report.files == 13
    summary = report.to_dict()
    assert summary["cells"] == 3
    assert summary["leaks"] == {"row": 8, "category": 1, "secret": 2}
    assert NAME not in json.dumps(summary)
    assert IDENT not in json.dumps(summary)


def test_a_lone_cell_is_reported_but_not_a_leak(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "minimum.json").write_text(
        json.dumps({"min": 0.1234567890123456}), encoding="utf-8"
    )
    report = leaks.scan(folder, leaks.probes(table(), DECLARED))
    assert [(f.kind, f.column) for f in report.findings] == [("cell", "a")]
    assert report.leaks() == []


def test_a_short_value_is_counted_unsearchable_not_dropped(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "a.txt").write_text("Bo", encoding="utf-8")
    frame = pandas.DataFrame({"name": ["Bo", "Alexandria"]})
    report = leaks.scan(folder, leaks.probes(frame, {}))
    assert report.probes == {"row": 2, "category": 0, "secret": 0, "cell": 0}
    assert report.unsearchable == {
        "row": 1,
        "category": 0,
        "secret": 0,
        "cell": 0,
    }
    assert report.leaks() == []


def test_a_damaged_container_is_still_searched_as_bytes(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "broken.zlib").write_bytes(b"\x78\x9c" + NAME.encode())
    (folder / "broken.zip").write_bytes(b"PK\x03\x04" + NAME.encode())
    (folder / "broken.parquet").write_bytes(b"PAR1" + NAME.encode())
    (folder / "broken.sqlite").write_bytes(b"SQLite format 3\x00" + b"x")
    report = leaks.scan(folder, leaks.probes(table(), DECLARED))
    assert sorted(f.path for f in report.leaks()) == [
        "broken.parquet",
        "broken.zip",
        "broken.zlib",
    ]


def nested(data, times):
    """`data` gzip-compressed `times` times over."""
    return functools.reduce(
        lambda inner, layer: gzip.compress(inner), range(times), data
    )


def test_nesting_stops_at_the_depth_limit(tmp_path):
    (tmp_path / "deep.gz").write_bytes(nested(NAME.encode(), leaks.DEPTH + 1))
    report = leaks.scan(tmp_path / "deep.gz", leaks.probes(table(), DECLARED))
    assert report.leaks() == []
    (tmp_path / "shallow.gz").write_bytes(nested(NAME.encode(), leaks.DEPTH))
    report = leaks.scan(
        tmp_path / "shallow.gz", leaks.probes(table(), DECLARED)
    )
    assert [f.path for f in report.leaks()] == ["shallow.gz"]


def test_environment_secrets_are_read_by_name(monkeypatch):
    monkeypatch.setenv("MY_API_TOKEN", "token-value-123")
    monkeypatch.setenv("PLAIN_SETTING", "not-a-secret")
    monkeypatch.setenv("CUSTOM", "custom-secret-value")
    monkeypatch.setenv("EMPTY_PASSWORD", "")
    found = leaks.environment(["CUSTOM"])
    assert "token-value-123" in found
    assert "custom-secret-value" in found
    assert "not-a-secret" not in found
    assert "" not in found


def test_declared_levels_come_from_the_bundle_schema(tmp_path):
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "categorical": ["colour", "size"],
                "categories": {"colour": ["red", "green"]},
            }
        ),
        encoding="utf-8",
    )
    assert leaks.declared(schema) == {"colour": ["red", "green"], "size": []}
    schema.write_text("{", encoding="utf-8")
    with pytest.raises(errors.InputError, match="cannot read schema"):
        leaks.declared(schema)


def test_scanning_nothing_is_an_error(tmp_path):
    with pytest.raises(errors.InputError, match="nothing to scan"):
        leaks.scan(tmp_path / "missing", [])


def test_the_command_exits_one_on_a_leak_and_prints_no_value(
    tmp_path, capsys, monkeypatch
):
    folder = written(tmp_path / "run")
    data = tmp_path / "train.csv"
    table().to_csv(data, index=False)
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps({"categorical": ["colour"], "categories": DECLARED}),
        encoding="utf-8",
    )
    monkeypatch.setenv("SCMLE_TEST_SIGNING_KEY", SECRET)
    assert (
        leaks.main(
            [
                str(folder),
                "--data",
                str(data),
                "--schema",
                str(schema),
                "--secret-env",
                "SCMLE_TEST_SIGNING_KEY",
            ]
        )
        == 1
    )
    printed = capsys.readouterr().out
    assert json.loads(printed)["leaks"] == {
        "row": 8,
        "category": 1,
        "secret": 2,
    }
    assert NAME not in printed
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "ok.json").write_text("{}", encoding="utf-8")
    assert leaks.main([str(clean), "--data", str(data)]) == 0
