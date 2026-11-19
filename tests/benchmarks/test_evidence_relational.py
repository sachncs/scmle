"""Claim C5 evidence, real part: relational datasets against the reference."""

import dataclasses
import hashlib
import math
import sqlite3

import pandas

from benchmarks.evidence import reading
from benchmarks.evidence import relational

TOL = 1e-8


def test_five_datasets_with_licence_and_checksum():
    assert len(relational.DATASETS) >= reading.REAL_DATASETS
    assert len({d.name for d in relational.DATASETS}) == len(
        relational.DATASETS
    )
    for dataset in relational.DATASETS:
        assert dataset.licence
        assert dataset.home.startswith("https://")
        for file in dataset.files:
            assert file.url.startswith("https://")
            assert len(file.sha256) == 64


def test_days_are_exact_and_parse_every_precision():
    assert relational.days("1970-01-02", False) == 1.0
    assert relational.days("2016-07-04", False) == 16_986.0
    half = relational.days("1970-01-01 12:00:00.000", False)
    assert half == 0.5
    assert relational.days("1970-01-01T00:00:00+01:00", False) == -1 / 24
    assert math.isnan(relational.days(None, False))
    assert math.isnan(relational.days("", False))
    assert relational.days("1871", True) == 1871.0


def test_cutoffs_tie_with_events_and_recorded_times():
    own = pandas.DataFrame(
        {"event": [1.0, 5.0, 9.0], "known": [2.0, math.nan, 12.0]},
        dtype=object,
    )
    cuts = relational.cutoffs(own, 7.0)
    assert cuts == [5.0, 12.0, 0.0, 12.0, 13.0]


def test_entities_span_light_and_heavy():
    rows = pandas.DataFrame({"key": [1, 2, 2, 3, 3, 3, None]}, dtype=object)
    assert relational.entities(rows) == [1, 2, 3]


def test_specs_cover_every_primitive_window_and_nested_pair():
    dataset = relational.DATASETS[0]
    found = relational.specs(dataset)
    plain = {(s.primitive, s.window) for s in found if s.inner is None}
    assert len(plain) == 40
    nested = {(s.inner, s.primitive, s.window) for s in found if s.inner}
    assert len(nested) == 36


def world(tmp_path, monkeypatch):
    """A small SQLite dataset with ties, late rows and a timeless leaf."""
    path = tmp_path / "shop.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE orders (oid INTEGER, cid INTEGER, at TEXT, rec TEXT,
                             total REAL);
        CREATE TABLE lines (lid INTEGER, oid INTEGER, price REAL);
        INSERT INTO orders VALUES
          (1, 1, '2024-01-01', '2024-01-01 06:00:00', 10.0),
          (2, 1, '2024-01-05 12:00:00', NULL, 20.0),
          (3, 1, '2024-01-05 12:00:00', '2024-03-01', 30.0),
          (4, 2, '2024-02-01', '2024-02-02', NULL),
          (5, 2, '2024-02-08', '2024-02-08', 5.5);
        INSERT INTO lines VALUES
          (1, 1, 1.0), (2, 1, 2.0), (3, 2, 4.0), (4, 3, NULL),
          (5, 4, 8.0), (6, 9, 16.0);
        """
    )
    connection.commit()
    connection.close()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset = relational.Dataset(
        "shop",
        "test",
        "https://example.invalid",
        "sqlite",
        (relational.File("shop.db", "https://example.invalid", digest),),
        relational.Table(
            "orders", "cid", "at", "rec", ("total",), "oid", "oid"
        ),
        relational.Table("lines", "oid", None, None, ("price",), order="lid"),
    )
    monkeypatch.setattr(relational, "CACHE", tmp_path / "cache")
    folder = relational.CACHE / "shop"
    folder.mkdir(parents=True)
    (folder / "shop.db").write_bytes(path.read_bytes())
    return dataset


def test_a_small_database_agrees_with_the_reference(tmp_path, monkeypatch):
    found = relational.compare(world(tmp_path, monkeypatch), TOL)
    assert found["error"] == ""
    assert found["features"] == 76
    assert found["equal"] == found["features"]
    assert found["targets"] == 10


def test_a_product_fault_is_reported_unequal(tmp_path, monkeypatch):
    dataset = world(tmp_path, monkeypatch)
    real = relational.fast

    def late(spec, data, tables, targets):
        moved = targets.assign(
            moment=targets["moment"] + pandas.Timedelta(days=1)
        )
        return real(spec, data, tables, moved)

    monkeypatch.setattr(relational, "fast", late)
    found = relational.compare(dataset, TOL)
    assert found["error"] == ""
    assert found["unequal"]
    assert found["equal"] < found["features"]


def test_a_changed_file_is_refused_and_counts_against(tmp_path, monkeypatch):
    dataset = world(tmp_path, monkeypatch)
    wrong = dataclasses.replace(
        dataset,
        files=(dataclasses.replace(dataset.files[0], sha256="0" * 64),),
    )
    found = relational.compare(wrong, TOL)
    assert "SHA-256" in found["error"]
    assert found["equal"] == 0
    assert not relational.agrees(found)


def test_agrees_needs_every_feature_equal_and_no_error():
    good = {"features": 3, "equal": 3, "error": ""}
    assert relational.agrees(good)
    assert not relational.agrees(good | {"equal": 2})
    assert not relational.agrees(good | {"error": "OSError: x"})
    assert not relational.agrees(good | {"features": 0, "equal": 0})
