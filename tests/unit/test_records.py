"""Properties of the versioned record layer (`scmle.contracts.records`)."""

from typing import ClassVar

import hypothesis
from hypothesis import strategies
import pydantic
import pytest

from scmle.contracts import errors
from scmle.contracts import records

TEXT = strategies.text(max_size=20)
NUMBER = strategies.integers(-(10**9), 10**9)


class Point(records.Record):
    """A kind at version 1: no migrations."""

    label: str
    count: int


def rename(old, new):
    """A version step that only renames one key."""
    return records.Migration(renames={old: new})


def double(layout):
    """Doubles the count."""
    return {**layout, "total": layout["total"] * 2}


def halve(layout):
    """Undoes `double`."""
    return {**layout, "total": layout["total"] // 2}


class Tally(records.Record):
    """A kind at version 3: a rename, then a rename with a change."""

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        rename("label", "name"),
        records.Migration(
            renames={"count": "total"}, change=double, revert=halve
        ),
    )

    name: str
    total: int


class Legacy(records.Record):
    """A kind whose old files spell the version key `format`."""

    legacy: ClassVar[str | None] = "format"

    label: str


@hypothesis.given(label=TEXT, count=NUMBER)
def test_a_record_survives_a_json_round_trip(label, count):
    record = Point(label=label, count=count)
    stored = record.model_dump(mode="json")
    assert stored == {"schema_version": 1, "label": label, "count": count}
    assert Point.model_validate(stored) == record
    assert Point.model_validate_json(record.model_dump_json()) == record


@hypothesis.given(label=TEXT, count=NUMBER)
def test_a_record_without_a_version_key_is_version_one(label, count):
    assert Point.model_validate({"label": label, "count": count}) == Point(
        label=label, count=count
    )


@hypothesis.given(label=TEXT, count=NUMBER)
def test_an_old_record_migrates_to_the_current_version(label, count):
    loaded = Tally.model_validate({"label": label, "count": count})
    assert loaded.schema_version == Tally.latest() == 3
    assert (loaded.name, loaded.total) == (label, count * 2)
    middle = {"schema_version": 2, "name": label, "count": count}
    assert Tally.model_validate(middle) == loaded


@hypothesis.given(label=TEXT, count=NUMBER)
def test_every_step_is_undone_by_its_downgrade(label, count):
    layout = {"label": label, "count": count}
    for step in Tally.migrations:
        moved = step.upgrade(layout)
        assert step.downgrade(moved) == layout
        layout = moved


def test_a_record_built_in_code_is_at_the_current_version():
    assert Tally(name="a", total=2).schema_version == Tally.latest()


def test_a_record_newer_than_the_build_is_refused():
    with pytest.raises(ValueError, match=r"Tally record has schema version 4"):
        Tally.model_validate({"schema_version": 4, "name": "a", "total": 1})
    with pytest.raises(ValueError, match="newer than this build's 1"):
        Point(label="a", count=1, schema_version=2)


@pytest.mark.parametrize("version", [0, -1, True, "1", 1.0])
def test_a_malformed_version_is_refused(version):
    with pytest.raises(ValueError, match="schema version must be"):
        Point.model_validate(
            {"schema_version": version, "label": "a", "count": 1}
        )


def test_an_unknown_key_is_refused():
    with pytest.raises(pydantic.ValidationError, match="extra_forbidden"):
        Point.model_validate({"label": "a", "count": 1, "colour": "red"})


def test_a_record_is_frozen():
    record = Point(label="a", count=1)
    with pytest.raises(pydantic.ValidationError, match="frozen"):
        record.count = 2


def test_the_legacy_spelling_is_read_only_when_declared():
    assert (
        Legacy.model_validate({"format": 1, "label": "a"}).schema_version == 1
    )
    with pytest.raises(ValueError, match="newer than this build"):
        Legacy.model_validate({"format": 2, "label": "a"})
    with pytest.raises(pydantic.ValidationError, match="extra_forbidden"):
        Point.model_validate({"format": 1, "label": "a", "count": 1})


def test_a_rename_that_would_overwrite_a_key_is_refused():
    with pytest.raises(errors.InputError, match="overwrite"):
        rename("label", "name").upgrade({"label": "a", "name": "b"})


def test_a_step_that_cannot_be_undone_is_refused():
    with pytest.raises(errors.InputError, match="injective"):
        records.Migration(renames={"a": "c", "b": "c"})
    with pytest.raises(errors.InputError, match="revert"):
        records.Migration(change=double)
    with pytest.raises(errors.InputError, match="revert"):
        records.Migration(revert=halve)


def test_a_part_carries_no_version():
    class Part(records.Schema):
        value: int

    assert Part(value=1).model_dump() == {"value": 1}
    with pytest.raises(pydantic.ValidationError, match="extra_forbidden"):
        Part.model_validate({"value": 1, "schema_version": 1})
