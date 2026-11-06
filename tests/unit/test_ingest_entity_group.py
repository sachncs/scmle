"""The resolved entity becomes the split's group column (F9)."""

import pandas as pd
import pytest

from scmle.contracts import errors
from scmle.core import ingest_stage
from scmle.data.charter import ingest


def spec():
    return ingest.EntitySpec(
        fields=(ingest.EntityField("name"), ingest.EntityField("email", False)),
        passes=(
            ingest.EntityPass("email"),
            ingest.EntityPass("name", "prefix"),
        ),
        column="entity_group",
    )


def duplicated_table():
    people = [
        ("Maria Gonzalez", "maria.g@example.com"),
        ("Jonathan Smith", "jon.smith@example.com"),
        ("Priya Natarajan", "priya.n@example.com"),
        ("Wei Zhang", "wei.zhang@example.com"),
        ("Fatima Okafor", "fatima.o@example.com"),
        ("Lars Eriksson", "lars.e@example.com"),
    ]
    rows = []
    for index, (name, email) in enumerate(people):
        rows.append((index * 2, name, email))
        rows.append((index * 2 + 1, name.replace("a", "A", 1), email))
    return pd.DataFrame(rows, columns=["customer_id", "name", "email"])


def test_duplicates_under_different_keys_share_a_group():
    frame = duplicated_table()
    groups, document = ingest_stage.resolve_entities(frame, spec())
    assert len(groups) == len(frame)
    for first in range(0, len(frame), 2):
        assert groups.iloc[first] == groups.iloc[first + 1]
    assert groups.nunique() == 6
    assert document["tier"] == "unverified"


def test_distinct_people_are_not_merged():
    frame = duplicated_table()
    groups, ignored = ingest_stage.resolve_entities(frame, spec())
    assert groups.nunique() > 1


def test_a_missing_entity_field_is_a_config_error():
    frame = duplicated_table().drop(columns=["email"])
    with pytest.raises(errors.ConfigError, match="email"):
        ingest_stage.resolve_entities(frame, spec())
