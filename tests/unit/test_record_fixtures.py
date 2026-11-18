"""Stored version-1 records still load, dump identically and keep their digests.

Each fixture in `tests/fixtures/records/v1` was written by the writer that
existed before the kind became a `records.Record` (the hand-written
`to_dict`). Both kinds are now at version 2 (single-word keys); loading a
fixture migrates it, its version-1 anchor (`records.anchor`) must give back
the stored record, and every pinned identity digest in
`tests/fixtures/record_digests.json` must be unchanged.
"""

import hashlib
import json
import pathlib
from typing import Any
from typing import ClassVar

import hypothesis
from hypothesis import strategies
import pytest

from scmle.contracts import evaluation
from scmle.contracts import explanations
from scmle.contracts import records

FIXTURES = pathlib.Path(__file__).parents[1] / "fixtures"
VERSION_ONE = FIXTURES / "records" / "v1"
DIGESTS = json.loads((FIXTURES / "record_digests.json").read_text("utf-8"))
KINDS = {
    "evaluation_policy": evaluation.Criteria,
    "explanation_evidence": explanations.Evidence,
}


def stored(name):
    """The text of a version-1 fixture."""
    return (VERSION_ONE / f"{name}.json").read_text("utf-8")


def write(record):
    """The bytes the run store writes for a record (`Store.write_json`)."""
    return json.dumps(record.model_dump(mode="json"), indent=2) + "\n"


def test_every_fixture_names_a_kind():
    assert {path.stem for path in VERSION_ONE.glob("*.json")} == set(KINDS)
    assert set(DIGESTS) == set(KINDS)


@pytest.mark.parametrize("name", sorted(KINDS))
def test_a_version_one_record_loads_and_anchors_to_the_same_record(name):
    text = stored(name)
    record = KINDS[name].model_validate(json.loads(text))
    assert record.schema_version == 2
    assert records.anchor(record) == json.loads(text)
    assert KINDS[name].model_validate_json(record.model_dump_json()) == record
    assert KINDS[name].model_validate_json(write(record)) == record


@pytest.mark.parametrize("name", sorted(KINDS))
def test_the_identity_digest_is_pinned(name):
    record = KINDS[name].model_validate_json(stored(name))
    pinned = DIGESTS[name]
    kind = KINDS[name]
    assert pinned["kind"] == f"{kind.__module__}.{kind.__qualname__}"
    digest = hashlib.sha256(record.canonize().encode("utf-8")).hexdigest()
    assert digest == pinned["sha256"]


POLICY_KEYS = ("task", "primary_metric", "schema_version")


def nest(layout):
    """Moves every policy key but the task and primary into `policy`."""
    return {
        "task": layout["task"],
        "primary": layout["primary"],
        "policy": {
            k: v
            for k, v in layout.items()
            if k not in (*POLICY_KEYS, "primary")
        },
    }


def flatten(layout):
    """Undoes `nest`."""
    return {
        "task": layout["task"],
        "primary": layout["primary"],
        **layout["policy"],
    }


class Successor(records.Record):
    """A later policy layout: the primary metric renamed, the rest nested."""

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={"primary_metric": "primary"}, change=nest, revert=flatten
        ),
    )

    task: str
    primary: str
    policy: dict[str, Any]


def test_an_old_version_fixture_loads_through_its_migration():
    layout = json.loads(stored("evaluation_policy"))
    successor = Successor.model_validate(layout)
    assert successor.schema_version == 2
    assert successor.primary == layout["primary_metric"]
    assert successor.policy["metrics"] == layout["metrics"]
    newer = successor.model_dump(mode="json")
    del newer["schema_version"]
    restored = Successor.migrations[0].downgrade(newer)
    assert {"schema_version": 1, **restored} == layout
    original = evaluation.Criteria.model_validate(layout)
    assert records.anchor(original) == json.loads(stored("evaluation_policy"))


@hypothesis.given(
    threshold=strategies.floats(0.5, 0.99),
    warning=strategies.integers(1, 1000),
)
def test_a_policy_survives_a_round_trip(threshold, warning):
    base = evaluation.Criteria.model_validate_json(stored("evaluation_policy"))
    layout = base.model_dump(mode="json")
    layout["release"]["metrics"][0]["threshold"] = threshold
    layout["small"] = {"warning": warning, "blocking": None}
    policy = evaluation.Criteria.model_validate(layout)
    assert (
        evaluation.Criteria.model_validate(
            json.loads(json.dumps(policy.model_dump(mode="json")))
        )
        == policy
    )


@hypothesis.given(
    importance=strategies.floats(-1e6, 1e6),
    repeats=strategies.integers(1, 100),
)
def test_explanation_evidence_survives_a_round_trip(importance, repeats):
    layout = json.loads(stored("explanation_evidence"))
    layout["output"] = {"features": [{"name": "age", "importance": importance}]}
    layout["method"]["parameters"] = {"repeats": repeats}
    evidence = explanations.Evidence.model_validate(layout)
    again = explanations.Evidence.model_validate_json(
        evidence.model_dump_json()
    )
    assert again == evidence
    assert again.canonize() == evidence.canonize()
