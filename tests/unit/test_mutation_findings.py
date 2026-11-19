"""Exact text and round trip of a data finding."""

import pytest

from scmle.contracts import errors
from scmle.contracts.data import findings


def test_a_gleaned_finding_asks_for_a_disposition():
    found = findings.Finding.glean("rows repeat", "Quality")

    assert found.remediation == (
        "Review the observation and record a disposition."
    )


def test_a_blank_identifier_is_refused_with_its_name():
    record = findings.Finding.glean("rows repeat", "quality").to_dict()

    with pytest.raises(errors.InputError) as caught:
        findings.Finding.from_dict(record | {"identifier": " "})

    assert str(caught.value) == "identifier must not be empty"


def test_from_dict_keeps_the_reviewer():
    record = findings.Finding.glean("rows repeat", "quality").to_dict()

    found = findings.Finding.from_dict(record | {"reviewer": "R. Lee"})

    assert found.reviewer == "R. Lee"
    assert found.to_dict() == record | {"reviewer": "R. Lee"}
