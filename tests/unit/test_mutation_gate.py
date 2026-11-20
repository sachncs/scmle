"""The release gate's refusals, word for word."""

import dataclasses

from scmle.deploy import gate
from scmle.deploy import livetests
from tests import support_release


def test_an_unsigned_model_is_named() -> None:
    candidate = support_release.candidate(
        model=support_release.version(signed=False)
    )
    assert gate.veto(candidate, support_release.KEY) == [
        "the model artifact is unsigned"
    ]


def test_every_missing_staging_check_is_listed() -> None:
    staged = support_release.staging_record()
    thin = dataclasses.replace(staged, evidence=staged.evidence[:1]).signed(
        support_release.KEY
    )
    candidate = support_release.candidate("production", record=thin)
    missing = ", ".join(livetests.CHECK_NAMES[1:])
    assert gate.audit(candidate, support_release.KEY) == [
        f"staging evidence is missing: {missing}"
    ]
