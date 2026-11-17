"""Guarantees of docs/PORTABLE_TRAINING.md that no other test enforced."""

import pytest

from scmle.contracts import errors
from scmle.runners import roles


def test_check_separation_refuses_each_overlap_on_its_own():
    """Enforces docs/PORTABLE_TRAINING.md:42."""
    roles.check_separation(roles.TRAINING_ROLE, roles.SERVING_ROLE)
    cases = (
        (
            roles.Role("training", (), ("evidence/",)),
            roles.Role("serving", (), ("evidence/x/",)),
            "both write",
        ),
        (
            roles.Role("training", (), ("bundles/",)),
            roles.Role("serving", (), ("audit/",)),
            "released bundles",
        ),
        (
            roles.Role("training", (), ("artifacts/",)),
            roles.Role("serving", (), ("runs/",)),
            "training records",
        ),
    )
    for training, serving, message in cases:
        with pytest.raises(errors.ConfigError, match=message):
            roles.check_separation(training, serving)
