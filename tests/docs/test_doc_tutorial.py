"""Guarantees of docs/TUTORIAL.md that no other test enforced."""

import pickle

import pytest
from sklearn import naive_bayes

from scmle.models import custom


class Shipped(naive_bayes.GaussianNB):
    """A module-level estimator class, importable by reference."""


def test_a_registered_class_reaches_trials_only_if_defined_at_module_level():
    """Enforces docs/TUTORIAL.md:289.

    Trial processes receive the adapter by pickle, which stores a class by
    its import path, so a class defined inside a function cannot travel.
    """

    class Local(naive_bayes.GaussianNB):
        pass

    shipped = custom.rebuild(custom.Registration("shipped", Shipped, None))
    copy = pickle.loads(pickle.dumps(shipped))
    assert copy.spec.classifier is Shipped
    stranded = custom.rebuild(custom.Registration("stranded", Local, None))
    with pytest.raises((pickle.PicklingError, AttributeError)):
        pickle.dumps(stranded)
