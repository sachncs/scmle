"""Any scikit-learn-compatible estimator can join a run in one call."""

import pickle

import numpy
import pandas
import pytest
from sklearn import base
from sklearn import linear_model
from sklearn import naive_bayes
import support

from scmle.contracts import errors
from scmle.contracts import search
import scmle.models.custom


class WrongLength(base.ClassifierMixin, base.BaseEstimator):
    """Breaks the contract: predictions are one row short."""

    def fit(self, features, target):
        self.classes_ = numpy.unique(target)
        return self

    def predict(self, features):
        return numpy.zeros(len(features) - 1)


@pytest.fixture(autouse=True)
def clean_registry():
    yield
    for name in ("gaussian_nb", "bayes_ridge", "broken"):
        scmle.models.custom.unregister_estimator(name)


def frame(rows=500, seed=0):
    rng = numpy.random.default_rng(seed)
    data = pandas.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    data["y"] = (data.a + data.b + rng.normal(0, 0.5, rows) > 0).astype(int)
    return data


def test_a_registered_classifier_competes_in_a_run(tmp_path):
    scmle.models.custom.register_estimator(
        "gaussian_nb", classifier=naive_bayes.GaussianNB
    )

    result = support.make_project(
        frame(), "y", time_budget=300, output_dir=tmp_path
    ).run()

    ran = [t for t in result.trials if t.model_name == "gaussian_nb"]
    assert ran and ran[0].status.value == "ok"


def test_a_registered_regressor_gets_a_search_space(tmp_path):
    space = search.SearchSpace(
        (search.FloatParameter("alpha_1", 1e-8, 1e-3, log=True),)
    )
    adapter = scmle.models.custom.register_estimator(
        "bayes_ridge", regressor=linear_model.BayesianRidge, search_space=space
    )

    rng = numpy.random.default_rng(1)
    data = pandas.DataFrame(
        {"a": rng.normal(size=400), "b": rng.normal(size=400)}
    )
    data["y"] = 3 * data.a - data.b + rng.normal(0, 0.1, 400)
    result = support.make_project(
        data,
        "y",
        task="regression",
        time_budget=300,
        output_dir=tmp_path,
    ).run()

    assert adapter.search_space is space
    assert any(t.model_name == "bayes_ridge" for t in result.trials)


def test_an_estimator_that_breaks_the_contract_is_refused_with_the_check():
    with pytest.raises(errors.ConfigError, match="wrong length"):
        scmle.models.custom.register_estimator("broken", classifier=WrongLength)

    from scmle.models import registry

    assert "broken" not in registry.USER_ADAPTERS


def test_builtin_names_and_duplicates_are_refused():
    with pytest.raises(errors.ConfigError, match="built-in"):
        scmle.models.custom.register_estimator(
            "random_forest", classifier=naive_bayes.GaussianNB
        )
    scmle.models.custom.register_estimator(
        "gaussian_nb", classifier=naive_bayes.GaussianNB
    )
    with pytest.raises(errors.ConfigError, match="already registered"):
        scmle.models.custom.register_estimator(
            "gaussian_nb", classifier=naive_bayes.GaussianNB
        )
    scmle.models.custom.register_estimator(
        "gaussian_nb", classifier=naive_bayes.GaussianNB, replace=True
    )


def test_a_class_is_required():
    with pytest.raises(errors.ConfigError, match="classifier class"):
        scmle.models.custom.register_estimator("gaussian_nb")


def test_the_adapter_survives_pickling_for_trial_processes():
    adapter = scmle.models.custom.register_estimator(
        "gaussian_nb", classifier=naive_bayes.GaussianNB
    )

    restored = pickle.loads(pickle.dumps(adapter))

    assert restored.name == "gaussian_nb"
    assert restored.capabilities == adapter.capabilities


def test_a_registered_model_runs_under_hard_enforcement(tmp_path):
    scmle.models.custom.register_estimator(
        "gaussian_nb", classifier=naive_bayes.GaussianNB
    )

    result = support.make_project(
        frame(),
        "y",
        time_budget=300,
        output_dir=tmp_path,
        enforcement="hard",
    ).run()

    ran = [t for t in result.trials if t.model_name == "gaussian_nb"]
    assert ran and ran[0].status.value == "ok"
