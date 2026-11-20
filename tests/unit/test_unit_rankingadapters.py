"""Ranking adapters build pointwise and library rankers."""

from sklearn import ensemble

from scmle.contracts import tasks
from scmle.models.ranking import adapters
from scmle.models.ranking import estimators


def test_a_pointwise_adapter_wraps_a_seeded_boosting_regressor():
    adapter = adapters.Pointwise()
    built = adapter.build(tasks.Task.RANKING, 3, {"max_leaf_nodes": 9})
    assert isinstance(built, estimators.Scorer)
    assert isinstance(built.regressor, ensemble.HistGradientBoostingRegressor)
    assert built.regressor.random_state == 3
    assert built.regressor.max_leaf_nodes == 9
    assert adapter.name == "pointwise_ranker"
    assert (
        adapter.capabilities.tasks == adapters.RANKING == {tasks.Task.RANKING}
    )
    assert adapter.capabilities.missing is True


def test_a_library_ranker_keeps_its_library_and_seed():
    built = adapters.LIGHTGBM_RANKER.build(
        tasks.Task.RANKING, 5, {"learning_rate": 0.05}
    )
    assert isinstance(built, estimators.Library)
    assert built.library == "lightgbm"
    assert built.random_state == 5
    assert built.learning_rate == 0.05
    assert (
        adapters.XGBOOST_RANKER.build(tasks.Task.RANKING, 0).library
        == "xgboost"
    )


def test_a_ranker_declares_its_name_library_and_requirements():
    assert adapters.LIGHTGBM_RANKER.name == "lightgbm_ranker"
    assert adapters.LIGHTGBM_RANKER.requires == ("lightgbm",)
    assert adapters.XGBOOST_RANKER.name == "xgboost_ranker"
    assert adapters.XGBOOST_RANKER.requires == ("xgboost",)
    bare = adapters.Ranker()
    assert bare.library == ""
    assert bare.family == "boosting"
    custom = adapters.Ranker(library="lightgbm")
    assert custom.library == "lightgbm"
    assert custom.requires == adapters.Ranker.requires
    names = list(adapters.Ranker.space.names())
    assert names == ["learning_rate", "max_leaves"]
    assert [type(a) for a in adapters.RANKING_ADAPTERS] == [
        adapters.Pointwise,
        adapters.Ranker,
        adapters.Ranker,
        adapters.Linear,
    ]
    assert adapters.RANKING_ADAPTERS[1:] == (
        adapters.LIGHTGBM_RANKER,
        adapters.XGBOOST_RANKER,
        adapters.PAIRWISE,
    )
