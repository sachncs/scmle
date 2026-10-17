"""The benchmark suites are well formed (no downloads)."""

from benchmarks import datasets
from mml.evaluation import metrics


def everything():
    return datasets.SUITES["everything"]


def test_names_are_unique_across_every_suite():
    names = [d.name for d in everything()]

    assert len(names) == len(set(names))


def test_every_dataset_names_a_known_task_and_metric():
    known = set(metrics.CATALOG.metrics)

    for dataset in everything():
        assert dataset.task in {"classification", "regression"}, dataset.name
        assert dataset.metric in known, dataset.name
        assert dataset.target and dataset.target not in dataset.drop


def test_the_default_suite_is_the_original_nine_and_stays_stable():
    assert [d.name for d in datasets.SUITES["core"]] == [
        "breast_cancer",
        "wine",
        "digits",
        "diabetes_progression",
        "credit-g",
        "titanic",
        "kc1_defects",
        "adult_income",
        "california_housing",
    ]


def test_hugging_face_tables_point_at_files_in_known_repositories():
    tables = datasets.SUITES["huggingface"]

    assert len(tables) >= 20 and all(t.online for t in tables)
    assert {t.repo for t in tables} == {
        "inria-soda/tabular-benchmark",
        "scikit-learn/adult-census-income",
        "scikit-learn/auto-mpg",
        "scikit-learn/Fish",
    }
    assert all(t.filename.endswith(".csv") for t in tables)


def test_suites_cover_the_data_structures_the_framework_claims():
    extended = {d.name: d for d in datasets.SUITES["extended"]}

    assert extended["electricity"].time_column == "date"
    assert extended["credit_card_fraud"].metric == "pr_auc"
    assert extended["letter"].task == "classification"
    assert sum(d.task == "regression" for d in extended.values()) >= 6


def test_dropped_columns_are_removed_when_loading(monkeypatch):
    import pandas as pd

    class Fake(datasets.BenchmarkDataset):
        def fetch(self):
            return pd.DataFrame(
                {"id": [1, 2, 3], "x": [1.0, 2.0, 3.0], "y": [0, 1, 0]}
            )

    loaded = Fake("f", "y", "classification", "roc_auc", drop=("id",)).load()

    assert list(loaded.columns) == ["x", "y"]
