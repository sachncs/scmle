"""The newer task types on real datasets, each against a simple baseline.

Every case loads a real public dataset, runs the full lifecycle, and scores a
simple baseline on exactly the same locked rows (the run's own validation
manifest), so the comparison is like for like. The point is to learn how each
task behaves on messy data (irregular gaps, rare events, confounded
treatment), not to claim wins: cases where the framework does not beat the
baseline are reported as they are.

Usage:
    python -m benchmarks.real_tasks [--only NAME ...] [--budget SECONDS]
        [--output benchmarks/results_real_tasks.json]
"""

import argparse
from collections.abc import Callable
import dataclasses
import json
import logging
import pathlib
import sys
import tempfile
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import multioutput

import mml
from mml.evaluation import concordance
from mml.evaluation import multitarget
from mml.evaluation import ordinal
from mml.evaluation import ranking
from mml.evaluation import uplift
from mml.models.survival import estimators as survival_estimators

LOGGER = logging.getLogger(__name__)
RDATASETS = "https://vincentarelbundock.github.io/Rdatasets/csv/"
CACHE = pathlib.Path(tempfile.gettempdir()) / "mml_real_data"


@dataclasses.dataclass
class Row:
    """One dataset's outcome.

    Attributes:
        dataset: Dataset name.
        task: Task type.
        rows: Rows used.
        metric: Metric name.
        higher_is_better: Orientation of the metric.
        framework: The framework's locked-test score.
        baseline: The baseline's score on the same rows.
        baseline_name: What the baseline is.
        selected: Selected model.
        seconds: Wall-clock time.
        note: What to keep in mind when reading the numbers.
        error: Traceback if the case failed.
    """

    dataset: str
    task: str
    rows: int = 0
    metric: str = ""
    higher_is_better: bool = True
    framework: float = float("nan")
    baseline: float = float("nan")
    baseline_name: str = ""
    selected: str = ""
    seconds: float = 0.0
    note: str = ""
    error: str = ""

    @property
    def verdict(self) -> str:
        """Ahead, behind or level, by the sign of the difference."""
        gap = self.framework - self.baseline
        if not self.higher_is_better:
            gap = -gap
        if np.isnan(gap):
            return "n/a"
        return "ahead" if gap > 0 else "behind" if gap < 0 else "level"


def cached(name: str, make: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Downloads a table once and keeps it as Parquet."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.parquet"
    if not path.exists():
        make().to_parquet(path)
    return pd.read_parquet(path)


def rdataset(path: str) -> pd.DataFrame:
    """A table from the Rdatasets collection."""
    return cached(
        path.replace("/", "_"),
        lambda: pd.read_csv(RDATASETS + path).drop(
            columns=["rownames"], errors="ignore"
        ),
    )


def openml(name: str, version: int | str = "active") -> pd.DataFrame:
    """A table from OpenML."""
    from sklearn.datasets import fetch_openml

    return cached(
        f"openml_{name}_{version}",
        lambda: fetch_openml(name=name, version=version, as_frame=True).frame,
    )


def split_ids(result: mml.RunResult) -> tuple[np.ndarray, np.ndarray]:
    """Development and locked row positions of a finished run."""
    manifest = json.loads(
        (result.run_dir / "validation_manifest.json").read_text()
    )
    return np.array(manifest["train_ids"]), np.array(manifest["test_ids"])


def encode(frame: pd.DataFrame) -> pd.DataFrame:
    """Numeric matrix for baselines: dummies, missing filled with medians."""
    coded = pd.get_dummies(frame, dtype=float)
    return coded.fillna(coded.median())


def project(
    frame: pd.DataFrame, target: Any, root: pathlib.Path, **kw: Any
) -> mml.MLProject:
    """A project with the suite's shared settings."""
    kw.setdefault("time_budget", BUDGET)
    kw.setdefault("workers", 2)
    return mml.MLProject(frame, target, output_dir=root, **kw)


BUDGET = 150


def forecast_case(
    name: str,
    frame: pd.DataFrame,
    value: str,
    clock: str,
    period: int,
    horizon: int,
    root: pathlib.Path,
) -> Row:
    """Forecasts a real series against last-value and seasonal-naive."""
    result = project(
        frame,
        value,
        root,
        task="forecast",
        horizon=horizon,
        time_column=clock,
        seasonal_period=period,
        folds=3,
    ).run()
    steps = result.evaluation["forecast_backtest"]["steps"]
    model = float(np.mean([s["model_mae"] for s in steps]))
    naive = {
        "last value": float(np.mean([s["last_value_mae"] for s in steps])),
        "seasonal naive": float(
            np.mean([s["seasonal_naive_mae"] for s in steps])
        ),
    }
    best = min(naive, key=lambda kind: naive[kind])
    return Row(
        name,
        "forecast",
        len(frame),
        "mean MAE over steps",
        False,
        model,
        naive[best],
        best,
        result.selected_model,
        note=f"{horizon} steps ahead, period {period}; baselines "
        + ", ".join(f"{k} {v:.2f}" for k, v in naive.items()),
    )


def survival_case(
    name: str,
    frame: pd.DataFrame,
    duration: str,
    event: str,
    root: pathlib.Path,
    note: str = "",
) -> Row:
    """Survival against an untuned Cox model on the same rows."""
    result = project(
        frame, duration, root, task="survival", event_column=event
    ).run()
    train, test = split_ids(result)
    features = frame.drop(columns=[duration, event])
    matrix = encode(features)
    matrix = (matrix - matrix.iloc[train].mean()) / matrix.iloc[
        train
    ].std().replace(0, 1)
    cox = survival_estimators.CoxRegressor().fit(
        matrix.iloc[train].to_numpy(),
        frame[duration].iloc[train].to_numpy(),
        frame[event].iloc[train].to_numpy(),
    )
    base = concordance.concordance_index(
        frame[duration].iloc[test].to_numpy(),
        frame[event].iloc[test].to_numpy(),
        cox.predict(matrix.iloc[test].to_numpy()),
    )
    return Row(
        name,
        "survival",
        len(frame),
        "c_index",
        True,
        result.evaluation["test_metrics"]["c_index"],
        base,
        "untuned ridge Cox",
        result.selected_model,
        note=note + f" events {int(frame[event].sum())}/{len(frame)}",
    )


def ordinal_case(
    name: str, frame: pd.DataFrame, target: str, root: pathlib.Path
) -> Row:
    """Ordered labels against a plain multiclass boosted classifier."""
    order = sorted(frame[target].unique(), key=lambda v: float(v))
    result = project(
        frame, target, root, task="ordinal", ordinal_order=order
    ).run()
    train, test = split_ids(result)
    ranks = frame[target].map({v: i for i, v in enumerate(order)}).to_numpy()
    features = encode(frame.drop(columns=[target]))
    classifier = ensemble.HistGradientBoostingClassifier(random_state=0).fit(
        features.iloc[train], ranks[train]
    )
    guess = classifier.predict(features.iloc[test])
    base = ordinal.quadratic_weighted_kappa(ranks[test], guess)
    return Row(
        name,
        "ordinal",
        len(frame),
        "qwk",
        True,
        result.evaluation["test_metrics"]["qwk"],
        base,
        "multiclass boosted classifier",
        result.selected_model,
        note=f"{len(order)} ordered levels; MAE in levels "
        f"{result.evaluation['test_metrics']['mace']:.3f} vs "
        f"{ordinal.mean_absolute_class_error(ranks[test], guess):.3f}",
    )


def multilabel_case(
    name: str, frame: pd.DataFrame, labels: list[str], root: pathlib.Path
) -> Row:
    """Several labels against one boosted classifier per label."""
    frame = frame.copy()
    for label in labels:
        frame[label] = (
            frame[label].astype(str).str.upper().isin(["TRUE", "1"]).astype(int)
        )
    result = project(frame, labels, root, task="multilabel").run()
    train, test = split_ids(result)
    features = frame.drop(columns=labels)
    model = multioutput.MultiOutputClassifier(
        ensemble.HistGradientBoostingClassifier(random_state=0)
    ).fit(features.iloc[train], frame[labels].iloc[train])
    guess = model.predict(features.iloc[test])
    base = multitarget.f1_macro_labels(
        frame[labels].iloc[test].to_numpy(), guess
    )
    return Row(
        name,
        "multilabel",
        len(frame),
        "macro F1",
        True,
        result.evaluation["test_metrics"]["f1_macro_labels"],
        base,
        "one boosted classifier per label",
        result.selected_model,
        note=f"{len(labels)} labels; split "
        f"{result.evaluation['split_strategy']}",
    )


def multioutput_case(
    name: str, frame: pd.DataFrame, targets: list[str], root: pathlib.Path
) -> Row:
    """Several numbers against one boosted regressor per target."""
    frame = frame.assign(**{t: frame[t].astype(float) for t in targets})
    result = project(frame, targets, root, task="multioutput").run()
    train, test = split_ids(result)
    features = frame.drop(columns=targets)
    model = multioutput.MultiOutputRegressor(
        ensemble.HistGradientBoostingRegressor(random_state=0)
    ).fit(features.iloc[train], frame[targets].iloc[train])
    base = multitarget.r2_mean(
        frame[targets].iloc[test].to_numpy(), model.predict(features.iloc[test])
    )
    return Row(
        name,
        "multioutput",
        len(frame),
        "mean R2",
        True,
        result.evaluation["test_metrics"]["r2_mean"],
        base,
        "one boosted regressor per target",
        result.selected_model,
    )


def uplift_case(
    name: str,
    frame: pd.DataFrame,
    outcome: str,
    treatment: str,
    root: pathlib.Path,
    note: str = "",
) -> Row:
    """Uplift against random ranking, with the randomization verdict."""
    result = project(
        frame, outcome, root, task="uplift", treatment_column=treatment
    ).run()
    test = split_ids(result)[1]
    arm = frame[treatment].to_numpy()[test]
    y = frame[outcome].to_numpy(float)[test]
    generator = np.random.default_rng(0)
    chance = float(
        np.mean(
            [
                uplift.qini_coefficient(y, arm, generator.normal(size=len(y)))
                for __ in range(50)
            ]
        )
    )
    report = result.evaluation["uplift"]
    return Row(
        name,
        "uplift",
        len(frame),
        "qini",
        True,
        result.evaluation["test_metrics"]["qini"],
        chance,
        "random ranking (mean of 50)",
        result.selected_model,
        note=note + f" propensity AUC {report['propensity']['auc']:.2f}; "
        f"overall effect {report['average_effect']:+.4f}",
    )


def ranking_case(root: pathlib.Path) -> Row:
    """Web-search ranking against pointwise boosting."""
    from huggingface_hub import hf_hub_download

    def load() -> pd.DataFrame:
        path = hf_hub_download(
            "philipphager/MSLR-WEB10k",
            "validation.parquet",
            repo_type="dataset",
            local_dir=str(CACHE),
        )
        raw = pd.read_parquet(path).head(700)
        rows = []
        for query, labels, features in zip(
            raw["query"], raw["labels"], raw["features"], strict=True
        ):
            block = pd.DataFrame(np.vstack(features)).add_prefix("f")
            block.insert(0, "relevance", np.asarray(labels))
            block.insert(0, "query", query)
            rows.append(block)
        return pd.concat(rows, ignore_index=True)

    frame = cached("mslr_700", load)
    result = project(
        frame, "relevance", root, task="ranking", query_column="query"
    ).run()
    train, test = split_ids(result)
    features = frame.drop(columns=["relevance", "query"])
    model = ensemble.HistGradientBoostingRegressor(random_state=0).fit(
        features.iloc[train], frame["relevance"].iloc[train]
    )
    base = ranking.mean_ndcg(
        frame["relevance"].iloc[test].to_numpy(),
        model.predict(features.iloc[test]),
        frame["query"].iloc[test],
    )
    return Row(
        "mslr-web10k (700 queries)",
        "ranking",
        len(frame),
        "ndcg@10",
        True,
        result.evaluation["test_metrics"]["ndcg"],
        base,
        "pointwise boosting",
        result.selected_model,
        note=f"{frame['query'].nunique()} queries, 136 features",
    )


def streaming_case(root: pathlib.Path) -> Row:
    """A real 581k-row table streamed under a tiny memory budget."""
    from sklearn.datasets import fetch_covtype

    CACHE.mkdir(exist_ok=True)
    path = CACHE / "covertype.parquet"
    if not path.exists():
        data = fetch_covtype(as_frame=True).frame
        data.to_parquet(path)
    result = mml.MLProject(
        str(path),
        "Cover_Type",
        task="classification",
        time_budget=BUDGET,
        workers=2,
        output_dir=root,
        memory_budget_gb=0.05,
    ).run()
    streamed = result.evaluation["streamed_locked"]
    # Baseline: boosting trained on every development row, scored on the
    # same hashed locked rows.
    from mml.data.readers import streaming

    frame = pd.read_parquet(path)
    side = streaming.side_of(np.arange(len(frame)), 0.2, 42)
    features = frame.drop(columns=["Cover_Type"])
    model = ensemble.HistGradientBoostingClassifier(random_state=0).fit(
        features[side == 0], frame["Cover_Type"][side == 0]
    )
    base = float(
        (
            model.predict(features[side == 1]) == frame["Cover_Type"][side == 1]
        ).mean()
    )
    return Row(
        "covertype (581k rows, 50 MB budget)",
        "streamed",
        len(frame),
        "accuracy on all locked rows",
        True,
        streamed["metrics"]["accuracy"],
        base,
        "boosting on every development row",
        result.selected_model,
        note=f"{streamed['rows']} locked rows scored by streaming; "
        f"{result.evaluation['n_dev']} development rows in memory",
    )


def text_case(root: pathlib.Path) -> Row:
    """SMS spam: sentence embeddings against TF-IDF terms."""

    def load() -> pd.DataFrame:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "ucirvine/sms_spam",
            "plain_text/train-00000-of-00001.parquet",
            repo_type="dataset",
            local_dir=str(CACHE),
        )
        return pd.read_parquet(path)

    frame = cached("sms_spam", load).rename(
        columns={"sms": "text", "label": "spam"}
    )
    runs = {}
    for encoding in ("tfidf", "embedding"):
        runs[encoding] = project(
            frame,
            "spam",
            root / encoding,
            task="binary_classification",
            text_encoding=encoding,
        ).run()
    return Row(
        "sms-spam",
        "text",
        len(frame),
        "roc_auc",
        True,
        runs["embedding"].evaluation["test_metrics"]["roc_auc"],
        runs["tfidf"].evaluation["test_metrics"]["roc_auc"],
        "TF-IDF terms",
        runs["embedding"].selected_model,
        note="same framework run with the text encoder switched",
    )


def build_cases(root: pathlib.Path) -> dict[str, Callable[[], Row]]:
    """Every case by name."""

    def r(name: str) -> pathlib.Path:
        return root / name

    def air() -> pd.DataFrame:
        return rdataset("datasets/AirPassengers.csv")

    cases: dict[str, Callable[[], Row]] = {
        "airpassengers": lambda: forecast_case(
            "AirPassengers (monthly, 144)",
            air(),
            "value",
            "time",
            12,
            6,
            r("air"),
        ),
        "nottem": lambda: forecast_case(
            "nottem (monthly temperature, 240)",
            rdataset("datasets/nottem.csv"),
            "value",
            "time",
            12,
            6,
            r("nottem"),
        ),
        "ukgas": lambda: forecast_case(
            "UKgas (quarterly, 108)",
            rdataset("datasets/UKgas.csv"),
            "value",
            "time",
            4,
            4,
            r("ukgas"),
        ),
        "rotterdam": lambda: survival_case(
            "rotterdam breast cancer (2982)",
            rdataset("survival/rotterdam.csv").drop(
                columns=["pid", "dtime", "death"]
            ),
            "rtime",
            "recur",
            r("rotterdam"),
            "recurrence-free time",
        ),
        "pbc": lambda: survival_case(
            "PBC liver (418)",
            rdataset("survival/pbc.csv")
            .drop(columns=["id"])
            .assign(status=lambda f: (f.status == 2).astype(int)),
            "time",
            "status",
            r("pbc"),
            "death vs censored/transplant",
        ),
        "wine-red": lambda: ordinal_case(
            "wine quality, red (1599)",
            openml("wine-quality-red").rename(columns={"class": "quality"}),
            "quality",
            r("winered"),
        ),
        "wine-white": lambda: ordinal_case(
            "wine quality, white (4898)",
            openml("wine-quality-white").rename(columns={"Class": "quality"}),
            "quality",
            r("winewhite"),
        ),
        "yeast": lambda: multilabel_case(
            "yeast (2417, 14 labels)",
            openml("yeast", 4),
            [f"Class{i}" for i in range(1, 15)],
            r("yeast"),
        ),
        "scene": lambda: multilabel_case(
            "scene (2407, 6 labels)",
            openml("scene"),
            ["Beach", "Sunset", "FallFoliage", "Field", "Mountain", "Urban"],
            r("scene"),
        ),
        "energy": lambda: multioutput_case(
            "energy efficiency (768, 2 loads)",
            openml("energy-efficiency"),
            ["y1", "y2"],
            r("energy"),
        ),
        "nsw": lambda: uplift_case(
            "NSW job training (randomized, 445)",
            rdataset("causaldata/nsw_mixtape.csv")
            .drop(columns=["data_id"])
            .assign(employed=lambda f: (f.re78 > 0).astype(int))
            .drop(columns=["re78"]),
            "employed",
            "treat",
            r("nsw"),
            "randomized experiment.",
        ),
        "lalonde": lambda: uplift_case(
            "Lalonde NSW vs PSID (observational, 614)",
            rdataset("MatchIt/lalonde.csv")
            .assign(employed=lambda f: (f.re78 > 0).astype(int))
            .drop(columns=["re78"]),
            "employed",
            "treat",
            r("lalonde"),
            "NOT randomized: controls come from a survey.",
        ),
        "criteo": lambda: uplift_case(
            "Criteo uplift (randomized, sampled 420k)",
            criteo(),
            "visit",
            "treatment",
            r("criteo"),
            "randomized, 3% sample.",
        ),
        "mslr": lambda: ranking_case(r("mslr")),
        "covertype": lambda: streaming_case(r("covertype")),
        "sms": lambda: text_case(r("sms")),
    }
    return cases


def criteo() -> pd.DataFrame:
    """A 3% random sample of the Criteo uplift table."""

    def load() -> pd.DataFrame:
        url = (
            "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/"
            "main/criteo-research-uplift-v2.1.csv.gz"
        )
        parts = []
        for chunk in pd.read_csv(url, chunksize=1_000_000):
            parts.append(chunk.sample(frac=0.03, random_state=0))
        return pd.concat(parts, ignore_index=True).drop(
            columns=["conversion", "exposure"]
        )

    return cached("criteo_3pct", load)


def format_table(rows: list[Row]) -> str:
    """Markdown table of the results."""
    lines = [
        "| dataset | task | rows | metric | framework | baseline | "
        "baseline is | verdict | selected |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row.error:
            lines.append(f"| {row.dataset} | {row.task} | | ERROR | | | | | |")
            continue
        lines.append(
            f"| {row.dataset} | {row.task} | {row.rows} | {row.metric} | "
            f"{row.framework:.4f} | {row.baseline:.4f} | {row.baseline_name} | "
            f"{row.verdict} | {row.selected} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Runs the selected real-data cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--budget", type=int, default=150)
    parser.add_argument(
        "--output", default="benchmarks/results_real_tasks.json"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING, format="%(message)s", stream=sys.stdout
    )
    LOGGER.setLevel(logging.INFO)
    global BUDGET
    BUDGET = args.budget
    rows: list[Row] = []
    with tempfile.TemporaryDirectory() as scratch:
        cases = build_cases(pathlib.Path(scratch))
        for name, case in cases.items():
            if args.only and name not in args.only:
                continue
            LOGGER.info("running %s ...", name)
            started = time.monotonic()
            try:
                row = case()
            except Exception:  # Report every case, even when one fails.
                row = Row(name, "?", error=traceback.format_exc(limit=5))
                LOGGER.error("%s failed:\n%s", name, row.error)
            row.seconds = time.monotonic() - started
            rows.append(row)
            LOGGER.info("%s", format_table([row]).splitlines()[-1])
    LOGGER.info("\n%s", format_table(rows))
    pathlib.Path(args.output).write_text(
        json.dumps({"results": [dataclasses.asdict(r) for r in rows]}, indent=2)
    )
    return 0 if all(not r.error for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
