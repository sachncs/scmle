# Tutorial

## 1. Plan, run, predict

The first script writes a small example table, `customers.parquet`, that the
later steps read. Training starts helper processes, so a script that trains
keeps its work under `if __name__ == "__main__":`.

```python
import logging

import numpy
import pandas

import scmle.api.project

if __name__ == "__main__":
    # A small example table: one row per customer.
    rng = numpy.random.default_rng(0)
    data = pandas.DataFrame(
        {
            "customer_id": rng.permutation(600),
            "event_time": pandas.date_range("2025-01-01", periods=600),
            "age": rng.integers(18, 80, 600),
            "tenure": rng.integers(1, 72, 600),
            "monthly_spend": rng.normal(60, 20, 600),
            "plan": rng.choice(["basic", "plus", "pro"], 600),
        }
    )
    risk = data.monthly_spend / 50 - data.tenure / 30 + (data.plan == "basic")
    data["churn"] = (risk + rng.normal(scale=0.7, size=600) > 0.5).astype(int)
    data.to_parquet("customers.parquet")

    logging.basicConfig(level=logging.INFO)
    project = scmle.api.project.Project(
        "customers.parquet", target="churn", metric="roc_auc", time_budget=60
    )
    logging.info(project.plan().summary())  # nothing is trained or written
    result = project.run()
    logging.info("\n%s", result.leaderboard()[["model", "status", "score"]])
    logging.info("\n%s", result.salience().head())
    # Sign with your own secret, or set SCMLE_SIGNING_KEY.
    result.save("artifacts/churn", signing_key="tutorial-signing-key")
```

Open `runs/run_*/report.html`, and read
`runs/run_*/diagnostics/decision_log.json` for the evidence behind every
choice.

## 2. Make decisions explicit

```python
import pandas

import scmle.api.project
from scmle.contracts import tasks

data = pandas.read_parquet("customers.parquet")
project = scmle.api.project.Project(
    data,
    "churn",
    decision=tasks.Threshold(  # tune the cutoff to your costs
        threshold_objective="cost", false_positive_cost=1, false_negative_cost=8
    ),
    selection=tasks.Selection(max_model_mb=50, max_p95_latency_ms=20),
    leakage_policy="block",  # opt-in stop; the default "warn" labels the score
)
```

## 3. Group and time structure

```python
import pandas

import scmle.api.project

data = pandas.read_parquet("customers.parquet")
scmle.api.project.Project(
    data, "churn", group_column="customer_id"
)  # no customer in both
scmle.api.project.Project(
    data, "churn", time_column="event_time"
)  # train on the past
```

## 4. Feature graph

```python
import numpy
import pandas

import scmle.api.project
from scmle.contracts import tasks

# Daily readings of four units.
rng = numpy.random.default_rng(0)
data = pandas.DataFrame(
    {
        "unit": numpy.repeat(["u1", "u2", "u3", "u4"], 60),
        "day": numpy.tile(pandas.date_range("2026-01-01", periods=60), 4),
        "price": rng.uniform(5, 15, 240),
        "area": rng.uniform(1, 3, 240),
        "load": rng.normal(10, 2, 240),
    }
)
data["y"] = data.load + rng.normal(size=240)

graph = tasks.Derivation(
    nodes=[
        tasks.Node(name="unit_price", op="ratio", inputs=["price", "area"]),
        tasks.Node(
            name="month",
            op="datepart",
            inputs=["day"],
            params={"part": "month"},
        ),
        tasks.Node(
            name="price_clip",
            op="clip",
            inputs=["price"],
            params={"upper": 0.98},
        ),
        tasks.Node(
            name="load_prev", op="lag", inputs=["load"], params={"k": 1}
        ),
        tasks.Node(
            name="load_avg3",
            op="rolling_mean",
            inputs=["load"],
            params={"window": 3},
        ),
    ]
)
project = scmle.api.project.Project(
    data, "y", time_column="day", group_column="unit", feature_graph=graph
)
```

Row-wise nodes run inside the pipeline and learn their statistics (`clip`,
`bin`, `standardize`) on the training fold only. Ordered nodes (`lag`, `diff`,
`rolling_*`, `expanding_mean`) are computed per group in time order from the
current and earlier rows, never the target. When you score, they look back on
the recent rows per entity that the bundle stored at training time, plus the
rows the serving adapters have scored so far, so a single new record still
gets its lags. The graph is part
of the configuration hash and is stored, with its fingerprint, in the bundle.
Declare columns known only after the outcome with
`unavailable_at_prediction=[...]`, and add `embargo_rows=` for temporal
validation gaps.

## 5. Approval gates, audit trail and replay

```python
import logging

import pandas

import scmle.api.project
from scmle.pipeline import policies


def reviewer(review):  # review.gate is PLANNING or SELECTING
    logging.info(review.summary)
    return input("approve? [y/N] ").lower() == "y"


data = pandas.read_parquet("customers.parquet")
project = scmle.api.project.Project(
    data, "churn", checkpoints=(policies.Callback(reviewer),)
)
```

Rejecting at `SELECTING` stops the run in `BLOCKED` with the test set still
locked. `diagnostics/data_access.json` and the `policy.*`, `split.created`,
`selection.frozen`, `test.opened` events in `events.jsonl` are the audit trail.

Replay re-executes a finished run's configuration and compares the two
runs (pass `data=` when the run was made from a DataFrame):

```python
import logging
import pathlib

import scmle.api.project

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    finished = sorted(pathlib.Path("runs").glob("run_*"))[0]  # step 1's run
    verdict = scmle.api.project.Project.verify_replay(finished)
    logging.info(verdict.level)  # exact | comparable | different
```

## 6. Interrupted runs

A cancelled, failed or crashed run continues from its directory; a run that
completed cannot be resumed. Here a timer cancels the run to stand in for the
interruption:

```python
import threading

import scmle.api.project
from scmle.contracts import errors

if __name__ == "__main__":
    project = scmle.api.project.Project(
        "customers.parquet", "churn", time_budget=120
    )
    threading.Timer(10, project.cancel).start()  # the interruption
    try:
        project.run()
    except errors.CancelledError as error:
        stopped = error.details["run_dir"]
    scmle.api.project.Project.resume(stopped)  # finished trials are reused
```

## 7. Serve and monitor

```python
import json
import logging
import urllib.request

import pandas

import scmle.artifacts.bundle
from scmle.monitoring import monitor
from scmle.serving import server

logging.basicConfig(level=logging.INFO)
bundle = scmle.artifacts.bundle.Bundle.load(
    "artifacts/churn", signing_key="tutorial-signing-key"
)
recent = pandas.read_parquet("customers.parquet").sample(100, random_state=1)
recent_rows = recent.drop(columns="churn")

api = server.HttpServing(bundle)  # loopback only without a gateway
api.start()  # POST /predict {"records": [...]}
records = recent_rows.head(2).to_dict(orient="records")
request = urllib.request.Request(
    api.address + "/predict",
    data=json.dumps({"records": records}, default=str).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request) as response:
    logging.info(json.load(response)["predictions"])
api.stop()

report = monitor.Watcher(bundle).assess(recent_rows, outcomes=recent["churn"])
logging.info(report.recommendation)  # none / investigate / retrain
```

A registry keeps versions and moves the champion only with a signed,
independent approval. `enlist` takes the result of a run:

```python
import logging

import pandas

import scmle.api.project
from scmle.governance import registry
from scmle.operations import lifecycle

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = pandas.read_parquet("customers.parquet")
    result = scmle.api.project.Project(
        data[:400], "churn", time_budget=60
    ).run()
    fresh_labeled_rows = data[400:]  # outcomes that arrived later

    store = registry.ModelRegistryStore("registry")
    manager = lifecycle.Maintainer(store, "churn", "work")
    record = manager.enlist(result)  # registers a staged version
    approval_key = "loaded-from-your-release-secret"
    approval = registry.Sanction(
        model="churn",
        version=record.version,
        content_id=record.content_id,
        reviewer="reviewer@example.com",
        decision=registry.Verdict.APPROVED,
        reason="reviewed evaluation, limitations, and recovery evidence",
        evidence=("evidence/release-review.json",),
        approved_at="2026-10-20T00:00:00+00:00",
    ).sign(approval_key)
    store.promote(
        "churn", record.version, approval=approval, signing_key=approval_key
    )
    outcome = manager.review(fresh_labeled_rows, target="churn")
    logging.info(outcome.notes)
    # A better challenger is staged, then promoted only after a second review.
    if outcome.comparison and outcome.comparison.decision == "promote":
        candidate = store.resolve("churn", "challenger")
        candidate_approval = registry.Sanction(
            model="churn",
            version=candidate.version,
            content_id=candidate.content_id,
            reviewer="reviewer@example.com",
            decision=registry.Verdict.APPROVED,
            reason="reviewed challenger evidence independently",
            evidence=("evidence/challenger-review.json",),
            approved_at="2026-10-20T00:00:00+00:00",
        ).sign(approval_key)
        manager.elevate(candidate_approval, signing_key=approval_key)
```

Inference is strict about column names unless you state a policy:

```python
import logging

import pandas

import scmle.artifacts.bundle
from scmle.inference import mapping

logging.basicConfig(level=logging.INFO)
bundle = scmle.artifacts.bundle.Bundle.load(
    "artifacts/churn", signing_key="tutorial-signing-key"
)
new_rows = pandas.read_parquet("customers.parquet").drop(columns="churn")
new_rows = new_rows.rename(columns={"monthly_spend": "spend", "plan": "Plan"})

policy = mapping.Mapper({"spend": "monthly_spend"}, case_insensitive=True)
logging.info(bundle.predict(new_rows.head(), policy))
```

Missing labels are handled explicitly: performance is reported only on rows
whose outcome has arrived, with the labeled fraction alongside.

## 8. Command line

```bash
scmle plan  customers.parquet --target churn
scmle run   customers.parquet --target churn --workers 4 --budget 60
scmle catalog customers.parquet --label-column churn --output artifacts/catalog.json
cat customers.parquet | scmle catalog --raw-bytes
export SCMLE_SIGNING_KEY=tutorial-signing-key   # the key step 1 signed with
scmle predict artifacts/churn customers.parquet --output scored.csv   # needs a signed bundle, or --trusted for your own
```

An interrupted run continues from its directory:

```bash synopsis
scmle resume runs/<run_id>
```

## 9. Beyond the basics

Options are declared on the project; each of these is checked when it is
made, and nothing trains until `run`:

```python
import numpy
import pandas

import scmle.api.project
import scmle.evaluation.metrics
from scmle.contracts import tasks


# Your own metric (a module-level function, so trial processes can import it).
def cost(truth, predicted):
    """Eight units for a missed churner, one for a false alarm."""
    truth, predicted = numpy.asarray(truth), numpy.asarray(predicted)
    return float(numpy.mean(8 * (truth > predicted) + 1 * (truth < predicted)))


data = pandas.read_parquet("customers.parquet")

# Skip whole families, or single models.
project = scmle.api.project.Project(
    data, "churn", models=tasks.Candidates(exclude_families=["deep_learning"])
)

# Prefer the simpler model unless the best is credibly better.
selection = tasks.Selection(uncertainty="bayesian", rope=0.005)

# Series data: rolling-origin validation and delayed columns.
project = scmle.api.project.Project(
    data,
    "churn",
    time_column="event_time",
    validation="rolling",
    horizon_rows=30,
    window_rows=300,
    feature_delays={"monthly_spend": 4},
)

scmle.evaluation.metrics.register_metric(
    "business_cost", cost, greater_is_better=False
)
project = scmle.api.project.Project(data, "churn", metric="business_cost")

# Domain rules and responsibility.
project = scmle.api.project.Project(
    data,
    "churn",
    sensitive_columns=["age"],
    monotonic_constraints={"tenure": -1},
    weight_column="monthly_spend",  # rows count by weight when training
)

# Rows without a label: set aside, or self-train when it measurably helps.
project = scmle.api.project.Project(data, "churn", unlabeled="pseudo_label")
# result.bundle.prioritize(pool, count=20) -> rows to label next

# Time to an event with censored rows (target = time, event_column = 0/1).
project = scmle.api.project.Project(
    data, "tenure", task="survival", event_column="churn"
)
# result.bundle.predict_survival(rows, times=[3, 6, 12]) -> P(no event yet)

# Ordered labels: predictions come back as your labels, in order.
project = scmle.api.project.Project(
    data, "plan", task="ordinal", ordinal_order=["basic", "plus", "pro"]
)
```

After a run (here on the first 300 customers, keeping the rest held out):

```python
import logging

import pandas

import scmle.api.project
import scmle.deploy.packaging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = pandas.read_parquet("customers.parquet")
    features = data.drop(columns="churn")
    result = scmle.api.project.Project(
        data[:300], "churn", time_budget=120
    ).run()
    result.save("artifacts/early", signing_key="tutorial-signing-key")

    # Deployment. Any model: a hardened service folder (Dockerfile pinned by
    # digest at build time, locked requirements, SBOM, signed provenance).
    # Sign with signing_key= or $SCMLE_SIGNING_KEY; production refuses
    # unsigned folders.
    scmle.deploy.packaging.scaffold(
        result.bundle, "service", signing_key="tutorial-signing-key"
    )
    # scmle serve service/model --port 8000

    # What would change this decision? Only the listed columns may change.
    row = features.head(1)
    flipped = 1 - result.predict(row)[0]  # the other decision
    found = result.bundle.counterfactual(
        row, flipped, actionable=["monthly_spend", "plan"], background=features
    )

    # Reliability checks, explanations and calibrated uncertainty.
    rows, background = features[300:305], features[:300]
    logging.info(result.evaluation["health"])  # healthy / overfit / underfit
    logging.info(result.diagnostics["robustness"])  # score under degraded input
    logging.info(result.diagnostics["slice_performance"])  # weakest subgroups
    logging.info(
        result.diagnostics["adversarial_validation"]
    )  # dev vs test shift
    logging.info(result.explain(rows, background))  # per-prediction attribution
    logging.info(
        result.bundle.predict_set(rows, coverage=0.9)
    )  # classification
```

A regression model gives prediction intervals:

```python
import logging

import pandas

import scmle.api.project

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = pandas.read_parquet("customers.parquet").drop(columns="churn")
    result = scmle.api.project.Project(
        data, "monthly_spend", time_budget=120
    ).run()
    rows = data.drop(columns="monthly_spend").head(5)
    logging.info(result.bundle.predict_interval(rows, coverage=0.9))
    # Adaptive width:
    logging.info(result.bundle.predict_interval(rows, 0.9, method="quantile"))
```

ONNX where the model allows it (forests, linear, SVM, LightGBM, XGBoost;
HistGradientBoosting is not expressible and goes through the service). It
needs the `onnx` extra:

```bash
pip install "scmle[onnx] @ git+https://github.com/sachncs/scmle"
```

The export is verified on at least 299 rows held out from training:

```python
import pandas

import scmle.artifacts.bundle

bundle = scmle.artifacts.bundle.Bundle.load(
    "artifacts/early", signing_key="tutorial-signing-key"
)
held_out = pandas.read_parquet("customers.parquet")[300:].drop(columns="churn")
bundle.export_onnx("model.onnx", check=held_out)
```

Register any scikit-learn-compatible estimator in one call; the class must be
defined at module level so trial processes can import it. Registered models
default to `cost=3`, and budgets under 300 s only search cheap models, so pass
`cost=1` for a fast one or raise `time_budget` (see
[`examples/custom_estimator.py`](../examples/custom_estimator.py)):

```python
from sklearn.naive_bayes import GaussianNB

import scmle.models.custom

scmle.models.custom.register_estimator(
    "gaussian_nb", classifier=GaussianNB, cost=1
)
```

## 10. More runnable examples

[`examples/`](../examples/README.md) holds self-contained scripts for signed
bundles, drift monitoring, custom estimators, regression intervals, temporal
validation and approval gates. [`ARTIFACTS.md`](ARTIFACTS.md) explains every
file a run writes, and [`CLI.md`](CLI.md) the shell interface.

## 11. Extending

Subclass `Model` and pass `comply`, then enrol it with
`plugins.REGISTRY.enrol(plugins.Kind.ESTIMATORS, name, adapter)` (from
`scmle.runtime import plugins`) or advertise it through the `scmle.models`
entry point. Metrics, readers, detectors, stages and transformers are enrolled
the same way under their own `plugins.Kind` and entry-point group. See
`docs/ARCHITECTURE.md` for every extension point.
