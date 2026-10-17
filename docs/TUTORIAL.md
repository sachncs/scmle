# Tutorial

## 1. Plan, run, predict

```python
import logging

import mml

logging.basicConfig(level=logging.INFO)
project = mml.MLProject("customers.parquet", target="churn", metric="roc_auc")
logging.info(project.plan().summary())  # nothing is trained or written
result = project.run()
logging.info(
    "\n%s", result.leaderboard()[["model_name", "status", "score_mean"]]
)
logging.info("\n%s", result.feature_importance().head())
result.save("artifacts/churn", signing_key="...")  # or set MAGICML_SIGNING_KEY
```

Open `runs/run_*/report.html`, and read
`runs/run_*/diagnostics/decision_log.json` for the evidence behind every
choice.

## 2. Make decisions explicit

```python
from mml.contracts import tasks

project = mml.MLProject(
    data,
    "churn",
    decision=tasks.DecisionSpec(  # tune the cutoff to your costs
        threshold_objective="cost", false_positive_cost=1, false_negative_cost=8
    ),
    selection=tasks.SelectionSpec(max_model_mb=50, max_p95_latency_ms=20),
    leakage_policy="block",  # refuse suspicious features
)
```

## 3. Group and time structure

```python
mml.MLProject(
    data, "default", group_column="customer_id"
)  # no customer in both
mml.MLProject(data, "default", time_column="event_time")  # train on the past
```

## 4. Feature graph

```python
from mml.contracts import tasks

graph = tasks.FeatureGraphSpec(
    nodes=[
        tasks.FeatureNode(
            name="unit_price", op="ratio", inputs=["price", "area"]
        ),
        tasks.FeatureNode(
            name="month",
            op="datepart",
            inputs=["when"],
            params={"part": "month"},
        ),
        tasks.FeatureNode(
            name="price_clip",
            op="clip",
            inputs=["price"],
            params={"upper": 0.98},
        ),
        tasks.FeatureNode(
            name="load_prev", op="lag", inputs=["load"], params={"k": 1}
        ),
        tasks.FeatureNode(
            name="load_avg3",
            op="rolling_mean",
            inputs=["load"],
            params={"window": 3},
        ),
    ]
)
project = mml.MLProject(
    data, "y", time_column="day", group_column="unit", feature_graph=graph
)
```

Row-wise nodes run inside the pipeline and learn their statistics (`clip`,
`bin`, `standardize`) on the training fold only. Ordered nodes (`lag`, `diff`,
`rolling_*`, `expanding_mean`) are computed per group in time order from the
current and earlier rows, never the target, and are recomputed on the batch
you score, so that batch must carry each entity's history. The graph is part
of the configuration hash and is stored, with its fingerprint, in the bundle.
Declare columns known only after the outcome with
`unavailable_at_prediction=[...]`, and add `embargo_rows=` for temporal
validation gaps.

## 5. Approval gates, audit trail and replay

```python
from mml.core import policies


def reviewer(review):  # review.gate is PLANNING or SELECTING
    logging.info(review.summary)
    return input("approve? [y/N] ").lower() == "y"


project = mml.MLProject(
    data, "churn", checkpoints=(policies.CallbackCheckpoint(reviewer),)
)
```

Rejecting at `SELECTING` stops the run in `BLOCKED` with the test set still
locked. `diagnostics/data_access.json` and the `policy.*`, `split.created`,
`selection.frozen`, `test.opened` events in `events.jsonl` are the audit trail.

```python
verdict = mml.MLProject.verify_replay("runs/run_20261015_001", data=data)
logging.info(verdict.level)  # exact | comparable | different
```

## 6. Interrupted runs

```python
mml.MLProject.resume("runs/run_20261015_001")  # finished trials are reused
```

## 7. Serve and monitor

```python
from mml.governance import lifecycle, registry
from mml.inference import serving
from mml.monitoring import monitor

bundle = mml.ModelBundle.load("artifacts/churn", signing_key="...")
api = serving.HttpServing(bundle)
api.start()  # POST /predict {"records": [...]}

report = monitor.ModelMonitor(bundle).assess(recent_rows, outcomes=known_labels)
logging.info(report.recommendation)  # none / investigate / retrain

store = registry.ModelRegistryStore("registry")
manager = lifecycle.LifecycleManager(store, "churn", "work")
manager.register_run(result)  # becomes the champion
manager.review(
    fresh_labeled_rows, target="churn"
)  # retrain + promote if clearly better
```

Inference is strict about column names unless you state a policy:

```python
policy = mml.inference.mapping.ColumnMapping(
    {"cust_age": "age"}, case_insensitive=True
)
bundle.predict(new_rows, policy)
```

Missing labels are handled explicitly: performance is reported only on rows
whose outcome has arrived, with the labeled fraction alongside.

## 8. Command line

```bash
magicml plan  data.csv --target churn
magicml run   data.csv --target churn --workers 4 --time-budget 600
magicml resume runs/run_20261015_001
magicml predict artifacts/churn new.csv --output scored.csv   # signed bundles
```

## 9. Beyond the basics

```python
from mml.contracts import tasks

# Skip whole families, or single models.
project = mml.MLProject(
    data, "churn", models=tasks.ModelSpec(exclude_families=["deep_learning"])
)

# Prefer the simpler model unless the best is credibly better.
selection = tasks.SelectionSpec(uncertainty="bayesian", rope=0.005)

# Series data: rolling-origin validation and delayed columns.
project = mml.MLProject(
    data,
    "demand",
    time_column="week",
    validation="rolling",
    horizon_rows=52,
    window_rows=520,
    feature_delays={"survey": 4},
)

# Your own metric (a module-level function, so trial processes can import it).
mml.register_metric("business_cost", my_module.cost, greater_is_better=False)
project = mml.MLProject(data, "churn", metric="business_cost")

# Domain rules and responsibility.
project = mml.MLProject(
    data,
    "default",
    sensitive_columns=["sex"],
    monotonic_constraints={"missed_payments": 1},
    weight_column="survey_weight",  # rows count by weight when training
)

# Rows without a label: set aside, or self-train when it measurably helps.
project = mml.MLProject(data, "churn", unlabeled="pseudo_label")
# result.bundle.rank_for_labeling(pool, count=20) -> rows to label next

# After a run: reliability checks, explanations and calibrated uncertainty.
result.evaluation["fit_health"]  # healthy / overfit / underfit
result.diagnostics["robustness"]  # score under degraded input
result.diagnostics["slice_performance"]  # weakest subgroups
result.diagnostics["adversarial_validation"]  # development vs test shift
result.explain(rows, background)  # per-prediction attribution
result.bundle.predict_interval(rows, coverage=0.9)  # regression
result.bundle.predict_interval(rows, 0.9, method="quantile")  # adaptive width
result.bundle.predict_set(rows, coverage=0.9)  # classification
```

## 10. Extending

Subclass `ModelAdapter` and pass `check_conformance`, or register it through
the `mml.models` entry point. See `docs/ARCHITECTURE.md` for every extension
point.
