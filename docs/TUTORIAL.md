# Tutorial

## 1. Plan, run, predict

```python
import mml

project = mml.MLProject("customers.parquet", target="churn", metric="roc_auc")
print(project.plan().summary())  # nothing is trained or written
result = project.run()
print(result.leaderboard()[["model_name", "strategy", "status", "score_mean"]])
print(result.feature_importance().head())
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

## 4. Approval gates, audit trail and replay

```python
from mml.core import checkpoints


def reviewer(review):  # review.gate is PLANNING or SELECTING
    print(review.summary)
    return input("approve? [y/N] ").lower() == "y"


project = mml.MLProject(
    data, "churn", checkpoints=(checkpoints.CallbackCheckpoint(reviewer),)
)
```

Rejecting at `SELECTING` stops the run in `BLOCKED` with the test set still
locked. `diagnostics/data_access.json` and the `policy.*`, `split.created`,
`selection.frozen`, `test.opened` events in `events.jsonl` are the audit trail.

```python
verdict = mml.MLProject.verify_replay("runs/run_20261015_001", data=data)
print(verdict.level)  # exact | comparable | different
```

## 5. Interrupted runs

```python
mml.MLProject.resume("runs/run_20261015_001")  # finished trials are reused
```

## 6. Serve and monitor

```python
from mml.deployment import lifecycle, registry, serving
from mml.monitoring import monitor

bundle = mml.ModelBundle.load("artifacts/churn", signing_key="...")
api = serving.HttpServing(bundle)
api.start()  # POST /predict {"records": [...]}

report = monitor.ModelMonitor(bundle).assess(recent_rows, outcomes=known_labels)
print(report.recommendation)  # none / investigate / retrain

store = registry.ModelRegistryStore("registry")
manager = lifecycle.LifecycleManager(store, "churn", "work")
manager.register_run(result)  # becomes the champion
manager.review(
    fresh_labeled_rows, target="churn"
)  # retrain + promote if clearly better
```

Inference is strict about column names unless you state a policy:

```python
policy = mml.artifacts.bundle.ColumnMapping(
    {"cust_age": "age"}, case_insensitive=True
)
bundle.predict(new_rows, policy)
```

Missing labels are handled explicitly: performance is reported only on rows
whose outcome has arrived, with the labeled fraction alongside.

## 7. Command line

```bash
magicml plan  data.csv --target churn
magicml run   data.csv --target churn --workers 4 --time-budget 600
magicml resume runs/run_20261015_001
magicml predict artifacts/churn new.csv --output scored.csv   # signed bundles
```

## 8. Extending

Subclass `ModelAdapter` and pass `check_conformance`, or register it through
the `mml.models` entry point. See `docs/ARCHITECTURE.md` for every extension
point.
