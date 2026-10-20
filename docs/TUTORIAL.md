# Tutorial

## 1. Plan, run, predict

```python
import logging

import scml

logging.basicConfig(level=logging.INFO)
project = scml.MLProject("customers.parquet", target="churn", metric="roc_auc")
logging.info(project.plan().summary())  # nothing is trained or written
result = project.run()
logging.info(
    "\n%s", result.leaderboard()[["model_name", "status", "score_mean"]]
)
logging.info("\n%s", result.feature_importance().head())
result.save("artifacts/churn", signing_key="...")  # or set SCML_SIGNING_KEY
```

Open `runs/run_*/report.html`, and read
`runs/run_*/diagnostics/decision_log.json` for the evidence behind every
choice.

## 2. Make decisions explicit

```python
from scml.contracts import tasks

project = scml.MLProject(
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
scml.MLProject(
    data, "default", group_column="customer_id"
)  # no customer in both
scml.MLProject(data, "default", time_column="event_time")  # train on the past
```

## 4. Feature graph

```python
from scml.contracts import tasks

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
project = scml.MLProject(
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
from scml.core import policies


def reviewer(review):  # review.gate is PLANNING or SELECTING
    logging.info(review.summary)
    return input("approve? [y/N] ").lower() == "y"


project = scml.MLProject(
    data, "churn", checkpoints=(policies.CallbackCheckpoint(reviewer),)
)
```

Rejecting at `SELECTING` stops the run in `BLOCKED` with the test set still
locked. `diagnostics/data_access.json` and the `policy.*`, `split.created`,
`selection.frozen`, `test.opened` events in `events.jsonl` are the audit trail.

```python
verdict = scml.MLProject.verify_replay("runs/run_20261015_001", data=data)
logging.info(verdict.level)  # exact | comparable | different
```

## 6. Interrupted runs

```python
scml.MLProject.resume("runs/run_20261015_001")  # finished trials are reused
```

## 7. Serve and monitor

```python
from scml.governance import lifecycle, registry
from scml.inference import serving
from scml.monitoring import monitor

bundle = scml.ModelBundle.load("artifacts/churn", signing_key="...")
api = serving.HttpServing(bundle)
api.start()  # POST /predict {"records": [...]}

report = monitor.ModelMonitor(bundle).assess(recent_rows, outcomes=known_labels)
logging.info(report.recommendation)  # none / investigate / retrain

store = registry.ModelRegistryStore("registry")
manager = lifecycle.LifecycleManager(store, "churn", "work")
record = manager.register_run(result)  # registers a staged version
approval_key = "loaded-from-your-release-secret"
approval = registry.PromotionApproval(
    model="churn",
    version=record.version,
    content_id=record.content_id,
    reviewer="reviewer@example.com",
    decision=registry.PromotionDecision.APPROVED,
    reason="reviewed evaluation, limitations, and recovery evidence",
    evidence=("evidence/release-review.json",),
    approved_at="2026-10-20T00:00:00+00:00",
).sign(approval_key)
store.promote(
    "churn", record.version, approval=approval, signing_key=approval_key
)
outcome = manager.review(fresh_labeled_rows, target="churn")
# A better challenger is staged, then promoted only after a second review.
if outcome.comparison and outcome.comparison.decision == "promote":
    candidate = store.resolve("churn", "challenger")
    candidate_approval = registry.PromotionApproval(
        model="churn",
        version=candidate.version,
        content_id=candidate.content_id,
        reviewer="reviewer@example.com",
        decision=registry.PromotionDecision.APPROVED,
        reason="reviewed challenger evidence independently",
        evidence=("evidence/challenger-review.json",),
        approved_at="2026-10-20T00:00:00+00:00",
    ).sign(approval_key)
    manager.promote_challenger(candidate_approval, signing_key=approval_key)
```

Inference is strict about column names unless you state a policy:

```python
policy = scml.inference.mapping.ColumnMapping(
    {"cust_age": "age"}, case_insensitive=True
)
bundle.predict(new_rows, policy)
```

Missing labels are handled explicitly: performance is reported only on rows
whose outcome has arrived, with the labeled fraction alongside.

## 8. Command line

```bash
scml plan  data.csv --target churn
scml run   data.csv --target churn --workers 4 --time-budget 600
scml resume runs/run_20261015_001
scml predict artifacts/churn new.csv --output scored.csv   # signed bundles
```

## 9. Beyond the basics

```python
from scml.contracts import tasks

# Skip whole families, or single models.
project = scml.MLProject(
    data, "churn", models=tasks.ModelSpec(exclude_families=["deep_learning"])
)

# Prefer the simpler model unless the best is credibly better.
selection = tasks.SelectionSpec(uncertainty="bayesian", rope=0.005)

# Series data: rolling-origin validation and delayed columns.
project = scml.MLProject(
    data,
    "demand",
    time_column="week",
    validation="rolling",
    horizon_rows=52,
    window_rows=520,
    feature_delays={"survey": 4},
)

# Your own metric (a module-level function, so trial processes can import it).
scml.register_metric("business_cost", my_module.cost, greater_is_better=False)
project = scml.MLProject(data, "churn", metric="business_cost")

# Domain rules and responsibility.
project = scml.MLProject(
    data,
    "default",
    sensitive_columns=["sex"],
    monotonic_constraints={"missed_payments": 1},
    weight_column="survey_weight",  # rows count by weight when training
)

# Rows without a label: set aside, or self-train when it measurably helps.
project = scml.MLProject(data, "churn", unlabeled="pseudo_label")
# result.bundle.rank_for_labeling(pool, count=20) -> rows to label next

# Time to an event with censored rows (target = time, event_column = 0/1).
project = scml.MLProject(
    data, "months_to_churn", task="survival", event_column="churned"
)
# result.bundle.predict_survival(rows, times=[3, 6, 12]) -> P(no event yet)

# Deployment. Any model: a ready-to-run service folder with a Dockerfile.
result.bundle.export_service("service")
# ONNX where the model allows it (forests, linear, SVM, LightGBM, XGBoost;
# HistGradientBoosting is not expressible and goes through the service).
result.bundle.export_onnx("model.onnx", check=sample_rows)
# scml serve runs/<id>/model --trusted --port 8000

# Ordered labels: predictions come back as your labels, in order.
project = scml.MLProject(
    data, "rating", task="ordinal", ordinal_order=["low", "medium", "high"]
)

# What would change this decision? Only the listed columns may change.
found = result.bundle.counterfactual(
    row, "approved", actionable=["income", "city"], background=features
)

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

Register any scikit-learn-compatible estimator in one call; the class must be
defined at module level so trial processes can import it:

```python
scml.register_estimator("gaussian_nb", classifier=GaussianNB)
```

## 10. Extending

Subclass `ModelAdapter` and pass `check_conformance`, or register it through
the `scml.models` entry point. See `docs/ARCHITECTURE.md` for every extension
point.
