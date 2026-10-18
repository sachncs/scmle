# Curriculum: the eight modelling stages and where each lives

Machine learning modelling is an end-to-end statistical and engineering
process. This page follows the eight stages of that process, states the
mathematics each rests on, and points at the module that implements it and the
test that proves it. Every `python` block below is executed by
`tests/unit/test_curriculum.py`, so the examples cannot rot.

The framework is supervised. Clustering (silhouette, Davies-Bouldin) is out of
scope by design, as are object detection and segmentation.

## Stage 1: problem formulation and statistical assumptions

A prediction problem fixes a target `Y`, features `X` available *at prediction
time*, a loss `L`, and the population `P(X, Y)` the model will face. The goal is
to minimise the expected risk `R(f) = E[L(Y, f(X))]` under that population, and
a success criterion states what value of the metric is good enough to ship.

* Modules: `scml/contracts/tasks.py` (`ProblemSpec`, task types, success
  criteria), `scml/core/stages.py` (criteria check),
  `scml/evaluation/criteria.py`.
* Tests: `tests/integration/test_problem_spec.py`.

```python
import scml
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
data = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
data["y"] = (data.a + 0.5 * data.b + rng.normal(0, 0.7, 400) > 0).astype(int)
project = scml.MLProject(data, "y", metric="roc_auc", time_budget=60)
print(project.plan().summary())
```

## Stage 2: sampling, curation and exploratory analysis

A sample is useful only if it is representative of the population.
Distinguishability of sample `S` from reference `R` is measured by the
out-of-fold AUC of a classifier trained to tell them apart: 0.5 means
indistinguishable, 1.0 means the sample is not the population. Label noise is
found by confident learning: row `i` is a suspect when the out-of-fold
probability of its given label is below the class mean `t_k = mean p(k | x)` over
rows labelled `k` while another class `j` has `p(j | x) >= t_j`.

* Modules: `scml/data/profiling/profiler.py` (distributions, quantiles,
  missingness), `scml/data/profiling/missingness.py` (MCAR/MAR/MNAR signal),
  `scml/data/profiling/label_noise.py`,
  `scml/data/profiling/representativeness.py`, `scml/data/splitting/duplicates.py`
  (deduplication and leakage), `scml/evaluation/shift.py` (covariate shift).
* Tests: `tests/unit/test_data_quality_checks.py`,
  `tests/integration/test_missingness_plan.py`.

```python
import numpy as np
import pandas as pd
from scml.data.profiling import label_noise, representativeness

rng = np.random.default_rng(0)
features = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
labels = pd.Series((features.a + features.b > 0).astype(int))
labels.iloc[::25] = 1 - labels.iloc[::25]
report = label_noise.find_label_noise(features, labels)
print(len(report.suspects), "suspect labels")

reference = features.sample(300, random_state=1).reset_index(drop=True) + 1.0
check = representativeness.compare_to_reference(features, reference)
print(check.auc, check.representative, check.gaps[0].column)
```

## Stage 3: partitioning and feature engineering

Every transformation is fitted on training rows only, inside the cross-validation
loop; otherwise the held-out rows leak into the statistics (preprocessing
leakage). Splits follow the prediction setting: stratified for classes, grouped
for dependent rows, temporal for the future. The locked test set is touched once.

* Modules: `scml/data/splitting/splitters.py`, `scml/features/feature_graph.py` (encoding,
  scaling, automatic construction and L1 selection),
  `scml/data/curation/preprocessing.py`.
* Tests: `tests/unit/test_splitting.py`, `tests/leakage/test_leakage.py`.

```python
import numpy as np
import pandas as pd
import scml

rng = np.random.default_rng(1)
data = pd.DataFrame(
    {"group": rng.integers(0, 40, 600), "x": rng.normal(size=600)}
)
data["y"] = (data.x + rng.normal(0, 0.8, 600) > 0).astype(int)
project = scml.MLProject(
    data, "y", metric="roc_auc", time_budget=60, group_column="group"
)
print(project.plan().summary())
```

## Stage 4: model selection and architecture design

Candidates range from baselines through linear models, trees, boosting and
neural networks. Capacity trades representational power against variance, so a
baseline always runs first and every candidate must beat it with statistical
support, not a lucky split. Selection compares paired fold scores and treats
overlapping intervals as ties.

* Modules: `scml/models/registry.py`, `scml/models/sklearn/`,
  `scml/models/boosting/`, `scml/models/deeplearning/` (MLP, residual,
  FT-Transformer), `scml/evaluation/selection.py`,
  `scml/evaluation/comparison.py`.
* Tests: `tests/integration/test_register_estimator.py`,
  `tests/unit/test_selection.py`.

```python
import numpy as np
import pandas as pd
import scml

rng = np.random.default_rng(2)
data = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
data["y"] = (data.a * data.b > 0).astype(int)
result = scml.MLProject(data, "y", metric="roc_auc", time_budget=90).run()
print(result.leaderboard()[["model_name", "status", "score_mean"]])
```

## Stage 5: training and optimisation

Parameters solve `theta* = argmin (1/n) sum_i L(y_i, f_theta(x_i)) + lambda R(theta)`.
Gradient methods update `theta <- theta - eta g`; momentum, RMSProp and Adam
rescale `g`, and AdamW decouples weight decay. Warm-up and cosine schedules
shape `eta`; accumulation emulates a larger batch; clipping bounds `||g||`.
Optimiser choice decides how parameters are estimated, architecture decides
what can be represented. A training-health report flags unstable loss,
plateaus and divergence.

* Modules: `scml/models/deeplearning/` (optimizers, losses, accumulation,
  health), `scml/models/deeplearning/keras_engine.py`.
* Tests: `tests/unit/test_optimizers.py`.

```python
import torch

weights = torch.zeros(2, requires_grad=True)
optimiser = torch.optim.AdamW([weights], lr=0.1, weight_decay=0.01)
target = torch.tensor([1.0, -2.0])
for step in range(200):
    optimiser.zero_grad()
    loss = ((weights - target) ** 2).sum()
    loss.backward()
    optimiser.step()
print(weights.detach().numpy().round(2))
```

## Stage 6: regularisation and generalisation

The squared-error risk decomposes as
`E[(Y - f_hat(X))^2] = Bias^2 + Variance + sigma^2`. Regularisation (L1, L2,
dropout, early stopping, ensembling) accepts more bias for less variance.
The bias-variance report refits a model on bootstrap resamples, takes the
spread of its predictions as variance, and estimates `sigma^2` from nearest
neighbours. The bootstrap understates variance, so read the verdict as
direction, not as an exact split. Learning curves show whether more rows would
help.

* Sensitivity: `scml/evaluation/sensitivity.py::analyze` separates how much
  the score moves with the training seed, with a bootstrap resample of the
  training rows, with the choice of split, and with noise added to the
  features. On scikit-learn's breast-cancer data (ROC AUC, 10 repeats) the
  standard deviations were 0.000 to 0.006, so score differences under about
  0.01 between models there are within noise.
* Modules: `scml/evaluation/bias_variance.py`,
  `scml/evaluation/learning_curve.py`, `scml/evaluation/fit_health.py`.
* Tests: `tests/unit/test_bias_variance.py`.

```python
import numpy as np
import pandas as pd
from sklearn import linear_model, tree
from scml.evaluation import bias_variance

rng = np.random.default_rng(3)
features = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
target = pd.Series(2 * features.a + rng.normal(0, 0.5, 300))
for build in (
    lambda: tree.DecisionTreeRegressor(),
    lambda: linear_model.Ridge(alpha=1.0),
):
    part = bias_variance.decompose(
        build, features, target, False, replicates=10
    )
    print(type(build()).__name__, part.verdict)
```

## Stage 7: evaluation, diagnostics and model selection

Metrics follow the task: MAE/RMSE for regression, PR-AUC and precision at k for
imbalance, log loss and Brier score for probabilities (`ECE = sum_b (n_b/n)
|acc_b - conf_b|` for calibration), NDCG/MAP/MRR for ranking, MASE with rolling
backtests for forecasting, concordance for survival. Uncertainty comes from
repeated and nested cross-validation, paired bootstrap intervals and conformal
prediction (coverage `>= 1 - alpha`). Error analysis slices by subgroup, stress
tests robustness and tests fairness.

* Modules: `scml/evaluation/` (`metrics.py`, `nested.py`, `conformal.py`,
  `reliability.py`, `slices.py`, `robustness.py`, `fairness.py`,
  `ranking.py`, `backtest.py`, `survival.py`), `scml/optimization/optimizers.py` (grid, random,
  Bayesian, successive halving, Hyperband, Pareto selection).
* Tests: `tests/integration/test_evaluation_extras.py`,
  `tests/integration/test_tuning_and_pareto.py`.

```python
import numpy as np
import pandas as pd
import scml

rng = np.random.default_rng(4)
data = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
data["y"] = (data.a + rng.normal(0, 0.8, 500) > 0).astype(int)
result = scml.MLProject(data, "y", metric="roc_auc", time_budget=90).run()
print(result.summary())
```

## Stage 8: deployment, monitoring and retraining

In production the joint distribution moves: covariate shift (`P(X)`), prior
shift (`P(Y)`) and concept drift (`P(Y|X)`). Feature drift is scored with the
population stability index `PSI = sum_b (a_b - e_b) ln(a_b / e_b)`. A challenger
runs in shadow on live traffic, an A/B test compares arms with an independent
bootstrap interval, and a promotion that regresses is rolled back to the
`previous` registry alias. Retraining is policy-driven with an audit trail.

* Modules: `scml/monitoring/monitor.py`, `scml/monitoring/shadow.py`,
  `scml/monitoring/abtest.py`, `scml/governance/lifecycle.py`
  (`check_rollback`, retraining policy), `scml/governance/registry.py`,
  `scml/inference/` (ONNX export, service package), `scml/artifacts/bundle.py`.
* Tests: `tests/unit/test_lifecycle_extras.py`,
  `tests/integration/test_serve_and_export.py`.

```python
import numpy as np
import pandas as pd
import scml
from scml.monitoring import monitor

rng = np.random.default_rng(5)
data = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
data["y"] = (data.a + rng.normal(0, 0.8, 500) > 0).astype(int)
result = scml.MLProject(data, "y", metric="roc_auc", time_budget=90).run()
live = data.drop(columns="y").copy()
live["a"] = live["a"] + 2.0
report = monitor.ModelMonitor(result.load_best_model()).drift(live)
print(report)
```
