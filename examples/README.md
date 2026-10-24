# Examples

Each file is self-contained, uses synthetic data, runs in about a minute on a
laptop and logs what it does. Install with
`pip install "scmle[boosting]"` (or `uv sync --extra boosting`) and run, for example:

```bash
python examples/quickstart.py
```

| Example | Shows |
| --- | --- |
| `quickstart.py` | Plan, run, read the leaderboard and predict. Start here. |
| `signed_bundle.py` | Save a signed bundle, reload it, and what a wrong key does. |
| `monitor_drift.py` | Assess a healthy and a drifted batch with delayed labels. |
| `custom_estimator.py` | Register your own scikit-learn estimator in the search. |
| `regression_intervals.py` | Regression with conformal intervals; checks stated against observed coverage. |
| `temporal_validation.py` | Time-ordered data: temporal validation, group column, lag and date features. |
| `approval_gate.py` | A human checkpoint that blocks a run before the test set opens. |

Runs write to `runs/` in the current directory (`runs/run_*/report.html` is the
human-readable summary; see [`docs/ARTIFACTS.md`](../docs/ARTIFACTS.md)).

## Command line

The same workflow without Python code:

```bash
python - <<'PY'
import pandas as pd, numpy as np
rng = np.random.default_rng(0)
x = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
x["y"] = (x.a - x.b + rng.normal(scale=0.5, size=500) > 0).astype(int)
x.to_csv("demo.csv", index=False)
PY
scmle plan demo.csv --target y
scmle run  demo.csv --target y --time-budget 60
```

Full reference: [`docs/CLI.md`](../docs/CLI.md).
