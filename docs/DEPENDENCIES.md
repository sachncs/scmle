# Dependency policy

* **Python:** CPython 3.14 only, both the standard and the free-threaded
  build (the suite passes on both).
* **Base install** is deliberately light: numpy, pandas, pyarrow, pydantic,
  scikit-learn, joblib, threadpoolctl. No GPU, service, database server or
  cloud account is required.
* **Optional extras** are never imported unless used:

| Extra | Adds | Notes |
| --- | --- | --- |
| `boosting` | LightGBM, XGBoost | adapters are rejected with a reason when absent |
| `optimization` | Optuna | `SearchSpec(optimizer="optuna")` |
| `sampling` | imbalanced-learn | SMOTE, Borderline-SMOTE, ADASYN strategies |
| `polars` | Polars | Polars frames and lazyframes as input (Arrow tables need only pyarrow) |
| `tracking` | MLflow | `MlflowExperimentStore` |
| `distributed` | Ray | `ResourceSpec(executor="ray")`; no free-threaded (3.14t) wheel yet |
| `catboost` | CatBoost | no free-threaded (3.14t) wheel yet; kept out of `boosting` for that reason |

* **Lower bounds are tested**, not guessed: CI resolves
  `--resolution lowest-direct` and runs the unit, property and leakage suites
  on those versions (pandas 3.0, scikit-learn 1.8, numpy 2.3, ...).
* **Upper bounds** are not pinned in the library; `uv.lock` records the
  combination that CI exercises.
* **Audit:** `pip-audit` runs in CI; a vulnerable release fails the build.
* **Licenses:** the project is Apache-2.0. All base dependencies are
  permissively licensed (BSD, Apache-2.0, MIT); review before adding any
  copyleft package to an extra.
