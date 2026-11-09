# Dependency policy

* **Python:** CPython 3.11 or newer, tested on 3.11, 3.12, 3.13 and 3.14,
  including the free-threaded 3.14 build.
* **Base install** is deliberately light: numpy, pandas, pyarrow, pydantic,
  scikit-learn, joblib, threadpoolctl. No GPU, service, database server or
  hosted account is required.
* **Optional extras** are never imported unless used:

| Extra | Adds | Notes |
| --- | --- | --- |
| `boosting` | LightGBM, XGBoost | adapters are rejected with a reason when absent |
| `optimization` | Optuna | `SearchSpec(optimizer="optuna")` |
| `sampling` | imbalanced-learn | SMOTE, Borderline-SMOTE, ADASYN strategies |
| `polars` | Polars | Polars frames and lazyframes as input (Arrow tables need only pyarrow) |
| `spreadsheets` | openpyxl | `.xlsx` and `.xlsm` input; first worksheet is read |
| `tracking` | MLflow | `MlflowExperimentStore` |
| `distributed` | Ray | `ResourceSpec(executor="ray")`; no free-threaded (3.14t) wheel yet |
| `catboost` | CatBoost | no free-threaded (3.14t) wheel yet; kept out of `boosting` for that reason |
| `deeplearning` | PyTorch, Keras 3 (PyTorch backend) | model families run in a separate worker process; CPU wheel on Linux via `uv` |
| `tabpfn` | TabPFN | needs an accepted Prior Labs license and `TABPFN_TOKEN`; extension-only, not in `all` |
| `embeddings` | sentence-transformers | downloads a model on first use; not in `all` |
| `onnx` | skl2onnx, onnxruntime, onnxmltools | for `scmle export-onnx` |
| `tracking` | MLflow | `MlflowExperimentStore` |
| `multimodal` | Pillow | image modality |
| `benchmarks` | huggingface_hub | Hugging Face benchmark suite |

* **Lower bounds are tested**, not guessed: CI resolves
  `--resolution lowest-direct` and runs the unit, property and leakage suites
  on those versions (pandas 3.0, scikit-learn 1.8, numpy 2.3, ...).
* **Upper bounds** are not pinned in the library; `uv.lock` records the
  combination that CI exercises.
* **No cloud SDK.** No cloud-provider SDK (for example boto3, google-cloud
  or azure packages) and no SDK for a hosted-compute service is a dependency,
  a base requirement or an optional extra, and none may be added. Code that
  talks to a hosted service belongs to the user's own adapter, written against
  the interfaces the package ships.
* **Audit:** `pip-audit` runs in CI; a vulnerable release fails the build.
* **Licenses:** the project is Apache-2.0. CI traverses the installed base
  runtime dependency closure and publishes `dependency-licenses.json`. A
  missing license record or copyleft license (`GPL`, `LGPL`, `AGPL`, `SSPL`,
  `BUSL`, or Elastic) fails the security job. Optional extras must be audited
  in the environment where they are enabled before a release claim includes
  them; review before adding any copyleft package to an extra.
