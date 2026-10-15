# magicml

Autonomous, leakage-safe ML lifecycle framework for tabular data.
Python 3.14 only. The architecture is specified in [`product.md`](product.md);
the importable package is `mml`.

```python
import mml

project = mml.MLProject("customers.parquet", target="churn", metric="roc_auc")
print(project.plan().summary())      # inspect before any training
result = project.run()               # profile, split, search, test, package
print(result.leaderboard())
result.save("artifacts/churn_model")
result.predict("new_customers.parquet")
```

CLI: `magicml plan|run|predict` (see `magicml --help`).

## What v0.1 guarantees

- **Leakage-safe by construction.** The test partition sits behind
  `LockedTestSet`, opened once, after selection. Preprocessing is part of the
  candidate pipeline and is fitted inside each cross-validation fold.
  Group and temporal data are split accordingly, never randomly.
- **Auditable.** Every run writes a directory (`config.json`, split manifest,
  `events.jsonl`, trials, leaderboard, evaluation, decision log, `report.html`,
  model bundle) and every state change is an event.
- **Conservative.** Ambiguous targets, unparseable time columns, too few
  groups or minority examples put the run in `BLOCKED` instead of guessing.
- **Safe artifacts.** Bundles carry a schema and SHA-256 checksums and verify
  dependency compatibility on load. They are pickle-based: load trusted
  bundles only.

## Layout

| Package | Responsibility |
| --- | --- |
| `mml.contracts` | Typed contracts (pydantic config, frozen dataclasses, ABCs) |
| `mml.data` | Readers, column kinds, profiling, split strategies, pipelines |
| `mml.models` | `ModelAdapter` hierarchy, registry, conformance checks |
| `mml.intelligence` | `Planner` and composable `PlanningRule`s |
| `mml.evaluation` | Metric catalog, selection policies |
| `mml.core` | State machine, `Stage` hierarchy, trial runner, orchestrator |
| `mml.artifacts` | Model bundle, run store, report |
| `mml.api` | `MLProject`, `RunResult` |

Extension points are polymorphic: add a `TableReader`, `ColumnKind`,
`SplitStrategy`, `ModelAdapter`, `PlanningRule`, `SelectionPolicy` or `Stage`
subclass; no core edits are needed. Third-party adapters can also register via
the `mml.models` entry-point group and must pass
`mml.models.conformance.check_conformance`.

## Conventions

Google Python style (80 columns, Google docstrings, module imports), enforced
by `ruff` and `mypy`. No underscore-prefixed names, enforced by
`tests/unit/test_conventions.py`.

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy mml
uv run pytest
```

## Not yet implemented

Hyperparameter search, boosting extras, imbalance strategies, feature
engineering, explainability, monitoring, deployment and distributed execution
(roadmap M2+ in `product.md`). The PyPI names `mlforge` and `magicml` are both
already taken; choose a distribution name before publishing.
