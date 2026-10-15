# Contributing

```bash
uv sync --extra boosting --extra optimization
uv run ruff check . && uv run ruff format --check .
uv run mypy mml benchmarks
uv run pytest
uv run python -m benchmarks.run_benchmarks --offline --output results.json
uv run python -m benchmarks.check_regression results.json
```

Rules (all enforced by CI):

* Python 3.11+ (keep the code importable on 3.11), Google style: 80 columns, Google docstrings, `import module`
  (not `from module import name`, except `typing`/`collections.abc`).
* **No underscore-prefixed names** for variables, functions, classes,
  attributes or files (`tests/unit/test_conventions.py`).
* Prefer a polymorphic class over a conditional on type; keep abstractions
  minimal and justified by two or more implementations.
* Data holders are frozen dataclasses (or pydantic models for configuration).
* Every behaviour change needs a test; every fixed bug needs a regression
  test. Leakage-relevant changes must keep `tests/leakage` green.
* Make small, atomic commits; do not add co-author trailers.
