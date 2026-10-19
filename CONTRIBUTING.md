# Contributing

Thanks for helping. Issues and pull requests are welcome; please follow the
[code of conduct](CODE_OF_CONDUCT.md) and report security problems through
[`SECURITY.md`](SECURITY.md).

Development works the same on Linux, macOS and Windows. Install
[uv](https://docs.astral.sh/uv/), then:

```bash
uv sync --extra boosting --extra optimization
uv run ruff check . && uv run ruff format --check .
uv run mypy scml benchmarks
uv run pytest
uv run python -m benchmarks.run_benchmarks --offline --output results.json
uv run python -m benchmarks.check_regression results.json
```

To reproduce the bounded release evidence locally, run
`./scripts/release_preflight.sh`. Set `SCML_PYTHON_VERSION` to test another
supported interpreter, for example `SCML_PYTHON_VERSION=3.11`.

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
* Cross-platform: no POSIX-only calls without a Windows branch, and every
  text file is opened with `encoding="utf-8"` (and `newline="\n"` when
  writing) so signed bundles hash identically everywhere.

## Layout

* `scml/`: the package (`core` orchestration, `data`, `models`,
  `evaluation`, `governance`, `inference`, ...); see `docs/ARCHITECTURE.md`.
* `tests/`: `unit`, `property`, `leakage`, `integration`, `spec`.
* `benchmarks/`: regression gate run in CI.

## Adding a model family

Implement the `ModelAdapter` contract, register it, and make
`scml.models.conformance.check_conformance` pass for it (see
`docs/TUTORIAL.md`).
