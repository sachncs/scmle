#!/usr/bin/env bash

# Reproduce the local release evidence without mutating tracked files.
set -euo pipefail

python_version="${SCML_PYTHON_VERSION:-3.14}"
benchmark_directory="$(mktemp -d "${TMPDIR:-/tmp}/scml-bench.XXXXXX")"
benchmark_output="$benchmark_directory/results.json"
preflight_directory="$(mktemp -d "${TMPDIR:-/tmp}/scml-preflight.XXXXXX")"

uv sync --python "$python_version" \
  --extra boosting --extra optimization --extra multimodal

echo "== static checks =="
uv run --python "$python_version" ruff check .
uv run --python "$python_version" ruff format --check .
uv run --python "$python_version" mypy scml benchmarks
uv run --python "$python_version" bandit -q -r scml
uv run --python "$python_version" pip-audit --skip-editable
uv run --python "$python_version" python scripts/check_api_compatibility.py \
  tests/fixtures/public_api.json

echo "== bounded contract suite =="
uv run --python "$python_version" pytest -q \
  -W error::ResourceWarning \
  --cov=scml --cov-report=xml --cov-fail-under=50 \
  tests/unit tests/property tests/leakage \
  tests/integration/test_cli.py \
  tests/integration/test_end_to_end.py \
  tests/integration/test_predict_api.py \
  tests/integration/test_deployment.py \
  tests/integration/test_monitoring.py \
  tests/integration/test_serve_and_export.py \
  --junitxml="$preflight_directory/bounded-suite.xml"
uv run --python "$python_version" python scripts/check_pytest_skips.py \
  "$preflight_directory/bounded-suite.xml"
uv run --python "$python_version" python scripts/check_test_stability.py \
  --runs 3 \
  --quarantine tests/quarantine.json \
  --output "$preflight_directory/test-stability.json" \
  tests/unit/test_api_surface.py \
  tests/unit/test_performance_contracts.py \
  tests/unit/test_test_stability.py
uv run --python "$python_version" coverage report \
  --include='scml/contracts/*,scml/data/*,scml/evaluation/*,scml/governance/*,scml/inference/*,scml/monitoring/*' \
  --fail-under=85
echo "== critical mutation gate =="
uv run --python "$python_version" mutmut run --max-children 4
scripts/check_mutation_score.sh 70

echo "== benchmark regression =="
uv run --python "$python_version" python -m benchmarks.run_benchmarks \
  --offline --seeds 5 --budget 120 --workers 2 --output "$benchmark_output"
uv run --python "$python_version" python -m benchmarks.check_regression \
  "$benchmark_output"

echo "== package =="
uv build --python "$python_version"
uv run --python "$python_version" twine check dist/*
scripts/verify_release_artifacts.sh

echo "== marketing site =="
(cd website \
  && npm ci \
  && npm audit --omit=dev --audit-level=high \
  && npm run build \
  && npm run check:a11y \
  && npm run check:links \
  && npx playwright install chromium \
  && npm run test:e2e)

echo "preflight passed; benchmark evidence: $benchmark_output"
