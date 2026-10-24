#!/usr/bin/env bash

# Verify release artifacts are reproducible and installable without an index.
set -euo pipefail

python_version="${SCMLE_PYTHON_VERSION:-3.14}"
source_date_epoch="${SOURCE_DATE_EPOCH:-1792497600}"
probe_directory="$(mktemp -d "${TMPDIR:-/tmp}/scmle-artifacts.XXXXXX")"

build_artifacts() {
  local output_directory="$1"
  SOURCE_DATE_EPOCH="$source_date_epoch" uv build \
    --python "$python_version" \
    --out-dir "$output_directory" \
    --clear
}

sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
}

artifact_path() {
  local directory="$1"
  local suffix="$2"
  find "$directory" -maxdepth 1 -type f -name "*$suffix" -print -quit
}

echo "== reproducible wheel and source distribution =="
build_artifacts "$probe_directory/first"
build_artifacts "$probe_directory/second"

wheel_one="$(artifact_path "$probe_directory/first" '.whl')"
wheel_two="$(artifact_path "$probe_directory/second" '.whl')"
sdist_one="$(artifact_path "$probe_directory/first" '.tar.gz')"
sdist_two="$(artifact_path "$probe_directory/second" '.tar.gz')"

test -n "$wheel_one"
test -n "$wheel_two"
test -n "$sdist_one"
test -n "$sdist_two"
test "$(basename "$wheel_one")" = "$(basename "$wheel_two")"
test "$(basename "$sdist_one")" = "$(basename "$sdist_two")"
test "$(sha256 "$wheel_one")" = "$(sha256 "$wheel_two")"
test "$(sha256 "$sdist_one")" = "$(sha256 "$sdist_two")"
echo "wheel: $(sha256 "$wheel_one")"
echo "sdist: $(sha256 "$sdist_one")"

echo "== network-isolated wheel installation =="
uv venv --python "$python_version" "$probe_directory/wheel-environment"
(cd /tmp && uv pip install \
  --python "$probe_directory/wheel-environment/bin/python" \
  --no-index --no-deps "$wheel_one")
(cd /tmp && "$probe_directory/wheel-environment/bin/python" -c \
  'import importlib.metadata as metadata; print(metadata.version("scmle"))')

echo "== network-isolated source distribution rebuild and installation =="
uv build --offline --wheel --python "$python_version" \
  --out-dir "$probe_directory/rebuilt-from-sdist" "$sdist_one"
rebuilt_wheel="$(artifact_path "$probe_directory/rebuilt-from-sdist" '.whl')"
test -n "$rebuilt_wheel"
uv venv --python "$python_version" "$probe_directory/sdist-environment"
(cd /tmp && uv pip install \
  --python "$probe_directory/sdist-environment/bin/python" \
  --no-index --no-deps "$rebuilt_wheel")
(cd /tmp && "$probe_directory/sdist-environment/bin/python" -c \
  'import importlib.metadata as metadata; print(metadata.version("scmle"))')

echo "release artifact verification passed; evidence directory: $probe_directory"
