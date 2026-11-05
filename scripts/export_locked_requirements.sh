#!/usr/bin/env bash
# Export the runtime dependencies of scmle, with hashes, from uv.lock.
#
#   scripts/export_locked_requirements.sh           # rewrite deploy/requirements.lock
#   scripts/export_locked_requirements.sh --check   # fail if it is stale
#
# uv.lock records every wheel and its hash for linux amd64 and arm64 as well as
# macOS and Windows; the export keeps the environment markers, so one file
# serves both image architectures. The check compares the pinned versions and
# the set of hashes, not the text, so a different uv version that formats the
# file differently does not make it look stale.
set -euo pipefail
cd "$(dirname "$0")/.."
target=deploy/requirements.lock
fresh="$(mktemp)"
trap 'rm -f "$fresh"' EXIT
uv export --frozen --no-dev --no-emit-project --format requirements-txt \
  --no-header --output-file "$fresh" >/dev/null
if [ "${1:-}" = "--check" ]; then
  python3 - "$fresh" "$target" <<'PY'
import re
import sys


def facts(path):
    text = open(path, encoding="utf-8").read()
    pins = set(re.findall(r"(?m)^([A-Za-z0-9._-]+==[^\s;\\]+)", text))
    hashes = set(re.findall(r"--hash=(sha256:[0-9a-f]{64})", text))
    return pins, hashes


fresh, current = facts(sys.argv[1]), facts(sys.argv[2])
if fresh != current:
    print("deploy/requirements.lock is stale: run scripts/export_locked_requirements.sh", file=sys.stderr)
    sys.exit(1)
print(f"deploy/requirements.lock is current ({len(current[0])} pins, {len(current[1])} hashes)")
PY
else
  cp "$fresh" "$target"
  echo "wrote $target ($(grep -c '==' "$target") pinned distributions)"
fi
