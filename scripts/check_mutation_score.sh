#!/usr/bin/env bash

# Enforce the release target for the configured critical mutation scope.
set -euo pipefail

minimum_score="${1:-70}"
results="$(uv run mutmut results --all true)"
killed="$(printf '%s\n' "$results" | awk '$NF == "killed" { count++ } END { print count + 0 }')"
scored="$(printf '%s\n' "$results" | awk '$NF == "killed" || $NF == "survived" || $NF == "timeout" { count++ } END { print count + 0 }')"

if (( scored == 0 )); then
  echo "mutation gate found no scored mutants" >&2
  exit 1
fi

score=$((100 * killed / scored))
printf 'mutation score: %s%% (%s/%s killed; %s minimum)\n' \
  "$score" "$killed" "$scored" "$minimum_score"

if (( score < minimum_score )); then
  echo "mutation score is below the configured release minimum" >&2
  exit 1
fi
