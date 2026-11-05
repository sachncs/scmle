#!/usr/bin/env bash
# Terraform/OpenTofu checks that need no credentials: fmt, init without a
# backend, validate. Runs only if `terraform` or `tofu` is installed; if not
# it says so and exits 0, so the Python policy check
# (scripts/check_terraform_policy.py) stays the check that always runs.
# `plan` needs provider credentials and is not run here; the cloud release
# workflow runs it for a protected environment.
set -euo pipefail

root="${1:-deploy/terraform}"
tool=""
for candidate in terraform tofu; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    tool="${candidate}"
    break
  fi
done
if [ -z "${tool}" ]; then
  echo "terraform_checks: neither terraform nor tofu is installed; skipped" >&2
  exit 0
fi

"${tool}" fmt -check -recursive "${root}"
for dir in "${root}"/modules/* "${root}"/*/environments/*; do
  [ -d "${dir}" ] || continue
  echo "validate ${dir}"
  "${tool}" -chdir="${dir}" init -backend=false -input=false >/dev/null
  "${tool}" -chdir="${dir}" validate
done
