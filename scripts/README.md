# Scripts

Release and repository controls. None of them is imported by the package; CI,
the release workflow and `release_preflight.sh` call them, and each has a unit
test under `tests/unit/`. Run any script with `uv run python scripts/<name>.py`.

| Script | What it enforces | Called by |
| --- | --- | --- |
| `check_api_compatibility.py` | The public API (`scmle.__all__`) matches the snapshot in `tests/fixtures/public_api.json`. | `ci.yml`, preflight |
| `check_pytest_skips.py` | Every skip is listed in `pytest_skip_manifest.json` with a reason; any xfail fails. | every test job |
| `pytest_skip_manifest.json` | The approved skips and their reasons (data for the script above). | `check_pytest_skips.py` |
| `check_test_stability.py` | Repeats tests and applies the quarantine policy in `tests/quarantine.json` (owner, reason, expiry). | `full-suite.yml` |
| `check_mutation_score.sh` | Mutation score of the critical modules in `setup.cfg` is at least the given minimum (default 70). | `full-suite.yml` |
| `check_dependency_licenses.py` | The runtime dependency closure has a license record and no copyleft license. | `ci.yml` |
| `vuln_gate.py` | A scanner report (pip-audit or Trivy JSON) has no blocking finding under `deploy/vuln-policy.json`; fails closed on an empty or unreadable report. Implementation: `scmle.deploy.vuln_gate`. | generated `build.sh` |
| `make_demo_data.py` | The datasets of the local demo in `deploy/local/data/` match their generator (`--check`). | tests |
| `export_locked_requirements.sh` | `deploy/requirements.lock` equals the hashed export of `uv.lock` (`--check`). | `make lock-check` |
| `make_serving_golden.py` | The golden expected outputs in `deploy/local/golden/` equal the closed form of the hand-written reference model (`--check`). | tests |
| `check_serving_parity.py` | A running service returns the golden labels and probabilities within the declared tolerance, with version and artifact digest. | local demo, tests |
| `run_object_store_conformance.py` | The object-store conformance suite (`scmle.data.store.conformance`) against a local directory, MinIO or another S3 endpoint, the GCS emulator or a bucket; the report is JSON and the exit status is 0 only if every check passed. | `tests/unit/test_object_store_conformance.py` (local), operators (cloud) |
| `make_phase15_evidence.py` | Regenerates `plan/evidence/phase-15/`: the object-store conformance reports, the interrupted-job, duplicate-run, partial-failure and missed-schedule logs, a lineage trace, the emitted workflow definitions with their structural check, the scenario test results and a summary of what did not run. | `tests/unit/test_phase15_evidence.py`, maintainers |
| `check_terraform_policy.py` | The Terraform under `deploy/terraform` declares what the policy requires (no public ingress without authentication, encryption, no wildcard IAM, budgets, images by digest, no secrets, limits, three environments per provider). Reads the files; needs no Terraform and no credentials. Implementation: `scmle.deploy.iac_policy`. | `ci.yml`, tests |
| `terraform_checks.sh` | `fmt -check` and `validate` (with `-backend=false`) of every module and environment, only if `terraform` or `tofu` is installed; otherwise it says so and exits 0. | `ci.yml` |
| `check_destroy_orphans.py` | Lists resources an environment left behind after `terraform destroy`, from saved provider listings and `terraform state list`; exit 0 only when nothing is left. Implementation: `scmle.deploy.orphans`. | operators, tests |
| `make_phase16_evidence.py` | Regenerates `plan/evidence/phase-16/`: the policy check and its seeded violations, whether Terraform was available, seeded orphan listings, the live checks and their seeded failures, the gate's decision for a complete release and each seeded failed gate, reconcile reports after a success, a failed release, a rollback and seeded disagreements, the workflow controls, the scenario tests and a summary of what did not run. | `tests/unit/test_phase16_evidence.py`, maintainers |
| `make_phase17_evidence.py` | Regenerates `plan/evidence/phase-17/local/`: an injected service failure and a controlled drift event with their classification, webhooks, retained telemetry and audit rows, the exporter payloads, a controlled degradation through the F13 loop to a gated replacement and a post-promotion check, the local recovery faults with measured RTO and RPO, the claim checker on the empty provider folders and seeded incomplete trees, the scenario tests and a summary of what did not run. | `tests/unit/test_phase17_evidence.py`, maintainers |
| `check_production_claim.py` | Refuses the production claim unless `plan/evidence/phase-17/{aws,gcp}/` hold every signed staging record for one commit, passing, covered by a sign-off from a reviewer who is not the author, and `docs/RELEASING.md` holds the fixed claim text. Implementation: `scmle.operations.claim`. | release review, tests |
| `verify_hosted_controls.py` | Workflows pin every action to a commit SHA, never use `pull_request_target`, mention no long-lived cloud key, and keep their required control steps (for the cloud release workflows: manual trigger only, every job behind the `SCMLE_CLOUD_RELEASE` guard, a protected environment and the gate steps); optionally probes a live Pages URL with unauthenticated GETs. | `ci.yml`, `pages.yml`, `site.yml` |
| `verify_release_artifacts.sh` | Two builds are byte-identical and the wheel installs and runs without an index. | `ci.yml`, `release.yml` |
| `verify_external_release.py` | After publishing, the public tag and the PyPI files match the locally built artifacts by digest. | `release.yml` |
| `verify_release_review.py` | An independent release-review record covers every required claim for an exact commit. | release review ([`docs/RELEASING.md`](../docs/RELEASING.md)) |
| `release_preflight.sh` | Runs the whole local release evidence chain: build, lint, types, bandit, pip-audit, API check, tests. | `release.yml`, maintainers |

Adding a script: give it an argparse interface and a docstring, add a test,
wire it into a workflow, and add a row here. Scripts that nothing calls are
deleted.
