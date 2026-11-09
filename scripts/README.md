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
| `trace_model.py` | Walks a trail folder as a reviewer would, using only the retained records and the standard library: sealed files, signatures, joins between records, thresholds, the ledger chains and the observed outcomes against the approved range. | `tests/unit/test_trace_model_script.py`, reviewers |
| `verify_hosted_controls.py` | Workflows pin every action to a commit SHA, never use `pull_request_target`, mention no long-lived cloud key, and keep their required control steps; optionally probes a live Pages URL with unauthenticated GETs. | `ci.yml`, `pages.yml`, `site.yml` |
| `verify_release_artifacts.sh` | Two builds are byte-identical and the wheel installs and runs without an index. | `ci.yml`, `release.yml` |
| `verify_external_release.py` | After publishing, the public tag and the PyPI files match the locally built artifacts by digest. | `release.yml` |
| `verify_release_review.py` | An independent release-review record covers every required claim for an exact commit. | release review ([`docs/RELEASING.md`](../docs/RELEASING.md)) |
| `release_preflight.sh` | Runs the whole local release evidence chain: build, lint, types, bandit, pip-audit, API check, tests. | `release.yml`, maintainers |
| `admit.py` | Runs a heavy command (tests, coverage, mutation, profiling) only when the one-minute load average and free memory allow, within a limited number of slots, with math-library threads set to 1. Not run by CI; a maintainer tool for a shared machine. | maintainers, `tests/unit/test_admit_script.py` |
| `check_justified.py` | Lists functions and classes that nothing justifies: unreferenced, referenced only by tests, pass-throughs, getters, stateless one-method classes and single-implementation interfaces. Static and by name, so it under-reports; `justified_dynamic.txt` declares names used through run-time-built names, each with a reason. Report-only today (`--strict` fails while candidates remain); not yet a CI gate. | maintainers, `tests/unit/test_check_justified_script.py` |
| `justified_dynamic.txt` | The declared run-time-built names (data for the script above). | `check_justified.py` |

Adding a script: give it an argparse interface and a docstring, add a test,
wire it into a workflow, and add a row here. Scripts that nothing calls are
deleted.
