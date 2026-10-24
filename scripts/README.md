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
| `verify_hosted_controls.py` | Workflows pin every action to a commit SHA, never use `pull_request_target`, and keep their required control steps; optionally probes a live Pages URL with unauthenticated GETs. | `ci.yml`, `pages.yml`, `site.yml` |
| `verify_release_artifacts.sh` | Two builds are byte-identical and the wheel installs and runs without an index. | `ci.yml`, `release.yml` |
| `verify_external_release.py` | After publishing, the public tag and the PyPI files match the locally built artifacts by digest. | `release.yml` |
| `verify_release_review.py` | An independent release-review record covers every required claim for an exact commit. | release review ([`docs/RELEASING.md`](../docs/RELEASING.md)) |
| `release_preflight.sh` | Runs the whole local release evidence chain: build, lint, types, bandit, pip-audit, API check, tests. | `release.yml`, maintainers |

Adding a script: give it an argparse interface and a docstring, add a test,
wire it into a workflow, and add a row here. Scripts that nothing calls are
deleted.
