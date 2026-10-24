# Releasing

A release is a version bump, a tag, and a pipeline that refuses to publish
unless the evidence holds. The checks live in [`scripts/`](../scripts/README.md).

## Before tagging

1. Set the version in `pyproject.toml` and `scmle/__init__.py`.
2. Move the `Unreleased` entries in `CHANGELOG.md` under `## <version>`.
3. If the public API changed deliberately, regenerate
   `tests/fixtures/public_api.json` with
   `python scripts/check_api_compatibility.py tests/fixtures/public_api.json --write` and note the change in the changelog and
   [`API_STABILITY.md`](API_STABILITY.md).
4. Run the full local evidence chain:

   ```bash
   ./scripts/release_preflight.sh        # SCMLE_PYTHON_VERSION=3.11 to vary
   ```

   It builds the artifacts and the site, then runs lint, types, `bandit`,
   `pip-audit`, the API check and the complete test suite.

## The pipeline (`release.yml`, on a `v*` tag)

1. **verify**: the tag equals the package version, `scmle.__version__` and a
   changelog section.
2. **test**: the bounded suite on Linux, macOS and Windows.
3. **release-checks**: `release_preflight.sh`.
4. **publish**: full test suite, benchmark regression gate, build, `twine check`,
   `verify_release_artifacts.sh` (reproducible, installs offline), a wheel and
   sdist smoke test, PyPI trusted publishing, then `verify_external_release.py`
   compares the public tag and PyPI files with the local build by digest.

PyPI publishing uses OIDC trusted publishing; no token is stored. The `pypi`
GitHub environment should require approval.

## Evidence the pipeline cannot produce

Two controls need people, and the release stays open without them:

* The operational rehearsal in [`OPERATIONS_REHEARSAL.md`](OPERATIONS_REHEARSAL.md),
  reviewed by someone other than the author.
* An independent review record, checked with

  ```bash
  python scripts/verify_release_review.py review.json --source-commit <sha>
  ```

## After publishing

Install the released version in a clean environment on each platform, run
`examples/quickstart.py`, and open the GitHub release to confirm the notes.
