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
   `pip-audit`, the API check, `python -m benchmarks.qualification --check
   all` (the detector, F6, screen, uncertainty and subgroup certificates plus
   the framing, ingestion, causal, decision and closed-loop families,
   including `loop_certificates.json`) and the complete test suite.

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

## Releasing a model to a cloud

That is a different release from the package. The model path, its gate and its
records are in [`CLOUD_RELEASE.md`](CLOUD_RELEASE.md); the infrastructure is in
[`INFRASTRUCTURE.md`](INFRASTRUCTURE.md). `cloud-release.yml` does nothing
unless `SCMLE_CLOUD_RELEASE` is set and runs only by hand.

## The production claim

One sentence is the claim, and nothing else may be said in its place:

<!-- claim:start -->
> SCMLE is production-qualified on AWS and GCP: every P0 acceptance
> criterion passed in staging on both providers, with evidence signed by a
> reviewer other than the author.
<!-- claim:end -->

**Status: this has NOT happened.** No acceptance run, recovery run or
reviewer sign-off exists for AWS or GCP. `plan/evidence/phase-17/aws/` and
`plan/evidence/phase-17/gcp/` hold no records, so the claim must not be made,
and neither the changelog, the README nor the site may suggest it.

The claim may be made only when this passes, with the pipeline's key and the
reviewer's own key (the pipeline must not hold the second):

```bash
python scripts/check_production_claim.py --root plan/evidence/phase-17 \
    --key-file pipeline.key --reviewer-key-file reviewer.key
```

It refuses unless, for each provider, `acceptance.json`, `deployment.json`,
`recovery.json`, `observability.json`, `continuous-evaluation.json` and
`signoff.json` are present, signed, from staging, for one commit and passing,
the recovery run covers every fault in the catalogue in that provider's own
staging environment within the declared RTO and RPO, and the sign-off names a
reviewer who is not the requester and covers the current digest of each
file. The local evidence in `plan/evidence/phase-17/local/` does not count:
it ran no provider ([`OPERATIONS.md`](OPERATIONS.md)). A passing local run is
not a step towards the sentence above.

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
