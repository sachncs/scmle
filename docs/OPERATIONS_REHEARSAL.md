# Operational rehearsal

Run this procedure against a disposable release candidate and a disposable
registry. Record the commit SHA, package version, Python version, platform,
operator, timestamps, and links to the resulting run directory and logs.

The rehearsal is complete only when every step has evidence and an operator
other than the author can explain the recovery decision.

## 1. Interruption and resume

1. Start a bounded run with a small dataset and a run output directory.
2. Interrupt it after at least one trial checkpoint exists.
3. Confirm the run ends in `CANCELLED` or `FAILED` and that its event log and
   recoverable trial list are present.
4. Resume the same run with the same data and configuration.
5. Confirm completed trials are reused, the final state is `COMPLETED`, and
   the event log contains both the original and resumed state transitions.

Automated evidence: `tests/integration/test_resume.py` and
`tests/integration/test_recovery.py`.

## 2. Rollback

1. Register and promote a known-good bundle.
2. Promote a challenger in the disposable registry.
3. Score the same sufficiently large labeled batch with both versions.
4. Run the rollback check with the declared tolerance.
5. Confirm a clearly worse champion restores the previous alias and writes a
   `lifecycle.rollback_check` audit event.

Automated evidence: `tests/unit/test_lifecycle_extras.py` and
`tests/integration/test_deployment.py`.

## 3. Signing-key rotation

1. Produce a signed bundle with the current key.
2. Run `ModelBundle.rotate_signing_key(bundle_dir, old_key, new_key)`.
3. Load the bundle with the new key and verify predictions against a saved
   parity sample.
4. Confirm the old key is rejected and no temporary signature file remains.
5. Only then retire the old key in the secret manager.

Automated evidence: `tests/unit/test_bundle.py::test_signing_key_rotation_preserves_bundle_and_rejects_old_key`.

## 4. Monitoring handoff

1. Assess a fresh labeled batch with the promoted bundle.
2. Record row count, labeled coverage, drift findings, recommendation, and
   the operator decision.
3. If retraining is recommended, require challenger comparison evidence before
   promotion; otherwise record why the recommendation was deferred.
4. Confirm the monitoring report and audit events are retained with the run or
   registry evidence.

Automated evidence: `tests/integration/test_monitoring.py`,
`tests/integration/test_deployment.py`, and `docs/OPERATING_LOOP.md`.

## Exit criteria

Attach the evidence bundle to the release review. The production-readiness
gate remains open if any step was skipped, used an untracked secret, relied
on `trusted=True` outside the disposable rehearsal, or lacked an independent
operator review.
