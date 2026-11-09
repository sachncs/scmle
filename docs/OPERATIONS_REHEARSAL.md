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

1. Register a known-good bundle and promote it with a signed
   `PromotionApproval` bound to its content digest, review decision, and
   evidence references. Pass the release verification key to the promotion
   call.
2. Stage a challenger under the `challenger` alias; do not move the serving
   alias as part of the training step.
3. Score the same sufficiently large labeled batch with both versions.
4. Run the rollback check with the declared tolerance and confirm a clearly
   worse champion remains in place while approval is pending.
5. Approve the previous version independently, rerun the rollback action, and
   confirm it restores the previous alias with a complete approval record in
   the `lifecycle.rollback_check` audit trail.

Automated evidence: `tests/unit/test_lifecycle_extras.py` and
`tests/integration/test_deployment.py`.

## 5. Fault evidence recording

1. Build the complete `Registry` for the disposable
   release candidate.
2. Use `FaultInjectionHarness` with a unique execution ID, reviewer, and
   evidence directory.
3. Run one scenario-specific injector and return one `FaultArtifact` for
   every declared evidence requirement.
4. Confirm each artifact is atomically persisted, its SHA-256 digest is
   recorded, and missing or unsafe observations remain inconclusive or
   failed.
5. Do not treat this recorder test as evidence that process death, disk
   exhaustion, or worker loss has actually been exercised; those injectors
   still require an environment-specific executor.

Automated evidence: `tests/unit/test_reliability_harness.py`.

## 3. Signing-key rotation

1. Produce a signed bundle with the current key.
2. Run `Bundle.rotate_signing_key(bundle_dir, old_key, new_key)`.
3. Load the bundle with the new key and verify predictions against a saved
   parity sample.
4. Confirm the old key is rejected and no temporary signature file remains.
5. Only then retire the old key in the secret manager.

Automated evidence: `tests/unit/test_bundle.py::test_signing_key_rotation_preserves_bundle_and_rejects_old_key`.

## 6. Monitoring handoff

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

Attach the evidence bundle to the release review. The release gate remains
open if any step was skipped, used an untracked secret, relied
on `trusted=True` outside the disposable rehearsal, or lacked an independent
operator review.
