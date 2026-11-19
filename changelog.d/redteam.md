### C8: a regression test per bypass attempt, and a red-team pass

- `tests/redteam/test_redteam_attempts.py` runs every attempt in
  `benchmarks/evidence/bypass.py` end to end as its own test. An attempt
  passes only when the control it attacks refuses it, or a control it lists
  as defence in depth (`Attempt.also`). An attempt that names a documented
  override may instead go through that override. The record names what
  stopped the attack: a registered gate, the raising site of an
  unregistered control (`envelope.authenticate`, `bundle.authenticate`,
  `charter.verify`), or for the CLI the label the refusal printed
  (`cli.blocked`, `cli.artifact`). An attack that raises `Broken` because
  it never reached its control is recorded as `error`, not as blocked.
  Nothing may get through without an override, and no attempt may fail to
  set up. An attempt that got through when it was added names the product
  code at fault (`Attempt.defect`). It is a strict xfail that expects only
  a bypass, so a broken or misdirected attack fails the suite.
- `pickle_products` now reaches the control it tests. It resumes with a
  checkpoint that approves selection, and it watches `envelope.read`, so
  the record shows the swapped stage products were read and refused.
- The search grew from 47 to 88 attempts. The new ones cover custody journal
  edits, custody token reuse in process and in the vault, pickle payloads in
  bundles, deep-learning checkpoints and stage products, a stripped charter
  signature, a plugin stage, `SCMLE_TRUSTED`, budgets of NaN and infinity,
  `pandas.NA` targets, and charter reruns into another output folder or
  after deleting `.confirmations`. A pickle payload or a plugin counts as
  through when it took effect (`witnessed`), even if the call was refused
  afterwards.
- Ten attempts got through. Each is a product defect, still open:
  - `scmle/envelope.py`: `read` accepts a signed envelope with no key to
    check it when `trusted=True`, and the attacker's pickle runs
    (`pickle_checkpoint_trusted_nokey`). `Bundle.load` refuses the same
    case (`pickle_bundle_trusted_nokey` is blocked), although the module
    docstring says both follow one rule.
  - `scmle/assurance/custody/journal.py`: the custody journal's hash chain
    has no key and no anchored head. Removing its last line
    (`journal_truncated`), recomputing the chain (`journal_rechained`,
    `checkpoint_rechained`) or appending a `started` entry, which
    `Journal.current` treats as a fresh start (`journal_restarted`,
    `checkpoint_restarted`), lets a completed run score its locked test set
    again. The same edits, or a forged approval (`checkpoint_approved`), let
    a run that a checkpoint rejected resume.
  - `scmle/assurance/ledger/usage.py`: the record of spent confirmation sets
    is unsigned files in the output folder. A charter rerun into another
    folder (`charter_rerun_elsewhere`), or after `.confirmations` is deleted
    (`charter_rerun_forgotten`), confirms on the same rows a second time.
  - `scmle/pipeline/stages/defaults.py`: a plugin stage gets the raw locked
    test set from the board and can bind and open it before selection
    (`plugin_peek`). Only `FinalEvaluationStage` wraps the set in the
    journal's `Custodian`.
- The C8 row stays `not measured`, or becomes `fail` once the lead
  re-measures, because one bypass disproves "cannot be bypassed".
  `bypass.json` must be re-measured (`--write`) for the new attempts.
