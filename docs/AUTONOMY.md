# Autonomous operation

What the system does without a person typing each command, what it still asks
a person for, and what has not been shown. Read
[What SCMLE cannot do](WHAT_SCMLE_CANNOT_DO.md) first if you are new.

**Status.** Built and tested against simulated worlds (`tests/simulation/`).
It has not run on a real deployment, and nothing here is a claim about
production.

## The rule

A human approves every promotion. The registry's `PromotionApproval`, bound to
the exact bundle digest, is unchanged. The controller never moves the champion
alias, and neither does an envelope.

## What runs by itself

`scmle operate tick` is one pass, safe to run from cron or a workflow engine.
It reads new files from an inbox:

* `inbox/batches/*.csv`: served cases with `prediction` and `decision` columns
  (and the optional `case_id`, `propensity`, `audit`, `set_size`, `novel`,
  `under_covered`, `bypass` columns `scmle loop observe` takes);
* `inbox/labels/*.csv`: arrived losses with `case_id` and `loss`, optionally
  `covered` and one `challenger:<name>` column per challenger.

It applies each file once, steps the loop, and journals what it did in the
deployment ledger. Alarms, challengers, promotion proposals and escalations are
the loop's own rules (F13); the controller adds only the following.

| Behaviour | How |
|---|---|
| Exactly once | A file is identified by the SHA-256 of its bytes. An `intent` row is chained before the loop is called and an `applied` row after. A tick that is killed and run again reads the loop's batch counter to see whether the file was applied; if the counter fits neither case it halts and escalates. |
| Bad input | An unreadable or malformed file is `rejected` once, escalated, and never retried. The rest of the inbox is still applied. |
| Silence | After `max_idle_ticks` ticks with no new batch, or `max_unlabelled_ticks` ticks with no usable labels, it opens one escalation. It does not stop the loop. |
| A halted loop | While the ledger is halted, or the saved state disagrees with the ledger, a tick changes nothing and reports `halted`. Only a passing recalibration releases a halt. |

## What needs a person

* **Every promotion.** The loop writes a `promotion_proposal`; a person's
  signed approval in the registry moves the alias.
* **Training a challenger when an alarm fires.** By default an alarm only
  opens the escalation `l1_alarm_investigate`, because the alarm design has not
  met its detection-delay gates (H2 and H10), so `alarm_gates_retrain` is off.
  Until it is certified, nobody should switch it on for a real deployment.
* **The other ten never-list actions**: reopening a sealed set, changing the
  charter, editing the ledger or registering late, relaxing a threshold, acting
  on an unverified stream, answering a flagged high-impact case, suppressing
  its own alert, changing the labelling policy, deleting or sending
  externally, and creating or signing an attestation. They cannot be put in an
  envelope.
* **A halt, a rejected file, silence, or a failed training.** Each opens an
  escalation that waits for a person.

## Envelopes

An envelope is authority a person signs in advance for an action that is not on
the never list. Today there is one such action: `train_challenger`.

```bash
# the person writes and signs it with their own key (Envelope.create)
scmle operate grant deploy/ --envelope envelope.json --key-file person.key
scmle operate status deploy/          # uses left per envelope
scmle operate revoke deploy/ e1 --reason "model retired"
scmle operate halt deploy/ --reason "drill"
```

An envelope has a scope (model version hash and stream id), a first and last
day, and a use limit per action. A use is chained in the ledger **before** the
action runs, and the ledger itself refuses a use that is unknown, revoked, out
of scope, not covered or over its limit. Anyone may revoke an envelope, because
revoking only removes authority.

Without an envelope the retrain hook trains nothing and escalates
(`retrain_not_authorized`). A training that raises is escalated
(`retrain_failed`) and the loop goes on; the use is still spent.

## Limits you must know

* **Simulation, not traffic.** The worlds are stable, abrupt and gradual drift,
  label delay, a label outage, a corrupted batch, and better and poisoned
  challengers. Passing them measures those worlds.
* **Kills between calls.** The test kills the controller at every checkpoint
  between calls and checks that it ends where an uninterrupted run ends. A
  crash inside a call leaves the loop's saved state behind the ledger; the loop
  then starts halted instead of guessing.
* **The key.** Envelopes and attestations are HMAC-signed. Whoever can verify
  can also sign, so keep the verification key where the system cannot read it.
  When you give the controller no keys it relies on the grant row, which was
  checked when it was written. A process with write access to the ledger file
  can still write rows; this is code around the entry points of the package,
  not a sandbox.
* **Not built yet** (see `plan/2-autonomous-operation.md`): alarm-gated
  retraining, shadow, canary and rollback as envelope actions, a per-day cap
  and a workflow-engine hook, and a run in staging.
