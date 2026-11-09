# Stage 2: Autonomous operation

Names follow [`00-canon.md`](00-canon.md) (principle P5). Item classes and
reasons: [`COMPLETION_AUDIT.md`](COMPLETION_AUDIT.md), "Stage 2". Feature: F16
in [`../product/features.md`](../product/features.md). What exists is described
in [`../docs/AUTONOMY.md`](../docs/AUTONOMY.md). Depends on stage 1 and on the
built closed loop (`scmle/governance/loop.py`).

## Decision (owner, recorded 2026-11-08)

A human approves every promotion. The registry's `PromotionApproval`, bound to
the exact bundle digest, is unchanged and stays the only way to move the
champion alias forward. The envelope covers everything up to the proposal.
Rollback to the previously approved champion is the one champion move an
envelope may later cover (item 2.8); it stays behind the existing guard and the
registry's rules. The other ten `never_list` actions stay human only. No
registry rule changes, so no guarantee of the product changes.

## Starting point (2026-11-08)

Built and tested against simulated worlds (`tests/unit/test_envelope.py`,
`tests/unit/test_controller.py`, `tests/simulation/`):

* an envelope (`scmle/governance/envelope.py`, `Envelope` in
  `deployment_ledger.py`) that authorises `train_challenger` with a use limit,
  a scope and dates, signed by a key the system does not hold, with ledger rows
  that the chain itself checks;
* a controller (`scmle/operations/controller.py`) and `scmle operate tick |
  status | grant | revoke | halt` that apply each inbox file exactly once,
  spend an envelope before a retrain, escalate when data or labels stop, and
  leave a halted loop alone.

Not built, and the reason each item below exists:

* `LoopPolicy.alarm_gates_retrain` is False because the H2 and H10
  detection-delay gates missed their floors, so an alarm escalates and the
  retrain hook is never called. The measured gap (`loop_certificates.json`,
  2000 replicates, batches of 50 cases, `D_max` of 50 batches): at a 2x rise of
  the loss rate the median delay is 76 batches and 0.35% of replicates detect
  within `D_max` (H2); H10's halt on a 2x rise of `r_star` is within `D_max` in
  13% of replicates, median 57. At a 3x rise both pass. The alarm is valid
  (false alarms stay under the bound) and detects; it is too slow at small
  rises, which is a power limit of one observation per batch, not a defect.
* Nothing calls `Schedule.tick`; a person or cron runs `operate tick`.
* Only `train_challenger` can be put in an envelope.
* The envelope and controller have no mutation survivor record.
* No staging deployment has run.

## Work packages

| WP | Item | Class | Scope | Closing criterion |
|---|---|---|---|---|
| 2.1 | 2.2 | A | Alarm-gated retraining (the former row F13-3a). Re-register the L1 and selective-alarm design (tighter bounds on batch means, per-case instead of per-batch observations, or a charter `D_max` the settings can meet) so the H2 and H10 floors pass on registered worlds; re-issue `loop_certificates.json` with the same seeds and floors; then allow `alarm_gates_retrain` | H2 and H10 pass at their pre-registered `D_max`; until then the controller only recommends retraining |
| 2.2 | 2.6 | A | A per-day action cap, an `operate start` action that loops ticks with a kill switch, and a workflow handler so `Schedule.tick` (slot-derived run ids) or a workflow engine drives `operate tick` | a double tick is one run; the cap refuses the extra action; tests in `tests/unit` |
| 2.3 | 2.7 | A | Add `envelope.py` and `controller.py` to `setup.cfg` `source_paths`; write the envelope's refusal floors; record survivors in `tests/unit/MUTANTS_envelope.md` and `MUTANTS_controller.md` | mutation score at the repo floor, each survivor justified |
| 2.4 | 2.8 | A | Shadow, canary and rollback to the previously approved champion as envelope actions, each with refusal tests; rollback keeps the registry's rules | `pytest tests/unit -k envelope`; the human-only actions remain inexpressible |
| 2.5 | 2.5 | C | The controller in a staging deployment with real label delay and a real business outcome, after item 4.29 | a retained record of the run, reviewed by someone other than the author |

## Exit gate

```bash
uv run pytest -q tests/unit tests/integration tests/simulation
uv run python -m benchmarks.qualification --check all
uv run ruff check . && uv run ruff format --check . && uv run mypy scmle benchmarks
```

Then push and read every CI job on the final head.

## Not claimed

Autonomous operation in production. Anything beyond the simulated worlds and
the declared envelope. A passing simulation is a measurement of those worlds,
and the documentation says so.

## Risks

| Risk | Mitigation |
|---|---|
| An envelope weakens the rule that the system never holds its approval key | The verification key is never held by the system; human-only actions are inexpressible; adversarial tests and a mutation record |
| The alarm design cannot meet the H2 and H10 floors | The controller stays recommend-only; the floors are not lowered |
| A controller bug takes a wrong action | Ledger before action, halt on disagreement, crash-and-resume tests |
| The simulation overstates real behaviour | Name it as a simulation everywhere; item 2.5 stays open |
