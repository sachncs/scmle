# plan.md has moved

The plan this file once held (a review of what separates SCMLE from an
autonomous ML/DS lifecycle, in phases 0 to 5) is now the five stages in
[`plan/README.md`](plan/README.md), with only what is still to be built, one
register of open items ([`plan/COMPLETION_AUDIT.md`](plan/COMPLETION_AUDIT.md))
and the features still to build ([`product/features.md`](product/features.md)).

Decisions recorded on 2026-11-08:

* A human approves every promotion; the envelope covers everything up to the
  proposal (`plan/2-autonomous-operation.md`).
* Built from that plan: LF text writes and a lint for them, the oracle-table
  floor, the envelope, the controller, `scmle operate`, the simulated worlds
  and `docs/AUTONOMY.md`.
* Not built: alarm-gated retraining (the H2 and H10 gates still miss their
  floors), shadow, canary and rollback as envelope actions, the workflow hook,
  mutation records and a staging run.
