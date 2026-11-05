# The business trail: from the question to the observed outcome

A trail follows one deployed model through the records that already exist:
the charter, the framing and decision-cost model, the claim ledger, the
registry and release state, and the deployment ledger of the closed loop. It
joins them into one folder that a reviewer who did not build the model can
check with nothing else.

`scmle trail build` copies those records into a folder and seals it.
`scmle trail verify` checks the folder against itself. `scmle trail show`
prints it. `scripts/trace_model.py` is the same walk written for a reviewer:
it imports nothing from SCMLE and reads nothing outside the folder.

## What is in a trail

```text
trail/
  trail.json                     ten steps, hash-chained, signed (analyst key)
  inputs.json                    the stakeholder's and the reviewer's statements
  reports/
    exploratory_report.json      what the exploration rows showed
    experiment_comparison.json   every candidate, ranked, with its spread
    outcome_tracking.json        observed cost per case against the expectation
  retained/
    run/         charter, claim ledger, custody log, evaluation, trials, bundle manifest
    framing/     framing draft, questions, decision-cost model
    registry/    registry.json, audit.jsonl, release_state.json
    deployment/  deployment_ledger.jsonl, loop_state.json, observed_outcomes.json
```

The ten steps, in order: the business question (the signed charter), framing
and the cost model, KPIs with acceptance criteria and the approved
expectation, the exploratory report, the experiment comparison, domain
quality thresholds with the reviewer's evidence, model selection with the
claim ledger and the bundle identity, deployment approval and release state,
the production decisions in the deployment ledger, and the observed outcomes
against the approved expectation.

## Who says what

`inputs.json` has two sections, each signed with its author's own HMAC key
(`scmle.contracts.signing`, the signing every other record uses):

* the **stakeholder** maps each business KPI to a metric, writes the
  acceptance criteria (`path`, `op`, `value`, `reason`) and approves an
  expectation: a range for the observed cost per case, with its basis;
* the **reviewer** states the domain quality thresholds and signs the
  evidence of the review, which names each record read and its digest. A
  record that changes afterwards is reported.

The charter is signed by the analyst, and the promotion is a
`PromotionApproval` signed with the reviewer's key. The trail is signed by the
analyst over its head and the digest of every retained record. A signature
proves possession of a key, not identity.

## What `verify` detects

Each finding has a kind. Every one has a seeded test in
`tests/unit/test_trail.py`.

| Kind | Meaning |
| --- | --- |
| `missing_link` | A record, a step or an input the trail names is absent, or no outcome can be compared. |
| `broken_chain` | A step does not continue the hash chain, or the deployment or claim ledger does not chain. |
| `unsigned_step` | The charter, the stakeholder or reviewer section, the approval or the trail carries no signature that verifies. |
| `unapproved_step` | No signed approval promoted this version, it was for another bundle, rejected, not independent, or made by the builder. |
| `outcome_outside_expectation` | The 95% interval of the observed cost per case does not overlap the approved range. |
| `record_tampered` | A retained file differs from its sealed digest, or changed after the reviewer signed. |
| `fact_mismatch` | A stored fact or derived report is not what the retained records give now. |
| `identity_mismatch` | Two records name different charters, bundles or runs. |
| `criterion_not_met` | An acceptance criterion or quality threshold fails on the retained numbers. |

`verify` re-derives every fact from the retained records; it does not trust
the numbers in `trail.json`. The exploratory report and the experiment
comparison use no random draw and no clock, so recomputing them gives the same
bytes and the same `sha256`.

## Observed outcomes and what they do not show

The cost of a case is the cell of the chartered cost model for the action the
cut takes and the outcome that arrived. The report gives the mean cost per
case with a normal-approximation 95% interval and compares the interval, not
the mean, with the approved range: `within`, `above`, `below`, `insufficient`
(fewer than 100 matured cases) or `absent`. It says in words that the interval
is not a forecast and that a cost inside the range does not show the model
caused the outcome, because no counterfactual was observed. The cost is that
of the rule on every case; cases the loop escalated to a person are counted by
the deployment ledger and still costed here.

## The local evidence and its limits

`plan/evidence/phase-18/local/` holds a trail for one model, produced by
`scripts/make_phase18_evidence.py` with the real commands: `scmle frame`,
`charter sign`, `run`, `package`, the local registry, `scmle loop`, then
`trail build`. Read these limits before relying on it.

* The analyst, stakeholder and reviewer were all the author, with three
  different keys. The reviewer-independence check compares names, and the
  names are role labels. **No independent person has traced or signed this
  trail.** The gate "a reviewer who did not build the model traces one
  deployed model" is open.
* The Titanic extract has no key and no timestamp. The framing step used a
  row number as the unit and a constant boarding date as the prediction
  moment, both assumptions. Because the extract is ordered, the row number
  correlates with the target, and the claim ledger tested it.
* The observed outcomes replay the held-out test rows through the loop. They
  agree with the expectation by construction, so they show the mechanism and
  not the model's cost in use. The cost ceiling was set after the held-out
  evaluation was seen.
* The confirmation set was smaller than the chartered power needs, so the run's
  scope is `exploratory_unconfirmed`; the trail carries that fact.
* The release state is `approved`, not `production`: no runtime confirmed a
  deployment. Nothing ran on AWS or GCP.

## Reviewing a trail

```bash
python scripts/trace_model.py trail --keys keys.json
scmle trail verify trail --keys keys.json
```

`keys.json` maps `analyst`, `stakeholder` and `reviewer` to the keys that
verify them. The keys in the committed evidence are published demonstration
keys: they show tamper evidence, not a statement by their holder. Without
keys the script exits 2 and says which signatures it did not check.
