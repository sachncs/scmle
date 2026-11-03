# F4: Admissibility and provenance

Vocabulary and principles: [`../../plan/00-canon.md`](../../plan/00-canon.md).
Built in [Phase 4](../../plan/phase-4-detectors-and-admissibility.md); needs
F1 (estimand) and F2 (exploration set).

## Purpose

Decide which fields may enter the model, on facts about how each value comes
to exist, and use statistics only to raise suspicion about what provenance
cannot settle.

## Decision it serves

Per field: `admit`, `quarantine` (excluded, reversible, reason logged) or
`admit-flagged`. Per dataset: whether reconciliation of the extract against
independent totals passes. Both feed the modelling specification.

## Behaviour

Proposed artifacts: `field_provenance.json` (data dictionary with a "how this
value comes to exist" column), `admissibility_table.json`,
`reconciliation.json`, `dag.json` (hashed, marked hypothesis),
`leakage_suspicions.json`.

Field record (proposed):

```
name, origin {instrument | system_default | derived | manual},
writer, written_at {event_time | batch_time | unknown},
overwritten {never | on_update | unknown}, definition_history[],
as_of {timestamp-rule | lag | unknown}, as_of_source {log | attested | none},
available_at_serving {yes | no | unknown}, tier
```

Tier of a field: `confirmed` only when as-of is derived from logs and also
passes the statistical checks; `attested` when a named person supplied it;
`unverified` otherwise. Strict mode (charter): `unverified` fields are
`quarantine`; permissive mode: `admit-flagged`.

## Method (one rigorous method per question)

**Admissibility rule depends on the estimand type (F1).**
- `predictive`: a field is admissible iff its value is known before the
  prediction moment (as-of <= moment minus any declared lag) **and** available
  at serving. Mediators, colliders and post-treatment variables known by
  then are **not** excluded: the causal labels do not govern a pure
  prediction task. Only timing and serving availability do.
- `causal`: additionally require a valid adjustment set for the estimand,
  computed from the declared DAG (backdoor criterion); colliders and
  descendants of treatment are excluded from the adjustment set.

**As-of times from raw material first.** Where change-data-capture or event
logs exist, derive each field's as-of time with a point-in-time (as-of) join
against the log: for each row, the latest value written at or before the
prediction moment. A field whose reconstructed value differs from the stored
value for a non-trivial share of rows is evidence of overwrite or backfill
(reported with a binomial interval). Where no log exists, the as-of is an
attestation, tier `attested`, never `confirmed`. Cut: no inference of as-of
times from the data distribution alone.

**Leakage is a suspicion test, not a proof.** Statistics cannot separate a
leak from a strong legitimate predictor (the existing profiler says the same,
`scmle/data/profiling/profiler.py:4-6`). Two tests, both on `X` only:
1. *Single-feature skill vs permutation null.* Out-of-fold score of a small
   model on one feature, compared to the distribution from `B` target
   permutations (within groups/blocks, as in F3), reported with a one-sided
   lower confidence bound on the skill. A feature is a *suspect* when the
   lower bound exceeds the charter's plausibility ceiling for the domain.
   This replaces the fixed 0.98 on in-sample tree accuracy.
2. *Conditional permutation test.* Permute the feature within strata of the
   other admitted features (or their model-based conditional); a feature that
   keeps large predictive value given the others and fails the as-of check is
   escalated. Missingness-as-leakage (recorded only for some outcomes) is
   tested with Fisher's exact test or chi-square plus Cramer's V.

A suspect with no as-of evidence is quarantined in strict mode; a suspect
with log-derived as-of evidence that it precedes the moment is admitted and
the suspicion kept in the report. Every test is a ledger entry (F3).

**DAG as hypothesis.** A DAG may come from domain experts or an LLM draft.
It is stored as a hashed input marked `hypothesis`, with the draft's source
and prompt. Conditional-independence tests on `X` can only *falsify* an
implied independence; they cannot confirm a DAG (Markov-equivalent graphs
are indistinguishable, tests have limited power). Each CI test is a ledger
hypothesis; a rejected implication is flagged, not auto-corrected.

**Reconciliation.** Row counts, key cardinalities and control totals of the
extract are compared with independent records (source counts, finance totals,
log counts). Gap = `|extract - source| / source`, with the explanation
required to be a named mechanism (filter, late arrival, dedupe) from
provenance; an unexplained gap blocks as `unverified`.

## Guarantees and assumptions

- Admissibility is only as good as the as-of facts; the tier says which.
- The skill test controls the rate of calling a feature suspect when it has
  no signal under the permutation null (size), given exchangeability within
  the permutation blocks. It says nothing about power against a true leak
  whose skill is below the ceiling. The detector's measured size and recall
  are its certificate (F5/F7), as P1 requires before it may gate.
- The lower confidence bound uses cross-fitted scores; fold dependence makes
  the interval approximate, which F7 checks by simulation.
- CI tests carry the usual faithfulness and sample-size assumptions; falsify
  only.

## Failure modes and degradation

| Situation | Behaviour |
|---|---|
| No logs, no attestation | Field `unverified`; strict quarantines, permissive flags. |
| Log disagrees with stored value | Field quarantined; difference rate reported. |
| Strong legitimate predictor | May be a suspect; logged evidence plus an attestation lets it through. |
| Few rows in `X` | Test reports low power; no quarantine from the statistic alone. |
| LLM-drafted DAG contradicted by data | Flag, keep hypothesis tier. |

## Repo integration (verified)

- Existing availability inputs: `unavailable_at_prediction` and
  `feature_delays` in `scmle/contracts/tasks.py:214-219`; roles built at
  `scmle/data/profiling/profiler.py:84-124` (exclusion at `:94`, time column
  at `:99`).
- Existing leakage heuristics to replace: `LEAK_ASSOCIATION = 0.98`,
  `MISSINGNESS_LEAK = 0.9` (`profiler.py:21-22`); `leakage_findings`
  (`profiler.py:278-310`); in-sample `threshold_association` with
  `TREE_LEAVES = 20` (`scmle/data/validation/columns.py:34,120-145`);
  `missingness_association` (`columns.py:190`).
- Existing gate: `leakage_policy` `warn|block` (`scmle/contracts/tasks.py:570`)
  checked by the marker string `LEAKAGE_MARKER = "leakage"`
  (`scmle/core/stages.py:93,498-503`). Proposed: replace the string match with
  typed suspicions and the admissibility table.
- Finding container: `CurationFinding` (`scmle/contracts/data.py:85`).
- Proposed new package: `scmle/data/admissibility/`.

## Acceptance criteria (machine-checkable)

1. Predictive estimand: a planted mediator known before the moment is
   admitted; a planted post-outcome field is quarantined. Causal estimand:
   a planted collider is excluded from the adjustment set.
2. Point-in-time join reconstructs planted overwrites (rate recovered within
   its binomial interval).
3. Skill test: on null features, suspect rate <= nominal (binomial tolerance);
   on planted leaks (threshold rule, outcome-conditioned recording) recall is
   measured and written to the certificate; Titanic `boat`/`body` is flagged.
4. Strict mode quarantines every `unverified` field; no field is `confirmed`
   without log-derived as-of.
5. Every CI test and every leakage test appears in the ledger.
6. DAG artifact is hashed, tier `hypothesis`; an edit changes the hash.
7. Reconciliation with an unexplained gap fails the gate; an explained one
   passes with the explanation recorded.

## External referent

Documented leakage cases (Titanic `boat`/`body`, Kaggle leakage
post-mortems) as a labelled corpus; the source system's own change logs and
DDL; a domain owner's recorded attestation; R `dagitty` for adjustment-set
oracles; permutation-test p-values compared with scipy.
