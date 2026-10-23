# Measure, observe, evaluate, evolve

Production quality is a feedback loop, not a one-time approval. The project
uses four kinds of evidence:

1. **Measure** — CI records test outcomes, coverage, benchmark results,
   dependency findings, package smoke tests, and dataset-curation findings.
2. **Observe** — each run writes event records, manifests, fingerprints,
   curation reports, model metadata, and monitoring reference profiles. These
   artifacts make failures and data changes inspectable after the process has
   moved on.
3. **Evaluate** — compare a candidate with its locked evaluation data, the
   benchmark baseline, the prior champion, and the declared operational
   thresholds. A green model score does not override a curation, leakage,
   reproducibility, or security failure.
4. **Evolve** — turn recurring findings into a test, a validation rule, a
   benchmark case, a documented operating decision, or a deliberately scoped
   product change. Every change should leave an auditable artifact and a small
   atomic commit.

## Evidence map

| Question | Primary evidence | Review trigger |
| --- | --- | --- |
| Did the code remain correct? | PR contract suite and scheduled full-suite coverage | Failed test, unexplained skip, or coverage regression |
| Did performance regress? | Offline benchmark results and regression checker | Any failed benchmark gate or material drift from baseline |
| Did the data remain trustworthy? | `curation_report.json`, manifest, fingerprint, leakage checks | New schema, missingness, duplicates, drift, or representativeness finding |
| Can the result be reproduced? | Run events, checkpoints, bundle checksums, environment metadata | Resume mismatch, corrupt artifact, or environment change |
| Is the deployed model still healthy? | Monitoring report, labeled coverage, drift and rollback decision | Threshold breach or insufficient outcome coverage |
| Is the release safe? | Bandit, pip-audit, package smoke tests, platform matrix | Any security finding or platform-specific failure |

## Review cadence

- Every pull request runs the bounded contract tier and records its coverage
  artifact; the enforced floor is currently 50%.
- Every scheduled full-suite run publishes coverage and benchmark evidence.
- Every dataset or schema change requires reviewing the curation report before
  model comparisons are interpreted.
- Every candidate promotion requires locked-data evaluation, monitoring
  reference approval, an independent `PromotionApproval` bound to the exact
  bundle digest, and a rollback decision.
- Every release requires the declared release evidence to be complete.

When a signal changes, preserve the evidence first, identify the smallest
reproducible cause, and then evolve the system with a focused test or control.
