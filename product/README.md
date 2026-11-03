# Product: the SCMLE qualification instrument

Canonical names and principles: [`../plan/00-canon.md`](../plan/00-canon.md).

## What it is

A data-stage instrument that turns a signed charter into confirmed,
tier-labelled claims about a dataset, with every detector carrying a measured
certificate. It runs unattended; a human is needed only to supply facts the
data cannot (field provenance, as-of times, domain sign-off), and the product
degrades visibly when they are absent.

## What it is not

- Not a larger EDA tool. A check that cannot change an action, or has no
  measured operating characteristic, is shown but never gates.
- Not a proof of correctness. Guarantees are probabilistic and conditional on
  recorded assumptions.
- Not adversary-proof custody on a single machine.

## Features

| ID | Feature | One line |
|---|---|---|
| F1 | [Charter and decision map](features/F1-charter-and-decision-map.md) | Question, estimand, unit, population, feasibility, error budget, signed. |
| F2 | [Custody and splits](features/F2-custody-and-splits.md) | Exploration / confirmation / forward sets, hashed manifest, sealed access. |
| F3 | [Claim ledger and confirmation](features/F3-claim-ledger-and-confirmation.md) | Online FDR ledger, negative controls, unlock-once confirmation. |
| F4 | [Admissibility and provenance](features/F4-admissibility-and-provenance.md) | As-of facts, field provenance, leakage tests, DAG falsification. |
| F5 | [Detectors with certificates](features/F5-detectors-with-certificates.md) | Missingness, labels, duplicates, sentinels, outliers, each with measured size and recall. |
| F6 | [Robustness and shift](features/F6-robustness-and-shift.md) | Multiverse stability, two-sample shift tests, forward check. |
| F7 | [Verification harness](features/F7-verification-harness.md) | Corpora, differential tests, mutation tests, predictive-validity backtest, red team. |

## Outputs of a run

`analysis_charter.json`, `split_manifest.json`, `error_budget.json`,
`hypothesis_ledger.jsonl`, `detector_certificates.json`,
`admissibility_table.json`, `stability_report.json`, `shift_report.json`,
`confirmed_claims.json`, `decision_log.jsonl`, `eda_report.html`, and the
modelling specification handed to the existing pipeline.

## Evidence tiers

`confirmed` > `attested` > `unverified`. Shown on every claim, in the CLI
summary and in the report header.
