# State-of-the-art comparison protocol (v1)

This document is committed and tagged (`sota-protocol-v1`) **before any result
exists**. It fixes the claim, the tasks, the systems, the budgets and the
statistics so the outcome can fail. Changing it after results exist requires a
dated amendment (see the end), and an amendment can never touch the hold-out
rule.

## 1. The claim being tested

At budget *B*, on the **hold-out** tasks, SCMLE is *close to the state of the
art* if both hold:

1. its median normalized score is within **0.02** of the best reference
   system's, and
2. the 95% bootstrap interval of (best reference - SCMLE) over tasks has an
   upper bound no larger than **0.05**.

The claim is evaluated separately for each budget actually run, and is reported
as met or not met. A budget that was not run supports no claim.

## 2. Tasks (verified against the public benchmark on 2026-10-24)

* Source: TabArena-v0.1 curated tasks (`autogluon/tabarena`, Apache-2.0),
  **51 OpenML tasks**: 30 binary, 8 multiclass, 13 regression; 748 to 150,000
  rows. The pinned table is `tasks_tabarena_v0pt1.csv`. Each task has 3 folds;
  most have 3 or 10 repeats.
* Split into **development** (31 tasks) and **hold-out** (20 tasks) by
  `split.py`: group by problem type, order by `sha256(seed:task_id)`, first 60%
  of each group is development. Seed `scmle-sota-protocol-v1`. Reproduce and
  verify with `python -m benchmarks.sota.split --check`.
* **Exclusions at freeze time: none.** SCMLE supports all three problem types.
  Any later exclusion must be justified in an amendment committed before the
  hold-out is opened, and may not depend on any SCMLE result.
* The hold-out is opened **once**, for the final evaluation in section 6. Until
  then no hold-out score of any system may be computed or looked at.

## 3. Systems

| System | Version (PyPI, checked 2026-10-24) | Notes |
| --- | --- | --- |
| SCMLE | the tagged commit under test | this repository |
| AutoGluon-Tabular | 1.6.3 | requires Python >=3.10,<3.14: run in a separate environment (the repository's own environment is 3.14); presets `best_quality` and `high_quality` |
| FLAML | 2.7.0 | |
| LightGBM / XGBoost / CatBoost | 4.7.0 / 3.4.1 (needs Python >=3.12) / 1.2.10 | each individually tuned with the same budget |
| TabPFN | 9.2.0 | only on tasks the benchmark marks as runnable; licensed, may need `TABPFN_TOKEN` |

Versions are re-checked at the start of Step 1 and recorded with every result
file. A newer release is used only through an amendment.

The comparison harness is TabArena's, so reference systems are run the way
their maintainers run them. TabArena also publishes cached validation and test
predictions for many methods; those are used for the offline simulations in
Step 2 only, never as a substitute for running a system under our budget.

## 4. Budgets and hardware

* Wall-clock budgets for the **whole** run (search, training, ensembling):
  **60 s, 600 s, 3600 s**, identical for every system.
* All systems run on the same machine class, one task at a time, with the same
  CPU and memory limits. The machine is recorded in every result file. No GPU
  unless every system uses it.
* **Tiers**, because full runs are expensive. Estimated cost for one system on
  development at all three budgets and 6 splits is about 9 days of
  single-machine time, so results are labelled by tier and claims are made only
  for the tier that was run:
  * **Lite**: first split of each task, budgets 60 s and 600 s.
  * **Full**: 6 splits on development (3 folds x repeats 0-1), 3 splits on
    hold-out (3 folds x repeat 0), all three budgets.

## 5. Scores and statistics

* Per-task metric: the task's own evaluation metric as defined by TabArena
  (checked against its metadata in Step 1).
* **Normalized score** per task and split: `(s - worst) / (best - worst)` over
  all compared systems, oriented so higher is better. A system that fails or
  times out scores the worst value for that task and is counted as a failure in
  the report; nothing is silently dropped.
* Headline: **median normalized score over tasks**, after averaging splits
  within a task.
* Intervals: bootstrap over tasks, 10,000 resamples, fixed seed
  `scmle-sota-bootstrap-v1`.
* Comparison tests: Friedman test with Wilcoxon-Holm post hoc across systems,
  alpha 0.05, plus the benchmark's own Elo-style ranking as a cross-check.
* Time: report the budget used and the time actually used by each system.

## 6. Rules that protect the result

1. Protocol tag before results (Step 0 acceptance).
2. Harness calibration (Step 1) must pass before SCMLE numbers are used.
3. All tuning, portfolio selection and ablation use the **development** tasks
   only.
4. The code evaluated on hold-out is a tagged commit that existed before the
   run; the statistics script is unchanged since this protocol was tagged.
5. Raw per-task results and logs are archived with SHA-256 checksums.
6. Losses are published with the wins.

## 7. Stop rules

* If the harness does not reproduce a reference system's published score within
  its interval and the difference cannot be explained, fix the harness; do not
  produce SCMLE numbers.
* If the median total deficit on development is within the noise level
  (95% interval contains zero), skip the improvement step and evaluate the
  claim as is.
* If no change clears the ablation criterion (real gain positive, 95% interval
  excluding zero), stop changing and evaluate as is.

## 8. Amendments

An amendment is a dated section appended here, committed before the
hold-out is opened, stating what changed and why. It cannot change the hold-out
assignment, the claim thresholds in section 1, or the seeds in section 5.

*(none)*
