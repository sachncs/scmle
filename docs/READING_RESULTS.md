# How to read your results

A run leaves a folder full of files. This page tells you which ones to open,
what each number means, and how to tell a result you can rely on from one that
only looks good.

All numbers come from the real Titanic run in [Start here](START_HERE.md).
Terms are in the [glossary](GLOSSARY.md).

## The rule of thumb

1. Read the **exam score** (the locked test), not the practice score.
2. Read its **band** (confidence interval) as well as the score.
3. Compare it with a **baseline** (a model that does nothing clever).
4. Ask **why** it is good: look at which features carry it.
5. If it looks too good, it probably is. Hunt for a leak.

## Which file to open first

| You want to know | Open | Look for |
| --- | --- | --- |
| The headline | the last line of the run output | `selected ... (cv ...); test: ...` |
| The story in one page | `report.html` | findings first, then the leaderboard; the long plan list is folded at the bottom |
| Every number | `evaluation.json` | `test_metrics`, `selected_cv_score`, `fit_health`, `cv_confidence` |
| Every model tried | `leaderboard.parquet` or `result.leaderboard()` | `score_mean`, `score_std`, `status`, `fit_health` |
| Why each choice was made | `diagnostics/decision_log.json` | `selection`, `outcomes` |
| What the model relies on | `diagnostics/permutation_importance.json` | the first rows |
| Warnings about your data | `diagnostics/recommendations.json` and the "Findings" part of `report.html` | duplicates, label noise, leakage suspects |
| Whether the set-aside rows stayed sealed | `events.jsonl` | `split.created` before `selection.frozen` before `test.opened` |

A complete list of files is in [`ARTIFACTS.md`](ARTIFACTS.md).

## The two scores: practice and exam

```text
practice score   : 0.848 +/- 0.023
practice 95% band: 0.8 to 0.896
exam score       : 0.885
exam 95% band    : [0.841, 0.927]
dummy baseline   : 0.5
```

* **Practice score** (`selected_cv_score`, "cv" in the run output): the
  average over cross-validation rounds on the learning rows. The `+/-` is the
  spread between rounds. SCMLE used it to pick the winner.
* **Exam score** (`test_metrics`): the winner scored once on held-out data,
  rows nobody touched. This is the fair preview.
* **Band**: with 216 exam rows the true score could be 0.84 or 0.93. Quote the
  band, not just the point. When the bands of two models overlap, you cannot
  say which is better.
* **Baseline**: the dummy model scores 0.5. Always check that you beat it by a
  margin that matters to you.

A trustworthy gap between practice and exam is small, and the exam is often a
little lower. If the exam is much lower than practice, the model memorised.
If the exam is higher, that is luck or a small exam set; do not count on it.

## The leaderboard

`result.leaderboard()` gives one row per trial. The columns you need:

| Column | Meaning |
| --- | --- |
| `model_name` | the model recipe |
| `status` | `ok`, `failed`, `pruned` (given up early), `skipped` (out of time) |
| `score_mean`, `score_std` | practice score and its spread over folds |
| `strategy` | the data treatment used, such as `iterative_imputation` or `default` |
| `fit_health` | `healthy`, `overfit` or `underfit` |
| `selected` | true for the winner |

From our run:

```text
                   model_name status  score_mean  score_std
0                    stacking     ok    0.848248   0.022959
1                    blending     ok    0.847238   0.023506
2  bagged_logistic_regression     ok    0.834896   0.007738
```

How to read it:

* The first two rows differ by 0.001, and the spread is 0.023. They are tied.
  When a simple model is tied with a complex one, prefer the simple one: it is
  easier to explain and to run. SCMLE breaks exact ties toward the simpler
  model, but a near-tie is your decision.
* A trial marked `overfit` has a big gap between its training and validation
  gain. Do not ship it even if its score is high.
* Failed or skipped trials are normal. Check the `error` column if a model you
  expected is missing.

## The diagnostics, one by one

All live in `diagnostics/`. They describe the winner on the exam rows.

| File | What it tells you | A good sign | A bad sign |
| --- | --- | --- | --- |
| `classification_report.json` | right and wrong guesses per class | both classes do reasonably | one class is nearly all wrong |
| `calibration.json` | whether "0.8" means 80 percent | predicted and observed frequencies close; low Brier score | a curve far from the diagonal |
| `threshold_analysis.json` | precision and recall at each cut | a cut that suits your costs exists | no cut gives acceptable both |
| `metric_uncertainty.json` | the band around the exam score | narrow | very wide: too few exam rows |
| `permutation_importance.json` | which features matter (the feature importance) | sensible features lead | an id, a date or a "result" column leads |
| `slice_performance.json` | score for each group, such as sex = male | gaps small | a big gap; ours: male 0.727 vs overall 0.885, worth a look |
| `robustness.json` | score when data is degraded (missing, noisy, unseen categories) | `fragile_under` empty | a scenario that collapses the score |
| `adversarial_validation.json` | can a classifier tell practice rows from exam rows? | AUC near 0.5 | AUC far above 0.5: the exam rows differ from practice; ours 0.61 "moderate shift" |
| `fairness.json` | gaps between groups of sensitive columns | gaps small | large gaps; `available: false` if you named none |
| `error_analysis.json` | the kinds of rows the model gets wrong | no big pattern | a cluster of failures to investigate |
| `recommendations.json` | advice from the profile | empty | a duplicates or group warning |

The run also writes `model_card.json`, the model card: a short card with the model's intended
use, scores and limits. Read it before you hand the model to anyone.

Read slices carefully. In our run the model is much worse for men (0.727) than
for women (0.884). That is a real finding: the overall 0.885 hides a weaker
group. Whether it matters depends on how you will use the model.

## The decision log

`diagnostics/decision_log.json` has three parts:

* `plan`: each technique SCMLE considered and whether it was `include`d,
  `reject`ed (with the reason) or `defer`red. Many `reject` lines such as
  `missing dependency: lightgbm` are normal.
* `outcomes`: what happened to each technique after trials, such as
  `class_weight ... not applicable: classes are balanced enough (1.6:1 < 3:1)`.
* `selection`: the model and settings that won, and the rule used
  (`best mean CV score; exact ties -> simpler_model; dummy baseline excluded`).

Use it to answer "why did it pick this?" and "why did it not try X?". If you
disagree with a choice, change the option and run again.

## Fit health

`fit_health` compares how much the model gained over the baseline in training
with how much it kept in validation.

* `healthy`: what it learned in training carried over (we kept 78 percent of
  the training gain).
* `overfit`: it kept little. The trial `hist_gradient_boosting` was marked
  this way in our run.
* `underfit` (underfitting): it learned too little even in training.

## What a trustworthy result looks like

* The exam score is clearly above the baseline.
* Practice and exam scores are close.
* The band is narrow enough for your decision.
* The top features make sense to someone who knows the subject.
* The fit health is `healthy`.
* The leaderboard has several models near the top rather than one miracle.
* There are no unexplained warnings: duplicates, leakage, label noise,
  shift. If there are, you can say why they do not matter.
* The scores do not change much when you run again with another seed.

## What a suspicious result looks like

| What you see | What it probably means | What to do |
| --- | --- | --- |
| Exam score near 1.0 (or an error near 0) on real, messy data | leakage | check `permutation_importance.json`; find the feature; see below |
| One feature dominates the importance | leakage, or an id that identifies the row | ask when that column is filled in |
| Practice far above exam | overfitting, or duplicates in the practice rows | more data, simpler models, set a group column |
| Exam far above practice | a lucky or unrepresentative exam set | check the band; check `adversarial_validation` |
| Exam score about equal to the baseline | no signal, or the wrong target | check the target and features; look at the data again |
| `fragile_under` lists scenarios | the model breaks when inputs are a bit worse | expect real-life data to be messier; fix inputs or retrain on messier data |
| A large `slice_performance` gap | the model is weaker for a group | decide if it is acceptable; collect more data for that group |
| `WARNING ... leakage suspect(s)` at the end of the last line, or entries in the "Findings" section at the top of the report | something needs a look | read the "Findings" section |

### The leak, as it looked

The author ran the same table twice. The only difference was `boat` and `body`:

| | Without them (`--unavailable-at-prediction boat body`) | With them (`--leakage-policy warn`) |
| --- | --- | --- |
| Exam ROC AUC | 0.885 | 0.988 |
| Top feature | `sex`, importance 0.242 | `boat`, importance 0.334 |
| Chosen model | stacking | bagged_logistic_regression, strategy `select_features` |

The second run only happened because the stop was turned off. By default
SCMLE refuses it (`blocked: unresolved leakage risk ...`). With `warn` the run
finished and the last line ends with
`WARNING 2 leakage suspect(s) kept in: boat, body`. The "Findings" section at
the top of its `report.html` and the plan warnings contain:

```text
feature 'boat' is a leakage suspect: cross-fitted skill 0.973 ... This is a suspicion, not proof ...
```

Two lessons. First, a score of 0.988 for "who survived the Titanic from a
ticket" should have rung a bell: the real answer is not that predictable.
Second, that is why `block` is the default: it stops the run instead of
warning. Turn it off (`--leakage-policy warn`) only when you have looked at
every suspect and decided it is a legitimate predictor.

A leakage suspicion is "not proof": a really strong, legitimate feature looks
the same to a statistical screen. The cure is a human question: "Would I know
this value at the moment of prediction?" If yes, you can state so with an
attestation (see [When things go wrong](WHEN_THINGS_GO_WRONG.md)). If no,
exclude it.

## Duplicates and noise

The run also told us:

```text
14% of rows are exact duplicates; duplicates can appear in both training and validation folds and inflate scores.
```

It means the exam score of 0.885 may be a little high, because a copy of an
exam row might sit among the learning rows. Cure: remove duplicates, or name a
group column if copies belong to one entity. In a charter run, SCMLE keeps
copies on the same side of the split automatically (the run logged
`110 groups of duplicate rows (339 rows) kept on one side of every split`).

## Reading calibration and thresholds

If you use the predicted chance (for example, "call everyone above 70
percent"), read `calibration.json`. Each bin lists the chance the model said
(`mean_predicted`) and how often it happened (`observed_frequency`). In our
run the bin around 0.14 saw 0.10: a model that is slightly too pessimistic.
Close enough for ranking, but do not quote its probabilities as exact.

`threshold_analysis.json` shows what each cut-off buys:

```text
threshold 0.5: precision 0.78, recall 0.76
threshold 0.8: precision 0.94, recall 0.55
threshold 0.3: precision 0.67, recall 0.85
```

A higher cut means fewer false alarms and more misses. Choose the cut from the
cost of each (see [Decisions](DECISIONS.md)).

## A second mode: the charter

Everything above is the ordinary mode. The careful mode adds a written plan
(the charter) and sealed rows. You use it when someone will act on a *finding*
("fare is linked to survival") and needs to know it was not luck. Create the
plan, check it, sign it and run with it:

```bash
scmle charter init charter.json
scmle charter check charter.json --rows 1093
scmle charter sign charter.json
scmle run titanic.csv --target survived --unavailable-at-prediction boat body --charter charter.json
```

(Edit the file *before* you sign it: `charter sign` wraps the plan inside a
`charter` key next to the hash and signature, so later edits must go inside
that key and break the signature anyway.)

(`charter init` writes a template that fails `check` until you fill in the
question, population, target, prediction moment, one intercurrent event with
its strategy, unit and deployment population. The template does not show the
shape of the event; it is `{"event": "...", "strategy": "..."}`.)

When the charter is filled in correctly, `check` prints something like:

```text
charter be7b4e8c968abdd66a37234df0280bacc2c9bfae4837ccaa6709b955fc90f813 signature verified
feasibility: feasible (273 rows >= 214 needed for power 0.8)
```

`--rows` is the number of development rows: the table minus the exam rows
(1309 minus 216 is 1093). With the default effect size of 0.2, the same table
gave
`feasibility: downgrade (273 rows < 333 needed for power 0.8 at effect 0.2; detectable effect is 0.221)`.
That is a warning that your sealed set is too small for your promise. You
either lower the promise (`min_effect` 0.25 made it feasible) or you get more
rows.

After the run, list the claims:

```bash
scmle claims list runs_charter/run_20261105_001
scmle claims verify runs_charter/run_20261105_001
```

Real output:

```text
h000001  confirmed   pclass                    adj.p=2.358e-06  effect=-0.2927
h000004  confirmed   parch                     adj.p=0.0005005  effect=0.2186
h000005  confirmed   fare                      adj.p=2.941e-08  effect=0.344
3 claim(s)
```

```text
PASS    ledger chain verifies and decisions replay: 31 rows, head 82d3ed82f6f8
PASS    claims file names the ledger head: head 82d3ed82f6f8
PASS    charter verifies and matches the ledger: be7b4e8c968a (signature verified)
SKIPPED chartered rules were used: the charter fixes no rules
PASS    claims agree with the sealed ledger: 3 confirmed of 3 claims
PASS    confirmation replays from the p-values on C: 3 hypotheses, rule by
SKIPPED second-stage accounting is consistent: no withdrawal rule ran
PASS    custody log: selection frozen before C opened, C opened once: 5 entries, chain verified
PASS    negative control matches the ledger: hash equal
7 of 9 steps passed
```

How to read the claims:

* Each row is a **claim**: a finding that survived exploration on rows `X` and
  was then tested once on the sealed rows `C`. Here: "`pclass`, `parch` and
  `fare` are each linked to survival".
* `confirmed` is the tier: it passed on fresh rows, with the stated error
  allowance for the whole list.
* `adj.p` is the p-value adjusted for testing several things. Small is good.
* `effect` is the size and direction (a rank correlation). `pclass` is
  negative: higher class number (lower class) goes with less survival.
* Note what is *not* claimed: that these features *cause* survival, or that
  the model is good. The claim is only an association.
* `SKIPPED` in `verify` is not a failure. It means the step did not apply
  (here, the charter fixes no extra rules). `SKIPPED` is never counted as
  `PASS`. A `FAIL` makes the command exit with code 1.

### The tiers in plain words

| Tier | Plain meaning | Example from this run |
| --- | --- | --- |
| `confirmed` | Passed a one-time test on rows that were sealed from you. | the three claims above |
| `attested` | A named person vouched for a fact the data cannot show, such as "this column is known before the outcome". | none yet in this run |
| `unverified` | SCMLE used a default and nobody has checked it. | every feature's availability in `admissibility_table.json`: `availability at serving is unknown` |

`unverified` is not a failure, and not a block. It is an honest label: "this
rests on an assumption nobody has checked". What to do with it is in
[When things go wrong](WHEN_THINGS_GO_WRONG.md).

The sealed-row sizes in this run: X had 819 rows, C had 273, and the final test
set 216. In `split_manifest.json` you can see which row ids went where.

### Other charter files worth a glance

* `feasibility.json`: `verdict: feasible`, `n_confirmation: 273`,
  `n_required: 214`, `power_at_min_effect: 0.907`. The sealed set was big
  enough.
* `hypothesis_ledger.jsonl`: the hypothesis ledger, the log of every idea
  tested on `X`.
* `negative_control.json`: SCMLE re-ran the search 20 times on shuffled
  targets. They found 0 discoveries, while the real data found 3. If the
  shuffled runs found as many, the discoveries would be noise.
* `stability_report.json`: `status: not_evaluated` because the charter declared
  no alternative choices to vary. That is honest, not a pass.
* `shift_report.json`: `status: unverifiable` because no sample of the
  deployment population was supplied. It never says `pass` without evidence.
* `admissibility_table.json`: per feature, `admit_flagged` (allowed, with a
  note), or `quarantine` (kept out). `boat` and `body` are quarantined.

## A last check before you rely on a result

* Can I explain, in one sentence, why this model should work?
* Did I look at the top features and recognise them?
* Does my exam band contain a value I would be happy with?
* Is every warning in the report either fixed or understood?
* Is there a written "what would make me stop" (see
  [Decisions](DECISIONS.md))?

If any answer is no, go back before you ship.
