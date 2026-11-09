# Glossary

Every word you will meet, in plain language. Each entry has a meaning, an
example and why it matters. The examples use the Titanic passenger table from
[Start here](START_HERE.md). Entries are in groups, from the first ideas to
the production ones. Use your browser's search to find a word.

* [The data and the question](#the-data-and-the-question)
* [Scores](#scores)
* [Honest testing](#honest-testing)
* [Charter, custody and claims](#charter-custody-and-claims)
* [Models and decisions](#models-and-decisions)
* [Watching a model in use](#watching-a-model-in-use)
* [Shipping and trust](#shipping-and-trust)
* [Serving and operations](#serving-and-operations)

## The data and the question

### target

**Meaning.** The one column you want to predict.
**Example.** `survived` (1 lived, 0 died).
**Why it matters.** Everything else is built around it. A wrong target means a
perfect answer to the wrong question. See [Decisions](DECISIONS.md).

### feature

**Meaning.** A column the model may use to guess the target.
**Example.** `sex`, `age`, `pclass`, `fare`.
**Why it matters.** The model can only learn from what you give it. A feature
you would not have at prediction time is a leak (see below).

### row, column, table

**Meaning.** A table has rows (one per thing, here one per passenger) and
columns (one fact about each thing).
**Example.** 1309 rows and 10 columns.
**Why it matters.** The unit of a row ("one passenger", "one customer-month")
decides how you must split the data. See "grouping" in
[Decisions](DECISIONS.md).

### task

**Meaning.** The kind of answer: a category (classification), a number
(regression), a time until something happens (survival), and so on.
**Example.** `binary_classification` for two possible answers.
**Why it matters.** It decides which scores and models make sense. SCMLE works
it out from the target and tells you in the plan; check it.

### classification, regression

**Meaning.** Classification predicts a category. Regression predicts a
number.
**Example.** "Lived or died" is classification. "Fare paid" is regression.
**Why it matters.** Using the wrong one gives meaningless scores.

### missing value

**Meaning.** A cell with no recorded value (shown as `NaN` or empty).
**Example.** `age` is missing for 263 passengers.
**Why it matters.** Missing is often not random. That `boat` is missing for the
people who died is a clue to a leak, not noise.

### duplicate

**Meaning.** A row that is an exact copy of another.
**Example.** 234 of the Titanic rows are copies.
**Why it matters.** If one copy lands in the practice part and another in the
exam part, the model has seen the exam answer. Scores come out too high.

## Scores

### metric

**Meaning.** The number that says how good the predictions are.
**Example.** ROC AUC, accuracy, RMSE.
**Why it matters.** The search picks the model that scores best on your metric.
A metric that does not match your real goal picks the wrong model. See
[Decisions](DECISIONS.md).

### ROC AUC

**Meaning.** For a yes/no prediction: the chance that the model gives a higher
score to a random "yes" than to a random "no". 0.5 is a coin flip, 1.0 is
perfect.
**Example.** 0.885 means a random survivor is ranked above a random
non-survivor 88.5 percent of the time.
**Why it matters.** It is the default for two-answer problems. It ignores the
threshold (below), so it says nothing about how many mistakes you make at the
cut you will actually use. A score above about 0.98 on real, messy data
deserves suspicion.

### accuracy

**Meaning.** The share of predictions that are right.
**Example.** 0.824: 82 of every 100 passengers guessed right.
**Why it matters.** It misleads when one answer is rare. If 1 in 100 cases is
fraud, a model that says "never fraud" is 99 percent accurate and useless.

### precision and recall

**Meaning.** Precision: of the ones you flagged, how many were truly "yes".
Recall: of all the true "yes", how many you flagged.
**Example.** At a cut of 0.5, precision 0.78 and recall 0.76 for survivors.
**Why it matters.** They trade off against each other. Which matters more
depends on the cost of each kind of mistake.

### confusion matrix

**Meaning.** A 2 by 2 table of right and wrong guesses. Rows are the truth,
columns are the guess.
**Example.** `[[113, 18], [20, 65]]`: 113 deaths right, 18 deaths guessed as
lived, 20 survivors guessed as died, 65 survivors right.
**Why it matters.** It shows which kind of mistake you make, which a single
number hides.

### log loss, Brier score

**Meaning.** Scores that punish confident wrong probabilities. Lower is better.
**Example.** Log loss 0.408, Brier 0.127 in our run.
**Why it matters.** Use them when you will act on the probability itself, not
only on the yes/no.

### baseline, dummy model

**Meaning.** A trivial model to beat: one that ignores the features.
**Example.** The `dummy` model scores ROC AUC 0.5.
**Why it matters.** A model that does not clearly beat the baseline has learned
nothing. Also compare against what your team does today (a "current process").

### confidence interval (band)

**Meaning.** A range that the true value probably lies in, given that you only
tested on a sample.
**Example.** Exam score 0.885, 95 percent band 0.841 to 0.927.
**Why it matters.** A small test set makes a wide band. Two models whose bands
overlap are not clearly different.

### calibration

**Meaning.** Whether stated probabilities match reality. Of all cases given 0.8,
about 80 percent should happen.
**Example.** In our run the model said 0.14 for a group where 0.10 lived: close.
**Why it matters.** If you act on the number ("call everyone above 70 percent")
it must mean what it says. Check `diagnostics/calibration.json`.

### threshold (cut-off)

**Meaning.** The probability above which you say "yes".
**Example.** 0.5 by default. At 0.8, precision is 0.94 and recall 0.55.
**Why it matters.** It sets how many true cases you catch against how many
false alarms you raise. Choose it from the cost of each mistake, not from habit.

### PR AUC

**Meaning.** PR AUC is a similar score that focuses on the rare answer.
**Example.** 0.854 in our run.
**Why it matters.** Prefer it when "yes" is rare.

## Honest testing

### training data, practice rows

**Meaning.** The rows the model learns from.
**Example.** 1093 of our 1309 rows.
**Why it matters.** The model will always look good on rows it has already
seen. Those scores say nothing about the future.

### held-out data (exam rows, test set)

**Meaning.** Rows kept hidden until the end and used once to score the winner.
**Example.** 216 rows were hidden. SCMLE logs `test.opened` when it uses them.
**Why it matters.** It is the closest thing to an honest preview of real use.
If you peek at it and then change the model, it stops being honest.

### cross-validation (CV)

**Meaning.** Practice rounds. Split the practice rows into several parts, train
on some, score on the rest, rotate, and average.
**Example.** 5 folds; the stacked model averaged 0.848 with a spread of 0.023.
**Why it matters.** It lets the search compare many models without touching the
exam rows. The "cv" number is the practice score.

### overfitting

**Meaning.** The model memorised the practice rows, quirks included, and does
worse on new ones.
**Example.** A gap such as practice 0.99 and exam 0.70. SCMLE's `fit_health`
says `overfit` for such trials; one of our trials was marked that way.
**Why it matters.** It is the most common reason a model that looked great in
the lab fails in use.

### underfitting

**Meaning.** The model is too simple to catch the pattern.
**Example.** Both practice and exam scores are poor.
**Why it matters.** Rarely the problem with this tool, but it looks like
"the data has no signal".

### leakage (leak)

**Meaning.** Information in a feature that you would not have when you really
use the model, usually because it was recorded after the outcome.
**Example.** `boat` (which lifeboat) and `body` (body recovered) are only known
after the sinking. With them the exam score was 0.988. Without them, 0.885.
**Why it matters.** It makes a model look brilliant and then fail. It is the
single biggest cause of too-good results. See
[Reading results](READING_RESULTS.md).

### leakage suspect

**Meaning.** A feature SCMLE's statistical screen flags as probably a leak.
**Example.** `boat` and `body`.
**Why it matters.** It is a suspicion, not proof. A genuinely strong feature
looks the same. By default a suspect does not stop the run (`leakage_policy`
is `warn`); the score is labelled provisional until a person settles it. See "unverified" and "attestation".

### false flag

**Meaning.** A column the leakage screen flags although it is legitimate: its
values really are known at prediction time.
**Example.** Two of the 30 breast-cancer measurements (`worst radius` and
`worst area`) when the whole table is screened.
**Why it matters.** The screen cannot avoid them, because a strong honest
predictor and a leak look alike in the data. See [Leakage](LEAKAGE.md).

### provisional

**Meaning.** A score that may change once a person settles an open question.
**Example.** `test (PROVISIONAL: 2 flagged column(s) not attested: boat, body)`.
**Why it matters.** It tells you not to quote the number yet.

### split

**Meaning.** How rows are divided into practice and exam parts.
**Example.** `stratified`: each part keeps the same share of survivors.
**Why it matters.** A wrong split lets exam information leak into practice.
Related rows must stay on the same side (see "group" and "time order").

### group, grouping

**Meaning.** Rows that belong together, such as all rows of one customer.
**Example.** `--group-column customer_id`.
**Why it matters.** If one customer is in both practice and exam, the model has
effectively seen the exam. Scores come out too high.

### time order, temporal split

**Meaning.** Training on the past and testing on the future.
**Example.** `--time-column event_time`.
**Why it matters.** Shuffled rows let the model learn from the future. A model
for the future must be tested on the future.

### dataset fingerprint

**Meaning.** A short code that identifies exactly these rows.
**Example.** `split_fingerprint: 4bb647ae69c0aca6`.
**Why it matters.** It proves that two runs used the same data and the same
split.

### seed, reproducible

**Meaning.** A seed is a starting number for randomness. Same seed, data and
settings should give the same result.
**Example.** The command-line run and the Python run in
[Start here](START_HERE.md) gave identical scores.
**Why it matters.** You can check someone else's result. Note that a time
budget makes the search depend on computer speed.

## Charter, custody and claims

These words belong to the careful, "pre-registered" way of working. You can
build your first model without them. You need them when someone must be able
to trust a finding, not just a model.

### charter

**Meaning.** A written, signed plan made before looking at the data: the
question, who the model is for, when it will be used, how many rows are held
back and how much risk of a false finding you accept.
**Example.** "Can we predict from the ticket who survived? Population:
passengers. Prediction moment: at boarding. Measure: ROC AUC."
**Why it matters.** It stops you changing the question after seeing the
answer. A change to even one byte breaks its signature. Create one with
`scmle charter init`.

### estimand

**Meaning.** The exact thing you are estimating: who, what, when and how it is
summarised.
**Example.** Population "passengers", target "survived", moment "at boarding".
**Why it matters.** Vague questions give answers that cannot be checked.

### custody (X, C and F)

**Meaning.** Custody means the data is divided into locked sets with rules for
who may look. **X** is the exploration set: you may look at it as much as you
like. **C** is the confirmation set: sealed, opened exactly once, after you
have decided what to test. **F** is the forward-in-time set, for data with
dates: the newest rows, kept for a final check.
**Example.** In the Titanic charter run, 819 rows were X and 273 were C.
**Why it matters.** Anything you find by looking is at risk of being luck.
Testing it once on rows you never looked at is the only way to know. The seal
is held in a separate helper program so ordinary code cannot read C.

### final test set

**Meaning.** The exam rows used to score the finished model (F for data with
dates).
**Example.** 216 rows.
**Why it matters.** It is separate from C, so a confirmed finding never shapes
the model that is scored.

### hypothesis ledger

**Meaning.** A log of every idea tested on X, whether it passed, and how much
"error allowance" it used.
**Example.** `hypothesis_ledger.jsonl` in a charter run.
**Why it matters.** If you try 100 ideas, a few will look good by luck. The
ledger counts the tries so luck is not mistaken for discovery.

### error budget, alpha

**Meaning.** How much risk of a false discovery you accept, stated up front.
**Example.** `alpha_confirm` of 1/20: about one in twenty confirmed findings
may be false.
**Why it matters.** It is a promise about the confirmed list as a whole.

### FDR (false discovery rate)

**Meaning.** The expected share of your "discoveries" that are false.
**Example.** With 3 confirmed claims at 5 percent FDR, you expect at most
about 0.15 of one false claim.
**Why it matters.** It is the right yardstick when you test many ideas.

### p-value

**Meaning.** How surprising the data would be if there were no real effect. Small
means surprising. It is not the chance that you are right.
**Example.** 5e-09 for "fare is associated with survival".
**Why it matters.** Many tests make small p-values appear by chance, which is
why the ledger and the error budget exist.

### e-value

**Meaning.** A different evidence score: a number you can keep multiplying as
data arrives, and which stays honest even if you look again and again.
**Example.** Used by the closed loop to decide whether a new model has beaten
the old one, in a way that stays valid however often you check.
**Why it matters.** Ordinary tests break if you peek repeatedly. E-values do
not, so they fit monitoring over time.

### claim

**Meaning.** A finding that survived exploration on X and was then tested once
on C.
**Example.** "fare is associated with survival", effect 0.344, confirmed.
**Why it matters.** Only claims, not hunches, carry the error guarantee. List
them with `scmle claims list`.

### tier (confirmed, attested, unverified)

**Meaning.** A label on every claim saying how much evidence stands behind it.
`confirmed`: it passed the one-time test on fresh rows. `attested`: a named
person vouched for a fact the data cannot show (for example, "this column is
known before the outcome"). `unverified`: SCMLE used a default and nobody has
checked it.
**Example.** In the Titanic charter run, the three claims are `confirmed`; the
feature entries in `admissibility_table.json` are `unverified` ("availability
at serving is unknown").
**Why it matters.** It tells you how far to trust a statement. `unverified` is
not an error. It is an honest "nobody has checked this yet". See
[Reading results](READING_RESULTS.md) and
[When things go wrong](WHEN_THINGS_GO_WRONG.md).

### attestation (attestor)

**Meaning.** A signed statement by a named person about something data cannot
prove.
**Example.** "I confirm that column `fare` is known at boarding."
**Why it matters.** It moves a claim from `unverified` to `attested`, and
makes a person responsible. SCMLE refuses an attestation signed with a key the
system itself holds.
**Attestor.** The person or role who signs. For a causal claim the charter
names the attestors in advance and says which assumption each may vouch for;
a statement from anyone else, for another charter, past its expiry date, or
with a signature that does not verify, does not count, and the assumption then
has no named attestor (the claim stays `unverified`). A valid signature shows
the statement was issued unchanged by someone holding the signing key. It
does not show who that was, or that the assumption is true.

### admissibility

**Meaning.** The check, per feature, of whether the model may use it:
`admit`, `admit_flagged` (allowed with a warning) or `quarantine` (kept out).
**Example.** `boat` was quarantined; `sex` was `admit_flagged`.
**Why it matters.** It is the formal record of which columns you trusted.

### causal claim

**Meaning.** A claim that changing X changes Y, not just that they move
together.
**Example.** "Giving the discount raised sales" versus "customers who got the
discount bought more".
**Why it matters.** Prediction does not tell you what would happen if you
acted. A causal claim needs a design (an experiment or assumptions that
someone names). SCMLE gives it a tier. Without a named attestor with a
verified signed statement for each assumption, it stays `unverified`. A held-out set alone can never confirm a
causal effect.

### negative control

**Meaning.** A test where the answer is known to be "no effect", to see if the
method wrongly finds one. SCMLE shuffles the target and re-runs the search.
**Example.** 20 shuffled re-runs found 0 discoveries. The real run found 3.
**Why it matters.** If shuffled data also gives discoveries, the method is
fooling you.

### certificate

**Meaning.** A measured record of how often a check is wrong, tested on data
with planted problems.
**Example.** The leakage screen used a certified detector.
**Why it matters.** Only a check with a passing certificate may block a run. Any
other check is shown but never stops anything ("descriptive").

### gate

**Meaning.** A check that can stop a run or a release.
**Example.** The leakage block.
**Why it matters.** A gate is a decision, so it is only allowed to use checks
whose error rates are known.

### descriptive (versus certified)

**Meaning.** Output that informs but does not decide anything.
**Example.** A shift report that says "unverifiable".
**Why it matters.** Do not treat descriptive output as a guarantee.

### conformal (prediction sets and intervals)

**Meaning.** A method that wraps any model's predictions in an interval or a
set that is right at a promised rate, such as 90 percent.
**Example.** `bundle.predict_interval(rows, coverage=0.9)` for a number;
`bundle.predict_set(rows, coverage=0.9)` for a category.
**Why it matters.** It tells you how unsure each single answer is. The promise
holds when new data looks like the data used for calibration.

### shift, drift

**Meaning.** Shift: the new data differs from the training data. Drift: the
same thing happening over time in production.
**Example.** All fares multiplied by eight.
**Why it matters.** A model is only good for the world it learned.

### PSI (population stability index)

**Meaning.** A number for how much one column's spread has moved. About 0.1 or
less is calm, 0.25 or more is a strong shift.
**Example.** `fare` PSI 6.32 in the changed batch.
**Why it matters.** It is the main drift signal in the monitor.

## Models and decisions

### model

**Meaning.** A recipe that turns features into a prediction, learned from
data.
**Example.** `logistic_regression`, `random_forest`, `hist_gradient_boosting`.
**Why it matters.** SCMLE tries many and keeps the best by cross-validation.

### stacking, blending

**Meaning.** Combining several models into one. SCMLE uses the combination only
if it beats its best single member.
**Example.** The Titanic winner was `stacking`.
**Why it matters.** It often adds a little accuracy at the cost of simplicity.

### hyperparameter, tuning

**Meaning.** A model's setting chosen by you, not learned (such as tree depth).
Tuning searches for good settings.
**Example.** `n_estimators`, `max_samples`.
**Why it matters.** It is why a time budget matters: more search can find better
settings, at the cost of time.

### feature importance (permutation importance)

**Meaning.** How much the score drops when one feature is shuffled.
**Example.** Shuffling `sex` cost 0.242 of AUC.
**Why it matters.** It shows what the model relies on. A surprising top
feature is the first sign of a leak or a bug.

### decision, cost of errors

**Meaning.** The action you take from a prediction and what each kind of wrong
action costs.
**Example.** Missing a fraud costs 8; a false alarm costs 1.
**Why it matters.** It, not the metric, decides the right threshold. See
[Decisions](DECISIONS.md).

### net benefit

**Meaning.** A way to count the value of a decision rule that weighs true hits
against false alarms at your costs.
**Example.** Compared against "treat everyone" and "treat no one".
**Why it matters.** A model that beats both across your plausible costs is
useful; one that does not is not.

### sensitive column

**Meaning.** A column such as sex or age group whose groups you want to compare
for fairness.
**Example.** `--sensitive-columns sex`.
**Why it matters.** A good overall score can hide a poor one for a group.
SCMLE reports the gap (see `fairness.json` and `slice_performance.json`).

## Watching a model in use

### monitoring

**Meaning.** Comparing production data and outcomes with what the model saw in
training.
**Example.** `ModelMonitor.assess`.
**Why it matters.** Models fail quietly. Monitoring is how you hear about it.

### delayed labels

**Meaning.** The true outcome arrives later than the prediction.
**Example.** You learn if a loan defaulted a year after approving it.
**Why it matters.** Performance can only be judged on rows whose outcome has
arrived. The report shows the labeled fraction.

### champion and challenger

**Meaning.** Champion: the model in use. Challenger: a candidate to replace it.
**Example.** v1 champion, v2 challenger.
**Why it matters.** The challenger must beat the champion on the same fresh
rows before it takes over.

### closed loop

**Meaning.** The cycle: watch, raise an alarm, retrain, test the challenger,
promote only if evidence allows, and ask a person when unsure.
**Example.** `scmle loop observe`, `loop step`.
**Why it matters.** It automates the routine and keeps people for the hard
calls. It is certified only in part (see
[What SCMLE cannot do](WHAT_SCMLE_CANNOT_DO.md)).

### escalation

**Meaning.** The system stops and asks a named person to decide.
**Example.** `scmle loop pending` lists open questions; `scmle loop reply`
records the answer.
**Why it matters.** Some moves (for example moving the champion) must not be
automatic. The answer is stored as an `attested` record.

### halt, incident

**Meaning.** A halt stops the loop. An incident is the record of why.
**Example.** The loop starts halted if its state file does not match its
ledger.
**Why it matters.** It is a safe stop that needs a person.

### rollback

**Meaning.** Going back to the previous model.
**Example.** `previous` alias becomes `champion` again.
**Why it matters.** It is your undo. Test it before you need it.

## Shipping and trust

### bundle

**Meaning.** The saved model as a folder: the fitted model, its input schema,
a reference profile for monitoring, and a manifest of file checksums.
**Example.** `artifacts/titanic`.
**Why it matters.** It is the unit you sign, register, serve and roll back.

### schema

**Meaning.** The list of columns and types the model expects.
**Example.** `pclass, sex, age, sibsp, parch, fare, embarked`.
**Why it matters.** Wrong columns are refused, not guessed.

### checksum (digest, hash)

**Meaning.** A short fingerprint of a file's exact contents. Change one byte
and it changes.
**Example.** `artifact_digest: sha256:17d1...`.
**Why it matters.** It proves two copies are identical.

### signature (HMAC)

**Meaning.** A stamp made with a secret key. Anyone with the key can check
that the content is unchanged and was stamped by a key holder.
**Example.** `signature.json` in the bundle.
**Why it matters.** Bundles can run code when opened, so SCMLE refuses an
unsigned one. A signature proves the key, not the person; it is not a public
certificate.

### pickle, trusted

**Meaning.** Pickle is the file format of the model. Opening one can run code.
`--trusted` tells SCMLE "I made this myself; open it anyway".
**Example.** `scmle predict ... --trusted`.
**Why it matters.** Never use `--trusted` on a model from someone else. It is
refused outright when `SCMLE_ENV=production`.

### SBOM (software bill of materials)

**Meaning.** A list of every library and version inside a package.
**Example.** `sbom.cdx.json`.
**Why it matters.** When a security flaw is announced, you can see in seconds
whether you are exposed.

### provenance

**Meaning.** A signed record of where something came from and what it was made
of.
**Example.** `provenance.json` lists the bundle digest and build files.
**Why it matters.** The service checks it at start-up. A package without a
valid one never becomes ready.

### registry

**Meaning.** A shelf of numbered model versions with labels (`champion`,
`previous`, `challenger`).
**Example.** `registry/` with versions 1 and 2.
**Why it matters.** It makes "which model is live" a recorded fact, and moving
the champion label needs a signed approval.

### promotion, approval

**Meaning.** Promotion makes a version the champion. An approval is a signed
note by an independent reviewer bound to one exact model.
**Example.** `PromotionApproval(...).sign(key)`.
**Why it matters.** It stops one person pushing an unreviewed model into use.

### model card

**Meaning.** A short description of a model's intended use, scores and limits.
**Example.** `model_card.json` in every run.
**Why it matters.** The next person should not have to guess what the model is
for.

## Serving and operations

### service, endpoint, liveness, readiness

**Meaning.** A service is a running program that answers requests. An endpoint
is one address on it. Liveness (`/livez`) says it runs. Readiness (`/readyz`)
says it is verified, loaded and can answer.
**Example.** `/predict` returns predictions.
**Why it matters.** A platform restarts a service that is not live and sends
traffic only to one that is ready.

### SLO (service level objective)

**Meaning.** A target for how the service should behave, set before measuring.
**Example.** 95th-percentile latency under 100 ms for one record.
**Why it matters.** It turns "fast enough" into a number you can check. See
[`SERVICE_SLO.md`](SERVICE_SLO.md).

### latency, p95

**Meaning.** The time one request takes. p95 means 95 of every 100 requests are
faster than this.
**Example.** `p95_latency_ms: 29.7` for our model.
**Why it matters.** Averages hide slow outliers that users feel.

### container, image

**Meaning.** An image is a frozen copy of a program and everything it needs. A
container is a running image.
**Example.** Built from `service/Dockerfile`.
**Why it matters.** It runs the same way on every computer.

### canary (staged traffic)

**Meaning.** Sending a small share of traffic to a new version first, watching,
then sending more.
**Example.** 10 percent, then 50, then 100.
**Why it matters.** A bad release hurts few people and can be reversed. The
staged-traffic helpers are library code; no release workflow in this
repository drives them.

### release gate

**Meaning.** The check that allows or refuses a release from its evidence.
**Example.** `scmle deploy gate admit manifest.json --key-file key`.
**Why it matters.** It refuses unsigned or unapproved models and lists every
reason, and exits with code 2.

### deployment record

**Meaning.** A signed record of what was deployed where, when, and who
approved it.
**Example.** `scmle deploy verify-record record.json`.
**Why it matters.** It is your audit trail.

### reconcile

**Meaning.** Comparing what the registry and ledger say with what the runtime
really runs.
**Example.** `scmle deploy reconcile`.
**Why it matters.** It catches "the label says v1 but v2 is running".

### RTO and RPO

**Meaning.** Recovery time objective: how long you may be down. Recovery point
objective: how much recent data you may lose.
**Example.** RTO 30 minutes, RPO 5 minutes.
**Why it matters.** You prove you can recover within them before you claim
production readiness.

### SCMLE_SIGNING_KEY

**Meaning.** The environment variable holding your secret signing key.
**Example.** `export SCMLE_SIGNING_KEY=...`
**Why it matters.** Without it signed bundles cannot be opened. Keep it out of
files and chat. Keys are read from files, not typed on a command line, in the
`deploy` commands.
